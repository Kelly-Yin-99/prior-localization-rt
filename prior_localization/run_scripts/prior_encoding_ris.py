from __future__ import annotations

from pathlib import Path
import json
import pickle
import warnings
import os
import ast
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from one.api import ONE

from prior_localization.my_rt import (
    load_wheel_data, calc_wheel_velocity, calc_trialwise_wheel, calc_movement_onset_times
)
from prior_localization.fit_data import fit_session_ephys


# -----------------------------
# USER SETTINGS
# -----------------------------
ROI_LIST = ["MOp", "MOs", "ACAd", "ORBvl"]
ROI_SET = set(ROI_LIST)

GROUPS = ["fast", "normal", "slow"]

# Input text on RIS
TXT_PATH = "/home/wg-yin/session_ids_for_behav_analysis.txt"

# Outputs on shared filesystem
OUT_ROOT = Path("/storage1/fs1/hiratani/Active/shared/ibl_space/derived/prior_localization_groups_output")

# Shared ONE cache root (we will create per-worker subfolders inside this)
CACHE_ROOT = Path("/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org")


# -----------------------------
# ONE setup for RIS
# -----------------------------
def make_one(cache_dir: Path) -> ONE:
    """
    Create ONE with:
    - shared cache directory (but per-worker subfolder to avoid parquet table race/corruption)
    - Alyx credentials from env: ALYX_LOGIN / ALYX_PASSWORD

    IMPORTANT: cache_tables=False prevents ONE from loading/writing global cache tables
    (the thing that caused datasets.pqt + snappy corruption errors).
    """
    # ONE.setup writes a params json; do it in the per-worker dir
    ONE.setup(
        base_url="https://openalyx.internationalbrainlab.org",
        silent=True,
        cache_dir=cache_dir,
    )
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        username=os.getenv("ALYX_LOGIN"),
        password=os.getenv("ALYX_PASSWORD"),
        silent=True,
        cache_dir=cache_dir,
        cache_tables=False,  # <<< critical for your errors
    )
    return one


# -----------------------------
# YOUR EXISTING HELPERS (unchanged)
# -----------------------------
def compute_trials_with_my_rt(one: ONE, eid: str):
    trials_obj = one.load_object(eid, "trials", collection="alf")
    n_raw_trials = len(trials_obj["stimOn_times"])  # BEFORE any dropna

    data = {}
    for k, v in trials_obj.items():
        arr = np.asarray(v)
        if k == "intervals" and arr.ndim == 2 and arr.shape[1] == 2:
            data["intervals_start"] = arr[:, 0]
            data["intervals_end"] = arr[:, 1]
        elif arr.ndim == 1:
            data[k] = arr
        else:
            data[k] = list(arr)

    df = pd.DataFrame(data)

    pos, ts = load_wheel_data(one, eid)
    if pos is None or ts is None:
        raise RuntimeError(f"No wheel data for eid={eid}")

    vel = calc_wheel_velocity(pos, ts)
    trial_pos, trial_ts, trial_vel = calc_trialwise_wheel(
        pos, ts, vel, df["stimOn_times"], df["feedback_times"]
    )

    (_, first_mo, _, _, _) = calc_movement_onset_times(trial_ts, trial_vel, df["stimOn_times"])
    df["first_movement_onset_times"] = first_mo
    df["reaction_time"] = df["first_movement_onset_times"] - df["stimOn_times"]

    return df, n_raw_trials


def make_fast_normal_slow_masks(df: pd.DataFrame, rt_col="reaction_time",
                               fast_thr=0.08, slow_thr=1.25):
    rt = df[rt_col].to_numpy()
    valid = np.isfinite(rt)

    fast = valid & (rt < fast_thr)
    slow = valid & (rt > slow_thr)
    normal = valid & (~fast) & (~slow)

    masks = {"fast": fast, "normal": normal, "slow": slow}
    meta = {
        "n_total": int(len(df)),
        "n_valid_rt": int(valid.sum()),
        "n_fast": int(fast.sum()),
        "n_normal": int(normal.sum()),
        "n_slow": int(slow.sum()),
    }
    return masks, meta


# -----------------------------
# Helper: extract R2 summary from a saved pkl
# -----------------------------
def summarize_region_pkl(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    fit = d["fit"]
    eid = d["eid"]
    subject = d["subject"]
    group = d.get("group_label", "unknown")
    probe = d.get("probe", "unknown")

    region = d["region"][0] if isinstance(d["region"], (list, tuple)) else str(d["region"])
    n_units = int(d["N_units"])

    session_scores = [fr["scores_test_full"] for fr in fit if fr.get("pseudo_id", None) == -1]
    pseudo_scores = [fr["scores_test_full"] for fr in fit if fr.get("pseudo_id", None) != -1]

    r2_unc = float(np.mean(session_scores)) if len(session_scores) else np.nan
    r2_pseudo = float(np.mean(pseudo_scores)) if len(pseudo_scores) else np.nan
    r2_corr = float(r2_unc - r2_pseudo) if np.isfinite(r2_unc) and np.isfinite(r2_pseudo) else np.nan

    return {
        "eid": eid,
        "subject": subject,
        "group": group,
        "probe": probe,
        "region": region,
        "N_units": n_units,
        "R2_uncorrected": r2_unc,
        "R2_pseudo_mean": r2_pseudo,
        "R2_corrected": r2_corr,
        "pkl_path": str(pkl_path),
    }


# -----------------------------
# Read EIDs
# -----------------------------
def read_session_ids_from_txt(txt_path: str) -> list[str]:
    session_ids: list[str] = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # supports lines like: ['eid1','eid2'] or just eid string
            if line.startswith("["):
                ids_in_line = ast.literal_eval(line)
                session_ids.extend([str(x) for x in ids_in_line])
            else:
                session_ids.append(line)

    # unique, preserve order
    session_ids = list(dict.fromkeys(session_ids))
    return session_ids


# -----------------------------
# Worker: run one eid
# -----------------------------
def run_one_eid_worker(eid: str, out_root_str: str, cache_root_str: str, n_pseudo: int, debug: bool):
    # Hide noisy warnings without relying on ONE's ALFWarning class
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=r".*Multiple revisions.*")
    warnings.filterwarnings("ignore", message=r".*Newer cache tables require ONE version.*")

    out_root = Path(out_root_str)
    out_root.mkdir(parents=True, exist_ok=True)

    cache_root = Path(cache_root_str)
    cache_root.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: per-worker cache dir to avoid concurrent cache table corruption
    pid = os.getpid()
    worker_cache = cache_root / f"worker_{pid}"
    worker_cache.mkdir(parents=True, exist_ok=True)

    one = make_one(cache_dir=worker_cache)

    try:
        probe_name = one.eid2pid(eid)[1]  # can be list -> merged_probes downstream
        ses = one.alyx.rest("sessions", "read", id=eid)
        subject = ses["subject"]
    except Exception as e:
        return {"eid": eid, "status": f"fail_one_setup: {e}", "n_raw_trials": -1, "rows": []}

    try:
        df, n_raw_trials = compute_trials_with_my_rt(one, eid)
    except Exception as e:
        return {"eid": eid, "status": f"fail_trials_or_wheel: {e}", "n_raw_trials": -1, "rows": []}

    # QC: >=401 raw trials
    if n_raw_trials < 401:
        return {"eid": eid, "status": "skip_trials", "n_raw_trials": int(n_raw_trials), "rows": []}

    masks, meta = make_fast_normal_slow_masks(df)
    (out_root / f"{eid}_rt_group_counts.json").write_text(json.dumps(meta, indent=2))

    if debug:
        print(f"[DEBUG] {eid}: raw_trials={n_raw_trials} meta={meta}")
        print(f"[DEBUG] {eid}: probe_name={probe_name}")
        print(f"[DEBUG] {eid}: worker_cache={worker_cache}")

    pseudo_ids = [-1] + list(range(1, n_pseudo + 1))

    # run groups
    for group_label in GROUPS:
        group_dir = out_root / eid / group_label
        group_dir.mkdir(parents=True, exist_ok=True)

        try:
            _ = fit_session_ephys(
                one=one,
                session_id=eid,
                subject=subject,
                probe_name=probe_name,
                output_dir=group_dir,
                pseudo_ids=pseudo_ids,
                target="pLeft",
                align_event="stimOn_times",
                time_window=(-0.6, -0.1),
                model="optBay",
                n_runs=2,
                min_rt=None,
                max_rt=None,
                trials_df=df,
                trial_mask=masks[group_label],
                group_label=group_label,
                debug=debug,
                # requires your patched fit_session_ephys
                roi_set=ROI_SET,
            )
        except Exception as e:
            # don't crash entire job because one group failed
            if debug:
                print(f"[DEBUG] {eid}: group {group_label} failed: {e}")

    # collect ROI pkls for this session
    rows = []
    for pkl_path in (out_root / eid).rglob("*.pkl"):
        try:
            row = summarize_region_pkl(pkl_path)
            if row["region"] in ROI_SET:
                rows.append(row)
        except Exception:
            continue

    status = "ok" if len(rows) else "skip_no_roi_or_failed"
    return {"eid": eid, "status": status, "n_raw_trials": int(n_raw_trials), "rows": rows}


# -----------------------------
# Main
# -----------------------------
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    eids = read_session_ids_from_txt(TXT_PATH)
    n_pseudo = int(os.getenv("N_PSEUDO", "100"))
    debug = bool(int(os.getenv("DEBUG", "0")))

    # Keep this modest (IO + ONE downloads will bottleneck)
    n_workers = int(os.getenv("N_WORKERS", "20"))

    all_rows = []
    run_log = []

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [
            ex.submit(run_one_eid_worker, eid, str(OUT_ROOT), str(CACHE_ROOT), n_pseudo, debug)
            for eid in eids
        ]
        for fut in as_completed(futs):
            r = fut.result()
            run_log.append({k: r.get(k) for k in ["eid", "status", "n_raw_trials"]})
            all_rows.extend(r.get("rows", []))
            print("[DONE]", r["eid"], r["status"], "rows=", len(r.get("rows", [])))

    # Save ONE big summary file
    summary_path = OUT_ROOT / "roi_summary.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(all_rows, f)

    # Save run log
    with open(OUT_ROOT / "run_log.json", "w") as f:
        json.dump(run_log, f, indent=2)

    print("Saved big ROI summary:", summary_path)


if __name__ == "__main__":
    main()
