from pathlib import Path
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from one.api import ONE
import one.alf.exceptions as alferr

from prior_localization.my_rt import (
    load_wheel_data, calc_wheel_velocity, calc_trialwise_wheel, calc_movement_onset_times
)
from prior_localization.fit_data import fit_session_ephys


# -----------------------------
# User settings
# -----------------------------
ROI_LIST = ["MOp", "MOs", "ACAd", "ORBvl", 'BMA','CEA']
ROI_SET = set(ROI_LIST)

GROUPS = ["fast", "normal", "slow"]

# -----------------------------
# Your existing helpers (unchanged)
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

    # region in file is like ['MOp'] or ['VPM']
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
# Worker: run one eid
# -----------------------------
def run_one_eid_worker(eid: str, out_root_str: str, n_pseudo: int, debug: bool):
    warnings.filterwarnings("ignore", category=alferr.ALFWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    out_root = Path(out_root_str)
    out_root.mkdir(parents=True, exist_ok=True)

    one = ONE()

    probe_name = one.eid2pid(eid)[1]
    ses = one.alyx.rest("sessions", "read", id=eid)
    subject = ses["subject"]

    df, n_raw_trials = compute_trials_with_my_rt(one, eid)

    # QC: >=401 raw trials
    if n_raw_trials < 401:
        return {"eid": eid, "status": "skip_trials", "n_raw_trials": int(n_raw_trials), "rows": []}

    masks, meta = make_fast_normal_slow_masks(df)
    (out_root / f"{eid}_rt_group_counts.json").write_text(json.dumps(meta, indent=2))

    if debug:
        print(f"[DEBUG] {eid}: raw_trials={n_raw_trials} meta={meta}")
        print(f"[DEBUG] {eid}: probe_name={probe_name}")

    pseudo_ids = [-1] + list(range(1, n_pseudo + 1))

    # run groups
    for group_label in GROUPS:
        group_dir = out_root / eid / group_label
        group_dir.mkdir(parents=True, exist_ok=True)

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

            # ROI filtering (NEW)
            roi_set=ROI_SET,
        )

    # After run, collect ROI pkls for this session
    rows = []
    # Find pkls under out_root/eid/*/*/*/*.pkl (group/subject/eid/*.pkl)
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
# Main: parallel + write ONE big file
# -----------------------------
def main():
    out_root = Path("./prior_localization_groups_output")
    out_root.mkdir(parents=True, exist_ok=True)

    # Put your list here
    eids = [
        "56956777-dca5-468c-87cb-78150432cc57",
        "3a3ea015-b5f4-4e8b-b189-9364d1fc7435",
        "d85c454e-8737-4cba-b6ad-b2339429d99b",
        "de905562-31c6-4c31-9ece-3ee87b97eab4",
        "e6594a5b-552c-421a-b376-1a1baa9dc4fd",
        "4e560423-5caf-4cda-8511-d1ab4cd2bf7d",
        # ...
    ]

    n_pseudo = 100
    debug = False
    n_workers = 4  # change based on CPU/RAM

    all_rows = []
    run_log = []

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(run_one_eid_worker, eid, str(out_root), n_pseudo, debug) for eid in eids]
        for fut in as_completed(futs):
            r = fut.result()
            run_log.append({k: r[k] for k in ["eid", "status", "n_raw_trials"]})
            all_rows.extend(r["rows"])
            print("[DONE]", r["eid"], r["status"], "rows=", len(r["rows"]))

    # Save ONE big summary file
    summary_path = out_root / "roi_summary.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(all_rows, f)

    # Optional: also save as parquet for easy viewing (uncomment if you want)
    try:
        df = pd.DataFrame(all_rows)
        df.to_parquet(out_root / "roi_summary.parquet", index=False)
    except Exception:
        pass

    # Save run log (small)
    with open(out_root / "run_log.json", "w") as f:
        json.dump(run_log, f, indent=2)

    print("Saved big ROI summary:", summary_path)


if __name__ == "__main__":
    main()
