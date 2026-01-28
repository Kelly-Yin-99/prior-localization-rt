


from __future__ import annotations

from pathlib import Path
import os
import json
import pickle
import warnings
import contextlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from one.api import ONE
import one.alf.exceptions as alferr

from prior_localization.my_rt import (
    load_wheel_data, calc_wheel_velocity, calc_trialwise_wheel, calc_movement_onset_times
)
from prior_localization.fit_data import fit_session_ephys



ROI_LIST = ["MOs"]          # plot this ROI
ROI_SET = set(ROI_LIST)
GROUPS = ["fast", "normal", "slow"]

OUT_ROOT = Path("./prior_localization_groups_output")
N_PSEUDO = 200
N_RUNS = 2
N_WORKERS = 4
DEBUG = False

ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.6, -0.1)



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
    _, trial_ts, trial_vel = calc_trialwise_wheel(
        pos, ts, vel, df["stimOn_times"], df["feedback_times"]
    )

    (_, first_mo, _, _, _) = calc_movement_onset_times(trial_ts, trial_vel, df["stimOn_times"])
    df["first_movement_onset_times"] = first_mo
    df["reaction_time"] = df["first_movement_onset_times"] - df["stimOn_times"]

    return df, n_raw_trials


def make_fast_normal_slow_masks(
    df: pd.DataFrame,
    rt_col: str = "reaction_time",
    fast_thr: float = 0.08,
    slow_thr: float = 1.25,
):
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
        "fast_thr": float(fast_thr),
        "slow_thr": float(slow_thr),
    }
    return masks, meta


# ============================================================
# 2) Plot decoded prior vs target (from region pkl only)
# ============================================================
def plot_decoded_prior_vs_target(region_pkl_path: Path, save_path: Path, title: str, run_idx: int = 0):
    with open(region_pkl_path, "rb") as f:
        d = pickle.load(f)

    fits_real = [fit for fit in d["fit"] if fit.get("pseudo_id", None) == -1]
    if len(fits_real) == 0:
        raise ValueError(f"No pseudo_id=-1 fits in {region_pkl_path}")

    preds = np.asarray([fit["predictions_test"] for fit in fits_real]).squeeze()
    targ  = np.asarray([fit["target"] for fit in fits_real]).squeeze()

    # Choose a run
    if preds.ndim == 1:
        yhat = preds
        y = targ
    else:
        run_idx = int(np.clip(run_idx, 0, preds.shape[0] - 1))
        yhat = preds[run_idx, :]
        y = targ[run_idx, :]

    plt.figure(figsize=(12, 4))
    plt.plot(yhat, label="decoded prior", linewidth=2)
    plt.plot(y, label="target (pLeft)", linewidth=2)
    plt.xlabel("trial")
    plt.ylabel("pLeft")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# 3) Find spikes/clusters collection robustly
# ============================================================
def load_spikes_and_clusters_any_collection(one: ONE, eid: str, prefer_probe: str | None = None):
    """
    Try to find a collection that contains spikes+clusters.
    prefer_probe can be 'probe00'/'probe01'/etc to prioritize those collections.
    Returns (spikes, clusters, collection_used).
    """
    cols = one.list_collections(eid)
    cols = [c for c in cols if "alf" in c]

    preferred: list[str] = []

    if prefer_probe:
        preferred += [
            f"alf/{prefer_probe}",
            f"alf/{prefer_probe}/pykilosort",
            f"alf/{prefer_probe}/ks2",
            f"alf/{prefer_probe}/kilosort",
            f"alf/{prefer_probe}/kilosort2",
            f"alf/{prefer_probe}/sorting",
        ]

    for c in cols:
        if "alf/probe" in c and c not in preferred:
            preferred.append(c)

    tried = []
    for c in preferred + cols:
        if c in tried:
            continue
        tried.append(c)
        try:
            spikes = one.load_object(eid, "spikes", collection=c)
            clusters = one.load_object(eid, "clusters", collection=c)
            return spikes, clusters, c
        except Exception:
            continue

    raise FileNotFoundError(
        f"[FR] Could not load spikes/clusters for eid={eid}. Tried {len(tried)} collections."
    )


# ============================================================
# 4) Trialwise mean FR from region pkl (supports saved ids = ints OR uuids)
# ============================================================
def compute_trialwise_mean_fr_from_region_pkl(
    one: ONE,
    eid: str,
    region_pkl_path: Path,
    stim_on_times: np.ndarray,
    t0: float = -0.6,
    t1: float = -0.1,
):
    """
    Mean FR per trial (spikes/unit/s) computed from the SAME units used by decoding.

    Supports BOTH cases:
      (A) saved field (d['cluster_uuids']) actually contains INT cluster IDs -> match spikes['clusters']
      (B) saved field contains UUIDs -> match clusters['uuids'] (or 'uuid')

    Returns: fr (n_trials,), n_units_used, collection_used
    """
    with open(region_pkl_path, "rb") as f:
        d = pickle.load(f)

    saved = d.get("cluster_uuids", None)
    if saved is None or len(saved) == 0:
        raise ValueError(f"[FR] region pkl has no cluster_uuids: {region_pkl_path}")

    saved = np.asarray(saved)

    probe_str = str(d.get("probe", "")).strip()
    prefer_probe = probe_str if probe_str.startswith("probe") else None

    spikes, clusters, col_used = load_spikes_and_clusters_any_collection(one, eid, prefer_probe=prefer_probe)

    st = np.asarray(spikes["times"])
    sc = np.asarray(spikes["clusters"])

    # ---------- Strategy A: saved are INT cluster ids ----------
    keep_cluster_ids = None
    try:
        saved_int = saved.astype(np.int64)
        # Check overlap with spikes cluster ids
        if np.intersect1d(np.unique(saved_int), np.unique(sc)).size > 0:
            keep_cluster_ids = np.unique(saved_int)
    except Exception:
        keep_cluster_ids = None

    if keep_cluster_ids is not None:
        keep_spike_mask = np.isin(sc, keep_cluster_ids)
        st2 = st[keep_spike_mask]
        sc2 = sc[keep_spike_mask]

        win_len = float(t1 - t0)
        fr = np.zeros(len(stim_on_times), dtype=float)
        for i, tstim in enumerate(stim_on_times):
            a = tstim + t0
            b = tstim + t1
            m = (st2 >= a) & (st2 < b)
            if not np.any(m):
                fr[i] = 0.0
                continue
            sc_win = sc2[m]
            counts = np.array([(sc_win == cid).sum() for cid in keep_cluster_ids], dtype=float)
            fr[i] = counts.mean() / win_len

        return fr, int(len(keep_cluster_ids)), col_used

    # ---------- Strategy B: saved are UUIDs ----------
    if "uuids" in clusters:
        clu_uuid = np.asarray(clusters["uuids"])
    elif "uuid" in clusters:
        clu_uuid = np.asarray(clusters["uuid"])
    else:
        raise KeyError(f"[FR] clusters missing uuid field. keys={list(clusters.keys())}")

    def _to_str_array(x):
        x = np.asarray(x)
        out = []
        for v in x:
            if isinstance(v, (bytes, np.bytes_)):
                try:
                    out.append(v.decode("utf-8"))
                except Exception:
                    out.append(str(v))
            else:
                out.append(str(v))
        return np.asarray(out, dtype=object)

    saved_str = _to_str_array(saved)
    clu_str = _to_str_array(clu_uuid)

    keep_cluster_ids = np.flatnonzero(np.isin(clu_str, saved_str))
    if keep_cluster_ids.size == 0:
        raise ValueError(
            f"[FR] Could not match saved unit ids to either spikes['clusters'] or clusters uuids. "
            f"collection={col_used}. Hint: saved ids likely come from a different collection."
        )

    keep_spike_mask = np.isin(sc, keep_cluster_ids)
    st2 = st[keep_spike_mask]
    sc2 = sc[keep_spike_mask]

    win_len = float(t1 - t0)
    fr = np.zeros(len(stim_on_times), dtype=float)
    for i, tstim in enumerate(stim_on_times):
        a = tstim + t0
        b = tstim + t1
        m = (st2 >= a) & (st2 < b)
        if not np.any(m):
            fr[i] = 0.0
            continue
        sc_win = sc2[m]
        counts = np.array([(sc_win == cid).sum() for cid in keep_cluster_ids], dtype=float)
        fr[i] = counts.mean() / win_len

    return fr, int(len(keep_cluster_ids)), col_used
def compute_trialwise_mean_fr_from_region_pkl(one, eid, region_pkl_path,
                                              stim_on_times, t0=-0.6, t1=-0.1):
    """
    Mean FR per trial computed from the SAME units used by decoding.
    Uses d["cluster_ids"] which MUST match spikes['clusters'] in the collection.
    """
    import numpy as np
    import pickle

    with open(region_pkl_path, "rb") as f:
        d = pickle.load(f)

    if "cluster_ids" not in d:
        raise ValueError(
            f"[FR] region pkl missing 'cluster_ids'. "
            f"Your pkls were generated before the prepare_ephys fix: {region_pkl_path}"
        )

    keep_cluster_ids = np.asarray(d["cluster_ids"], dtype=int)
    if keep_cluster_ids.size == 0:
        raise ValueError(f"[FR] Empty cluster_ids in {region_pkl_path}")

    # Prefer probe if available
    probe_str = str(d.get("probe", "")).strip()
    prefer_probe = probe_str if probe_str.startswith("probe") else None

    spikes, clusters, col_used = load_spikes_and_clusters_any_collection(one, eid, prefer_probe=prefer_probe)

    st = np.asarray(spikes["times"])
    sc = np.asarray(spikes["clusters"]).astype(int)

    # sanity: ensure overlap
    if np.intersect1d(keep_cluster_ids, np.unique(sc)).size == 0:
        raise ValueError(
            f"[FR] cluster_ids from pkl do not overlap spikes['clusters'] in collection={col_used}. "
            f"Likely you loaded the wrong collection OR the session uses a different probe collection."
        )

    # Keep spikes from these units only
    keep_spike_mask = np.isin(sc, keep_cluster_ids)
    st2 = st[keep_spike_mask]
    sc2 = sc[keep_spike_mask]

    win_len = float(t1 - t0)
    fr = np.zeros(len(stim_on_times), dtype=float)

    # compute mean spikes/unit/s per trial
    for i, tstim in enumerate(stim_on_times):
        a = tstim + t0
        b = tstim + t1
        m = (st2 >= a) & (st2 < b)
        if not np.any(m):
            fr[i] = 0.0
            continue
        sc_win = sc2[m]
        counts = np.array([(sc_win == cid).sum() for cid in keep_cluster_ids], dtype=float)
        fr[i] = counts.mean() / win_len

    return fr, int(len(keep_cluster_ids)), col_used


def plot_fr_across_trials(fr: np.ndarray, save_path: Path, mask: np.ndarray | None, title: str):
    x = np.arange(len(fr))
    ok = np.isfinite(fr)
    if mask is not None:
        ok = ok & np.asarray(mask, dtype=bool)

    plt.figure(figsize=(12, 4))
    plt.plot(x[ok], fr[ok], ".", markersize=3)
    plt.xlabel("trial")
    plt.ylabel("mean firing rate (spikes / unit / s)")
    plt.title(title)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# 5) Summary helper
# ============================================================
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


# ============================================================
# 6) Worker: run decoding + make plots
# ============================================================
def run_one_eid_worker(eid: str, out_root_str: str, n_pseudo: int, debug: bool):
    warnings.filterwarnings("ignore", category=alferr.ALFWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    out_root = Path(out_root_str)
    out_root.mkdir(parents=True, exist_ok=True)

    one = ONE()

    # metadata
    probe_name = one.eid2pid(eid)[1]
    ses = one.alyx.rest("sessions", "read", id=eid)
    subject = ses["subject"]

    # trials + rt
    df, n_raw_trials = compute_trials_with_my_rt(one, eid)

    if n_raw_trials < 401:
        return {"eid": eid, "status": "skip_trials", "n_raw_trials": int(n_raw_trials), "rows": []}

    masks, meta = make_fast_normal_slow_masks(df)
    (out_root / f"{eid}_rt_group_counts.json").write_text(json.dumps(meta, indent=2))

    if debug:
        print(f"[DEBUG] {eid}: raw_trials={n_raw_trials} meta={meta}")
        print(f"[DEBUG] {eid}: probe_name={probe_name}")

    pseudo_ids = [-1] + list(range(1, n_pseudo + 1))
    target_roi = ROI_LIST[0]

    # -------------------------
    # Run decoding per group
    # -------------------------
    for group_label in GROUPS:
        group_dir = out_root / eid / group_label
        group_dir.mkdir(parents=True, exist_ok=True)

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            _ = fit_session_ephys(
                one=one,
                session_id=eid,
                subject=subject,
                probe_name=probe_name,
                output_dir=group_dir,
                pseudo_ids=pseudo_ids,
                target="pLeft",
                align_event=ALIGN_EVENT,
                time_window=TIME_WINDOW,
                model="optBay",
                n_runs=N_RUNS,
                min_rt=None,
                max_rt=None,
                trials_df=df,
                trial_mask=masks[group_label],
                group_label=group_label,
                debug=False,
                roi_set=ROI_SET,
            )

    # -------------------------
    # Make plots per group
    # -------------------------
    stim_on = df[ALIGN_EVENT].to_numpy()

    def _norm_region(reg):
        if isinstance(reg, (list, tuple, np.ndarray)) and len(reg) > 0:
            reg = reg[0]
        return str(reg).strip()

    for group_label in GROUPS:
        group_dir = out_root / eid / group_label
        if not group_dir.exists():
            continue

        # Find region pkl (recursive, robust)
        pkl_candidates = sorted(list(group_dir.rglob("*.pkl")))
        roi_pkl_path = None
        for p in pkl_candidates:
            try:
                with open(p, "rb") as f:
                    d = pickle.load(f)
                if _norm_region(d.get("region", "")) == target_roi:
                    roi_pkl_path = p
                    break
            except Exception:
                continue

        if roi_pkl_path is None:
            if debug:
                print(f"[PLOT] No ROI='{target_roi}' pkl under {group_dir}")
            continue

        # 1) decoded prior vs target
        try:
            out1 = group_dir / f"{target_roi}_decodedPrior_vs_target.png"
            plot_decoded_prior_vs_target(
                roi_pkl_path,
                out1,
                title=f"{subject} | {eid} | {target_roi} | {group_label}",
                run_idx=0,
            )
            if debug:
                print(f"[PLOT] Saved decoded plot: {out1}")
        except Exception as e:
            print(f"[PLOT ERROR] decoded plot failed for {eid} {group_label}: {e}")

        # 2) mean FR across trials
        # NOTE: this can still fail if the saved unit-ids don't match any local spikes/clusters collection.
        try:
            fr, n_units_used, col_used = compute_trialwise_mean_fr_from_region_pkl(
                one=one,
                eid=eid,
                region_pkl_path=roi_pkl_path,
                stim_on_times=stim_on,
                t0=TIME_WINDOW[0],
                t1=TIME_WINDOW[1],
            )
            out2 = group_dir / f"{target_roi}_meanFR_across_trials.png"
            plot_fr_across_trials(
                fr,
                out2,
                mask=masks[group_label],
                title=f"{subject} | {eid} | {target_roi} | {group_label} | units={n_units_used} | {col_used}",
            )
            if debug:
                print(f"[PLOT] Saved FR plot: {out2}")
        except Exception as e:
            print(f"[PLOT ERROR] FR plot failed for {eid} {group_label}: {e}")

    # -------------------------
    # Collect ROI pkls for summary
    # -------------------------
    rows = []
    for pkl_path in (out_root / eid).rglob("*.pkl"):
        try:
            row = summarize_region_pkl(pkl_path)
            if row["region"] in ROI_SET:
                rows.append(row)
        except Exception:
            continue

    status = "ok" if rows else "skip_no_roi_or_failed"
    return {"eid": eid, "status": status, "n_raw_trials": int(n_raw_trials), "rows": rows}



def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    eids = [
        "2c44a360-5a56-4971-8009-f469fb59de98"
        # add more eids here
    ]

    all_rows = []
    run_log = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(run_one_eid_worker, eid, str(OUT_ROOT), N_PSEUDO, DEBUG) for eid in eids]
        for fut in as_completed(futs):
            r = fut.result()
            run_log.append({k: r.get(k) for k in ["eid", "status", "n_raw_trials"]})
            all_rows.extend(r.get("rows", []))
            print("[DONE]", r.get("eid"), r.get("status"), "rows=", len(r.get("rows", [])))

    roi_tag = "ALL" if not ROI_LIST else "_".join(ROI_LIST)

    summary_path = OUT_ROOT / f"roi_summary_{roi_tag}.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(all_rows, f)

    try:
        df_all = pd.DataFrame(all_rows)
        df_all.to_parquet(OUT_ROOT / f"roi_summary_{roi_tag}.parquet", index=False)
    except Exception:
        pass

    run_log_path = OUT_ROOT / f"run_log_{roi_tag}.json"
    with open(run_log_path, "w") as f:
        json.dump(run_log, f, indent=2)

    print("Saved big ROI summary:", summary_path)
    print("Saved run log:", run_log_path)
    print("\nPlots (if created) are under:")
    for eid in eids:
        for g in GROUPS:
            print(OUT_ROOT.resolve() / eid / g)


if __name__ == "__main__":
    main()
