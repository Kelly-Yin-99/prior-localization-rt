


from __future__ import annotations

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


ROI_LIST = ["MOp"]
ROI_SET = set(ROI_LIST)
GROUPS = ["fast", "normal", "slow"]

OUT_ROOT = Path("./prior_localization_sessionfit_output")
N_PSEUDO = 200
N_RUNS = 2
N_WORKERS = 4
DEBUG = False

ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.6, -0.1)
FAST_THR = 0.08
SLOW_THR = 1.25

# Exclude last N trials at FIT TIME (applies to real + pseudo)
DROP_LAST_N = 40
MIN_RAW_TRIALS = 401



def compute_trials_with_my_rt(one: ONE, eid: str):
    trials_obj = one.load_object(eid, "trials", collection="alf")
    n_raw_trials = len(trials_obj["stimOn_times"])

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
    _, trial_ts, trial_vel = calc_trialwise_wheel(pos, ts, vel, df["stimOn_times"], df["feedback_times"])
    (_, first_mo, _, _, _) = calc_movement_onset_times(trial_ts, trial_vel, df["stimOn_times"])
    df["first_movement_onset_times"] = first_mo
    df["reaction_time"] = df["first_movement_onset_times"] - df["stimOn_times"]

    return df, int(n_raw_trials)


def make_fast_normal_slow_masks(df: pd.DataFrame, rt_col="reaction_time", fast_thr=FAST_THR, slow_thr=SLOW_THR):
    rt = df[rt_col].to_numpy()
    valid = np.isfinite(rt)
    fast = valid & (rt < fast_thr)
    slow = valid & (rt > slow_thr)
    normal = valid & (~fast) & (~slow)
    return {"fast": fast, "normal": normal, "slow": slow}, {
        "n_total": int(len(df)),
        "n_valid_rt": int(valid.sum()),
        "n_fast": int(fast.sum()),
        "n_normal": int(normal.sum()),
        "n_slow": int(slow.sum()),
        "fast_thr": float(fast_thr),
        "slow_thr": float(slow_thr),
    }


def make_drop_last_mask(n_trials: int, drop_last_n: int):
    m = np.ones(int(n_trials), dtype=bool)
    if drop_last_n and drop_last_n > 0:
        m[max(0, n_trials - drop_last_n):] = False
    return m


def _trial_scalar_list(x_list):
    """list-of-arrays -> scalar per trial (mean over bins). returns vec (n_trials,) with NaNs for bad."""
    out = np.full(len(x_list), np.nan, float)
    for i, xi in enumerate(x_list):
        if xi is None:
            continue
        arr = np.asarray(xi).reshape(-1)
        if arr.size == 0 or (not np.all(np.isfinite(arr))):
            continue
        out[i] = float(np.mean(arr))
    return out


def _pearsonr_safe(a, b):
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]; b = b[ok]
    if a.size < 3:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _r2_safe(y_true, y_pred):
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]; y_pred = y_pred[ok]
    if y_true.size < 3:
        return np.nan
    # if variance is 0, R^2 is undefined (would divide by 0)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return np.nan
    sse = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - sse / denom)


def _z_and_p_from_null(real_val, null_vals, one_sided="greater_equal"):
    """Return z-score and empirical p based on null distribution."""
    null_vals = np.asarray(null_vals, float)
    null_vals = null_vals[np.isfinite(null_vals)]
    if (not np.isfinite(real_val)) or null_vals.size < 5:
        return np.nan, np.nan, np.nan, np.nan

    mu = float(np.mean(null_vals))
    sd = float(np.std(null_vals, ddof=1)) if null_vals.size > 1 else float(np.std(null_vals))
    z = float((real_val - mu) / sd) if sd > 0 else np.nan

    if one_sided == "greater_equal":
        p = float((1 + np.sum(null_vals >= real_val)) / (1 + null_vals.size))
    elif one_sided == "less_equal":
        p = float((1 + np.sum(null_vals <= real_val)) / (1 + null_vals.size))
    else:
        raise ValueError("one_sided must be 'greater_equal' or 'less_equal'")

    return mu, sd, z, p


def _corrected_r2_findling(r2_real, r2_fake_mean):
    """
    Findling-style correction:
        R2_corr = (R2_real - mean(R2_fake)) / (1 - mean(R2_fake))
    """
    if (not np.isfinite(r2_real)) or (not np.isfinite(r2_fake_mean)):
        return np.nan
    denom = 1.0 - float(r2_fake_mean)
    if denom == 0:
        return np.nan
    return float((r2_real - float(r2_fake_mean)) / denom)


def compute_group_stats_from_pkl(region_pkl: Path, group_mask_full: np.ndarray):
    """
    group_mask_full: boolean mask in FULL df trial space (len = len(df))
    Uses d['keep_idx_full'] to map full-trial mask into decoded-trial order.

    Returns:
      n_used,
      pearson: r_real, r_fake, z_r, p_emp_r
      r2: r2_real, r2_fake, z_r2, p_emp_r2, r2_corr
    """
    with open(region_pkl, "rb") as f:
        d = pickle.load(f)

    keep_idx_full = np.asarray(d["keep_idx_full"], dtype=int)

    # Map full-space group mask -> decoded-trial space
    group_mask_sub = np.asarray(group_mask_full, dtype=bool)[keep_idx_full]
    sub_idx = np.flatnonzero(group_mask_sub)
    n_used = int(sub_idx.size)

    # collect per pseudo_id across runs
    pid_to_r = {}
    pid_to_r2 = {}

    for fr in d["fit"]:
        pid = int(fr.get("pseudo_id", -999))
        preds = fr.get("predictions_test", None)
        targ  = fr.get("target", None)
        if preds is None or targ is None:
            continue

        yhat = _trial_scalar_list(preds)
        y    = _trial_scalar_list(targ)

        y_sub = y[sub_idx]
        yhat_sub = yhat[sub_idx]

        r = _pearsonr_safe(yhat_sub, y_sub)
        r2 = _r2_safe(y_sub, yhat_sub)

        if np.isfinite(r):
            pid_to_r.setdefault(pid, []).append(r)
        if np.isfinite(r2):
            pid_to_r2.setdefault(pid, []).append(r2)

    def mean_or_nan(x):
        return float(np.mean(x)) if len(x) else np.nan

    # REAL
    r_real = mean_or_nan(pid_to_r.get(-1, []))
    r2_real = mean_or_nan(pid_to_r2.get(-1, []))

    # FAKE arrays (mean across runs for each pseudo)
    fake_ids_r = sorted([pid for pid in pid_to_r.keys() if pid != -1])
    r_fake = np.asarray([mean_or_nan(pid_to_r[pid]) for pid in fake_ids_r], float)
    r_fake = r_fake[np.isfinite(r_fake)]

    fake_ids_r2 = sorted([pid for pid in pid_to_r2.keys() if pid != -1])
    r2_fake = np.asarray([mean_or_nan(pid_to_r2[pid]) for pid in fake_ids_r2], float)
    r2_fake = r2_fake[np.isfinite(r2_fake)]

    # z + p for pearson
    r_fake_mean, r_fake_sd, z_r, p_emp_r = _z_and_p_from_null(r_real, r_fake, one_sided="greater_equal")

    # z + p for R2
    r2_fake_mean, r2_fake_sd, z_r2, p_emp_r2 = _z_and_p_from_null(r2_real, r2_fake, one_sided="greater_equal")

    # corrected R2
    r2_corr = _corrected_r2_findling(r2_real, r2_fake_mean)

    return {
        "n_used": n_used,
        "r_real": r_real,
        "r_fake": r_fake,
        "r_fake_mean": r_fake_mean,
        "r_fake_std": r_fake_sd,
        "z_corr": z_r,
        "p_emp": p_emp_r,
        "r2_real": r2_real,
        "r2_fake": r2_fake,
        "r2_fake_mean": r2_fake_mean,
        "r2_fake_std": r2_fake_sd,
        "z_r2": z_r2,
        "p_emp_r2": p_emp_r2,
        "r2_corr": r2_corr,
    }


def find_roi_pkl(session_dir: Path, roi: str):
    for p in sorted(session_dir.rglob("*.pkl")):
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
            reg = d.get("region", "")
            reg = reg[0] if isinstance(reg, (list, tuple)) else str(reg)
            if str(reg).strip() == str(roi).strip():
                return p
        except Exception:
            continue
    return None



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

    if n_raw_trials < MIN_RAW_TRIALS:
        return {"eid": eid, "status": "skip_trials", "n_raw_trials": n_raw_trials, "rows": []}

    # Build RT masks in FULL trial space
    masks, meta = make_fast_normal_slow_masks(df)

    # Fit-time drop last N trials (real + pseudo)
    drop_last_mask = make_drop_last_mask(len(df), DROP_LAST_N)

    meta2 = dict(meta)
    meta2["drop_last_n"] = int(DROP_LAST_N)
    meta2["n_after_drop_last"] = int(drop_last_mask.sum())
    meta2["n_fast_after_drop_last"] = int((masks["fast"] & drop_last_mask).sum())
    meta2["n_normal_after_drop_last"] = int((masks["normal"] & drop_last_mask).sum())
    meta2["n_slow_after_drop_last"] = int((masks["slow"] & drop_last_mask).sum())
    (out_root / f"{eid}_rt_group_counts.json").write_text(json.dumps(meta2, indent=2))

    pseudo_ids = [-1] + list(range(1, n_pseudo + 1))

    # FIT ONCE (restricted trial set)
    session_dir = out_root / eid / "session_fit"
    session_dir.mkdir(parents=True, exist_ok=True)

    _ = fit_session_ephys(
        one=one,
        session_id=eid,
        subject=subject,
        probe_name=probe_name,
        output_dir=session_dir,
        pseudo_ids=pseudo_ids,
        target="pLeft",
        align_event=ALIGN_EVENT,
        time_window=TIME_WINDOW,
        model="optBay",
        n_runs=N_RUNS,
        trials_df=df,
        trial_mask=drop_last_mask,
        group_label="session",
        debug=bool(debug),
        roi_set=ROI_SET,
    )

    roi = ROI_LIST[0]
    roi_pkl = find_roi_pkl(session_dir, roi)
    if roi_pkl is None:
        return {"eid": eid, "status": "skip_no_roi", "n_raw_trials": n_raw_trials, "rows": []}

    # IMPORTANT: do NOT drop last 40 again here.
    rows = []
    for g in GROUPS:
        stats = compute_group_stats_from_pkl(roi_pkl, group_mask_full=masks[g])

        rows.append({
            "eid": eid,
            "subject": subject,
            "probe": probe_name if isinstance(probe_name, str) else str(probe_name),
            "roi": roi,
            "group": g,
            "n_trials_group_used": int(stats["n_used"]),

            # Pearson
            "r_real": stats["r_real"],
            "r_fake_mean": stats["r_fake_mean"],
            "r_fake_std": stats["r_fake_std"],
            "z_corr": stats["z_corr"],
            "p_emp": stats["p_emp"],

            # R^2
            "r2_real": stats["r2_real"],
            "r2_fake_mean": stats["r2_fake_mean"],
            "r2_fake_std": stats["r2_fake_std"],
            "r2_corr": stats["r2_corr"],
            "z_r2": stats["z_r2"],
            "p_emp_r2": stats["p_emp_r2"],

            # bookkeeping
            "drop_last_n": int(DROP_LAST_N),
            "roi_pkl_path": str(roi_pkl),

            # (optional) keep full nulls if you want:
            # "r_fake": stats["r_fake"],
            # "r2_fake": stats["r2_fake"],
        })

    (out_root / eid / f"pearson_summary_{roi}.json").write_text(
        json.dumps(rows, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    )

    return {"eid": eid, "status": "ok", "n_raw_trials": n_raw_trials, "rows": rows}


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    eids = [
        "ae8787b1-4229-4d56-b0c2-566b61a25b77",
        # add more
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

    roi_tag = "_".join(ROI_LIST) if ROI_LIST else "ALL"
    summary_path = OUT_ROOT / f"pearson_summary_{roi_tag}_2.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(all_rows, f)

    run_log_path = OUT_ROOT / f"run_log_{roi_tag}_2.json"
    with open(run_log_path, "w") as f:
        json.dump(run_log, f, indent=2)

    print("Saved pearson summary:", summary_path)
    print("Saved run log:", run_log_path)


if __name__ == "__main__":
    main()
