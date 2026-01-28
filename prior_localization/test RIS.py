from __future__ import annotations

from pathlib import Path
import json
import pickle
import warnings
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from one.api import ONE
import one as one_pkg
import one.alf.exceptions as alferr

from prior_localization.my_rt import (
    load_wheel_data, calc_wheel_velocity, calc_trialwise_wheel, calc_movement_onset_times
)
from prior_localization.fit_data import fit_session_ephys


# =========================
# "RIS-like" determinism knobs
# =========================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# =========================
# USER SETTINGS (match RIS)
# =========================
ROI_LIST = ["MOp"]
ROI_SET = set(ROI_LIST)
GROUPS = ["fast", "normal", "slow"]

RUN_TAG = os.getenv("RUN_TAG", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))

# LOCAL outputs
OUT_BASE = Path("./_local_ris_mirror_outputs").resolve()
OUT_ROOT = OUT_BASE / f"run_{RUN_TAG}"

# LOCAL cache root (per-worker subfolders created here)
CACHE_ROOT = Path("./_local_ris_mirror_ONE_cache").resolve()

N_PSEUDO = int(os.getenv("N_PSEUDO", "200"))
N_RUNS = int(os.getenv("N_RUNS", "2"))
N_WORKERS = int(os.getenv("N_WORKERS", "4"))
DEBUG = bool(int(os.getenv("DEBUG", "0")))

ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.6, -0.1)
FAST_THR = float(os.getenv("FAST_THR", "0.08"))
SLOW_THR = float(os.getenv("SLOW_THR", "1.25"))

DROP_LAST_N = int(os.getenv("DROP_LAST_N", "40"))
MIN_RAW_TRIALS = int(os.getenv("MIN_RAW_TRIALS", "401"))


# =========================
# ONE helper (RIS-style per-worker cache dirs)
# =========================
def make_one(worker_cache: Path) -> ONE:
    worker_cache = Path(worker_cache)
    tables_dir = worker_cache / "tables"
    dl_cache = worker_cache / "downloads"
    tables_dir.mkdir(parents=True, exist_ok=True)
    dl_cache.mkdir(parents=True, exist_ok=True)

    # If you already have ONE configured locally, you can remove username/password
    # and just do ONE(cache_dir=..., tables_dir=..., silent=True).
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        username=os.getenv("ALYX_LOGIN"),
        password=os.getenv("ALYX_PASSWORD"),
        silent=True,
        cache_dir=dl_cache,
        tables_dir=tables_dir,
        cache_rest=None,
    )
    return one


# =========================
# Trial + RT helpers (same as your RIS version)
# =========================
def compute_trials_with_my_rt(one: ONE, eid: str):
    trials_obj = one.load_object(eid, "trials", collection="alf")

    required = ["stimOn_times", "feedback_times"]
    for k in required:
        if k not in trials_obj:
            raise RuntimeError(f"[TRIALS] Missing key '{k}' in trials object for eid={eid}")

    n_raw_trials = len(trials_obj["stimOn_times"])
    if n_raw_trials <= 0:
        raise RuntimeError(f"[TRIALS] stimOn_times is empty for eid={eid}")

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
        raise RuntimeError(f"[WHEEL] No wheel data for eid={eid}")

    vel = calc_wheel_velocity(pos, ts)
    _, trial_ts, trial_vel = calc_trialwise_wheel(pos, ts, vel, df["stimOn_times"], df["feedback_times"])
    (_, first_mo, _, _, _) = calc_movement_onset_times(trial_ts, trial_vel, df["stimOn_times"])
    df["first_movement_onset_times"] = first_mo
    df["reaction_time"] = df["first_movement_onset_times"] - df["stimOn_times"]

    if not np.isfinite(df["reaction_time"]).any():
        raise RuntimeError(f"[RT] All reaction_time are non-finite for eid={eid}")

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


# =========================
# Stats helpers (same as your RIS version)
# =========================
def _trial_scalar_list(x_list):
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
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return np.nan
    sse = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - sse / denom)


def _z_and_p_from_null(real_val, null_vals, one_sided="greater_equal"):
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
    if (not np.isfinite(r2_real)) or (not np.isfinite(r2_fake_mean)):
        return np.nan
    denom = 1.0 - float(r2_fake_mean)
    if denom == 0:
        return np.nan
    return float((r2_real - float(r2_fake_mean)) / denom)


def compute_group_stats_from_pkl(region_pkl: Path, group_mask_full: np.ndarray):
    with open(region_pkl, "rb") as f:
        d = pickle.load(f)

    keep_idx_full = np.asarray(d["keep_idx_full"], dtype=int)
    group_mask_sub = np.asarray(group_mask_full, dtype=bool)[keep_idx_full]
    sub_idx = np.flatnonzero(group_mask_sub)
    n_used = int(sub_idx.size)

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

    r_real = mean_or_nan(pid_to_r.get(-1, []))
    r2_real = mean_or_nan(pid_to_r2.get(-1, []))

    fake_ids_r = sorted([pid for pid in pid_to_r.keys() if pid != -1])
    r_fake = np.asarray([mean_or_nan(pid_to_r[pid]) for pid in fake_ids_r], float)
    r_fake = r_fake[np.isfinite(r_fake)]

    fake_ids_r2 = sorted([pid for pid in pid_to_r2.keys() if pid != -1])
    r2_fake = np.asarray([mean_or_nan(pid_to_r2[pid]) for pid in fake_ids_r2], float)
    r2_fake = r2_fake[np.isfinite(r2_fake)]

    r_fake_mean, r_fake_sd, z_r, p_emp_r = _z_and_p_from_null(r_real, r_fake, one_sided="greater_equal")
    r2_fake_mean, r2_fake_sd, z_r2, p_emp_r2 = _z_and_p_from_null(r2_real, r2_fake, one_sided="greater_equal")

    r2_corr = _corrected_r2_findling(r2_real, r2_fake_mean)

    return {
        "n_used": n_used,
        "r_real": r_real,
        "r_fake_mean": r_fake_mean,
        "r_fake_std": r_fake_sd,
        "z_corr": z_r,
        "p_emp": p_emp_r,
        "r2_real": r2_real,
        "r2_fake_mean": r2_fake_mean,
        "r2_fake_std": r2_fake_sd,
        "r2_corr": r2_corr,
        "z_r2": z_r2,
        "p_emp_r2": p_emp_r2,
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


# =========================
# Probe pick (same as RIS)
# =========================
def _pick_probe_name_like_local(one: ONE, eid: str):
    pids = one.eid2pid(eid)
    return pids, (pids[1] if len(pids) > 1 else pids[0])


# =========================
# Worker (RIS-style)
# =========================
def run_one_eid_worker(eid: str, out_root_str: str, n_pseudo: int, debug: bool, cache_root_str: str):
    warnings.filterwarnings("ignore", category=alferr.ALFWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    out_root = Path(out_root_str)
    out_root.mkdir(parents=True, exist_ok=True)

    cache_root = Path(cache_root_str)
    cache_root.mkdir(parents=True, exist_ok=True)
    worker_cache = cache_root / f"worker_{os.getpid()}"
    one = make_one(worker_cache)

    print(f"[ONE] version={getattr(one_pkg,'__version__','?')} worker_cache={worker_cache}", flush=True)

    pids, probe_name = _pick_probe_name_like_local(one, eid)
    print(f"[PIDS] {eid} -> {pids} | picked_probe={probe_name}", flush=True)

    ses = one.alyx.rest("sessions", "read", id=eid)
    subject = ses["subject"]

    df, n_raw_trials = compute_trials_with_my_rt(one, eid)
    print(f"[TRIALS] eid={eid} n_raw_trials={n_raw_trials}", flush=True)

    if n_raw_trials < MIN_RAW_TRIALS:
        return {"eid": eid, "status": "skip_trials", "n_raw_trials": n_raw_trials, "rows": []}

    masks, meta = make_fast_normal_slow_masks(df)
    drop_last_mask = make_drop_last_mask(len(df), DROP_LAST_N)

    print(
        f"[RT] finite_rt={meta['n_valid_rt']} fast={meta['n_fast']} normal={meta['n_normal']} slow={meta['n_slow']} "
        f"drop_last_n={DROP_LAST_N} n_after_drop_last={int(drop_last_mask.sum())}",
        flush=True
    )

    meta2 = dict(meta)
    meta2.update({
        "drop_last_n": int(DROP_LAST_N),
        "n_after_drop_last": int(drop_last_mask.sum()),
        "n_fast_after_drop_last": int((masks["fast"] & drop_last_mask).sum()),
        "n_normal_after_drop_last": int((masks["normal"] & drop_last_mask).sum()),
        "n_slow_after_drop_last": int((masks["slow"] & drop_last_mask).sum()),
        "eid2pid_list": [str(x) for x in pids],
        "probe_picked": str(probe_name),
        "worker_cache": str(worker_cache),
        "one_version": str(getattr(one_pkg, "__version__", "?")),
    })
    (out_root / f"{eid}_rt_group_counts.json").write_text(json.dumps(meta2, indent=2))

    pseudo_ids = [-1] + list(range(1, n_pseudo + 1))

    session_dir = out_root / eid / "session_fit"
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"[FIT] eid={eid} session_dir={session_dir} debug={bool(debug)}", flush=True)

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
    print(f"[ROI] requested={roi} found_pkl={roi_pkl}", flush=True)

    if roi_pkl is None or (not Path(roi_pkl).exists()):
        return {"eid": eid, "status": "skip_no_roi", "n_raw_trials": n_raw_trials, "rows": []}

    d = pickle.load(open(roi_pkl, "rb"))
    real_entries = [fr for fr in d.get("fit", []) if int(fr.get("pseudo_id", -999)) == -1]
    if len(real_entries) == 0:
        raise RuntimeError(f"[PKL] No real pseudo_id=-1 entry in {roi_pkl}")

    fr = real_entries[0]
    yhat_all = _trial_scalar_list(fr["predictions_test"])
    y_all = _trial_scalar_list(fr["target"])

    print(
        f"[SANITY] yhat finite={int(np.isfinite(yhat_all).sum())}/{len(yhat_all)} std={float(np.nanstd(yhat_all))} "
        f"min={float(np.nanmin(yhat_all)) if np.isfinite(yhat_all).any() else np.nan} "
        f"max={float(np.nanmax(yhat_all)) if np.isfinite(yhat_all).any() else np.nan}",
        flush=True
    )
    print(
        f"[SANITY] y    finite={int(np.isfinite(y_all).sum())}/{len(y_all)} std={float(np.nanstd(y_all))}",
        flush=True
    )

    if int(np.isfinite(yhat_all).sum()) < 10:
        raise RuntimeError(f"[SANITY] Too few finite predictions in REAL fit (finite={int(np.isfinite(yhat_all).sum())})")
    if (not np.isfinite(np.nanstd(yhat_all))) or (np.nanstd(yhat_all) == 0):
        raise RuntimeError("[SANITY] REAL predictions are constant/invalid -> Pearson will be NaN (likely wrong probe or no units)")

    rows = []
    print("\n=== PER-SESSION / PER-GROUP (LOCAL RIS-MIRROR) ===", flush=True)
    for g in GROUPS:
        stats = compute_group_stats_from_pkl(roi_pkl, group_mask_full=masks[g])
        row = {
            "eid": eid,
            "subject": subject,
            "probe": str(probe_name),
            "roi": roi,
            "group": g,
            "n_trials_group_used": int(stats["n_used"]),
            "r_real": stats["r_real"],
            "r_fake_mean": stats["r_fake_mean"],
            "r_fake_std": stats["r_fake_std"],
            "z_corr": stats["z_corr"],
            "p_emp": stats["p_emp"],
            "r2_real": stats["r2_real"],
            "r2_fake_mean": stats["r2_fake_mean"],
            "r2_fake_std": stats["r2_fake_std"],
            "r2_corr": stats["r2_corr"],
            "z_r2": stats["z_r2"],
            "p_emp_r2": stats["p_emp_r2"],
            "drop_last_n": int(DROP_LAST_N),
            "roi_pkl_path": str(roi_pkl),
        }
        rows.append(row)

        print(
            f"{eid} | {g:<6} | n={row['n_trials_group_used']:4d} | "
            f"Pearson: r={row['r_real']} | null= {row['r_fake_mean']}± {row['r_fake_std']} | "
            f"z={row['z_corr']} | p={row['p_emp']} || "
            f"R2: r2={row['r2_real']} | null={row['r2_fake_mean']}± {row['r2_fake_std']} | "
            f"r2_corr={row['r2_corr']} | z={row['z_r2']} | p={row['p_emp_r2']}",
            flush=True
        )

    (out_root / eid / f"pearson_summary_{roi}.json").write_text(json.dumps(rows, indent=2))

    return {"eid": eid, "status": "ok", "n_raw_trials": n_raw_trials, "rows": rows}


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    # same eid you used
    eids = [
        "ae8787b1-4229-4d56-b0c2-566b61a25b77",
    ]

    all_rows = []
    run_log = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [
            ex.submit(run_one_eid_worker, eid, str(OUT_ROOT), N_PSEUDO, DEBUG, str(CACHE_ROOT))
            for eid in eids
        ]
        for fut in as_completed(futs):
            r = fut.result()
            run_log.append({k: r.get(k) for k in ["eid", "status", "n_raw_trials"]})
            all_rows.extend(r.get("rows", []))
            print("[DONE]", r.get("eid"), r.get("status"), "rows=", len(r.get("rows", [])), flush=True)

    roi_tag = "_".join(ROI_LIST) if ROI_LIST else "ALL"
    summary_path = OUT_ROOT / f"pearson_summary_{roi_tag}.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(all_rows, f)

    run_log_path = OUT_ROOT / f"run_log_{roi_tag}.json"
    with open(run_log_path, "w") as f:
        json.dump(run_log, f, indent=2)

    print("Saved pearson summary:", summary_path, flush=True)
    print("Saved run log:", run_log_path, flush=True)
    print("Local cache root:", CACHE_ROOT, flush=True)


if __name__ == "__main__":
    main()
