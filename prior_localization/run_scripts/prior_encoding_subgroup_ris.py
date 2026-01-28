# # run_sessionfit_groups_corr_r2_remote.py
# from __future__ import annotations
#
# from pathlib import Path
# import json
# import pickle
# import warnings
# import os
# import ast
# import contextlib
# import numpy as np
# import pandas as pd
# from concurrent.futures import ProcessPoolExecutor, as_completed
#
# from one.api import ONE
# import one.alf.exceptions as alferr
#
# from prior_localization.my_rt import (
#     load_wheel_data, calc_wheel_velocity, calc_trialwise_wheel, calc_movement_onset_times
# )
# from prior_localization.fit_data import fit_session_ephys
#
#
# # =========================
# # USER SETTINGS
# # =========================
# ROI_LIST = ["MOp", "MOs", "ACAd", "ORBvl"]
# ROI_SET = set(ROI_LIST)
# GROUPS = ["fast", "normal", "slow"]
#
# # Input txt on RIS
# TXT_PATH = "/home/wg-yin/session_ids_for_behav_analysis.txt"
#
# # Outputs on shared filesystem
# OUT_BASE = Path("/storage1/fs1/hiratani/Active/shared/ibl_space/derived/prior_localization_groups_output")
#
# # Shared cache root
# CACHE_ROOT = Path("/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org")
#
# RUN_TAG = os.getenv("RUN_TAG", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
# OUT_ROOT = OUT_BASE / f"run_{RUN_TAG}"
#
# # Decoding / analysis config
# N_PSEUDO = int(os.getenv("N_PSEUDO", "200"))
# N_RUNS = int(os.getenv("N_RUNS", "2"))
# N_WORKERS = int(os.getenv("N_WORKERS", "10"))
# DEBUG = bool(int(os.getenv("DEBUG", "0")))
#
# ALIGN_EVENT = "stimOn_times"
# TIME_WINDOW = (-0.6, -0.1)
# FAST_THR = float(os.getenv("FAST_THR", "0.08"))
# SLOW_THR = float(os.getenv("SLOW_THR", "1.25"))
#
# # Exclude last N trials at FIT TIME (applies to real + pseudo)
# DROP_LAST_N = int(os.getenv("DROP_LAST_N", "40"))
# MIN_RAW_TRIALS = int(os.getenv("MIN_RAW_TRIALS", "401"))
#
#
# # =========================
# # ONE helper (per-worker cache)
# # =========================
# def make_one(cache_dir: Path) -> ONE:
#     cache_dir = Path(cache_dir)
#     cache_dir.mkdir(parents=True, exist_ok=True)
#
#     tables_dir = cache_dir / "tables"
#     dl_cache = cache_dir / "downloads"
#     tables_dir.mkdir(parents=True, exist_ok=True)
#     dl_cache.mkdir(parents=True, exist_ok=True)
#
#     one = ONE(
#         base_url="https://openalyx.internationalbrainlab.org",
#         username=os.getenv("ALYX_LOGIN"),
#         password=os.getenv("ALYX_PASSWORD"),
#         silent=True,
#         cache_dir=dl_cache,
#         tables_dir=tables_dir,
#         cache_rest=None,
#     )
#     return one
#
#
# # =========================
# # IO helpers
# # =========================
# def read_session_ids_from_txt(txt_path: str) -> list[str]:
#     session_ids: list[str] = []
#     with open(txt_path, "r") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             if line.startswith("["):
#                 ids_in_line = ast.literal_eval(line)
#                 session_ids.extend([str(x) for x in ids_in_line])
#             else:
#                 session_ids.append(line)
#     # unique preserve order
#     return list(dict.fromkeys(session_ids))
#
#
# # =========================
# # Trial + RT helpers
# # =========================
# def compute_trials_with_my_rt(one: ONE, eid: str):
#     trials_obj = one.load_object(eid, "trials", collection="alf")
#     n_raw_trials = len(trials_obj["stimOn_times"])
#
#     data = {}
#     for k, v in trials_obj.items():
#         arr = np.asarray(v)
#         if k == "intervals" and arr.ndim == 2 and arr.shape[1] == 2:
#             data["intervals_start"] = arr[:, 0]
#             data["intervals_end"] = arr[:, 1]
#         elif arr.ndim == 1:
#             data[k] = arr
#         else:
#             data[k] = list(arr)
#     df = pd.DataFrame(data)
#
#     pos, ts = load_wheel_data(one, eid)
#     if pos is None or ts is None:
#         raise RuntimeError(f"No wheel data for eid={eid}")
#
#     vel = calc_wheel_velocity(pos, ts)
#     _, trial_ts, trial_vel = calc_trialwise_wheel(
#         pos, ts, vel, df["stimOn_times"], df["feedback_times"]
#     )
#     (_, first_mo, _, _, _) = calc_movement_onset_times(trial_ts, trial_vel, df["stimOn_times"])
#     df["first_movement_onset_times"] = first_mo
#     df["reaction_time"] = df["first_movement_onset_times"] - df["stimOn_times"]
#
#     return df, int(n_raw_trials)
#
#
# def make_fast_normal_slow_masks(
#     df: pd.DataFrame, rt_col="reaction_time",
#     fast_thr=FAST_THR, slow_thr=SLOW_THR
# ):
#     rt = df[rt_col].to_numpy()
#     valid = np.isfinite(rt)
#     fast = valid & (rt < fast_thr)
#     slow = valid & (rt > slow_thr)
#     normal = valid & (~fast) & (~slow)
#     return {"fast": fast, "normal": normal, "slow": slow}, {
#         "n_total": int(len(df)),
#         "n_valid_rt": int(valid.sum()),
#         "n_fast": int(fast.sum()),
#         "n_normal": int(normal.sum()),
#         "n_slow": int(slow.sum()),
#         "fast_thr": float(fast_thr),
#         "slow_thr": float(slow_thr),
#     }
#
#
# def make_drop_last_mask(n_trials: int, drop_last_n: int):
#     m = np.ones(int(n_trials), dtype=bool)
#     if drop_last_n and drop_last_n > 0:
#         m[max(0, n_trials - drop_last_n):] = False
#     return m
#
#
# # =========================
# # Stats helpers
# # =========================
# def _trial_scalar_list(x_list):
#     """list-of-arrays -> scalar per trial (mean over bins). returns vec (n_trials,) with NaNs for bad."""
#     out = np.full(len(x_list), np.nan, float)
#     for i, xi in enumerate(x_list):
#         if xi is None:
#             continue
#         arr = np.asarray(xi).reshape(-1)
#         if arr.size == 0 or (not np.all(np.isfinite(arr))):
#             continue
#         out[i] = float(np.mean(arr))
#     return out
#
#
# def _pearsonr_safe(a, b):
#     a = np.asarray(a, float).reshape(-1)
#     b = np.asarray(b, float).reshape(-1)
#     ok = np.isfinite(a) & np.isfinite(b)
#     a = a[ok]; b = b[ok]
#     if a.size < 3:
#         return np.nan
#     if np.std(a) == 0 or np.std(b) == 0:
#         return np.nan
#     return float(np.corrcoef(a, b)[0, 1])
#
#
# def _r2_safe(y_true, y_pred):
#     y_true = np.asarray(y_true, float).reshape(-1)
#     y_pred = np.asarray(y_pred, float).reshape(-1)
#     ok = np.isfinite(y_true) & np.isfinite(y_pred)
#     y_true = y_true[ok]; y_pred = y_pred[ok]
#     if y_true.size < 3:
#         return np.nan
#     denom = np.sum((y_true - np.mean(y_true)) ** 2)
#     if denom <= 0:
#         return np.nan
#     sse = np.sum((y_true - y_pred) ** 2)
#     return float(1.0 - sse / denom)
#
#
# def _z_and_p_from_null(real_val, null_vals, one_sided="greater_equal"):
#     null_vals = np.asarray(null_vals, float)
#     null_vals = null_vals[np.isfinite(null_vals)]
#     if (not np.isfinite(real_val)) or null_vals.size < 5:
#         return np.nan, np.nan, np.nan, np.nan
#
#     mu = float(np.mean(null_vals))
#     sd = float(np.std(null_vals, ddof=1)) if null_vals.size > 1 else float(np.std(null_vals))
#     z = float((real_val - mu) / sd) if sd > 0 else np.nan
#
#     if one_sided == "greater_equal":
#         p = float((1 + np.sum(null_vals >= real_val)) / (1 + null_vals.size))
#     elif one_sided == "less_equal":
#         p = float((1 + np.sum(null_vals <= real_val)) / (1 + null_vals.size))
#     else:
#         raise ValueError("one_sided must be 'greater_equal' or 'less_equal'")
#
#     return mu, sd, z, p
#
#
# def _corrected_r2_findling(r2_real, r2_fake_mean):
#     # Findling-style correction:
#     #   R2_corr = (R2_real - mean(R2_fake)) / (1 - mean(R2_fake))
#     if (not np.isfinite(r2_real)) or (not np.isfinite(r2_fake_mean)):
#         return np.nan
#     denom = 1.0 - float(r2_fake_mean)
#     if denom == 0:
#         return np.nan
#     return float((r2_real - float(r2_fake_mean)) / denom)
#
#
# def find_roi_pkl(session_dir: Path, roi: str):
#     for p in sorted(session_dir.rglob("*.pkl")):
#         try:
#             with open(p, "rb") as f:
#                 d = pickle.load(f)
#             reg = d.get("region", "")
#             reg = reg[0] if isinstance(reg, (list, tuple)) else str(reg)
#             if str(reg).strip() == str(roi).strip():
#                 return p
#         except Exception:
#             continue
#     return None
#
#
# def compute_group_stats_from_pkl(region_pkl: Path, group_mask_full: np.ndarray):
#     """
#     Uses d['keep_idx_full'] to map full-trial mask into decoded-trial order.
#     Returns a dict with Pearson + R2 + corrected R2.
#     """
#     with open(region_pkl, "rb") as f:
#         d = pickle.load(f)
#
#     keep_idx_full = np.asarray(d["keep_idx_full"], dtype=int)
#
#     group_mask_sub = np.asarray(group_mask_full, dtype=bool)[keep_idx_full]
#     sub_idx = np.flatnonzero(group_mask_sub)
#     n_used = int(sub_idx.size)
#
#     pid_to_r = {}
#     pid_to_r2 = {}
#
#     for fr in d["fit"]:
#         pid = int(fr.get("pseudo_id", -999))
#         preds = fr.get("predictions_test", None)
#         targ  = fr.get("target", None)
#         if preds is None or targ is None:
#             continue
#
#         yhat = _trial_scalar_list(preds)
#         y    = _trial_scalar_list(targ)
#
#         y_sub = y[sub_idx]
#         yhat_sub = yhat[sub_idx]
#
#         r = _pearsonr_safe(yhat_sub, y_sub)
#         r2 = _r2_safe(y_sub, yhat_sub)
#
#         if np.isfinite(r):
#             pid_to_r.setdefault(pid, []).append(r)
#         if np.isfinite(r2):
#             pid_to_r2.setdefault(pid, []).append(r2)
#
#     def mean_or_nan(x):
#         return float(np.mean(x)) if len(x) else np.nan
#
#     r_real = mean_or_nan(pid_to_r.get(-1, []))
#     r2_real = mean_or_nan(pid_to_r2.get(-1, []))
#
#     fake_ids_r = sorted([pid for pid in pid_to_r.keys() if pid != -1])
#     r_fake = np.asarray([mean_or_nan(pid_to_r[pid]) for pid in fake_ids_r], float)
#     r_fake = r_fake[np.isfinite(r_fake)]
#
#     fake_ids_r2 = sorted([pid for pid in pid_to_r2.keys() if pid != -1])
#     r2_fake = np.asarray([mean_or_nan(pid_to_r2[pid]) for pid in fake_ids_r2], float)
#     r2_fake = r2_fake[np.isfinite(r2_fake)]
#
#     r_fake_mean, r_fake_sd, z_r, p_emp_r = _z_and_p_from_null(r_real, r_fake, one_sided="greater_equal")
#     r2_fake_mean, r2_fake_sd, z_r2, p_emp_r2 = _z_and_p_from_null(r2_real, r2_fake, one_sided="greater_equal")
#
#     r2_corr = _corrected_r2_findling(r2_real, r2_fake_mean)
#
#     return {
#         "n_used": n_used,
#         # Pearson
#         "r_real": r_real,
#         "r_fake_mean": r_fake_mean,
#         "r_fake_std": r_fake_sd,
#         "z_corr": z_r,
#         "p_emp": p_emp_r,
#         # R2
#         "r2_real": r2_real,
#         "r2_fake_mean": r2_fake_mean,
#         "r2_fake_std": r2_fake_sd,
#         "r2_corr": r2_corr,
#         "z_r2": z_r2,
#         "p_emp_r2": p_emp_r2,
#     }
#
#
# # =========================
# # Worker
# # =========================
# def run_one_eid_worker(eid: str, out_root_str: str, cache_root_str: str, n_pseudo: int, debug: bool):
#     warnings.filterwarnings("ignore", category=alferr.ALFWarning)
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     warnings.filterwarnings("ignore", category=RuntimeWarning)
#     warnings.filterwarnings("ignore", message=r".*Multiple revisions.*")
#     warnings.filterwarnings("ignore", message=r".*Newer cache tables require ONE version.*")
#
#     out_root = Path(out_root_str)
#     out_root.mkdir(parents=True, exist_ok=True)
#
#     cache_root = Path(cache_root_str)
#     cache_root.mkdir(parents=True, exist_ok=True)
#
#     pid = os.getpid()
#     worker_cache = cache_root / f"worker_{pid}"
#     worker_cache.mkdir(parents=True, exist_ok=True)
#
#     one = make_one(cache_dir=worker_cache)
#
#     # metadata
#     try:
#         probe_name = one.eid2pid(eid)[1]
#         ses = one.alyx.rest("sessions", "read", id=eid)
#         subject = ses["subject"]
#     except Exception as e:
#         return {"eid": eid, "status": f"fail_one_setup: {e}", "n_raw_trials": -1, "rows": []}
#
#     # trials + wheel RT
#     try:
#         df, n_raw_trials = compute_trials_with_my_rt(one, eid)
#     except Exception as e:
#         return {"eid": eid, "status": f"fail_trials_or_wheel: {e}", "n_raw_trials": -1, "rows": []}
#
#     if int(n_raw_trials) < int(MIN_RAW_TRIALS):
#         return {"eid": eid, "status": "skip_trials", "n_raw_trials": int(n_raw_trials), "rows": []}
#
#     masks, meta = make_fast_normal_slow_masks(df)
#     drop_last_mask = make_drop_last_mask(len(df), DROP_LAST_N)
#
#     # Save counts sanity check
#     meta2 = dict(meta)
#     meta2["drop_last_n"] = int(DROP_LAST_N)
#     meta2["n_after_drop_last"] = int(drop_last_mask.sum())
#     meta2["n_fast_after_drop_last"] = int((masks["fast"] & drop_last_mask).sum())
#     meta2["n_normal_after_drop_last"] = int((masks["normal"] & drop_last_mask).sum())
#     meta2["n_slow_after_drop_last"] = int((masks["slow"] & drop_last_mask).sum())
#     (out_root / f"{eid}_rt_group_counts.json").write_text(json.dumps(meta2, indent=2))
#
#     if debug:
#         print(f"[DEBUG] {eid}: raw_trials={n_raw_trials} meta={meta2}")
#         print(f"[DEBUG] {eid}: probe_name={probe_name}")
#         print(f"[DEBUG] {eid}: worker_cache={worker_cache}")
#
#     pseudo_ids = [-1] + list(range(1, n_pseudo + 1))
#
#     # FIT ONCE (restricted trial set = drop last N at fit time)
#     session_dir = out_root / eid / "session_fit"
#     session_dir.mkdir(parents=True, exist_ok=True)
#
#     try:
#         with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
#             _ = fit_session_ephys(
#                 one=one,
#                 session_id=eid,
#                 subject=subject,
#                 probe_name=probe_name,
#                 output_dir=session_dir,
#                 pseudo_ids=pseudo_ids,
#                 target="pLeft",
#                 align_event=ALIGN_EVENT,
#                 time_window=TIME_WINDOW,
#                 model="optBay",
#                 n_runs=N_RUNS,
#                 trials_df=df,
#                 trial_mask=drop_last_mask,     # ✅ drop last 40 at FIT TIME (real+pseudo)
#                 group_label="session",
#                 debug=bool(debug),
#                 roi_set=ROI_SET,
#             )
#     except Exception as e:
#         return {"eid": eid, "status": f"fail_fit: {e}", "n_raw_trials": int(n_raw_trials), "rows": []}
#
#     rows = []
#
#     for roi in ROI_LIST:
#         roi_pkl = find_roi_pkl(session_dir, roi)
#         if roi_pkl is None:
#             continue
#
#         # IMPORTANT: do NOT drop last 40 again here; keep_idx_full already enforces it.
#         for g in GROUPS:
#             stats = compute_group_stats_from_pkl(roi_pkl, group_mask_full=masks[g])
#
#             rows.append({
#                 "eid": eid,
#                 "subject": subject,
#                 "probe": probe_name if isinstance(probe_name, str) else str(probe_name),
#                 "roi": roi,
#                 "group": g,
#                 "n_trials_group_used": int(stats["n_used"]),
#
#                 # Pearson
#                 "r_real": stats["r_real"],
#                 "r_fake_mean": stats["r_fake_mean"],
#                 "r_fake_std": stats["r_fake_std"],
#                 "z_corr": stats["z_corr"],
#                 "p_emp": stats["p_emp"],
#
#                 # R^2
#                 "r2_real": stats["r2_real"],
#                 "r2_fake_mean": stats["r2_fake_mean"],
#                 "r2_fake_std": stats["r2_fake_std"],
#                 "r2_corr": stats["r2_corr"],
#                 "z_r2": stats["z_r2"],
#                 "p_emp_r2": stats["p_emp_r2"],
#
#                 # bookkeeping
#                 "drop_last_n": int(DROP_LAST_N),
#                 "roi_pkl_path": str(roi_pkl),
#             })
#
#     status = "ok" if rows else "skip_no_roi_or_failed"
#     return {"eid": eid, "status": status, "n_raw_trials": int(n_raw_trials), "rows": rows}
#
#
# def main():
#     OUT_ROOT.mkdir(parents=True, exist_ok=True)
#     CACHE_ROOT.mkdir(parents=True, exist_ok=True)
#
#     print(f"[RUN] RUN_TAG={RUN_TAG}")
#     print(f"[RUN] Writing outputs to: {OUT_ROOT}")
#     print(f"[RUN] N_PSEUDO={N_PSEUDO} N_RUNS={N_RUNS} N_WORKERS={N_WORKERS} DROP_LAST_N={DROP_LAST_N} "
#           f"FAST_THR={FAST_THR} SLOW_THR={SLOW_THR}")
#
#     eids = read_session_ids_from_txt(TXT_PATH)
#
#     all_rows = []
#     run_log = []
#
#     with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
#         futs = [
#             ex.submit(run_one_eid_worker, eid, str(OUT_ROOT), str(CACHE_ROOT), N_PSEUDO, DEBUG)
#             for eid in eids
#         ]
#         for fut in as_completed(futs):
#             r = fut.result()
#             run_log.append({k: r.get(k) for k in ["eid", "status", "n_raw_trials"]})
#             all_rows.extend(r.get("rows", []))
#             print("[DONE]", r.get("eid"), r.get("status"), "rows=", len(r.get("rows", [])))
#
#     # run-tagged summary files
#     summary_path = OUT_ROOT / f"pearson_r2_summary_{RUN_TAG}.pkl"
#     with open(summary_path, "wb") as f:
#         pickle.dump(all_rows, f)
#
#     run_log_path = OUT_ROOT / f"run_log_{RUN_TAG}.json"
#     with open(run_log_path, "w") as f:
#         json.dump(run_log, f, indent=2)
#
#     print("Saved big summary:", summary_path)
#     print("Saved run log:", run_log_path)
#
#
# if __name__ == "__main__":
#     main()

# run_sessionfit_groups_corr_r2_remote.py
# from __future__ import annotations
#
# from pathlib import Path
# import json
# import pickle
# import warnings
# import os
# import ast
# import contextlib
# import numpy as np
# import pandas as pd
# from concurrent.futures import ProcessPoolExecutor, as_completed
#
# from one.api import ONE
# import one.alf.exceptions as alferr
#
# from prior_localization.my_rt import (
#     load_wheel_data, calc_wheel_velocity, calc_trialwise_wheel, calc_movement_onset_times
# )
# from prior_localization.fit_data import fit_session_ephys
#
#
# # =========================
# # USER SETTINGS
# # =========================
# ROI_LIST = ["MOp", "MOs", "ACAd", "ORBvl"]
# ROI_SET = set(ROI_LIST)
# GROUPS = ["fast", "normal", "slow"]
#
# TXT_PATH = "/home/wg-yin/session_ids_for_behav_analysis.txt"
#
# OUT_BASE = Path("/storage1/fs1/hiratani/Active/shared/ibl_space/derived/prior_localization_groups_output")
# CACHE_ROOT = Path("/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org")
#
# RUN_TAG = os.getenv("RUN_TAG", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
# OUT_ROOT = OUT_BASE / f"run_{RUN_TAG}"
#
# N_PSEUDO = int(os.getenv("N_PSEUDO", "200"))
# N_RUNS = int(os.getenv("N_RUNS", "2"))
# N_WORKERS = int(os.getenv("N_WORKERS", "10"))
# DEBUG = bool(int(os.getenv("DEBUG", "0")))
#
# ALIGN_EVENT = "stimOn_times"
# TIME_WINDOW = (-0.6, -0.1)
# FAST_THR = float(os.getenv("FAST_THR", "0.08"))
# SLOW_THR = float(os.getenv("SLOW_THR", "1.25"))
#
# DROP_LAST_N = int(os.getenv("DROP_LAST_N", "40"))
# MIN_RAW_TRIALS = int(os.getenv("MIN_RAW_TRIALS", "401"))
#
# # If True, also save a per-session JSON like the local script
# SAVE_PER_SESSION_JSON = bool(int(os.getenv("SAVE_PER_SESSION_JSON", "0")))
#
#
# # =========================
# # ONE helper (per-worker cache)
# # =========================
# def make_one(cache_dir: Path) -> ONE:
#     cache_dir = Path(cache_dir)
#     cache_dir.mkdir(parents=True, exist_ok=True)
#
#     tables_dir = cache_dir / "tables"
#     dl_cache = cache_dir / "downloads"
#     tables_dir.mkdir(parents=True, exist_ok=True)
#     dl_cache.mkdir(parents=True, exist_ok=True)
#
#     one = ONE(
#         base_url="https://openalyx.internationalbrainlab.org",
#         username=os.getenv("ALYX_LOGIN"),
#         password=os.getenv("ALYX_PASSWORD"),
#         silent=True,
#         cache_dir=dl_cache,
#         tables_dir=tables_dir,
#         cache_rest=None,
#     )
#     return one
#
#
# # =========================
# # IO helpers
# # =========================
# def read_session_ids_from_txt(txt_path: str) -> list[str]:
#     session_ids: list[str] = []
#     with open(txt_path, "r") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             if line.startswith("["):
#                 ids_in_line = ast.literal_eval(line)
#                 session_ids.extend([str(x) for x in ids_in_line])
#             else:
#                 session_ids.append(line)
#     return list(dict.fromkeys(session_ids))
#
#
# # =========================
# # Trial + RT helpers
# # =========================
# def compute_trials_with_my_rt(one: ONE, eid: str):
#     trials_obj = one.load_object(eid, "trials", collection="alf")
#     n_raw_trials = len(trials_obj["stimOn_times"])
#
#     data = {}
#     for k, v in trials_obj.items():
#         arr = np.asarray(v)
#         if k == "intervals" and arr.ndim == 2 and arr.shape[1] == 2:
#             data["intervals_start"] = arr[:, 0]
#             data["intervals_end"] = arr[:, 1]
#         elif arr.ndim == 1:
#             data[k] = arr
#         else:
#             data[k] = list(arr)
#     df = pd.DataFrame(data)
#
#     pos, ts = load_wheel_data(one, eid)
#     if pos is None or ts is None:
#         raise RuntimeError(f"No wheel data for eid={eid}")
#
#     vel = calc_wheel_velocity(pos, ts)
#     _, trial_ts, trial_vel = calc_trialwise_wheel(
#         pos, ts, vel, df["stimOn_times"], df["feedback_times"]
#     )
#     (_, first_mo, _, _, _) = calc_movement_onset_times(trial_ts, trial_vel, df["stimOn_times"])
#     df["first_movement_onset_times"] = first_mo
#     df["reaction_time"] = df["first_movement_onset_times"] - df["stimOn_times"]
#
#     return df, int(n_raw_trials)
#
#
# def make_fast_normal_slow_masks(df: pd.DataFrame, rt_col="reaction_time",
#                                fast_thr=FAST_THR, slow_thr=SLOW_THR):
#     rt = df[rt_col].to_numpy()
#     valid = np.isfinite(rt)
#     fast = valid & (rt < fast_thr)
#     slow = valid & (rt > slow_thr)
#     normal = valid & (~fast) & (~slow)
#     return {"fast": fast, "normal": normal, "slow": slow}, {
#         "n_total": int(len(df)),
#         "n_valid_rt": int(valid.sum()),
#         "n_fast": int(fast.sum()),
#         "n_normal": int(normal.sum()),
#         "n_slow": int(slow.sum()),
#         "fast_thr": float(fast_thr),
#         "slow_thr": float(slow_thr),
#     }
#
#
# def make_drop_last_mask(n_trials: int, drop_last_n: int):
#     m = np.ones(int(n_trials), dtype=bool)
#     if drop_last_n and drop_last_n > 0:
#         m[max(0, n_trials - drop_last_n):] = False
#     return m
#
#
# # =========================
# # Stats helpers
# # =========================
# def _trial_scalar_list(x_list):
#     out = np.full(len(x_list), np.nan, float)
#     for i, xi in enumerate(x_list):
#         if xi is None:
#             continue
#         arr = np.asarray(xi).reshape(-1)
#         if arr.size == 0 or (not np.all(np.isfinite(arr))):
#             continue
#         out[i] = float(np.mean(arr))
#     return out
#
#
# def _pearsonr_safe(a, b):
#     a = np.asarray(a, float).reshape(-1)
#     b = np.asarray(b, float).reshape(-1)
#     ok = np.isfinite(a) & np.isfinite(b)
#     a = a[ok]; b = b[ok]
#     if a.size < 3:
#         return np.nan
#     if np.std(a) == 0 or np.std(b) == 0:
#         return np.nan
#     return float(np.corrcoef(a, b)[0, 1])
#
#
# def _r2_safe(y_true, y_pred):
#     y_true = np.asarray(y_true, float).reshape(-1)
#     y_pred = np.asarray(y_pred, float).reshape(-1)
#     ok = np.isfinite(y_true) & np.isfinite(y_pred)
#     y_true = y_true[ok]; y_pred = y_pred[ok]
#     if y_true.size < 3:
#         return np.nan
#     denom = np.sum((y_true - np.mean(y_true)) ** 2)
#     if denom <= 0:
#         return np.nan
#     sse = np.sum((y_true - y_pred) ** 2)
#     return float(1.0 - sse / denom)
#
#
# def _z_and_p_from_null(real_val, null_vals, one_sided="greater_equal"):
#     null_vals = np.asarray(null_vals, float)
#     null_vals = null_vals[np.isfinite(null_vals)]
#     if (not np.isfinite(real_val)) or null_vals.size < 5:
#         return np.nan, np.nan, np.nan, np.nan
#
#     mu = float(np.mean(null_vals))
#     sd = float(np.std(null_vals, ddof=1)) if null_vals.size > 1 else float(np.std(null_vals))
#     z = float((real_val - mu) / sd) if sd > 0 else np.nan
#
#     if one_sided == "greater_equal":
#         p = float((1 + np.sum(null_vals >= real_val)) / (1 + null_vals.size))
#     elif one_sided == "less_equal":
#         p = float((1 + np.sum(null_vals <= real_val)) / (1 + null_vals.size))
#     else:
#         raise ValueError("one_sided must be 'greater_equal' or 'less_equal'")
#
#     return mu, sd, z, p
#
#
# def _corrected_r2_findling(r2_real, r2_fake_mean):
#     if (not np.isfinite(r2_real)) or (not np.isfinite(r2_fake_mean)):
#         return np.nan
#     denom = 1.0 - float(r2_fake_mean)
#     if denom == 0:
#         return np.nan
#     return float((r2_real - float(r2_fake_mean)) / denom)
#
#
# def find_roi_pkl(session_dir: Path, roi: str):
#     for p in sorted(session_dir.rglob("*.pkl")):
#         try:
#             with open(p, "rb") as f:
#                 d = pickle.load(f)
#             reg = d.get("region", "")
#             reg = reg[0] if isinstance(reg, (list, tuple)) else str(reg)
#             if str(reg).strip() == str(roi).strip():
#                 return p
#         except Exception:
#             continue
#     return None
#
#
# def _pick_real_pid(fit_list) -> int:
#     """Prefer -1, else 0, else smallest pid seen. Avoids silent NaNs if real pid differs."""
#     pids = []
#     for fr in fit_list:
#         pid = fr.get("pseudo_id", None)
#         if pid is None:
#             continue
#         try:
#             pids.append(int(pid))
#         except Exception:
#             continue
#     if not pids:
#         return -1
#     if -1 in pids:
#         return -1
#     if 0 in pids:
#         return 0
#     return int(min(pids))
#
#
# def compute_group_stats_from_pkl(region_pkl: Path, group_mask_full: np.ndarray):
#     with open(region_pkl, "rb") as f:
#         d = pickle.load(f)
#
#     keep_idx_full = np.asarray(d["keep_idx_full"], dtype=int)
#     group_mask_sub = np.asarray(group_mask_full, dtype=bool)[keep_idx_full]
#     sub_idx = np.flatnonzero(group_mask_sub)
#     n_used = int(sub_idx.size)
#
#     pid_to_r = {}
#     pid_to_r2 = {}
#
#     real_pid = _pick_real_pid(d.get("fit", []))
#
#     for fr in d["fit"]:
#         pid_raw = fr.get("pseudo_id", -999)
#         try:
#             pid = int(pid_raw)
#         except Exception:
#             pid = -999
#
#         preds = fr.get("predictions_test", None)
#         targ  = fr.get("target", None)
#         if preds is None or targ is None:
#             continue
#
#         yhat = _trial_scalar_list(preds)
#         y    = _trial_scalar_list(targ)
#
#         y_sub = y[sub_idx]
#         yhat_sub = yhat[sub_idx]
#
#         r = _pearsonr_safe(yhat_sub, y_sub)
#         r2 = _r2_safe(y_sub, yhat_sub)
#
#         if np.isfinite(r):
#             pid_to_r.setdefault(pid, []).append(r)
#         if np.isfinite(r2):
#             pid_to_r2.setdefault(pid, []).append(r2)
#
#     def mean_or_nan(x):
#         return float(np.mean(x)) if len(x) else np.nan
#
#     r_real = mean_or_nan(pid_to_r.get(real_pid, []))
#     r2_real = mean_or_nan(pid_to_r2.get(real_pid, []))
#
#     fake_ids_r = sorted([pid for pid in pid_to_r.keys() if pid != real_pid])
#     r_fake = np.asarray([mean_or_nan(pid_to_r[pid]) for pid in fake_ids_r], float)
#     r_fake = r_fake[np.isfinite(r_fake)]
#
#     fake_ids_r2 = sorted([pid for pid in pid_to_r2.keys() if pid != real_pid])
#     r2_fake = np.asarray([mean_or_nan(pid_to_r2[pid]) for pid in fake_ids_r2], float)
#     r2_fake = r2_fake[np.isfinite(r2_fake)]
#
#     r_fake_mean, r_fake_sd, z_r, p_emp_r = _z_and_p_from_null(r_real, r_fake, one_sided="greater_equal")
#     r2_fake_mean, r2_fake_sd, z_r2, p_emp_r2 = _z_and_p_from_null(r2_real, r2_fake, one_sided="greater_equal")
#
#     r2_corr = _corrected_r2_findling(r2_real, r2_fake_mean)
#
#     return {
#         "n_used": n_used,
#         "r_real": r_real,
#         "r_fake_mean": r_fake_mean,
#         "r_fake_std": r_fake_sd,
#         "z_corr": z_r,
#         "p_emp": p_emp_r,
#         "r2_real": r2_real,
#         "r2_fake_mean": r2_fake_mean,
#         "r2_fake_std": r2_fake_sd,
#         "r2_corr": r2_corr,
#         "z_r2": z_r2,
#         "p_emp_r2": p_emp_r2,
#         "real_pid": int(real_pid),
#         "n_fake_r": int(r_fake.size),
#         "n_fake_r2": int(r2_fake.size),
#     }
#
#
# # =========================
# # Pretty printing (local-style)
# # =========================
# def _fmt(x, width=7, prec=4):
#     if x is None or (isinstance(x, float) and not np.isfinite(x)):
#         return " " * (width - 3) + "NaN"
#     return f"{float(x):{width}.{prec}f}"
#
#
# def print_session_summary(eid: str, rows: list[dict]):
#     print("\n=== PER-SESSION / PER-GROUP ===")
#     for row in rows:
#         g = row["group"]
#         n = int(row.get("n_trials_group_used", -1))
#
#         r = row.get("r_real", np.nan)
#         rmu = row.get("r_fake_mean", np.nan)
#         rsd = row.get("r_fake_std", np.nan)
#         z  = row.get("z_corr", np.nan)
#         p  = row.get("p_emp", np.nan)
#
#         r2 = row.get("r2_real", np.nan)
#         r2mu = row.get("r2_fake_mean", np.nan)
#         r2sd = row.get("r2_fake_std", np.nan)
#         r2c = row.get("r2_corr", np.nan)
#         z2 = row.get("z_r2", np.nan)
#         p2 = row.get("p_emp_r2", np.nan)
#
#         line = (
#             f"{eid} | {g:<6s} | n={n:4d} | "
#             f"Pearson: r={_fmt(r)} | null={_fmt(rmu)}±{_fmt(rsd)} | z={_fmt(z, width=6, prec=3)} | p={_fmt(p, width=7, prec=4)} || "
#             f"R2: r2={_fmt(r2)} | null={_fmt(r2mu)}±{_fmt(r2sd)} | r2_corr={_fmt(r2c)} | z={_fmt(z2, width=6, prec=3)} | p={_fmt(p2, width=7, prec=4)}"
#         )
#         print(line)
#
#
# # =========================
# # Worker
# # =========================
# def run_one_eid_worker(eid: str, out_root_str: str, cache_root_str: str, n_pseudo: int, debug: bool):
#     warnings.filterwarnings("ignore", category=alferr.ALFWarning)
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     warnings.filterwarnings("ignore", category=RuntimeWarning)
#     warnings.filterwarnings("ignore", message=r".*Multiple revisions.*")
#     warnings.filterwarnings("ignore", message=r".*Newer cache tables require ONE version.*")
#
#     out_root = Path(out_root_str)
#     out_root.mkdir(parents=True, exist_ok=True)
#
#     cache_root = Path(cache_root_str)
#     cache_root.mkdir(parents=True, exist_ok=True)
#
#     pid = os.getpid()
#     worker_cache = cache_root / f"worker_{pid}"
#     worker_cache.mkdir(parents=True, exist_ok=True)
#
#     one = make_one(cache_dir=worker_cache)
#
#     # metadata
#     try:
#         probe_name = one.eid2pid(eid)[1]
#         ses = one.alyx.rest("sessions", "read", id=eid)
#         subject = ses["subject"]
#     except Exception as e:
#         return {"eid": eid, "status": f"fail_one_setup: {e}", "n_raw_trials": -1, "rows": []}
#
#     # trials + wheel RT
#     try:
#         df, n_raw_trials = compute_trials_with_my_rt(one, eid)
#     except Exception as e:
#         return {"eid": eid, "status": f"fail_trials_or_wheel: {e}", "n_raw_trials": -1, "rows": []}
#
#     if int(n_raw_trials) < int(MIN_RAW_TRIALS):
#         return {"eid": eid, "status": "skip_trials", "n_raw_trials": int(n_raw_trials), "rows": []}
#
#     masks, meta = make_fast_normal_slow_masks(df)
#     drop_last_mask = make_drop_last_mask(len(df), DROP_LAST_N)
#
#     # Save counts sanity check
#     meta2 = dict(meta)
#     meta2["drop_last_n"] = int(DROP_LAST_N)
#     meta2["n_after_drop_last"] = int(drop_last_mask.sum())
#     meta2["n_fast_after_drop_last"] = int((masks["fast"] & drop_last_mask).sum())
#     meta2["n_normal_after_drop_last"] = int((masks["normal"] & drop_last_mask).sum())
#     meta2["n_slow_after_drop_last"] = int((masks["slow"] & drop_last_mask).sum())
#     (out_root / f"{eid}_rt_group_counts.json").write_text(json.dumps(meta2, indent=2))
#
#     pseudo_ids = [-1] + list(range(1, n_pseudo + 1))
#
#     session_dir = out_root / eid / "session_fit"
#     session_dir.mkdir(parents=True, exist_ok=True)
#
#     try:
#         with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
#             _ = fit_session_ephys(
#                 one=one,
#                 session_id=eid,
#                 subject=subject,
#                 probe_name=probe_name,
#                 output_dir=session_dir,
#                 pseudo_ids=pseudo_ids,
#                 target="pLeft",
#                 align_event=ALIGN_EVENT,
#                 time_window=TIME_WINDOW,
#                 model="optBay",
#                 n_runs=N_RUNS,
#                 trials_df=df,
#                 trial_mask=drop_last_mask,
#                 group_label="session",
#                 debug=bool(debug),
#                 roi_set=ROI_SET,
#             )
#     except Exception as e:
#         return {"eid": eid, "status": f"fail_fit: {e}", "n_raw_trials": int(n_raw_trials), "rows": []}
#
#     rows = []
#     for roi in ROI_LIST:
#         roi_pkl = find_roi_pkl(session_dir, roi)
#         if roi_pkl is None:
#             continue
#
#         for g in GROUPS:
#             stats = compute_group_stats_from_pkl(roi_pkl, group_mask_full=masks[g])
#
#             row = {
#                 "eid": eid,
#                 "subject": subject,
#                 "probe": probe_name if isinstance(probe_name, str) else str(probe_name),
#                 "roi": roi,
#                 "group": g,
#                 "n_trials_group_used": int(stats["n_used"]),
#
#                 # Pearson
#                 "r_real": stats["r_real"],
#                 "r_fake_mean": stats["r_fake_mean"],
#                 "r_fake_std": stats["r_fake_std"],
#                 "z_corr": stats["z_corr"],
#                 "p_emp": stats["p_emp"],
#
#                 # R^2
#                 "r2_real": stats["r2_real"],
#                 "r2_fake_mean": stats["r2_fake_mean"],
#                 "r2_fake_std": stats["r2_fake_std"],
#                 "r2_corr": stats["r2_corr"],
#                 "z_r2": stats["z_r2"],
#                 "p_emp_r2": stats["p_emp_r2"],
#
#                 # bookkeeping
#                 "drop_last_n": int(DROP_LAST_N),
#                 "roi_pkl_path": str(roi_pkl),
#
#                 # diagnostics
#                 "real_pid": int(stats["real_pid"]),
#                 "n_fake_r": int(stats["n_fake_r"]),
#                 "n_fake_r2": int(stats["n_fake_r2"]),
#             }
#             rows.append(row)
#
#     if SAVE_PER_SESSION_JSON and rows:
#         roi_tag = "_".join(ROI_LIST) if ROI_LIST else "ALL"
#         (out_root / eid / f"pearson_summary_{roi_tag}.json").write_text(
#             json.dumps(rows, indent=2)
#         )
#
#     status = "ok" if rows else "skip_no_roi_or_failed"
#     return {"eid": eid, "status": status, "n_raw_trials": int(n_raw_trials), "rows": rows}
#
#
# def main():
#     OUT_ROOT.mkdir(parents=True, exist_ok=True)
#     CACHE_ROOT.mkdir(parents=True, exist_ok=True)
#
#     print(f"[RUN] RUN_TAG={RUN_TAG}")
#     print(f"[RUN] Writing outputs to: {OUT_ROOT}")
#     print(f"[RUN] N_PSEUDO={N_PSEUDO} N_RUNS={N_RUNS} N_WORKERS={N_WORKERS} DROP_LAST_N={DROP_LAST_N} "
#           f"FAST_THR={FAST_THR} SLOW_THR={SLOW_THR}")
#
#     eids = read_session_ids_from_txt(TXT_PATH)
#
#     all_rows = []
#     run_log = []
#     printed_keys = False
#
#     with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
#         futs = [ex.submit(run_one_eid_worker, eid, str(OUT_ROOT), str(CACHE_ROOT), N_PSEUDO, DEBUG)
#                 for eid in eids]
#
#         for fut in as_completed(futs):
#             r = fut.result()
#             eid = r.get("eid")
#             status = r.get("status")
#             nrt = r.get("n_raw_trials")
#             rows = r.get("rows", [])
#
#             run_log.append({"eid": eid, "status": status, "n_raw_trials": nrt})
#             all_rows.extend(rows)
#
#             print("[DONE]", eid, status, "rows=", len(rows))
#
#             # Print keys once (from first non-empty result)
#             if (not printed_keys) and rows:
#                 print("Keys:", sorted(list(rows[0].keys())))
#                 printed_keys = True
#
#             # Print per-session summary like local script
#             if rows:
#                 # If you want per-ROI separation, you can group here, but
#                 # simplest is to print all rows (ROI can be included in format if desired).
#                 # For now, print per ROI separately for readability:
#                 for roi in sorted({rr["roi"] for rr in rows}):
#                     roi_rows = [rr for rr in rows if rr["roi"] == roi]
#                     print(f"\n--- ROI: {roi} ---")
#                     print_session_summary(eid, roi_rows)
#
#     summary_path = OUT_ROOT / f"pearson_r2_summary_{RUN_TAG}.pkl"
#     with open(summary_path, "wb") as f:
#         pickle.dump(all_rows, f)
#
#     run_log_path = OUT_ROOT / f"run_log_{RUN_TAG}.json"
#     with open(run_log_path, "w") as f:
#         json.dump(run_log, f, indent=2)
#
#     print("Saved big summary:", summary_path)
#     print("Saved run log:", run_log_path)
#
#
# if __name__ == "__main__":
#     main()

#
# from __future__ import annotations
#
# from pathlib import Path
# import json
# import pickle
# import warnings
# import os
# import contextlib
# import numpy as np
# import pandas as pd
#
# from one.api import ONE
# import one.alf.exceptions as alferr
#
# from prior_localization.my_rt import (
#     load_wheel_data, calc_wheel_velocity, calc_trialwise_wheel, calc_movement_onset_times
# )
# from prior_localization.fit_data import fit_session_ephys
#
#
# # =========================
# # DEBUG TARGET (single session)
# # =========================
# EID = "ae8787b1-4229-4d56-b0c2-566b61a25b77"
#
# ROI_LIST = ["MOp"]
# ROI_SET = set(ROI_LIST)
# GROUPS = ["fast", "normal", "slow"]
#
# # Outputs on shared filesystem
# OUT_BASE = Path("/storage1/fs1/hiratani/Active/shared/ibl_space/derived/prior_localization_groups_output_debug")
# RUN_TAG = os.getenv("RUN_TAG", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
# OUT_ROOT = OUT_BASE / f"run_{RUN_TAG}"
#
# # Shared cache root
# CACHE_ROOT = Path("/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org_debug")
#
# N_PSEUDO = int(os.getenv("N_PSEUDO", "200"))
# N_RUNS = int(os.getenv("N_RUNS", "2"))
# DEBUG = True
#
# ALIGN_EVENT = "stimOn_times"
# TIME_WINDOW = (-0.6, -0.1)
# FAST_THR = float(os.getenv("FAST_THR", "0.08"))
# SLOW_THR = float(os.getenv("SLOW_THR", "1.25"))
#
# DROP_LAST_N = int(os.getenv("DROP_LAST_N", "40"))
# MIN_RAW_TRIALS = int(os.getenv("MIN_RAW_TRIALS", "401"))
#
#
# # =========================
# # ONE helper (per-run cache)
# # =========================
# def make_one(cache_dir: Path) -> ONE:
#     cache_dir = Path(cache_dir)
#     cache_dir.mkdir(parents=True, exist_ok=True)
#
#     tables_dir = cache_dir / "tables"
#     dl_cache = cache_dir / "downloads"
#     tables_dir.mkdir(parents=True, exist_ok=True)
#     dl_cache.mkdir(parents=True, exist_ok=True)
#
#     one = ONE(
#         base_url="https://openalyx.internationalbrainlab.org",
#         username=os.getenv("ALYX_LOGIN"),
#         password=os.getenv("ALYX_PASSWORD"),
#         silent=True,
#         cache_dir=dl_cache,
#         tables_dir=tables_dir,
#         cache_rest=None,
#     )
#     return one
#
#
# # =========================
# # Trial + RT helpers
# # =========================
# def compute_trials_with_my_rt(one: ONE, eid: str):
#     trials_obj = one.load_object(eid, "trials", collection="alf")
#     n_raw_trials = len(trials_obj["stimOn_times"])
#
#     data = {}
#     for k, v in trials_obj.items():
#         arr = np.asarray(v)
#         if k == "intervals" and arr.ndim == 2 and arr.shape[1] == 2:
#             data["intervals_start"] = arr[:, 0]
#             data["intervals_end"] = arr[:, 1]
#         elif arr.ndim == 1:
#             data[k] = arr
#         else:
#             data[k] = list(arr)
#     df = pd.DataFrame(data)
#
#     pos, ts = load_wheel_data(one, eid)
#     if pos is None or ts is None:
#         raise RuntimeError(f"No wheel data for eid={eid}")
#
#     vel = calc_wheel_velocity(pos, ts)
#     _, trial_ts, trial_vel = calc_trialwise_wheel(
#         pos, ts, vel, df["stimOn_times"], df["feedback_times"]
#     )
#     (_, first_mo, _, _, _) = calc_movement_onset_times(trial_ts, trial_vel, df["stimOn_times"])
#     df["first_movement_onset_times"] = first_mo
#     df["reaction_time"] = df["first_movement_onset_times"] - df["stimOn_times"]
#
#     return df, int(n_raw_trials)
#
#
# def make_fast_normal_slow_masks(df: pd.DataFrame, rt_col="reaction_time",
#                                fast_thr=FAST_THR, slow_thr=SLOW_THR):
#     rt = df[rt_col].to_numpy()
#     valid = np.isfinite(rt)
#     fast = valid & (rt < fast_thr)
#     slow = valid & (rt > slow_thr)
#     normal = valid & (~fast) & (~slow)
#     return {"fast": fast, "normal": normal, "slow": slow}
#
#
# def make_drop_last_mask(n_trials: int, drop_last_n: int):
#     m = np.ones(int(n_trials), dtype=bool)
#     if drop_last_n and drop_last_n > 0:
#         m[max(0, n_trials - drop_last_n):] = False
#     return m
#
#
# # =========================
# # Stats helpers
# # =========================
# def _trial_scalar_list_lenient(x_list):
#     """
#     list-of-arrays -> scalar per trial (mean over finite bins).
#     Only returns NaN if a trial has *no* finite bins.
#     """
#     out = np.full(len(x_list), np.nan, float)
#     for i, xi in enumerate(x_list):
#         if xi is None:
#             continue
#         arr = np.asarray(xi).reshape(-1).astype(float, copy=False)
#         arr = arr[np.isfinite(arr)]
#         if arr.size == 0:
#             continue
#         out[i] = float(np.mean(arr))
#     return out
#
#
# def _pearsonr_safe(a, b):
#     a = np.asarray(a, float).reshape(-1)
#     b = np.asarray(b, float).reshape(-1)
#     ok = np.isfinite(a) & np.isfinite(b)
#     a = a[ok]; b = b[ok]
#     if a.size < 3:
#         return np.nan, int(a.size), float(np.std(a)) if a.size else np.nan, float(np.std(b)) if b.size else np.nan
#     if np.std(a) == 0 or np.std(b) == 0:
#         return np.nan, int(a.size), float(np.std(a)), float(np.std(b))
#     return float(np.corrcoef(a, b)[0, 1]), int(a.size), float(np.std(a)), float(np.std(b))
#
#
# def _r2_safe(y_true, y_pred):
#     y_true = np.asarray(y_true, float).reshape(-1)
#     y_pred = np.asarray(y_pred, float).reshape(-1)
#     ok = np.isfinite(y_true) & np.isfinite(y_pred)
#     y_true = y_true[ok]; y_pred = y_pred[ok]
#     if y_true.size < 3:
#         return np.nan, int(y_true.size)
#     denom = np.sum((y_true - np.mean(y_true)) ** 2)
#     if denom <= 0:
#         return np.nan, int(y_true.size)
#     sse = np.sum((y_true - y_pred) ** 2)
#     return float(1.0 - sse / denom), int(y_true.size)
#
#
# def _z_and_p_from_null(real_val, null_vals, one_sided="greater_equal"):
#     null_vals = np.asarray(null_vals, float)
#     null_vals = null_vals[np.isfinite(null_vals)]
#     if (not np.isfinite(real_val)) or null_vals.size < 5:
#         return np.nan, np.nan, np.nan, np.nan
#
#     mu = float(np.mean(null_vals))
#     sd = float(np.std(null_vals, ddof=1)) if null_vals.size > 1 else float(np.std(null_vals))
#     z = float((real_val - mu) / sd) if sd > 0 else np.nan
#
#     if one_sided == "greater_equal":
#         p = float((1 + np.sum(null_vals >= real_val)) / (1 + null_vals.size))
#     elif one_sided == "less_equal":
#         p = float((1 + np.sum(null_vals <= real_val)) / (1 + null_vals.size))
#     else:
#         raise ValueError("one_sided must be 'greater_equal' or 'less_equal'")
#
#     return mu, sd, z, p
#
#
# def _corrected_r2_findling(r2_real, r2_fake_mean):
#     if (not np.isfinite(r2_real)) or (not np.isfinite(r2_fake_mean)):
#         return np.nan
#     denom = 1.0 - float(r2_fake_mean)
#     if denom == 0:
#         return np.nan
#     return float((r2_real - float(r2_fake_mean)) / denom)
#
#
# def find_roi_pkl(session_dir: Path, roi: str):
#     # more forgiving: match substring too (common on RIS)
#     roi = str(roi).strip()
#     for p in sorted(session_dir.rglob("*.pkl")):
#         try:
#             with open(p, "rb") as f:
#                 d = pickle.load(f)
#             reg = d.get("region", "")
#             reg = reg[0] if isinstance(reg, (list, tuple)) else str(reg)
#             reg = str(reg).strip()
#             if (reg == roi) or (roi in reg):
#                 return p
#         except Exception:
#             continue
#     return None
#
#
# def pick_real_pid(fit_list) -> int:
#     pids = []
#     for fr in fit_list:
#         pid = fr.get("pseudo_id", None)
#         if pid is None:
#             continue
#         try:
#             pids.append(int(pid))
#         except Exception:
#             continue
#     if not pids:
#         return -1
#     if -1 in pids:
#         return -1
#     if 0 in pids:
#         return 0
#     return int(min(pids))
#
#
# def compute_group_stats_from_pkl(region_pkl: Path, group_mask_full: np.ndarray):
#     with open(region_pkl, "rb") as f:
#         d = pickle.load(f)
#
#     keep_idx_full = np.asarray(d["keep_idx_full"], dtype=int)
#     group_mask_sub = np.asarray(group_mask_full, dtype=bool)[keep_idx_full]
#     sub_idx = np.flatnonzero(group_mask_sub)
#     n_used = int(sub_idx.size)
#
#     fit_list = d.get("fit", [])
#     real_pid = pick_real_pid(fit_list)
#
#     pid_to_r = {}
#     pid_to_r2 = {}
#
#     # diagnostics: count how many trials became NaN after scalarization
#     diag = {
#         "real_pid": int(real_pid),
#         "n_fit_entries": int(len(fit_list)),
#         "n_used_trials_in_group": int(n_used),
#     }
#
#     for fr in fit_list:
#         try:
#             pid = int(fr.get("pseudo_id", -999))
#         except Exception:
#             pid = -999
#
#         preds = fr.get("predictions_test", None)
#         targ  = fr.get("target", None)
#         if preds is None or targ is None:
#             continue
#
#         yhat = _trial_scalar_list_lenient(preds)
#         y    = _trial_scalar_list_lenient(targ)
#
#         # map to subgroup
#         y_sub = y[sub_idx]
#         yhat_sub = yhat[sub_idx]
#
#         r, n_ok_r, sd_yhat, sd_y = _pearsonr_safe(yhat_sub, y_sub)
#         r2, n_ok_r2 = _r2_safe(y_sub, yhat_sub)
#
#         if np.isfinite(r):
#             pid_to_r.setdefault(pid, []).append(r)
#         if np.isfinite(r2):
#             pid_to_r2.setdefault(pid, []).append(r2)
#
#         # record a little extra info for the REAL fit only
#         if pid == real_pid:
#             diag.update({
#                 "real_n_ok_for_r": int(n_ok_r),
#                 "real_sd_yhat": float(sd_yhat) if np.isfinite(sd_yhat) else np.nan,
#                 "real_sd_y": float(sd_y) if np.isfinite(sd_y) else np.nan,
#                 "real_n_ok_for_r2": int(n_ok_r2),
#                 "real_nan_frac_yhat": float(np.mean(~np.isfinite(yhat_sub))) if yhat_sub.size else np.nan,
#                 "real_nan_frac_y": float(np.mean(~np.isfinite(y_sub))) if y_sub.size else np.nan,
#             })
#
#     def mean_or_nan(x):
#         return float(np.mean(x)) if len(x) else np.nan
#
#     r_real = mean_or_nan(pid_to_r.get(real_pid, []))
#     r2_real = mean_or_nan(pid_to_r2.get(real_pid, []))
#
#     fake_ids_r = sorted([pid for pid in pid_to_r.keys() if pid != real_pid])
#     r_fake = np.asarray([mean_or_nan(pid_to_r[pid]) for pid in fake_ids_r], float)
#     r_fake = r_fake[np.isfinite(r_fake)]
#
#     fake_ids_r2 = sorted([pid for pid in pid_to_r2.keys() if pid != real_pid])
#     r2_fake = np.asarray([mean_or_nan(pid_to_r2[pid]) for pid in fake_ids_r2], float)
#     r2_fake = r2_fake[np.isfinite(r2_fake)]
#
#     r_fake_mean, r_fake_sd, z_r, p_emp_r = _z_and_p_from_null(r_real, r_fake)
#     r2_fake_mean, r2_fake_sd, z_r2, p_emp_r2 = _z_and_p_from_null(r2_real, r2_fake)
#
#     r2_corr = _corrected_r2_findling(r2_real, r2_fake_mean)
#
#     return {
#         "n_used": n_used,
#         "r_real": r_real,
#         "r_fake_mean": r_fake_mean,
#         "r_fake_std": r_fake_sd,
#         "z_corr": z_r,
#         "p_emp": p_emp_r,
#         "r2_real": r2_real,
#         "r2_fake_mean": r2_fake_mean,
#         "r2_fake_std": r2_fake_sd,
#         "r2_corr": r2_corr,
#         "z_r2": z_r2,
#         "p_emp_r2": p_emp_r2,
#         "diag": diag,
#     }
#
#
# # =========================
# # Main
# # =========================
# def main():
#     warnings.filterwarnings("ignore", category=alferr.ALFWarning)
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     warnings.filterwarnings("ignore", category=RuntimeWarning)
#
#     OUT_ROOT.mkdir(parents=True, exist_ok=True)
#     CACHE_ROOT.mkdir(parents=True, exist_ok=True)
#
#     print(f"[RUN] EID={EID}")
#     print(f"[RUN] OUT_ROOT={OUT_ROOT}")
#     print(f"[RUN] CACHE_ROOT={CACHE_ROOT}")
#     print(f"[RUN] ROI_LIST={ROI_LIST} N_PSEUDO={N_PSEUDO} N_RUNS={N_RUNS} DROP_LAST_N={DROP_LAST_N}")
#
#     worker_cache = CACHE_ROOT / f"worker_single_{os.getpid()}"
#     one = make_one(worker_cache)
#
#     # metadata
#     probe_name = one.eid2pid(EID)[1]
#     ses = one.alyx.rest("sessions", "read", id=EID)
#     subject = ses["subject"]
#     print(f"[META] subject={subject} probe={probe_name}")
#
#     # trials + RT
#     df, n_raw_trials = compute_trials_with_my_rt(one, EID)
#     print(f"[TRIALS] n_raw_trials={n_raw_trials}")
#     if n_raw_trials < MIN_RAW_TRIALS:
#         print("[SKIP] too few trials")
#         return
#
#     masks = make_fast_normal_slow_masks(df)
#     drop_last_mask = make_drop_last_mask(len(df), DROP_LAST_N)
#
#     # fit
#     pseudo_ids = [-1] + list(range(1, N_PSEUDO + 1))
#     session_dir = OUT_ROOT / EID / "session_fit"
#     session_dir.mkdir(parents=True, exist_ok=True)
#
#     print(f"[FIT] writing to {session_dir}")
#
#     # NOTE: do NOT redirect stdout while debugging — we want messages
#     _ = fit_session_ephys(
#         one=one,
#         session_id=EID,
#         subject=subject,
#         probe_name=probe_name,
#         output_dir=session_dir,
#         pseudo_ids=pseudo_ids,
#         target="pLeft",
#         align_event=ALIGN_EVENT,
#         time_window=TIME_WINDOW,
#         model="optBay",
#         n_runs=N_RUNS,
#         trials_df=df,
#         trial_mask=drop_last_mask,
#         group_label="session",
#         debug=True,
#         roi_set=ROI_SET,
#     )
#
#     # find ROI pkl
#     roi = ROI_LIST[0]
#     roi_pkl = find_roi_pkl(session_dir, roi)
#     print(f"[ROI] requested={roi} found_pkl={roi_pkl}")
#
#     if roi_pkl is None:
#         # list what regions exist
#         print("[ROI] Could not match ROI. Listing regions found in pkls:")
#         for p in sorted(session_dir.rglob("*.pkl"))[:50]:
#             try:
#                 d = pickle.load(open(p, "rb"))
#                 reg = d.get("region", None)
#                 if isinstance(reg, (list, tuple)) and reg:
#                     reg = reg[0]
#                 print(" ", p.name, "region=", reg)
#             except Exception:
#                 pass
#         return
#
#     # compute stats per group
#     rows = []
#     for g in GROUPS:
#         stats = compute_group_stats_from_pkl(roi_pkl, group_mask_full=masks[g])
#         row = {
#             "eid": EID,
#             "subject": subject,
#             "probe": str(probe_name),
#             "roi": roi,
#             "group": g,
#             "n_trials_group_used": int(stats["n_used"]),
#             "r_real": stats["r_real"],
#             "r_fake_mean": stats["r_fake_mean"],
#             "r_fake_std": stats["r_fake_std"],
#             "z_corr": stats["z_corr"],
#             "p_emp": stats["p_emp"],
#             "r2_real": stats["r2_real"],
#             "r2_fake_mean": stats["r2_fake_mean"],
#             "r2_fake_std": stats["r2_fake_std"],
#             "r2_corr": stats["r2_corr"],
#             "z_r2": stats["z_r2"],
#             "p_emp_r2": stats["p_emp_r2"],
#             "roi_pkl_path": str(roi_pkl),
#             "diag": stats["diag"],
#         }
#         rows.append(row)
#
#     # pretty print
#     print("\n=== PER-SESSION / PER-GROUP (RIS debug) ===")
#     for r in rows:
#         d = r["diag"]
#         print(
#             f"{EID} | {r['group']:<6} | n={r['n_trials_group_used']:4d} | "
#             f"Pearson r={r['r_real']} (null {r['r_fake_mean']}±{r['r_fake_std']}) z={r['z_corr']} p={r['p_emp']} || "
#             f"R2={r['r2_real']} (null {r['r2_fake_mean']}±{r['r2_fake_std']}) r2_corr={r['r2_corr']} z={r['z_r2']} p={r['p_emp_r2']}\n"
#             f"    diag: real_pid={d.get('real_pid')} n_ok_for_r={d.get('real_n_ok_for_r')} "
#             f"nan_frac_yhat={d.get('real_nan_frac_yhat')} nan_frac_y={d.get('real_nan_frac_y')} "
#             f"sd_yhat={d.get('real_sd_yhat')} sd_y={d.get('real_sd_y')}"
#         )
#
#     # save a json for you to inspect
#     out_json = OUT_ROOT / EID / "pearson_summary_MOp_debug.json"
#     out_json.write_text(json.dumps(rows, indent=2, default=str))
#     print("\nSaved:", out_json)
#
#
# if __name__ == "__main__":
#     main()
#

# from __future__ import annotations
#
# from pathlib import Path
# import json
# import pickle
# import warnings
# import os
# import numpy as np
# import pandas as pd
# from concurrent.futures import ProcessPoolExecutor, as_completed
#
# from one.api import ONE
# import one.alf.exceptions as alferr
#
# from prior_localization.my_rt import (
#     load_wheel_data, calc_wheel_velocity, calc_trialwise_wheel, calc_movement_onset_times
# )
# from prior_localization.fit_data import fit_session_ephys
#
#
# # =========================
# # RIS: make runtime deterministic-ish (avoid BLAS thread weirdness)
# # =========================
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
#
#
# # =========================
# # USER SETTINGS (match your local)
# # =========================
# ROI_LIST = ["MOp"]
# ROI_SET = set(ROI_LIST)
# GROUPS = ["fast", "normal", "slow"]
#
# # RIS outputs (shared filesystem)
# OUT_BASE = Path("/storage1/fs1/hiratani/Active/shared/ibl_space/derived/prior_localization_sessionfit_output")
# RUN_TAG = os.getenv("RUN_TAG", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
# OUT_ROOT = OUT_BASE / f"run_{RUN_TAG}"
#
# # RIS cache root (shared filesystem)
# CACHE_ROOT = Path("/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org_cache")
#
# N_PSEUDO = int(os.getenv("N_PSEUDO", "200"))
# N_RUNS = int(os.getenv("N_RUNS", "2"))
# N_WORKERS = int(os.getenv("N_WORKERS", "4"))
# DEBUG = bool(int(os.getenv("DEBUG", "0")))
#
# ALIGN_EVENT = "stimOn_times"
# TIME_WINDOW = (-0.6, -0.1)
# FAST_THR = float(os.getenv("FAST_THR", "0.08"))
# SLOW_THR = float(os.getenv("SLOW_THR", "1.25"))
#
# DROP_LAST_N = int(os.getenv("DROP_LAST_N", "40"))
# MIN_RAW_TRIALS = int(os.getenv("MIN_RAW_TRIALS", "401"))
#
#
# # =========================
# # ONE helper (per-worker cache dirs)
# # =========================
# def make_one(worker_cache: Path) -> ONE:
#     worker_cache = Path(worker_cache)
#     tables_dir = worker_cache / "tables"
#     dl_cache = worker_cache / "downloads"
#     tables_dir.mkdir(parents=True, exist_ok=True)
#     dl_cache.mkdir(parents=True, exist_ok=True)
#
#     # IMPORTANT: keep this stable; do NOT rely on local user config on RIS
#     one = ONE(
#         base_url="https://openalyx.internationalbrainlab.org",
#         username=os.getenv("ALYX_LOGIN"),
#         password=os.getenv("ALYX_PASSWORD"),
#         silent=True,
#         cache_dir=dl_cache,
#         tables_dir=tables_dir,
#         cache_rest=None,
#     )
#     return one
#
#
# # =========================
# # Trial + RT helpers (IDENTICAL to your local)
# # =========================
# def compute_trials_with_my_rt(one: ONE, eid: str):
#     trials_obj = one.load_object(eid, "trials", collection="alf")
#     n_raw_trials = len(trials_obj["stimOn_times"])
#
#     data = {}
#     for k, v in trials_obj.items():
#         arr = np.asarray(v)
#         if k == "intervals" and arr.ndim == 2 and arr.shape[1] == 2:
#             data["intervals_start"] = arr[:, 0]
#             data["intervals_end"] = arr[:, 1]
#         elif arr.ndim == 1:
#             data[k] = arr
#         else:
#             data[k] = list(arr)
#     df = pd.DataFrame(data)
#
#     pos, ts = load_wheel_data(one, eid)
#     if pos is None or ts is None:
#         raise RuntimeError(f"No wheel data for eid={eid}")
#
#     vel = calc_wheel_velocity(pos, ts)
#     _, trial_ts, trial_vel = calc_trialwise_wheel(pos, ts, vel, df["stimOn_times"], df["feedback_times"])
#     (_, first_mo, _, _, _) = calc_movement_onset_times(trial_ts, trial_vel, df["stimOn_times"])
#     df["first_movement_onset_times"] = first_mo
#     df["reaction_time"] = df["first_movement_onset_times"] - df["stimOn_times"]
#
#     return df, int(n_raw_trials)
#
#
# def make_fast_normal_slow_masks(df: pd.DataFrame, rt_col="reaction_time", fast_thr=FAST_THR, slow_thr=SLOW_THR):
#     rt = df[rt_col].to_numpy()
#     valid = np.isfinite(rt)
#     fast = valid & (rt < fast_thr)
#     slow = valid & (rt > slow_thr)
#     normal = valid & (~fast) & (~slow)
#     return {"fast": fast, "normal": normal, "slow": slow}, {
#         "n_total": int(len(df)),
#         "n_valid_rt": int(valid.sum()),
#         "n_fast": int(fast.sum()),
#         "n_normal": int(normal.sum()),
#         "n_slow": int(slow.sum()),
#         "fast_thr": float(fast_thr),
#         "slow_thr": float(slow_thr),
#     }
#
#
# def make_drop_last_mask(n_trials: int, drop_last_n: int):
#     m = np.ones(int(n_trials), dtype=bool)
#     if drop_last_n and drop_last_n > 0:
#         m[max(0, n_trials - drop_last_n):] = False
#     return m
#
#
# # =========================
# # Stats helpers (IDENTICAL to your local)
# # =========================
# def _trial_scalar_list(x_list):
#     """list-of-arrays -> scalar per trial (mean over bins). returns vec (n_trials,) with NaNs for bad."""
#     out = np.full(len(x_list), np.nan, float)
#     for i, xi in enumerate(x_list):
#         if xi is None:
#             continue
#         arr = np.asarray(xi).reshape(-1)
#         if arr.size == 0 or (not np.all(np.isfinite(arr))):
#             continue
#         out[i] = float(np.mean(arr))
#     return out
#
#
# def _pearsonr_safe(a, b):
#     a = np.asarray(a, float).reshape(-1)
#     b = np.asarray(b, float).reshape(-1)
#     ok = np.isfinite(a) & np.isfinite(b)
#     a = a[ok]; b = b[ok]
#     if a.size < 3:
#         return np.nan
#     if np.std(a) == 0 or np.std(b) == 0:
#         return np.nan
#     return float(np.corrcoef(a, b)[0, 1])
#
#
# def _r2_safe(y_true, y_pred):
#     y_true = np.asarray(y_true, float).reshape(-1)
#     y_pred = np.asarray(y_pred, float).reshape(-1)
#     ok = np.isfinite(y_true) & np.isfinite(y_pred)
#     y_true = y_true[ok]; y_pred = y_pred[ok]
#     if y_true.size < 3:
#         return np.nan
#     denom = np.sum((y_true - np.mean(y_true)) ** 2)
#     if denom <= 0:
#         return np.nan
#     sse = np.sum((y_true - y_pred) ** 2)
#     return float(1.0 - sse / denom)
#
#
# def _z_and_p_from_null(real_val, null_vals, one_sided="greater_equal"):
#     """Return z-score and empirical p based on null distribution."""
#     null_vals = np.asarray(null_vals, float)
#     null_vals = null_vals[np.isfinite(null_vals)]
#     if (not np.isfinite(real_val)) or null_vals.size < 5:
#         return np.nan, np.nan, np.nan, np.nan
#
#     mu = float(np.mean(null_vals))
#     sd = float(np.std(null_vals, ddof=1)) if null_vals.size > 1 else float(np.std(null_vals))
#     z = float((real_val - mu) / sd) if sd > 0 else np.nan
#
#     if one_sided == "greater_equal":
#         p = float((1 + np.sum(null_vals >= real_val)) / (1 + null_vals.size))
#     elif one_sided == "less_equal":
#         p = float((1 + np.sum(null_vals <= real_val)) / (1 + null_vals.size))
#     else:
#         raise ValueError("one_sided must be 'greater_equal' or 'less_equal'")
#
#     return mu, sd, z, p
#
#
# def _corrected_r2_findling(r2_real, r2_fake_mean):
#     """
#     Findling-style correction:
#         R2_corr = (R2_real - mean(R2_fake)) / (1 - mean(R2_fake))
#     """
#     if (not np.isfinite(r2_real)) or (not np.isfinite(r2_fake_mean)):
#         return np.nan
#     denom = 1.0 - float(r2_fake_mean)
#     if denom == 0:
#         return np.nan
#     return float((r2_real - float(r2_fake_mean)) / denom)
#
#
# def compute_group_stats_from_pkl(region_pkl: Path, group_mask_full: np.ndarray):
#     """
#     group_mask_full: boolean mask in FULL df trial space (len = len(df))
#     Uses d['keep_idx_full'] to map full-trial mask into decoded-trial order.
#     """
#     with open(region_pkl, "rb") as f:
#         d = pickle.load(f)
#
#     keep_idx_full = np.asarray(d["keep_idx_full"], dtype=int)
#
#     group_mask_sub = np.asarray(group_mask_full, dtype=bool)[keep_idx_full]
#     sub_idx = np.flatnonzero(group_mask_sub)
#     n_used = int(sub_idx.size)
#
#     pid_to_r = {}
#     pid_to_r2 = {}
#
#     for fr in d["fit"]:
#         pid = int(fr.get("pseudo_id", -999))
#         preds = fr.get("predictions_test", None)
#         targ  = fr.get("target", None)
#         if preds is None or targ is None:
#             continue
#
#         yhat = _trial_scalar_list(preds)
#         y    = _trial_scalar_list(targ)
#
#         y_sub = y[sub_idx]
#         yhat_sub = yhat[sub_idx]
#
#         r = _pearsonr_safe(yhat_sub, y_sub)
#         r2 = _r2_safe(y_sub, yhat_sub)
#
#         if np.isfinite(r):
#             pid_to_r.setdefault(pid, []).append(r)
#         if np.isfinite(r2):
#             pid_to_r2.setdefault(pid, []).append(r2)
#
#     def mean_or_nan(x):
#         return float(np.mean(x)) if len(x) else np.nan
#
#     r_real = mean_or_nan(pid_to_r.get(-1, []))
#     r2_real = mean_or_nan(pid_to_r2.get(-1, []))
#
#     fake_ids_r = sorted([pid for pid in pid_to_r.keys() if pid != -1])
#     r_fake = np.asarray([mean_or_nan(pid_to_r[pid]) for pid in fake_ids_r], float)
#     r_fake = r_fake[np.isfinite(r_fake)]
#
#     fake_ids_r2 = sorted([pid for pid in pid_to_r2.keys() if pid != -1])
#     r2_fake = np.asarray([mean_or_nan(pid_to_r2[pid]) for pid in fake_ids_r2], float)
#     r2_fake = r2_fake[np.isfinite(r2_fake)]
#
#     r_fake_mean, r_fake_sd, z_r, p_emp_r = _z_and_p_from_null(r_real, r_fake, one_sided="greater_equal")
#     r2_fake_mean, r2_fake_sd, z_r2, p_emp_r2 = _z_and_p_from_null(r2_real, r2_fake, one_sided="greater_equal")
#
#     r2_corr = _corrected_r2_findling(r2_real, r2_fake_mean)
#
#     return {
#         "n_used": n_used,
#         "r_real": r_real,
#         "r_fake": r_fake,
#         "r_fake_mean": r_fake_mean,
#         "r_fake_std": r_fake_sd,
#         "z_corr": z_r,
#         "p_emp": p_emp_r,
#         "r2_real": r2_real,
#         "r2_fake": r2_fake,
#         "r2_fake_mean": r2_fake_mean,
#         "r2_fake_std": r2_fake_sd,
#         "z_r2": z_r2,
#         "p_emp_r2": p_emp_r2,
#         "r2_corr": r2_corr,
#     }
#
#
# def find_roi_pkl(session_dir: Path, roi: str):
#     for p in sorted(session_dir.rglob("*.pkl")):
#         try:
#             with open(p, "rb") as f:
#                 d = pickle.load(f)
#             reg = d.get("region", "")
#             reg = reg[0] if isinstance(reg, (list, tuple)) else str(reg)
#             if str(reg).strip() == str(roi).strip():
#                 return p
#         except Exception:
#             continue
#     return None
#
#
# # =========================
# # RIS worker (same logic as local; only ONE/cache differs)
# # =========================
# def _pick_probe_name_like_local(one: ONE, eid: str):
#     """
#     Your local code used: one.eid2pid(eid)[1]
#     On RIS, ordering can differ; we keep the SAME rule but also log the list.
#     """
#     pids = one.eid2pid(eid)
#     # Log for reproducibility
#     return pids, (pids[1] if len(pids) > 1 else pids[0])
#
#
# def run_one_eid_worker(eid: str, out_root_str: str, n_pseudo: int, debug: bool, cache_root_str: str):
#     warnings.filterwarnings("ignore", category=alferr.ALFWarning)
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     warnings.filterwarnings("ignore", category=RuntimeWarning)
#
#     out_root = Path(out_root_str)
#     out_root.mkdir(parents=True, exist_ok=True)
#
#     cache_root = Path(cache_root_str)
#     cache_root.mkdir(parents=True, exist_ok=True)
#     worker_cache = cache_root / f"worker_{os.getpid()}"
#     one = make_one(worker_cache)
#
#     # Probe/subject metadata
#     pids, probe_name = _pick_probe_name_like_local(one, eid)
#     ses = one.alyx.rest("sessions", "read", id=eid)
#     subject = ses["subject"]
#
#     # Trials + RT
#     df, n_raw_trials = compute_trials_with_my_rt(one, eid)
#     if n_raw_trials < MIN_RAW_TRIALS:
#         return {"eid": eid, "status": "skip_trials", "n_raw_trials": n_raw_trials, "rows": []}
#
#     masks, meta = make_fast_normal_slow_masks(df)
#     drop_last_mask = make_drop_last_mask(len(df), DROP_LAST_N)
#
#     meta2 = dict(meta)
#     meta2.update({
#         "drop_last_n": int(DROP_LAST_N),
#         "n_after_drop_last": int(drop_last_mask.sum()),
#         "n_fast_after_drop_last": int((masks["fast"] & drop_last_mask).sum()),
#         "n_normal_after_drop_last": int((masks["normal"] & drop_last_mask).sum()),
#         "n_slow_after_drop_last": int((masks["slow"] & drop_last_mask).sum()),
#         "eid2pid_list": [str(x) for x in pids],
#         "probe_picked": str(probe_name),
#         "worker_cache": str(worker_cache),
#     })
#     (out_root / f"{eid}_rt_group_counts.json").write_text(json.dumps(meta2, indent=2))
#
#     pseudo_ids = [-1] + list(range(1, n_pseudo + 1))
#
#     # Fit (same params as local)
#     session_dir = out_root / eid / "session_fit"
#     session_dir.mkdir(parents=True, exist_ok=True)
#
#     _ = fit_session_ephys(
#         one=one,
#         session_id=eid,
#         subject=subject,
#         probe_name=probe_name,
#         output_dir=session_dir,
#         pseudo_ids=pseudo_ids,
#         target="pLeft",
#         align_event=ALIGN_EVENT,
#         time_window=TIME_WINDOW,
#         model="optBay",
#         n_runs=N_RUNS,
#         trials_df=df,
#         trial_mask=drop_last_mask,   # drop last at FIT TIME
#         group_label="session",
#         debug=bool(debug),
#         roi_set=ROI_SET,
#     )
#
#     roi = ROI_LIST[0]
#     roi_pkl = find_roi_pkl(session_dir, roi)
#     if roi_pkl is None:
#         return {"eid": eid, "status": "skip_no_roi", "n_raw_trials": n_raw_trials, "rows": []}
#
#     # IMPORTANT: do NOT drop last again here.
#     rows = []
#     for g in GROUPS:
#         stats = compute_group_stats_from_pkl(roi_pkl, group_mask_full=masks[g])
#         rows.append({
#             "eid": eid,
#             "subject": subject,
#             "probe": str(probe_name),
#             "roi": roi,
#             "group": g,
#             "n_trials_group_used": int(stats["n_used"]),
#             "r_real": stats["r_real"],
#             "r_fake_mean": stats["r_fake_mean"],
#             "r_fake_std": stats["r_fake_std"],
#             "z_corr": stats["z_corr"],
#             "p_emp": stats["p_emp"],
#             "r2_real": stats["r2_real"],
#             "r2_fake_mean": stats["r2_fake_mean"],
#             "r2_fake_std": stats["r2_fake_std"],
#             "r2_corr": stats["r2_corr"],
#             "z_r2": stats["z_r2"],
#             "p_emp_r2": stats["p_emp_r2"],
#             "drop_last_n": int(DROP_LAST_N),
#             "roi_pkl_path": str(roi_pkl),
#         })
#
#     (out_root / eid / f"pearson_summary_{roi}.json").write_text(
#         json.dumps(rows, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
#     )
#
#     return {"eid": eid, "status": "ok", "n_raw_trials": n_raw_trials, "rows": rows}
#
#
# def main():
#     OUT_ROOT.mkdir(parents=True, exist_ok=True)
#     CACHE_ROOT.mkdir(parents=True, exist_ok=True)
#
#     # Put your EIDs here (same as local)
#     eids = [
#         "ae8787b1-4229-4d56-b0c2-566b61a25b77",
#         # add more
#     ]
#
#     all_rows = []
#     run_log = []
#
#     with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
#         futs = [
#             ex.submit(run_one_eid_worker, eid, str(OUT_ROOT), N_PSEUDO, DEBUG, str(CACHE_ROOT))
#             for eid in eids
#         ]
#         for fut in as_completed(futs):
#             r = fut.result()
#             run_log.append({k: r.get(k) for k in ["eid", "status", "n_raw_trials"]})
#             all_rows.extend(r.get("rows", []))
#             print("[DONE]", r.get("eid"), r.get("status"), "rows=", len(r.get("rows", [])))
#
#     roi_tag = "_".join(ROI_LIST) if ROI_LIST else "ALL"
#     summary_path = OUT_ROOT / f"pearson_summary_{roi_tag}.pkl"
#     with open(summary_path, "wb") as f:
#         pickle.dump(all_rows, f)
#
#     run_log_path = OUT_ROOT / f"run_log_{roi_tag}.json"
#     with open(run_log_path, "w") as f:
#         json.dump(run_log, f, indent=2)
#
#     print("Saved pearson summary:", summary_path)
#     print("Saved run log:", run_log_path)
#
#
# if __name__ == "__main__":
#     main()

#
# from __future__ import annotations
#
# from pathlib import Path
# import json
# import pickle
# import warnings
# import os
# import numpy as np
# import pandas as pd
# from concurrent.futures import ProcessPoolExecutor, as_completed
#
# from one.api import ONE
# import one.alf.exceptions as alferr
#
# from prior_localization.my_rt import (
#     load_wheel_data, calc_wheel_velocity, calc_trialwise_wheel, calc_movement_onset_times
# )
# from prior_localization.fit_data import fit_session_ephys
#
#
# # =========================
# # RIS: make runtime deterministic-ish (avoid BLAS thread weirdness)
# # =========================
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
#
#
# # =========================
# # USER SETTINGS (match your local)
# # =========================
# ROI_LIST = ["MOp"]
# ROI_SET = set(ROI_LIST)
# GROUPS = ["fast", "normal", "slow"]
#
# # RIS outputs (shared filesystem)
# OUT_BASE = Path("/storage1/fs1/hiratani/Active/shared/ibl_space/derived/prior_localization_sessionfit_output")
# RUN_TAG = os.getenv("RUN_TAG", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
# OUT_ROOT = OUT_BASE / f"run_{RUN_TAG}"
#
# # RIS cache root (shared filesystem)
# CACHE_ROOT = Path("/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org_cache")
#
# N_PSEUDO = int(os.getenv("N_PSEUDO", "200"))
# N_RUNS = int(os.getenv("N_RUNS", "2"))
# N_WORKERS = int(os.getenv("N_WORKERS", "4"))
# DEBUG = bool(int(os.getenv("DEBUG", "0")))
#
# ALIGN_EVENT = "stimOn_times"
# TIME_WINDOW = (-0.6, -0.1)
# FAST_THR = float(os.getenv("FAST_THR", "0.08"))
# SLOW_THR = float(os.getenv("SLOW_THR", "1.25"))
#
# DROP_LAST_N = int(os.getenv("DROP_LAST_N", "40"))
# MIN_RAW_TRIALS = int(os.getenv("MIN_RAW_TRIALS", "401"))
#
#
# # =========================
# # ONE helper (per-worker cache dirs)
# # =========================
# def make_one(worker_cache: Path) -> ONE:
#     worker_cache = Path(worker_cache)
#     tables_dir = worker_cache / "tables"
#     dl_cache = worker_cache / "downloads"
#     tables_dir.mkdir(parents=True, exist_ok=True)
#     dl_cache.mkdir(parents=True, exist_ok=True)
#
#     one = ONE(
#         base_url="https://openalyx.internationalbrainlab.org",
#         username=os.getenv("ALYX_LOGIN"),
#         password=os.getenv("ALYX_PASSWORD"),
#         silent=True,
#         cache_dir=dl_cache,
#         tables_dir=tables_dir,
#         cache_rest=None,
#     )
#     return one
#
#
# # =========================
# # Trial + RT helpers (IDENTICAL to your local)
# # =========================
# def compute_trials_with_my_rt(one: ONE, eid: str):
#     trials_obj = one.load_object(eid, "trials", collection="alf")
#     n_raw_trials = len(trials_obj["stimOn_times"])
#
#     data = {}
#     for k, v in trials_obj.items():
#         arr = np.asarray(v)
#         if k == "intervals" and arr.ndim == 2 and arr.shape[1] == 2:
#             data["intervals_start"] = arr[:, 0]
#             data["intervals_end"] = arr[:, 1]
#         elif arr.ndim == 1:
#             data[k] = arr
#         else:
#             data[k] = list(arr)
#     df = pd.DataFrame(data)
#
#     pos, ts = load_wheel_data(one, eid)
#     if pos is None or ts is None:
#         raise RuntimeError(f"No wheel data for eid={eid}")
#
#     vel = calc_wheel_velocity(pos, ts)
#     _, trial_ts, trial_vel = calc_trialwise_wheel(pos, ts, vel, df["stimOn_times"], df["feedback_times"])
#     (_, first_mo, _, _, _) = calc_movement_onset_times(trial_ts, trial_vel, df["stimOn_times"])
#     df["first_movement_onset_times"] = first_mo
#     df["reaction_time"] = df["first_movement_onset_times"] - df["stimOn_times"]
#
#     return df, int(n_raw_trials)
#
#
# def make_fast_normal_slow_masks(df: pd.DataFrame, rt_col="reaction_time", fast_thr=FAST_THR, slow_thr=SLOW_THR):
#     rt = df[rt_col].to_numpy()
#     valid = np.isfinite(rt)
#     fast = valid & (rt < fast_thr)
#     slow = valid & (rt > slow_thr)
#     normal = valid & (~fast) & (~slow)
#     return {"fast": fast, "normal": normal, "slow": slow}, {
#         "n_total": int(len(df)),
#         "n_valid_rt": int(valid.sum()),
#         "n_fast": int(fast.sum()),
#         "n_normal": int(normal.sum()),
#         "n_slow": int(slow.sum()),
#         "fast_thr": float(fast_thr),
#         "slow_thr": float(slow_thr),
#     }
#
#
# def make_drop_last_mask(n_trials: int, drop_last_n: int):
#     m = np.ones(int(n_trials), dtype=bool)
#     if drop_last_n and drop_last_n > 0:
#         m[max(0, n_trials - drop_last_n):] = False
#     return m
#
#
# # =========================
# # Stats helpers (IDENTICAL to your local)
# # =========================
# def _trial_scalar_list(x_list):
#     """list-of-arrays -> scalar per trial (mean over bins). returns vec (n_trials,) with NaNs for bad."""
#     out = np.full(len(x_list), np.nan, float)
#     for i, xi in enumerate(x_list):
#         if xi is None:
#             continue
#         arr = np.asarray(xi).reshape(-1)
#         if arr.size == 0 or (not np.all(np.isfinite(arr))):
#             continue
#         out[i] = float(np.mean(arr))
#     return out
#
#
# def _pearsonr_safe(a, b):
#     a = np.asarray(a, float).reshape(-1)
#     b = np.asarray(b, float).reshape(-1)
#     ok = np.isfinite(a) & np.isfinite(b)
#     a = a[ok]; b = b[ok]
#     if a.size < 3:
#         return np.nan
#     if np.std(a) == 0 or np.std(b) == 0:
#         return np.nan
#     return float(np.corrcoef(a, b)[0, 1])
#
#
# def _r2_safe(y_true, y_pred):
#     y_true = np.asarray(y_true, float).reshape(-1)
#     y_pred = np.asarray(y_pred, float).reshape(-1)
#     ok = np.isfinite(y_true) & np.isfinite(y_pred)
#     y_true = y_true[ok]; y_pred = y_pred[ok]
#     if y_true.size < 3:
#         return np.nan
#     denom = np.sum((y_true - np.mean(y_true)) ** 2)
#     if denom <= 0:
#         return np.nan
#     sse = np.sum((y_true - y_pred) ** 2)
#     return float(1.0 - sse / denom)
#
#
# def _z_and_p_from_null(real_val, null_vals, one_sided="greater_equal"):
#     """Return z-score and empirical p based on null distribution."""
#     null_vals = np.asarray(null_vals, float)
#     null_vals = null_vals[np.isfinite(null_vals)]
#     if (not np.isfinite(real_val)) or null_vals.size < 5:
#         return np.nan, np.nan, np.nan, np.nan
#
#     mu = float(np.mean(null_vals))
#     sd = float(np.std(null_vals, ddof=1)) if null_vals.size > 1 else float(np.std(null_vals))
#     z = float((real_val - mu) / sd) if sd > 0 else np.nan
#
#     if one_sided == "greater_equal":
#         p = float((1 + np.sum(null_vals >= real_val)) / (1 + null_vals.size))
#     elif one_sided == "less_equal":
#         p = float((1 + np.sum(null_vals <= real_val)) / (1 + null_vals.size))
#     else:
#         raise ValueError("one_sided must be 'greater_equal' or 'less_equal'")
#
#     return mu, sd, z, p
#
#
# def _corrected_r2_findling(r2_real, r2_fake_mean):
#     if (not np.isfinite(r2_real)) or (not np.isfinite(r2_fake_mean)):
#         return np.nan
#     denom = 1.0 - float(r2_fake_mean)
#     if denom == 0:
#         return np.nan
#     return float((r2_real - float(r2_fake_mean)) / denom)
#
#
# def compute_group_stats_from_pkl(region_pkl: Path, group_mask_full: np.ndarray):
#     with open(region_pkl, "rb") as f:
#         d = pickle.load(f)
#
#     keep_idx_full = np.asarray(d["keep_idx_full"], dtype=int)
#     group_mask_sub = np.asarray(group_mask_full, dtype=bool)[keep_idx_full]
#     sub_idx = np.flatnonzero(group_mask_sub)
#     n_used = int(sub_idx.size)
#
#     pid_to_r = {}
#     pid_to_r2 = {}
#
#     for fr in d["fit"]:
#         pid = int(fr.get("pseudo_id", -999))
#         preds = fr.get("predictions_test", None)
#         targ  = fr.get("target", None)
#         if preds is None or targ is None:
#             continue
#
#         yhat = _trial_scalar_list(preds)
#         y    = _trial_scalar_list(targ)
#
#         y_sub = y[sub_idx]
#         yhat_sub = yhat[sub_idx]
#
#         r = _pearsonr_safe(yhat_sub, y_sub)
#         r2 = _r2_safe(y_sub, yhat_sub)
#
#         if np.isfinite(r):
#             pid_to_r.setdefault(pid, []).append(r)
#         if np.isfinite(r2):
#             pid_to_r2.setdefault(pid, []).append(r2)
#
#     def mean_or_nan(x):
#         return float(np.mean(x)) if len(x) else np.nan
#
#     r_real = mean_or_nan(pid_to_r.get(-1, []))
#     r2_real = mean_or_nan(pid_to_r2.get(-1, []))
#
#     fake_ids_r = sorted([pid for pid in pid_to_r.keys() if pid != -1])
#     r_fake = np.asarray([mean_or_nan(pid_to_r[pid]) for pid in fake_ids_r], float)
#     r_fake = r_fake[np.isfinite(r_fake)]
#
#     fake_ids_r2 = sorted([pid for pid in pid_to_r2.keys() if pid != -1])
#     r2_fake = np.asarray([mean_or_nan(pid_to_r2[pid]) for pid in fake_ids_r2], float)
#     r2_fake = r2_fake[np.isfinite(r2_fake)]
#
#     r_fake_mean, r_fake_sd, z_r, p_emp_r = _z_and_p_from_null(r_real, r_fake, one_sided="greater_equal")
#     r2_fake_mean, r2_fake_sd, z_r2, p_emp_r2 = _z_and_p_from_null(r2_real, r2_fake, one_sided="greater_equal")
#
#     r2_corr = _corrected_r2_findling(r2_real, r2_fake_mean)
#
#     return {
#         "n_used": n_used,
#         "r_real": r_real,
#         "r_fake_mean": r_fake_mean,
#         "r_fake_std": r_fake_sd,
#         "z_corr": z_r,
#         "p_emp": p_emp_r,
#         "r2_real": r2_real,
#         "r2_fake_mean": r2_fake_mean,
#         "r2_fake_std": r2_fake_sd,
#         "r2_corr": r2_corr,
#         "z_r2": z_r2,
#         "p_emp_r2": p_emp_r2,
#     }
#
#
# def find_roi_pkl(session_dir: Path, roi: str):
#     for p in sorted(session_dir.rglob("*.pkl")):
#         try:
#             with open(p, "rb") as f:
#                 d = pickle.load(f)
#             reg = d.get("region", "")
#             reg = reg[0] if isinstance(reg, (list, tuple)) else str(reg)
#             if str(reg).strip() == str(roi).strip():
#                 return p
#         except Exception:
#             continue
#     return None
#
#
# # =========================
# # RIS worker
# # =========================
# def _pick_probe_name_like_local(one: ONE, eid: str):
#     pids = one.eid2pid(eid)
#     return pids, (pids[1] if len(pids) > 1 else pids[0])
#
#
# def run_one_eid_worker(eid: str, out_root_str: str, n_pseudo: int, debug: bool, cache_root_str: str):
#     warnings.filterwarnings("ignore", category=alferr.ALFWarning)
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     warnings.filterwarnings("ignore", category=RuntimeWarning)
#
#     out_root = Path(out_root_str)
#     out_root.mkdir(parents=True, exist_ok=True)
#
#     cache_root = Path(cache_root_str)
#     cache_root.mkdir(parents=True, exist_ok=True)
#     worker_cache = cache_root / f"worker_{os.getpid()}"
#     one = make_one(worker_cache)
#
#     pids, probe_name = _pick_probe_name_like_local(one, eid)
#     ses = one.alyx.rest("sessions", "read", id=eid)
#     subject = ses["subject"]
#
#     df, n_raw_trials = compute_trials_with_my_rt(one, eid)
#
#     if n_raw_trials < MIN_RAW_TRIALS:
#         return {"eid": eid, "status": "skip_trials", "n_raw_trials": n_raw_trials, "rows": []}
#
#     masks, meta = make_fast_normal_slow_masks(df)
#     drop_last_mask = make_drop_last_mask(len(df), DROP_LAST_N)
#
#     meta2 = dict(meta)
#     meta2.update({
#         "drop_last_n": int(DROP_LAST_N),
#         "n_after_drop_last": int(drop_last_mask.sum()),
#         "n_fast_after_drop_last": int((masks["fast"] & drop_last_mask).sum()),
#         "n_normal_after_drop_last": int((masks["normal"] & drop_last_mask).sum()),
#         "n_slow_after_drop_last": int((masks["slow"] & drop_last_mask).sum()),
#         "eid2pid_list": [str(x) for x in pids],
#         "probe_picked": str(probe_name),
#         "worker_cache": str(worker_cache),
#     })
#     (out_root / f"{eid}_rt_group_counts.json").write_text(json.dumps(meta2, indent=2))
#
#     pseudo_ids = [-1] + list(range(1, n_pseudo + 1))
#     session_dir = out_root / eid / "session_fit"
#     session_dir.mkdir(parents=True, exist_ok=True)
#
#     _ = fit_session_ephys(
#         one=one,
#         session_id=eid,
#         subject=subject,
#         probe_name=probe_name,
#         output_dir=session_dir,
#         pseudo_ids=pseudo_ids,
#         target="pLeft",
#         align_event=ALIGN_EVENT,
#         time_window=TIME_WINDOW,
#         model="optBay",
#         n_runs=N_RUNS,
#         trials_df=df,
#         trial_mask=drop_last_mask,
#         group_label="session",
#         debug=bool(debug),
#         roi_set=ROI_SET,
#     )
#
#     roi = ROI_LIST[0]
#     roi_pkl = find_roi_pkl(session_dir, roi)
#     if roi_pkl is None:
#         return {"eid": eid, "status": "skip_no_roi", "n_raw_trials": n_raw_trials, "rows": []}
#
#     rows = []
#     print(f"\n=== PER-SESSION / PER-GROUP (RIS) ===", flush=True)
#     print(f"[META] eid={eid} subject={subject} probe_picked={probe_name} eid2pid_list={pids}", flush=True)
#
#     for g in GROUPS:
#         stats = compute_group_stats_from_pkl(roi_pkl, group_mask_full=masks[g])
#
#         row = {
#             "eid": eid,
#             "subject": subject,
#             "probe": str(probe_name),
#             "roi": roi,
#             "group": g,
#             "n_trials_group_used": int(stats["n_used"]),
#             "r_real": stats["r_real"],
#             "r_fake_mean": stats["r_fake_mean"],
#             "r_fake_std": stats["r_fake_std"],
#             "z_corr": stats["z_corr"],
#             "p_emp": stats["p_emp"],
#             "r2_real": stats["r2_real"],
#             "r2_fake_mean": stats["r2_fake_mean"],
#             "r2_fake_std": stats["r2_fake_std"],
#             "r2_corr": stats["r2_corr"],
#             "z_r2": stats["z_r2"],
#             "p_emp_r2": stats["p_emp_r2"],
#             "drop_last_n": int(DROP_LAST_N),
#             "roi_pkl_path": str(roi_pkl),
#         }
#         rows.append(row)
#
#         # Print like your local (raw floats; nan will show as 'nan')
#         print(
#             f"{eid} | {g:<6} | n={row['n_trials_group_used']:4d} | "
#             f"Pearson: r={row['r_real']} | null= {row['r_fake_mean']}± {row['r_fake_std']} | "
#             f"z={row['z_corr']} | p={row['p_emp']} || "
#             f"R2: r2={row['r2_real']} | null={row['r2_fake_mean']}± {row['r2_fake_std']} | "
#             f"r2_corr={row['r2_corr']} | z={row['z_r2']} | p={row['p_emp_r2']}",
#             flush=True
#         )
#
#     (out_root / eid / f"pearson_summary_{roi}.json").write_text(
#         json.dumps(rows, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
#     )
#
#     return {"eid": eid, "status": "ok", "n_raw_trials": n_raw_trials, "rows": rows}
#
#
# def main():
#     OUT_ROOT.mkdir(parents=True, exist_ok=True)
#     CACHE_ROOT.mkdir(parents=True, exist_ok=True)
#
#     eids = [
#         "ae8787b1-4229-4d56-b0c2-566b61a25b77",
#         # add more
#     ]
#
#     all_rows = []
#     run_log = []
#
#     with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
#         futs = [
#             ex.submit(run_one_eid_worker, eid, str(OUT_ROOT), N_PSEUDO, DEBUG, str(CACHE_ROOT))
#             for eid in eids
#         ]
#         for fut in as_completed(futs):
#             r = fut.result()
#             run_log.append({k: r.get(k) for k in ["eid", "status", "n_raw_trials"]})
#             all_rows.extend(r.get("rows", []))
#             print("[DONE]", r.get("eid"), r.get("status"), "rows=", len(r.get("rows", [])), flush=True)
#
#     roi_tag = "_".join(ROI_LIST) if ROI_LIST else "ALL"
#     summary_path = OUT_ROOT / f"pearson_summary_{roi_tag}.pkl"
#     with open(summary_path, "wb") as f:
#         pickle.dump(all_rows, f)
#
#     run_log_path = OUT_ROOT / f"run_log_{roi_tag}.json"
#     with open(run_log_path, "w") as f:
#         json.dump(run_log, f, indent=2)
#
#     print("Saved pearson summary:", summary_path, flush=True)
#     print("Saved run log:", run_log_path, flush=True)
#
#
# if __name__ == "__main__":
#     main()

# from __future__ import annotations
#
# from pathlib import Path
# import json
# import pickle
# import warnings
# import os
# import numpy as np
# import pandas as pd
# from concurrent.futures import ProcessPoolExecutor, as_completed
#
# from one.api import ONE
# import one as one_pkg
# import one.alf.exceptions as alferr
#
# from prior_localization.my_rt import (
#     load_wheel_data, calc_wheel_velocity, calc_trialwise_wheel, calc_movement_onset_times
# )
# from prior_localization.fit_data import fit_session_ephys
#
#
# # =========================
# # RIS: make runtime deterministic-ish (avoid BLAS thread weirdness)
# # =========================
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
#
#
# # =========================
# # USER SETTINGS (match your local)
# # =========================
# ROI_LIST = ["MOp"]
# ROI_SET = set(ROI_LIST)
# GROUPS = ["fast", "normal", "slow"]
#
# # RIS outputs (shared filesystem)
# OUT_BASE = Path("/storage1/fs1/hiratani/Active/shared/ibl_space/derived/prior_localization_sessionfit_output")
# RUN_TAG = os.getenv("RUN_TAG", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
# OUT_ROOT = OUT_BASE / f"run_{RUN_TAG}"
#
# # RIS cache root (shared filesystem)
# CACHE_ROOT = Path("/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org_cache")
#
# N_PSEUDO = int(os.getenv("N_PSEUDO", "200"))
# N_RUNS = int(os.getenv("N_RUNS", "2"))
# N_WORKERS = int(os.getenv("N_WORKERS", "4"))
# DEBUG = bool(int(os.getenv("DEBUG", "0")))
#
# ALIGN_EVENT = "stimOn_times"
# TIME_WINDOW = (-0.6, -0.1)
# FAST_THR = float(os.getenv("FAST_THR", "0.08"))
# SLOW_THR = float(os.getenv("SLOW_THR", "1.25"))
#
# DROP_LAST_N = int(os.getenv("DROP_LAST_N", "40"))
# MIN_RAW_TRIALS = int(os.getenv("MIN_RAW_TRIALS", "401"))
#
#
# # =========================
# # ONE helper (per-worker cache dirs)
# # =========================
# def make_one(worker_cache: Path) -> ONE:
#     worker_cache = Path(worker_cache)
#     tables_dir = worker_cache / "tables"
#     dl_cache = worker_cache / "downloads"
#     tables_dir.mkdir(parents=True, exist_ok=True)
#     dl_cache.mkdir(parents=True, exist_ok=True)
#
#     one = ONE(
#         base_url="https://openalyx.internationalbrainlab.org",
#         username=os.getenv("ALYX_LOGIN"),
#         password=os.getenv("ALYX_PASSWORD"),
#         silent=True,
#         cache_dir=dl_cache,
#         tables_dir=tables_dir,
#         cache_rest=None,
#     )
#     return one
#
#
# # =========================
# # Trial + RT helpers (as local, with sanity checks)
# # =========================
# def compute_trials_with_my_rt(one: ONE, eid: str):
#     trials_obj = one.load_object(eid, "trials", collection="alf")
#
#     # --- sanity: required keys
#     required = ["stimOn_times", "feedback_times"]
#     for k in required:
#         if k not in trials_obj:
#             raise RuntimeError(f"[TRIALS] Missing key '{k}' in trials object for eid={eid}")
#
#     n_raw_trials = len(trials_obj["stimOn_times"])
#     if n_raw_trials <= 0:
#         raise RuntimeError(f"[TRIALS] stimOn_times is empty for eid={eid}")
#
#     data = {}
#     for k, v in trials_obj.items():
#         arr = np.asarray(v)
#         if k == "intervals" and arr.ndim == 2 and arr.shape[1] == 2:
#             data["intervals_start"] = arr[:, 0]
#             data["intervals_end"] = arr[:, 1]
#         elif arr.ndim == 1:
#             data[k] = arr
#         else:
#             data[k] = list(arr)
#     df = pd.DataFrame(data)
#
#     pos, ts = load_wheel_data(one, eid)
#     if pos is None or ts is None:
#         raise RuntimeError(f"[WHEEL] No wheel data for eid={eid}")
#
#     vel = calc_wheel_velocity(pos, ts)
#     _, trial_ts, trial_vel = calc_trialwise_wheel(pos, ts, vel, df["stimOn_times"], df["feedback_times"])
#     (_, first_mo, _, _, _) = calc_movement_onset_times(trial_ts, trial_vel, df["stimOn_times"])
#     df["first_movement_onset_times"] = first_mo
#     df["reaction_time"] = df["first_movement_onset_times"] - df["stimOn_times"]
#
#     # --- sanity: RT should not be all NaN
#     if not np.isfinite(df["reaction_time"]).any():
#         raise RuntimeError(f"[RT] All reaction_time are non-finite for eid={eid}")
#
#     return df, int(n_raw_trials)
#
#
# def make_fast_normal_slow_masks(df: pd.DataFrame, rt_col="reaction_time", fast_thr=FAST_THR, slow_thr=SLOW_THR):
#     rt = df[rt_col].to_numpy()
#     valid = np.isfinite(rt)
#     fast = valid & (rt < fast_thr)
#     slow = valid & (rt > slow_thr)
#     normal = valid & (~fast) & (~slow)
#     return {"fast": fast, "normal": normal, "slow": slow}, {
#         "n_total": int(len(df)),
#         "n_valid_rt": int(valid.sum()),
#         "n_fast": int(fast.sum()),
#         "n_normal": int(normal.sum()),
#         "n_slow": int(slow.sum()),
#         "fast_thr": float(fast_thr),
#         "slow_thr": float(slow_thr),
#     }
#
#
# def make_drop_last_mask(n_trials: int, drop_last_n: int):
#     m = np.ones(int(n_trials), dtype=bool)
#     if drop_last_n and drop_last_n > 0:
#         m[max(0, n_trials - drop_last_n):] = False
#     return m
#
#
# # =========================
# # Stats helpers (same as local)
# # =========================
# def _trial_scalar_list(x_list):
#     """list-of-arrays -> scalar per trial (mean over bins). returns vec (n_trials,) with NaNs for bad."""
#     out = np.full(len(x_list), np.nan, float)
#     for i, xi in enumerate(x_list):
#         if xi is None:
#             continue
#         arr = np.asarray(xi).reshape(-1)
#         if arr.size == 0 or (not np.all(np.isfinite(arr))):
#             continue
#         out[i] = float(np.mean(arr))
#     return out
#
#
# def _pearsonr_safe(a, b):
#     a = np.asarray(a, float).reshape(-1)
#     b = np.asarray(b, float).reshape(-1)
#     ok = np.isfinite(a) & np.isfinite(b)
#     a = a[ok]; b = b[ok]
#     if a.size < 3:
#         return np.nan
#     if np.std(a) == 0 or np.std(b) == 0:
#         return np.nan
#     return float(np.corrcoef(a, b)[0, 1])
#
#
# def _r2_safe(y_true, y_pred):
#     y_true = np.asarray(y_true, float).reshape(-1)
#     y_pred = np.asarray(y_pred, float).reshape(-1)
#     ok = np.isfinite(y_true) & np.isfinite(y_pred)
#     y_true = y_true[ok]; y_pred = y_pred[ok]
#     if y_true.size < 3:
#         return np.nan
#     denom = np.sum((y_true - np.mean(y_true)) ** 2)
#     if denom <= 0:
#         return np.nan
#     sse = np.sum((y_true - y_pred) ** 2)
#     return float(1.0 - sse / denom)
#
#
# def _z_and_p_from_null(real_val, null_vals, one_sided="greater_equal"):
#     null_vals = np.asarray(null_vals, float)
#     null_vals = null_vals[np.isfinite(null_vals)]
#     if (not np.isfinite(real_val)) or null_vals.size < 5:
#         return np.nan, np.nan, np.nan, np.nan
#
#     mu = float(np.mean(null_vals))
#     sd = float(np.std(null_vals, ddof=1)) if null_vals.size > 1 else float(np.std(null_vals))
#     z = float((real_val - mu) / sd) if sd > 0 else np.nan
#
#     if one_sided == "greater_equal":
#         p = float((1 + np.sum(null_vals >= real_val)) / (1 + null_vals.size))
#     elif one_sided == "less_equal":
#         p = float((1 + np.sum(null_vals <= real_val)) / (1 + null_vals.size))
#     else:
#         raise ValueError("one_sided must be 'greater_equal' or 'less_equal'")
#
#     return mu, sd, z, p
#
#
# def _corrected_r2_findling(r2_real, r2_fake_mean):
#     if (not np.isfinite(r2_real)) or (not np.isfinite(r2_fake_mean)):
#         return np.nan
#     denom = 1.0 - float(r2_fake_mean)
#     if denom == 0:
#         return np.nan
#     return float((r2_real - float(r2_fake_mean)) / denom)
#
#
# def compute_group_stats_from_pkl(region_pkl: Path, group_mask_full: np.ndarray):
#     with open(region_pkl, "rb") as f:
#         d = pickle.load(f)
#
#     keep_idx_full = np.asarray(d["keep_idx_full"], dtype=int)
#     group_mask_sub = np.asarray(group_mask_full, dtype=bool)[keep_idx_full]
#     sub_idx = np.flatnonzero(group_mask_sub)
#     n_used = int(sub_idx.size)
#
#     pid_to_r = {}
#     pid_to_r2 = {}
#
#     for fr in d["fit"]:
#         pid = int(fr.get("pseudo_id", -999))
#         preds = fr.get("predictions_test", None)
#         targ  = fr.get("target", None)
#         if preds is None or targ is None:
#             continue
#
#         yhat = _trial_scalar_list(preds)
#         y    = _trial_scalar_list(targ)
#
#         y_sub = y[sub_idx]
#         yhat_sub = yhat[sub_idx]
#
#         r = _pearsonr_safe(yhat_sub, y_sub)
#         r2 = _r2_safe(y_sub, yhat_sub)
#
#         if np.isfinite(r):
#             pid_to_r.setdefault(pid, []).append(r)
#         if np.isfinite(r2):
#             pid_to_r2.setdefault(pid, []).append(r2)
#
#     def mean_or_nan(x):
#         return float(np.mean(x)) if len(x) else np.nan
#
#     r_real = mean_or_nan(pid_to_r.get(-1, []))
#     r2_real = mean_or_nan(pid_to_r2.get(-1, []))
#
#     fake_ids_r = sorted([pid for pid in pid_to_r.keys() if pid != -1])
#     r_fake = np.asarray([mean_or_nan(pid_to_r[pid]) for pid in fake_ids_r], float)
#     r_fake = r_fake[np.isfinite(r_fake)]
#
#     fake_ids_r2 = sorted([pid for pid in pid_to_r2.keys() if pid != -1])
#     r2_fake = np.asarray([mean_or_nan(pid_to_r2[pid]) for pid in fake_ids_r2], float)
#     r2_fake = r2_fake[np.isfinite(r2_fake)]
#
#     r_fake_mean, r_fake_sd, z_r, p_emp_r = _z_and_p_from_null(r_real, r_fake, one_sided="greater_equal")
#     r2_fake_mean, r2_fake_sd, z_r2, p_emp_r2 = _z_and_p_from_null(r2_real, r2_fake, one_sided="greater_equal")
#
#     r2_corr = _corrected_r2_findling(r2_real, r2_fake_mean)
#
#     return {
#         "n_used": n_used,
#         "r_real": r_real,
#         "r_fake_mean": r_fake_mean,
#         "r_fake_std": r_fake_sd,
#         "z_corr": z_r,
#         "p_emp": p_emp_r,
#         "r2_real": r2_real,
#         "r2_fake_mean": r2_fake_mean,
#         "r2_fake_std": r2_fake_sd,
#         "r2_corr": r2_corr,
#         "z_r2": z_r2,
#         "p_emp_r2": p_emp_r2,
#     }
#
#
# def find_roi_pkl(session_dir: Path, roi: str):
#     for p in sorted(session_dir.rglob("*.pkl")):
#         try:
#             with open(p, "rb") as f:
#                 d = pickle.load(f)
#             reg = d.get("region", "")
#             reg = reg[0] if isinstance(reg, (list, tuple)) else str(reg)
#             if str(reg).strip() == str(roi).strip():
#                 return p
#         except Exception:
#             continue
#     return None
#
#
# # =========================
# # RIS worker
# # =========================
# def _pick_probe_name_like_local(one: ONE, eid: str):
#     pids = one.eid2pid(eid)
#     return pids, (pids[1] if len(pids) > 1 else pids[0])
#
#
# def run_one_eid_worker(eid: str, out_root_str: str, n_pseudo: int, debug: bool, cache_root_str: str):
#     warnings.filterwarnings("ignore", category=alferr.ALFWarning)
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     warnings.filterwarnings("ignore", category=RuntimeWarning)
#
#     out_root = Path(out_root_str)
#     out_root.mkdir(parents=True, exist_ok=True)
#
#     cache_root = Path(cache_root_str)
#     cache_root.mkdir(parents=True, exist_ok=True)
#     worker_cache = cache_root / f"worker_{os.getpid()}"
#     one = make_one(worker_cache)
#
#     # --- sanity: ONE version + cache
#     print(f"[ONE] version={getattr(one_pkg,'__version__','?')} worker_cache={worker_cache}", flush=True)
#
#     # Probe/subject metadata
#     pids, probe_name = _pick_probe_name_like_local(one, eid)
#     print(f"[PIDS] {eid} -> {pids} | picked_probe={probe_name}", flush=True)
#
#     ses = one.alyx.rest("sessions", "read", id=eid)
#     subject = ses["subject"]
#
#     # Trials + RT
#     df, n_raw_trials = compute_trials_with_my_rt(one, eid)
#     print(f"[TRIALS] eid={eid} n_raw_trials={n_raw_trials}", flush=True)
#
#     if n_raw_trials < MIN_RAW_TRIALS:
#         return {"eid": eid, "status": "skip_trials", "n_raw_trials": n_raw_trials, "rows": []}
#
#     masks, meta = make_fast_normal_slow_masks(df)
#     drop_last_mask = make_drop_last_mask(len(df), DROP_LAST_N)
#
#     print(
#         f"[RT] finite_rt={meta['n_valid_rt']} fast={meta['n_fast']} normal={meta['n_normal']} slow={meta['n_slow']} "
#         f"drop_last_n={DROP_LAST_N} n_after_drop_last={int(drop_last_mask.sum())}",
#         flush=True
#     )
#
#     meta2 = dict(meta)
#     meta2.update({
#         "drop_last_n": int(DROP_LAST_N),
#         "n_after_drop_last": int(drop_last_mask.sum()),
#         "n_fast_after_drop_last": int((masks["fast"] & drop_last_mask).sum()),
#         "n_normal_after_drop_last": int((masks["normal"] & drop_last_mask).sum()),
#         "n_slow_after_drop_last": int((masks["slow"] & drop_last_mask).sum()),
#         "eid2pid_list": [str(x) for x in pids],
#         "probe_picked": str(probe_name),
#         "worker_cache": str(worker_cache),
#         "one_version": str(getattr(one_pkg, "__version__", "?")),
#     })
#     (out_root / f"{eid}_rt_group_counts.json").write_text(json.dumps(meta2, indent=2))
#
#     pseudo_ids = [-1] + list(range(1, n_pseudo + 1))
#
#     # Fit
#     session_dir = out_root / eid / "session_fit"
#     session_dir.mkdir(parents=True, exist_ok=True)
#
#     print(f"[FIT] eid={eid} session_dir={session_dir} debug={bool(debug)}", flush=True)
#
#     _ = fit_session_ephys(
#         one=one,
#         session_id=eid,
#         subject=subject,
#         probe_name=probe_name,
#         output_dir=session_dir,
#         pseudo_ids=pseudo_ids,
#         target="pLeft",
#         align_event=ALIGN_EVENT,
#         time_window=TIME_WINDOW,
#         model="optBay",
#         n_runs=N_RUNS,
#         trials_df=df,
#         trial_mask=drop_last_mask,
#         group_label="session",
#         debug=bool(debug),
#         roi_set=ROI_SET,
#     )
#
#     # find ROI pkl
#     roi = ROI_LIST[0]
#     roi_pkl = find_roi_pkl(session_dir, roi)
#     print(f"[ROI] requested={roi} found_pkl={roi_pkl}", flush=True)
#
#     if roi_pkl is None or (not Path(roi_pkl).exists()):
#         return {"eid": eid, "status": "skip_no_roi", "n_raw_trials": n_raw_trials, "rows": []}
#
#     # --- sanity: ROI pkl contents + prediction const-ness (REAL fit)
#     d = pickle.load(open(roi_pkl, "rb"))
#     if ("fit" not in d) or (len(d["fit"]) == 0):
#         raise RuntimeError(f"[PKL] No fit entries in {roi_pkl}")
#     if "keep_idx_full" not in d:
#         raise RuntimeError(f"[PKL] Missing keep_idx_full in {roi_pkl}")
#
#     real_entries = [fr for fr in d["fit"] if int(fr.get("pseudo_id", -999)) == -1]
#     if len(real_entries) == 0:
#         raise RuntimeError(f"[PKL] No real pseudo_id=-1 entry in {roi_pkl}")
#
#     fr = real_entries[0]
#     if fr.get("predictions_test", None) is None or fr.get("target", None) is None:
#         raise RuntimeError(f"[PKL] Missing predictions_test/target in real entry for {roi_pkl}")
#
#     yhat_all = _trial_scalar_list(fr["predictions_test"])
#     y_all = _trial_scalar_list(fr["target"])
#     print(
#         f"[SANITY] yhat finite={int(np.isfinite(yhat_all).sum())}/{len(yhat_all)} std={float(np.nanstd(yhat_all))} "
#         f"min={float(np.nanmin(yhat_all)) if np.isfinite(yhat_all).any() else np.nan} "
#         f"max={float(np.nanmax(yhat_all)) if np.isfinite(yhat_all).any() else np.nan}",
#         flush=True
#     )
#     print(
#         f"[SANITY] y    finite={int(np.isfinite(y_all).sum())}/{len(y_all)} std={float(np.nanstd(y_all))}",
#         flush=True
#     )
#
#     if int(np.isfinite(yhat_all).sum()) < 10:
#         raise RuntimeError(f"[SANITY] Too few finite predictions in REAL fit (finite={int(np.isfinite(yhat_all).sum())})")
#     if (not np.isfinite(np.nanstd(yhat_all))) or (np.nanstd(yhat_all) == 0):
#         raise RuntimeError("[SANITY] REAL predictions are constant/invalid -> Pearson will be NaN (likely wrong probe or no units)")
#
#     # Compute stats per group + print
#     rows = []
#     print("\n=== PER-SESSION / PER-GROUP (RIS) ===", flush=True)
#     for g in GROUPS:
#         stats = compute_group_stats_from_pkl(roi_pkl, group_mask_full=masks[g])
#         row = {
#             "eid": eid,
#             "subject": subject,
#             "probe": str(probe_name),
#             "roi": roi,
#             "group": g,
#             "n_trials_group_used": int(stats["n_used"]),
#             "r_real": stats["r_real"],
#             "r_fake_mean": stats["r_fake_mean"],
#             "r_fake_std": stats["r_fake_std"],
#             "z_corr": stats["z_corr"],
#             "p_emp": stats["p_emp"],
#             "r2_real": stats["r2_real"],
#             "r2_fake_mean": stats["r2_fake_mean"],
#             "r2_fake_std": stats["r2_fake_std"],
#             "r2_corr": stats["r2_corr"],
#             "z_r2": stats["z_r2"],
#             "p_emp_r2": stats["p_emp_r2"],
#             "drop_last_n": int(DROP_LAST_N),
#             "roi_pkl_path": str(roi_pkl),
#         }
#         rows.append(row)
#
#         print(
#             f"{eid} | {g:<6} | n={row['n_trials_group_used']:4d} | "
#             f"Pearson: r={row['r_real']} | null= {row['r_fake_mean']}± {row['r_fake_std']} | "
#             f"z={row['z_corr']} | p={row['p_emp']} || "
#             f"R2: r2={row['r2_real']} | null={row['r2_fake_mean']}± {row['r2_fake_std']} | "
#             f"r2_corr={row['r2_corr']} | z={row['z_r2']} | p={row['p_emp_r2']}",
#             flush=True
#         )
#
#     (out_root / eid / f"pearson_summary_{roi}.json").write_text(
#         json.dumps(rows, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
#     )
#
#     return {"eid": eid, "status": "ok", "n_raw_trials": n_raw_trials, "rows": rows}
#
#
# def main():
#     warnings.filterwarnings("ignore", category=alferr.ALFWarning)
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     warnings.filterwarnings("ignore", category=RuntimeWarning)
#
#     OUT_ROOT.mkdir(parents=True, exist_ok=True)
#     CACHE_ROOT.mkdir(parents=True, exist_ok=True)
#
#     eids = [
#         "ae8787b1-4229-4d56-b0c2-566b61a25b77",
#         # add more
#     ]
#
#     all_rows = []
#     run_log = []
#
#     with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
#         futs = [
#             ex.submit(run_one_eid_worker, eid, str(OUT_ROOT), N_PSEUDO, DEBUG, str(CACHE_ROOT))
#             for eid in eids
#         ]
#         for fut in as_completed(futs):
#             r = fut.result()
#             run_log.append({k: r.get(k) for k in ["eid", "status", "n_raw_trials"]})
#             all_rows.extend(r.get("rows", []))
#             print("[DONE]", r.get("eid"), r.get("status"), "rows=", len(r.get("rows", [])), flush=True)
#
#     roi_tag = "_".join(ROI_LIST) if ROI_LIST else "ALL"
#     summary_path = OUT_ROOT / f"pearson_summary_{roi_tag}.pkl"
#     with open(summary_path, "wb") as f:
#         pickle.dump(all_rows, f)
#
#     run_log_path = OUT_ROOT / f"run_log_{roi_tag}.json"
#     with open(run_log_path, "w") as f:
#         json.dump(run_log, f, indent=2)
#
#     print("Saved pearson summary:", summary_path, flush=True)
#     print("Saved run log:", run_log_path, flush=True)
#
#
# if __name__ == "__main__":
#     main()

from __future__ import annotations

from pathlib import Path
import json
import pickle
import warnings
import os
import numpy as np
import pandas as pd

from one.api import ONE
import one as one_pkg
import one.alf.exceptions as alferr

from prior_localization.my_rt import (
    load_wheel_data, calc_wheel_velocity, calc_trialwise_wheel, calc_movement_onset_times
)
from prior_localization.fit_data import fit_session_ephys


# =========================
# RIS: make runtime deterministic-ish (avoid BLAS thread weirdness)
# =========================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# =========================
# USER SETTINGS (match your local)
# =========================
ROI_LIST = ["MOp"]
ROI_SET = set(ROI_LIST)
GROUPS = ["fast", "normal", "slow"]

# RIS outputs (shared filesystem)
OUT_BASE = Path("/storage1/fs1/hiratani/Active/shared/ibl_space/derived/prior_localization_sessionfit_output")
RUN_TAG = os.getenv("RUN_TAG", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
OUT_ROOT = OUT_BASE / f"run_{RUN_TAG}"

# RIS cache root (shared filesystem)
CACHE_ROOT = Path("/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org_cache")

N_PSEUDO = int(os.getenv("N_PSEUDO", "200"))
N_RUNS = int(os.getenv("N_RUNS", "2"))
DEBUG = bool(int(os.getenv("DEBUG", "0")))

ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.6, -0.1)
FAST_THR = float(os.getenv("FAST_THR", "0.08"))
SLOW_THR = float(os.getenv("SLOW_THR", "1.25"))

DROP_LAST_N = int(os.getenv("DROP_LAST_N", "40"))
MIN_RAW_TRIALS = int(os.getenv("MIN_RAW_TRIALS", "401"))


# =========================
# ONE helper (per-worker cache dirs)
# =========================
def make_one(worker_cache: Path) -> ONE:
    worker_cache = Path(worker_cache)
    tables_dir = worker_cache / "tables"
    dl_cache = worker_cache / "downloads"
    tables_dir.mkdir(parents=True, exist_ok=True)
    dl_cache.mkdir(parents=True, exist_ok=True)

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
# Trial + RT helpers
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
# Stats helpers
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


def _pick_probe_name_like_local(one: ONE, eid: str):
    pids = one.eid2pid(eid)
    return pids, (pids[1] if len(pids) > 1 else pids[0])


def run_one_eid_worker(eid: str, out_root_str: str, n_pseudo: int, debug: bool, cache_root_str: str):
    warnings.filterwarnings("ignore", category=alferr.ALFWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    out_root = Path(out_root_str)
    out_root.mkdir(parents=True, exist_ok=True)

    cache_root = Path(cache_root_str)
    cache_root.mkdir(parents=True, exist_ok=True)

    # SERIAL version: still keep per-run cache folder, but not per-process
    worker_cache = cache_root / f"serial_{RUN_TAG}"
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
    real_entries = [fr for fr in d["fit"] if int(fr.get("pseudo_id", -999)) == -1]
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
    print("\n=== PER-SESSION / PER-GROUP (SERIAL) ===", flush=True)
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
    warnings.filterwarnings("ignore", category=alferr.ALFWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    eids = [
        "ae8787b1-4229-4d56-b0c2-566b61a25b77",
        # add more
    ]

    all_rows = []
    run_log = []

    # SERIAL loop (no multiprocessing)
    for eid in eids:
        try:
            r = run_one_eid_worker(eid, str(OUT_ROOT), N_PSEUDO, DEBUG, str(CACHE_ROOT))
        except Exception as e:
            print(f"[ERROR] eid={eid} exception={repr(e)}", flush=True)
            r = {"eid": eid, "status": "error", "n_raw_trials": None, "rows": []}

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


if __name__ == "__main__":
    main()
