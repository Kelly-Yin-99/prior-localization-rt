# from pathlib import Path
# import json
# import pickle
# import warnings
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
# # -----------------------------
# # User settings
# # -----------------------------
# ROI_LIST = ["ACAd"]
# ROI_SET = set(ROI_LIST)
#
# GROUPS = ["fast", "normal", "slow"]
#
# # -----------------------------
# # Your existing helpers (unchanged)
# # -----------------------------
# def compute_trials_with_my_rt(one: ONE, eid: str):
#     trials_obj = one.load_object(eid, "trials", collection="alf")
#     n_raw_trials = len(trials_obj["stimOn_times"])  # BEFORE any dropna
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
#
#     df = pd.DataFrame(data)
#
#     pos, ts = load_wheel_data(one, eid)
#     if pos is None or ts is None:
#         raise RuntimeError(f"No wheel data for eid={eid}")
#
#     vel = calc_wheel_velocity(pos, ts)
#     trial_pos, trial_ts, trial_vel = calc_trialwise_wheel(
#         pos, ts, vel, df["stimOn_times"], df["feedback_times"]
#     )
#
#     (_, first_mo, _, _, _) = calc_movement_onset_times(trial_ts, trial_vel, df["stimOn_times"])
#
#     df["first_movement_onset_times"] = first_mo
#     df["reaction_time"] = df["first_movement_onset_times"] - df["stimOn_times"]
#
#     return df, n_raw_trials
#
#
# def make_fast_normal_slow_masks(df: pd.DataFrame, rt_col="reaction_time",
#                                fast_thr=0.08, slow_thr=1.25):
#     rt = df[rt_col].to_numpy()
#     valid = np.isfinite(rt)
#
#     fast = valid & (rt < fast_thr)
#     slow = valid & (rt > slow_thr)
#     normal = valid & (~fast) & (~slow)
#
#     masks = {"fast": fast, "normal": normal, "slow": slow}
#     meta = {
#         "n_total": int(len(df)),
#         "n_valid_rt": int(valid.sum()),
#         "n_fast": int(fast.sum()),
#         "n_normal": int(normal.sum()),
#         "n_slow": int(slow.sum()),
#     }
#     return masks, meta
#
# import numpy as np
# import pandas as pd
#
# def add_my_rt_to_sl_trials(one, sl):
#     """
#     Adds:
#       - first_movement_onset_times
#       - reaction_time
#     directly onto sl.trials, preserving the exact row indexing used by load_trials_and_mask.
#     """
#     df = sl.trials.copy().reset_index(drop=True)
#
#     pos, ts = load_wheel_data(one, sl.eid)
#     if pos is None or ts is None:
#         raise RuntimeError(f"No wheel data for eid={sl.eid}")
#
#     vel = calc_wheel_velocity(pos, ts)
#
#     trial_pos, trial_ts, trial_vel = calc_trialwise_wheel(
#         pos, ts, vel, df["stimOn_times"], df["feedback_times"]
#     )
#
#     (_, first_mo, _, _, _) = calc_movement_onset_times(trial_ts, trial_vel, df["stimOn_times"])
#
#     df["first_movement_onset_times"] = first_mo
#     df["reaction_time"] = df["first_movement_onset_times"] - df["stimOn_times"]
#
#     sl.trials = df
#     return sl
#
#
# def make_fast_normal_slow_masks_from_sl(sl, rt_col="reaction_time", fast_thr=0.08, slow_thr=1.25):
#     rt = sl.trials[rt_col].to_numpy()
#     valid = np.isfinite(rt)
#
#     fast = valid & (rt < fast_thr)
#     slow = valid & (rt > slow_thr)
#     normal = valid & (~fast) & (~slow)
#
#     masks = {"fast": fast, "normal": normal, "slow": slow}
#     meta = {
#         "n_total": int(len(sl.trials)),
#         "n_valid_rt": int(valid.sum()),
#         "n_fast": int(fast.sum()),
#         "n_normal": int(normal.sum()),
#         "n_slow": int(slow.sum()),
#     }
#     return masks, meta
#
# # -----------------------------
# # Helper: extract R2 summary from a saved pkl
# # -----------------------------
# def summarize_region_pkl(pkl_path: Path):
#     with open(pkl_path, "rb") as f:
#         d = pickle.load(f)
#
#     fit = d["fit"]
#     eid = d["eid"]
#     subject = d["subject"]
#     group = d.get("group_label", "unknown")
#     probe = d.get("probe", "unknown")
#
#     # region in file is like ['MOp'] or ['VPM']
#     region = d["region"][0] if isinstance(d["region"], (list, tuple)) else str(d["region"])
#     n_units = int(d["N_units"])
#
#     session_scores = [fr["scores_test_full"] for fr in fit if fr.get("pseudo_id", None) == -1]
#     pseudo_scores = [fr["scores_test_full"] for fr in fit if fr.get("pseudo_id", None) != -1]
#
#     r2_unc = float(np.mean(session_scores)) if len(session_scores) else np.nan
#     r2_pseudo = float(np.mean(pseudo_scores)) if len(pseudo_scores) else np.nan
#     r2_corr = float(r2_unc - r2_pseudo) if np.isfinite(r2_unc) and np.isfinite(r2_pseudo) else np.nan
#
#     return {
#         "eid": eid,
#         "subject": subject,
#         "group": group,
#         "probe": probe,
#         "region": region,
#         "N_units": n_units,
#         "R2_uncorrected": r2_unc,
#         "R2_pseudo_mean": r2_pseudo,
#         "R2_corrected": r2_corr,
#         "pkl_path": str(pkl_path),
#     }
#
#
#
# import numpy as np
#
#
# def run_one_eid_worker(eid: str, out_root_str: str, n_pseudo: int, debug: bool):
#     import os
#     import contextlib
#     import warnings
#     import json
#     from pathlib import Path
#
#
#
#     warnings.filterwarnings("ignore", category=alferr.ALFWarning)
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     warnings.filterwarnings("ignore", category=RuntimeWarning)
#
#     out_root = Path(out_root_str)
#     out_root.mkdir(parents=True, exist_ok=True)
#
#
#     one = ONE()
#
#     probe_name = one.eid2pid(eid)[1]
#     ses = one.alyx.rest("sessions", "read", id=eid)
#     subject = ses["subject"]
#
#     df, n_raw_trials = compute_trials_with_my_rt(one, eid)
#
#     # QC: require enough raw trials
#     if n_raw_trials < 401:
#         return {
#             "eid": eid,
#             "status": "skip_trials",
#             "n_raw_trials": int(n_raw_trials),
#             "rows": [],
#         }
#
#     # -----------------------------
#     # Fast / normal / slow masks
#     # -----------------------------
#     masks, meta = make_fast_normal_slow_masks(df)
#     (out_root / f"{eid}_rt_group_counts.json").write_text(
#         json.dumps(meta, indent=2)
#     )
#
#     if debug:
#         print(f"[DEBUG] {eid}: raw_trials={n_raw_trials} meta={meta}")
#         print(f"[DEBUG] {eid}: probe_name={probe_name}")
#
#     # -----------------------------
#     # Pseudo sessions
#     # -----------------------------
#     pseudo_ids = [-1] + list(range(1, n_pseudo + 1))
#
#     # -----------------------------
#     # Run decoding per RT group
#     # -----------------------------
#     for group_label in GROUPS:
#         group_dir = out_root / eid / group_label
#         group_dir.mkdir(parents=True, exist_ok=True)
#
#         # SILENCE all prints from fit_session_ephys / decode_cv
#         with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
#             _ = fit_session_ephys(
#                 one=one,
#                 session_id=eid,
#                 subject=subject,
#                 probe_name=probe_name,
#                 output_dir=group_dir,
#                 pseudo_ids=pseudo_ids,
#                 target="pLeft",
#                 align_event="stimOn_times",
#                 time_window=(-0.6, -0.1),
#                 model="optBay",
#                 n_runs=2,
#                 min_rt=None,
#                 max_rt=None,
#                 trials_df=df,
#                 trial_mask=masks[group_label],
#                 group_label=group_label,
#                 debug=False,          # force debug off
#                 roi_set=ROI_SET,
#             )
#
#     # -----------------------------
#     # Collect ROI results
#     # -----------------------------
#     rows = []
#     for pkl_path in (out_root / eid).rglob("*.pkl"):
#         try:
#             row = summarize_region_pkl(pkl_path)
#             if row["region"] in ROI_SET:
#                 rows.append(row)
#         except Exception:
#             continue
#
#     status = "ok" if rows else "skip_no_roi_or_failed"
#
#     return {
#         "eid": eid,
#         "status": status,
#         "n_raw_trials": int(n_raw_trials),
#         "rows": rows,
#     }
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# def plot_roi_distributions(summary_pkl_path: Path, roi_list, group_order=("fast", "normal", "slow")):
#     with open(summary_pkl_path, "rb") as f:
#         rows = pickle.load(f)
#
#     df = pd.DataFrame(rows)
#     if df.empty:
#         print("[plot] roi_summary.pkl is empty. Nothing to plot.")
#         return
#
#     # keep only desired ROIs
#     df = df[df["region"].isin(set(roi_list))].copy()
#     if df.empty:
#         print("[plot] No rows match ROI_LIST. Nothing to plot.")
#         return
#
#     df["group"] = pd.Categorical(df["group"], categories=list(group_order), ordered=True)
#
#     # --- Figure 1: Uncorrected ---
#     fig1, ax1 = plt.subplots(figsize=(7.5, 5.0))
#     sns.violinplot(data=df, x="group", y="R2_uncorrected", order=group_order, inner="box", ax=ax1)
#     sns.stripplot(data=df, x="group", y="R2_uncorrected", order=group_order,
#                   color="black", alpha=0.55, jitter=True, size=4, ax=ax1)
#     ax1.axhline(0, color="k", linestyle="--", linewidth=1)
#     ax1.set_title(f"Uncorrected R² by RT group ({', '.join(roi_list)})")
#     ax1.set_xlabel("")
#     ax1.set_ylabel("R² (uncorrected)")
#     fig1.tight_layout()
#
#     # --- Figure 2: Corrected (unc - pseudo mean) ---
#     fig2, ax2 = plt.subplots(figsize=(7.5, 5.0))
#     sns.violinplot(data=df, x="group", y="R2_corrected", order=group_order, inner="box", ax=ax2)
#     sns.stripplot(data=df, x="group", y="R2_corrected", order=group_order,
#                   color="black", alpha=0.55, jitter=True, size=4, ax=ax2)
#     ax2.axhline(0, color="k", linestyle="--", linewidth=1)
#     ax2.set_title(f"Corrected R² (unc − pseudo) by RT group ({', '.join(roi_list)})")
#     ax2.set_xlabel("")
#     ax2.set_ylabel("R² (corrected)")
#     fig2.tight_layout()
#
#     plt.show()
#
#     # optional: save to disk next to the pkl
#     out_dir = summary_pkl_path.parent
#     fig1.savefig(out_dir / "roi_uncorrected_distribution.png", dpi=200)
#     fig2.savefig(out_dir / "roi_corrected_distribution.png", dpi=200)
#     print("[plot] Saved:",
#           out_dir / "roi_uncorrected_distribution.png",
#           out_dir / "roi_corrected_distribution.png")
#
#
# def main():
#     import json
#     import pickle
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     from pathlib import Path
#     from concurrent.futures import ProcessPoolExecutor, as_completed
#
#     out_root = Path("./prior_localization_groups_output")
#     out_root.mkdir(parents=True, exist_ok=True)
#
#     # ---- Put your list here ----
#     eids = [
#         #ORBvl example sessions
#         # "1191f865-b10a-45c8-9c48-24a980fd9402",
#         # "2e6e179c-fccc-4e8f-9448-ce5b6858a183",
#         # "4d8c7767-981c-4347-8e5e-5d5fffe38534",
#         # "d33baf74-263c-4b37-a0d0-b79dcb80a764",
#         # "dd4da095-4a99-4bf3-9727-f735077dba66",
#         # "dfbe628d-365b-461c-a07f-8b9911ba83aa",
#
#         #MOp example sessions
#
#         # "36280321-555b-446d-9b7d-c2e17991e090",
#         # "4aa1d525-5c7d-4c50-a147-ec53a9014812",
#         # "5455a21c-1be7-4cae-ae8e-8853a8d5f55e",
#         # "81a78eac-9d36-4f90-a73a-7eb3ad7f770b",
#         # "9e9c6fc0-4769-4d83-9ea4-b59a1230510e",
#         # "bd456d8f-d36e-434a-8051-ff3997253802",
#         # "cf43dbb1-6992-40ec-a5f9-e8e838d0f643",
#
#         #ACAd example sessions
#
#         "78b4fff5-c5ec-44d9-b5f9-d59493063f00",
#         "a4000c2f-fa75-4b3e-8f06-a7cf599b87ad"
#     ]
#
#     n_pseudo = 200
#     debug = False
#     n_workers = 4  # change based on CPU/RAM
#
#     all_rows = []
#     run_log = []
#
#
#     with ProcessPoolExecutor(max_workers=n_workers) as ex:
#         futs = [ex.submit(run_one_eid_worker, eid, str(out_root), n_pseudo, debug) for eid in eids]
#         for fut in as_completed(futs):
#             r = fut.result()
#             run_log.append({k: r.get(k) for k in ["eid", "status", "n_raw_trials"]})
#             all_rows.extend(r.get("rows", []))
#             print("[DONE]", r.get("eid"), r.get("status"), "rows=", len(r.get("rows", [])))
#
#
#     summary_path = out_root / "roii_summary_MOp.pkl"
#     with open(summary_path, "wb") as f:
#         pickle.dump(all_rows, f)
#
#
#     try:
#         df_all = pd.DataFrame(all_rows)
#         df_all.to_parquet(out_root / "roi_summary_MOp.parquet", index=False)
#     except Exception:
#         pass
#
#     #
#     with open(out_root / "run_log.json", "w") as f:
#         json.dump(run_log, f, indent=2)
#
#     print("Saved big ROI summary:", summary_path)
#
#
#
# if __name__ == "__main__":
#     main()



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



ROI_LIST = ["ACAd"]               # <-- set this to whichever ROI(s) you want
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
    _, trial_ts, trial_vel = calc_trialwise_wheel(
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


def run_one_eid_worker(eid: str, out_root_str: str, n_pseudo: int, debug: bool):
    import os
    import contextlib
    import warnings
    import json
    from pathlib import Path

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

    # df from load_object + wheel RT
    df, n_raw_trials = compute_trials_with_my_rt(one, eid)

    # QC: require enough raw trials
    if n_raw_trials < 401:
        return {
            "eid": eid,
            "status": "skip_trials",
            "n_raw_trials": int(n_raw_trials),
            "rows": [],
        }

    # fast/normal/slow masks
    masks, meta = make_fast_normal_slow_masks(df)
    (out_root / f"{eid}_rt_group_counts.json").write_text(json.dumps(meta, indent=2))

    if debug:
        print(f"[DEBUG] {eid}: raw_trials={n_raw_trials} meta={meta}")
        print(f"[DEBUG] {eid}: probe_name={probe_name}")

    pseudo_ids = [-1] + list(range(1, n_pseudo + 1))

    # run groups (silence stdout)
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
                align_event="stimOn_times",
                time_window=(-0.6, -0.1),
                model="optBay",
                n_runs=2,
                min_rt=None,
                max_rt=None,
                trials_df=df,
                trial_mask=masks[group_label],
                group_label=group_label,
                debug=False,
                roi_set=ROI_SET,
            )

    # collect ROI pkls
    rows = []
    for pkl_path in (out_root / eid).rglob("*.pkl"):
        try:
            row = summarize_region_pkl(pkl_path)
            if row["region"] in ROI_SET:
                rows.append(row)
        except Exception:
            continue

    status = "ok" if rows else "skip_no_roi_or_failed"
    return {
        "eid": eid,
        "status": status,
        "n_raw_trials": int(n_raw_trials),
        "rows": rows,
    }



def main():
    out_root = Path("./prior_localization_groups_output")
    out_root.mkdir(parents=True, exist_ok=True)


    eids = [
        # ORBvl example sessions
        # "1191f865-b10a-45c8-9c48-24a980fd9402",
        # "2e6e179c-fccc-4e8f-9448-ce5b6858a183",
        # "4d8c7767-981c-4347-8e5e-5d5fffe38534",
        # "d33baf74-263c-4b37-a0d0-b79dcb80a764",
        # "dd4da095-4a99-4bf3-9727-f735077dba66",
        # "dfbe628d-365b-461c-a07f-8b9911ba83aa",
        #
        #MOp example sessions
        #
        # "36280321-555b-446d-9b7d-c2e17991e090",
        # "4aa1d525-5c7d-4c50-a147-ec53a9014812",
        # "5455a21c-1be7-4cae-ae8e-8853a8d5f55e",
        # "81a78eac-9d36-4f90-a73a-7eb3ad7f770b",
        # "9e9c6fc0-4769-4d83-9ea4-b59a1230510e",
        # "bd456d8f-d36e-434a-8051-ff3997253802",
        # "cf43dbb1-6992-40ec-a5f9-e8e838d0f643",
        #
        # ACAd example sessions
        "78b4fff5-c5ec-44d9-b5f9-d59493063f00",
        "a4000c2f-fa75-4b3e-8f06-a7cf599b87ad",
    ]

    n_pseudo = 200
    debug = False
    n_workers = 4

    all_rows = []
    run_log = []

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(run_one_eid_worker, eid, str(out_root), n_pseudo, debug) for eid in eids]
        for fut in as_completed(futs):
            r = fut.result()
            run_log.append({k: r.get(k) for k in ["eid", "status", "n_raw_trials"]})
            all_rows.extend(r.get("rows", []))
            print("[DONE]", r.get("eid"), r.get("status"), "rows=", len(r.get("rows", [])))

    #  name outputs by ROI_LIST
    roi_tag = "ALL" if not ROI_LIST else "_".join(ROI_LIST)

    summary_path = out_root / f"roi_summary_{roi_tag}.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(all_rows, f)

    # optional parquet
    try:
        df_all = pd.DataFrame(all_rows)
        df_all.to_parquet(out_root / f"roi_summary_{roi_tag}.parquet", index=False)
    except Exception:
        pass

    # log also tagged (avoid overwriting between different ROI runs)
    run_log_path = out_root / f"run_log_{roi_tag}.json"
    with open(run_log_path, "w") as f:
        json.dump(run_log, f, indent=2)

    print("Saved big ROI summary:", summary_path)
    print("Saved run log:", run_log_path)


if __name__ == "__main__":
    main()
