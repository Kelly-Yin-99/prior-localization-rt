import logging
import pickle
import numpy as np
import os
import traceback
from typing import Optional, Union
import pandas as pd


from sklearn import linear_model as sklm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import RidgeCV, Ridge, Lasso, LassoCV
from sklearn.utils.class_weight import compute_sample_weight

from brainbox.io.one import SessionLoader
from brainwidemap.bwm_loading import load_trials_and_mask

from prior_localization.prepare_data import (
    prepare_ephys,
    prepare_behavior,
    prepare_motor,
    prepare_pupil,
    prepare_widefield,
    prepare_widefield_old,
)
from prior_localization.functions.behavior_targets import add_target_to_trials
from prior_localization.functions.neurometric import get_neurometric_parameters
from prior_localization.functions.utils import (
    check_inputs,
    check_config,
    compute_mask,
    subtract_motor_residuals,
    format_data_for_decoding,
    logisticreg_criteria,
    str2int,
)

# Set up logger
logger = logging.getLogger('prior_localization')

config = check_config()


def _subset_trials(trials: pd.DataFrame,
                   trials_df: Optional[pd.DataFrame] = None,
                   trial_mask: Optional[Union[np.ndarray, list]] = None) -> pd.DataFrame:
    """
    If trials_df is provided, use it instead of internally loaded trials.
    Then apply trial_mask (boolean mask or integer indices).
    """
    if trials_df is not None:
        trials = trials_df.copy()

    if trial_mask is not None:
        m = np.asarray(trial_mask)
        if m.dtype == bool:
            if len(m) != len(trials):
                raise ValueError(f"Boolean trial_mask length {len(m)} != n_trials {len(trials)}")
            trials = trials.loc[m].copy()
        else:
            trials = trials.iloc[m].copy()

    return trials.reset_index(drop=True)



def _trialwise_mean_fr_from_binned(binned, t0, t1):
    """
    binned: (n_trials, n_units) spike counts in [t0,t1] relative to align_event
    returns: (n_trials,) mean firing rate in spikes/unit/s
    """
    import numpy as np
    binned = np.asarray(binned)
    if binned.ndim != 2:
        raise ValueError(f"Expected binned 2D (n_trials,n_units), got {binned.shape}")
    n_trials, n_units = binned.shape
    win_len = float(t1 - t0)
    if win_len <= 0:
        raise ValueError("Invalid window length")
    # total spikes per trial across units
    spikes_per_trial = np.nansum(binned, axis=1)
    # mean spikes per unit per second
    fr = spikes_per_trial / (max(1, n_units) * win_len)
    return fr

#
#
# def fit_session_ephys(
#         one, session_id, subject, probe_name, output_dir, pseudo_ids=None,
#         target='pLeft', align_event='stimOn_times',
#         min_rt=0.08, max_rt=None,
#         time_window=(-0.6, -0.1),
#         binsize=None, n_bins_lag=None, n_bins=None,
#         model='optBay', n_runs=10,
#         compute_neurometrics=False, motor_residuals=False,
#         stage_only=False,
#         # New
#         trials_df=None,
#         trial_mask=None,
#         group_label="session",
#         debug=False,
#         roi_set=None,   # <-- NEW: set like {"MOp","MOs","ACAd","ORBvl"} or None
#
# ):
#     """
#     Fits a single session for ephys data, patched to support decoding on trial subgroups
#     (e.g., fast/normal/slow within a session) WITHOUT breaking the behavior pipeline.
#
#     Additional patch:
#     - If roi_set is provided:
#         * Early skip sessions that have NO ROI regions
#         * Only decode regions that are in roi_set
#     """
#
#
#     pseudo_ids, output_dir = check_inputs(
#         model, pseudo_ids, target, output_dir, config, logger, compute_neurometrics, motor_residuals
#     )
#
#     # Ensure Path-like output_dir
#     try:
#         _ = output_dir.joinpath
#     except Exception:
#         from pathlib import Path
#         output_dir = Path(output_dir)
#
#
#     sl = SessionLoader(one=one, eid=session_id)
#     sl.load_trials()
#
#     if debug:
#         print(f"[DEBUG] {session_id}: loaded sl.trials rows = {len(sl.trials)}")
#
#     # If user provides external trials_df, replace sl.trials
#     if trials_df is not None:
#         sl.trials = trials_df.copy()
#         if debug:
#             print(f"[DEBUG] {session_id}: replaced trials with trials_df rows = {len(sl.trials)}")
#
#     # Keep a full copy for behavior computations (IMPORTANT)
#     trials_full = sl.trials.copy()
#
#     min_rt = None
#     max_rt = None
#
#
#     # Compute base mask (choice/QC etc.)
#
#     _, base_mask = load_trials_and_mask(
#         one=one, eid=session_id, sess_loader=sl, min_rt=min_rt, max_rt=max_rt,
#         min_trial_len=None, max_trial_len=None,
#         exclude_nochoice=True, exclude_unbiased=False,
#     )
#
#     # Apply group mask (fast/normal/slow) on top of base mask
#
#     trials_mask_full = base_mask.copy()
#
#     if trial_mask is not None:
#         m = np.asarray(trial_mask)
#
#         # allow boolean mask or indices
#         if m.dtype != bool:
#             idx = m.astype(int)
#             m2 = np.zeros(len(sl.trials), dtype=bool)
#             m2[idx] = True
#             m = m2
#
#         if len(m) != len(sl.trials):
#             raise ValueError(f"trial_mask length {len(m)} != n_trials {len(sl.trials)}")
#
#         trials_mask_full = trials_mask_full & m
#
#     min_trials_group = 10
#     n_good = int(np.sum(trials_mask_full))
#
#     if debug:
#         print(f"[DEBUG] {session_id} group={group_label}: n_trials after mask = {n_good}")
#         print(f"[DEBUG] {session_id}: trials_mask sum after AND = {int(np.sum(trials_mask_full))}")
#
#     if n_good < min_trials_group:
#         if debug:
#             print(f"[DEBUG] {session_id} group={group_label}: too few trials (<{min_trials_group}), skipping")
#         return []
#
#     if int(np.sum(trials_mask_full)) <= config['min_trials']:
#         raise ValueError(
#             f"Session {session_id} has {int(np.sum(trials_mask_full))} good trials after masking, "
#             f"less than {config['min_trials']}."
#         )
#
#     # Indices of subgroup trials in FULL trial space
#     keep_idx = np.flatnonzero(trials_mask_full)
#
#     # ----------------------------
#     # SUBSET TRIALS for ephys & intervals (subgroup only)
#
#     sl.trials = trials_full.loc[trials_mask_full].reset_index(drop=True)
#     trials_mask = np.ones(len(sl.trials), dtype=bool)
#
#     align_times = sl.trials[align_event].to_numpy()
#     intervals = np.vstack([
#         align_times + time_window[0],
#         align_times + time_window[1]
#     ]).T
#
#     if debug:
#         print(f"[DEBUG] {session_id}: subgroup trials = {len(sl.trials)}")
#         print(f"[DEBUG] {session_id}: intervals shape (SUBGROUP) = {intervals.shape}")
#
#     if not np.all(np.isfinite(intervals)):
#         bad = np.where(~np.isfinite(intervals).all(axis=1))[0][:10]
#         raise ValueError(f"Non-finite intervals in subgroup (showing up to 10 rows): {bad}")
#
#     # If decoding wheel-based target, add behavior signal and update mask (subgroup space)
#     if target in ['wheel-speed', 'wheel-velocity']:
#         if binsize is None:
#             raise ValueError(f"If target is wheel-speed or wheel-velocity, binsize cannot be None")
#
#         sl.trials, trials_mask = add_target_to_trials(
#             session_loader=sl, target=target, intervals=intervals, binsize=binsize,
#             interval_len=time_window[1] - time_window[0], mask=trials_mask
#         )
#
#         if int(np.sum(trials_mask)) <= config['min_trials']:
#             raise ValueError(
#                 f"Session {session_id} has {int(np.sum(trials_mask))} good trials after wheel target masking, "
#                 f"less than {config['min_trials']}."
#             )
#
#         # Rebuild intervals again after trials_mask changed
#         align_times = sl.trials.loc[trials_mask, align_event].to_numpy()
#         intervals = np.vstack([align_times + time_window[0], align_times + time_window[1]]).T
#
#         if debug:
#             print(f"[DEBUG] {session_id}: intervals shape after wheel target masking = {intervals.shape}")
#
#
#     # Prepare ephys data (subgroup)
#
#     data_epoch, actual_regions, n_units, cluster_ids = prepare_ephys(
#         one, session_id, probe_name, config['regions'], intervals,
#         binsize=binsize, n_bins_lag=n_bins_lag, n_bins=n_bins,
#         qc=config['unit_qc'], min_units=config['min_units'], stage_only=stage_only,
#     )
#
#     if debug:
#         print(f"[DEBUG] {session_id}: actual_regions = {actual_regions}")
#         print(f"[DEBUG] {session_id}: n_units = {n_units}")
#
#     # ----------------------------
#     # NEW: ROI early skip BEFORE expensive prepare_behavior
#
#     if roi_set is not None and actual_regions is not None:
#         present = {r[0] if isinstance(r, (list, tuple)) else str(r) for r in actual_regions}
#         if len(present.intersection(set(roi_set))) == 0:
#             if debug:
#                 print(f"[DEBUG] {session_id} group={group_label}: no ROI regions present -> SKIP prepare_behavior")
#             return []
#
#     n_pseudo_sets = 1 if actual_regions is None else len(actual_regions)
#
#     if debug:
#         try:
#             print(f"[DEBUG] {session_id}: n_regions={len(data_epoch)} n_pseudo_sets={n_pseudo_sets}")
#             print(f"[DEBUG] {session_id}: actual_regions is None? {actual_regions is None}")
#             if isinstance(data_epoch, list) and len(data_epoch) > 0:
#                 print(f"[DEBUG] {session_id}: data_epoch[0] type={type(data_epoch[0])}")
#         except Exception as _e:
#             print("[DEBUG] could not summarize data_epoch:", _e)
#
#     # ----------------------------
#     # NOW compute behavior targets/masks in FULL space, slice to subgroup via keep_idx
#
#     all_trials, all_targets, all_masks, all_neurometrics = prepare_behavior(
#         session_id, subject,
#         trials_full,                 # FULL
#         trials_mask_full,            # FULL mask incl subgroup
#         pseudo_ids=pseudo_ids,
#         n_pseudo_sets=n_pseudo_sets,
#         output_dir=output_dir,
#         model=model,
#         target=target,
#         compute_neurometrics=compute_neurometrics,
#         keep_idx=keep_idx,           # slices to subgroup length
#     )
#
#     if debug:
#         try:
#             ex_lens_t = [len(all_targets[0][k]) for k in range(min(3, len(all_targets[0])))]
#             ex_lens_m = [len(all_masks[0][k]) for k in range(min(3, len(all_masks[0])))]
#             print(f"[DEBUG] {session_id}: behavior lens example targets={ex_lens_t} masks={ex_lens_m}")
#             print(
#                 f"[DEBUG] {session_id}: prepare_behavior returned lens:",
#                 len(all_trials), len(all_targets), len(all_masks), len(all_neurometrics)
#             )
#         except Exception as _e:
#             print("[DEBUG] could not summarize behavior outputs:", _e)
#
#     # Remove motor residuals if indicated
#     if motor_residuals:
#         motor_signals = prepare_motor(one, session_id, time_window=time_window)
#         all_targets, all_masks = subtract_motor_residuals(motor_signals, all_targets, all_masks)
#
#     # If staging only, stop here
#     if stage_only:
#         return
#
#
#     pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if len(pseudo_ids) > 1 else str(pseudo_ids[0])
#     probe_str = 'merged_probes' if (isinstance(probe_name, list) and len(probe_name) > 1) else probe_name
#
#
#     filenames = []
#     for i in range(len(data_epoch)):
#
#         # NEW: skip non-ROI regions to save time
#         if roi_set is not None and actual_regions is not None:
#             region_name = actual_regions[i][0] if isinstance(actual_regions[i], (list, tuple)) else str(actual_regions[i])
#             if region_name not in roi_set:
#                 continue
#         trialwise_mean_fr = None
#         fr_t0, fr_t1 = time_window  # e.g. (-0.6, -0.1)
#
#         if binsize is None:
#             # data_epoch[i] is (n_trials_subgroup, n_units) of spike COUNTS in that window
#             try:
#                 # mean spikes per unit per second, per trial
#                 trialwise_mean_fr = data_epoch[i].mean(axis=1) / (fr_t1 - fr_t0)
#             except Exception as e:
#                 if debug:
#                     print(f"[DEBUG] FR computation failed for region {actual_regions[i]}: {e}")
#                 trialwise_mean_fr = None
#         else:
#             # binsize != None: data_epoch[i] is lagged predictor matrices, not raw counts
#             trialwise_mean_fr = None
#         # Apply mask to targets
#         if isinstance(all_targets[i][0], list):
#             targets_masked = [
#                 [t[m_] for m_ in np.squeeze(np.where(m))] for t, m in zip(all_targets[i], all_masks[i])
#             ]
#         else:
#             targets_masked = [t[m] for t, m in zip(all_targets[i], all_masks[i])]
#
#         # Apply mask to ephys data
#         if isinstance(data_epoch[0], list):
#             data_masked = [
#                 [data_epoch[i][m_] for m_ in np.squeeze(np.where(m))] for m in all_masks[i]
#             ]
#         else:
#             data_masked = [data_epoch[i][m] for m in all_masks[i]]
#
#         # Fit
#         fit_results = fit_target(
#             all_data=data_masked,
#             all_targets=targets_masked,
#             all_trials=all_trials[i],
#             n_runs=n_runs,
#             all_neurometrics=all_neurometrics[i],
#             pseudo_ids=pseudo_ids,
#             base_rng_seed=str2int(session_id + '_'.join(actual_regions[i])),
#             group_label=group_label,
#             debug=True,
#             min_trials_subgroup=10,
#         )
#
#         # Add the mask to fit results (if enabled in config)
#         for fit_result in fit_results:
#             fit_result['mask'] = all_masks[i] if config['save_predictions'] else None
#
#         # Save
#         region_str = config['regions'] if (config['regions'] == 'all_regions') or (
#                 config['regions'] in config['region_defaults'].keys()) else '_'.join(actual_regions[i])
#
#         filename = output_dir.joinpath(
#             subject, session_id,
#             f'{region_str}_{probe_str}_pseudo_ids_{pseudo_str}_{group_label}.pkl'
#         )
#         filename.parent.mkdir(parents=True, exist_ok=True)
#
#         outdict = {
#             "fit": fit_results,
#             "subject": subject,
#             "eid": session_id,
#             "probe": probe_str,
#             "region": actual_regions[i],
#             "N_units": n_units[i],
#             "cluster_uuids": cluster_ids[i],
#             "group_label": group_label,
#             "trialwise_mean_fr": trialwise_mean_fr,  # (n_trials_subgroup,)
#             "fr_window": tuple(time_window),
#             "fr_align_event": align_event,
#         }
#
#         with open(filename, "wb") as fw:
#             pickle.dump(outdict, fw)
#
#         filenames.append(filename)
#
#         if debug:
#             print(f"[DEBUG] {session_id}: saved {filename}")
#
#     return filenames
#
#
#
#
# def fit_session_ephys(
#         one, session_id, subject, probe_name, output_dir, pseudo_ids=None,
#         target='pLeft', align_event='stimOn_times',
#         min_rt=0.08, max_rt=None,
#         time_window=(-0.6, -0.1),
#         binsize=None, n_bins_lag=None, n_bins=None,
#         model='optBay', n_runs=10,
#         compute_neurometrics=False, motor_residuals=False,
#         stage_only=False,
#         # New
#         trials_df=None,
#         trial_mask=None,
#         group_label="session",
#         debug=False,
#         roi_set=None,   # <-- NEW: set like {"MOp","MOs","ACAd","ORBvl"} or None
#
# ):
#     """
#     Fits a single session for ephys data, patched to support decoding on trial subgroups
#     (e.g., fast/normal/slow within a session) WITHOUT breaking the behavior pipeline.
#
#     Additional patch:
#     - If roi_set is provided:
#         * Early skip sessions that have NO ROI regions
#         * Only decode regions that are in roi_set
#     """
#
#
#     pseudo_ids, output_dir = check_inputs(
#         model, pseudo_ids, target, output_dir, config, logger, compute_neurometrics, motor_residuals
#     )
#
#     # Ensure Path-like output_dir
#     try:
#         _ = output_dir.joinpath
#     except Exception:
#         from pathlib import Path
#         output_dir = Path(output_dir)
#
#
#     sl = SessionLoader(one=one, eid=session_id)
#     sl.load_trials()
#
#     if debug:
#         print(f"[DEBUG] {session_id}: loaded sl.trials rows = {len(sl.trials)}")
#
#     # If user provides external trials_df, replace sl.trials
#     if trials_df is not None:
#         sl.trials = trials_df.copy()
#         if debug:
#             print(f"[DEBUG] {session_id}: replaced trials with trials_df rows = {len(sl.trials)}")
#
#     # Keep a full copy for behavior computations (IMPORTANT)
#     trials_full = sl.trials.copy()
#
#     min_rt = None
#     max_rt = None
#
#
#     # Compute base mask (choice/QC etc.)
#
#     _, base_mask = load_trials_and_mask(
#         one=one, eid=session_id, sess_loader=sl, min_rt=min_rt, max_rt=max_rt,
#         min_trial_len=None, max_trial_len=None,
#         exclude_nochoice=True, exclude_unbiased=False,
#     )
#
#     # Apply group mask (fast/normal/slow) on top of base mask
#
#     trials_mask_full = base_mask.copy()
#
#     if trial_mask is not None:
#         m = np.asarray(trial_mask)
#
#         # allow boolean mask or indices
#         if m.dtype != bool:
#             idx = m.astype(int)
#             m2 = np.zeros(len(sl.trials), dtype=bool)
#             m2[idx] = True
#             m = m2
#
#         if len(m) != len(sl.trials):
#             raise ValueError(f"trial_mask length {len(m)} != n_trials {len(sl.trials)}")
#
#         trials_mask_full = trials_mask_full & m
#
#     min_trials_group = 10
#     n_good = int(np.sum(trials_mask_full))
#
#     if debug:
#         print(f"[DEBUG] {session_id} group={group_label}: n_trials after mask = {n_good}")
#         print(f"[DEBUG] {session_id}: trials_mask sum after AND = {int(np.sum(trials_mask_full))}")
#
#     if n_good < min_trials_group:
#         if debug:
#             print(f"[DEBUG] {session_id} group={group_label}: too few trials (<{min_trials_group}), skipping")
#         return []
#
#     if int(np.sum(trials_mask_full)) <= config['min_trials']:
#         raise ValueError(
#             f"Session {session_id} has {int(np.sum(trials_mask_full))} good trials after masking, "
#             f"less than {config['min_trials']}."
#         )
#
#     # Indices of subgroup trials in FULL trial space
#     keep_idx = np.flatnonzero(trials_mask_full)
#
#     # ----------------------------
#     # SUBSET TRIALS for ephys & intervals (subgroup only)
#
#     sl.trials = trials_full.loc[trials_mask_full].reset_index(drop=True)
#     trials_mask = np.ones(len(sl.trials), dtype=bool)
#
#     align_times = sl.trials[align_event].to_numpy()
#     intervals = np.vstack([
#         align_times + time_window[0],
#         align_times + time_window[1]
#     ]).T
#
#     if debug:
#         print(f"[DEBUG] {session_id}: subgroup trials = {len(sl.trials)}")
#         print(f"[DEBUG] {session_id}: intervals shape (SUBGROUP) = {intervals.shape}")
#
#     if not np.all(np.isfinite(intervals)):
#         bad = np.where(~np.isfinite(intervals).all(axis=1))[0][:10]
#         raise ValueError(f"Non-finite intervals in subgroup (showing up to 10 rows): {bad}")
#
#     # If decoding wheel-based target, add behavior signal and update mask (subgroup space)
#     if target in ['wheel-speed', 'wheel-velocity']:
#         if binsize is None:
#             raise ValueError(f"If target is wheel-speed or wheel-velocity, binsize cannot be None")
#
#         sl.trials, trials_mask = add_target_to_trials(
#             session_loader=sl, target=target, intervals=intervals, binsize=binsize,
#             interval_len=time_window[1] - time_window[0], mask=trials_mask
#         )
#
#         if int(np.sum(trials_mask)) <= config['min_trials']:
#             raise ValueError(
#                 f"Session {session_id} has {int(np.sum(trials_mask))} good trials after wheel target masking, "
#                 f"less than {config['min_trials']}."
#             )
#
#         # Rebuild intervals again after trials_mask changed
#         align_times = sl.trials.loc[trials_mask, align_event].to_numpy()
#         intervals = np.vstack([align_times + time_window[0], align_times + time_window[1]]).T
#
#         if debug:
#             print(f"[DEBUG] {session_id}: intervals shape after wheel target masking = {intervals.shape}")
#
#
#     # Prepare ephys data (subgroup)
#
#     data_epoch, actual_regions, n_units, cluster_uuids_list, cluster_ids_list = prepare_ephys(
#         one, session_id, probe_name, config['regions'], intervals,
#         binsize=binsize, n_bins_lag=n_bins_lag, n_bins=n_bins,
#         qc=config['unit_qc'], min_units=config['min_units'], stage_only=stage_only,
#     )
#
#     if debug:
#         print(f"[DEBUG] {session_id}: actual_regions = {actual_regions}")
#         print(f"[DEBUG] {session_id}: n_units = {n_units}")
#
#     # ----------------------------
#     # NEW: ROI early skip BEFORE expensive prepare_behavior
#
#     if roi_set is not None and actual_regions is not None:
#         present = {r[0] if isinstance(r, (list, tuple)) else str(r) for r in actual_regions}
#         if len(present.intersection(set(roi_set))) == 0:
#             if debug:
#                 print(f"[DEBUG] {session_id} group={group_label}: no ROI regions present -> SKIP prepare_behavior")
#             return []
#
#     n_pseudo_sets = 1 if actual_regions is None else len(actual_regions)
#
#     if debug:
#         try:
#             print(f"[DEBUG] {session_id}: n_regions={len(data_epoch)} n_pseudo_sets={n_pseudo_sets}")
#             print(f"[DEBUG] {session_id}: actual_regions is None? {actual_regions is None}")
#             if isinstance(data_epoch, list) and len(data_epoch) > 0:
#                 print(f"[DEBUG] {session_id}: data_epoch[0] type={type(data_epoch[0])}")
#         except Exception as _e:
#             print("[DEBUG] could not summarize data_epoch:", _e)
#
#     # ----------------------------
#     # NOW compute behavior targets/masks in FULL space, slice to subgroup via keep_idx
#
#     all_trials, all_targets, all_masks, all_neurometrics = prepare_behavior(
#         session_id, subject,
#         trials_full,                 # FULL
#         trials_mask_full,            # FULL mask incl subgroup
#         pseudo_ids=pseudo_ids,
#         n_pseudo_sets=n_pseudo_sets,
#         output_dir=output_dir,
#         model=model,
#         target=target,
#         compute_neurometrics=compute_neurometrics,
#         keep_idx=keep_idx,           # slices to subgroup length
#     )
#
#     if debug:
#         try:
#             ex_lens_t = [len(all_targets[0][k]) for k in range(min(3, len(all_targets[0])))]
#             ex_lens_m = [len(all_masks[0][k]) for k in range(min(3, len(all_masks[0])))]
#             print(f"[DEBUG] {session_id}: behavior lens example targets={ex_lens_t} masks={ex_lens_m}")
#             print(
#                 f"[DEBUG] {session_id}: prepare_behavior returned lens:",
#                 len(all_trials), len(all_targets), len(all_masks), len(all_neurometrics)
#             )
#         except Exception as _e:
#             print("[DEBUG] could not summarize behavior outputs:", _e)
#
#     # Remove motor residuals if indicated
#     if motor_residuals:
#         motor_signals = prepare_motor(one, session_id, time_window=time_window)
#         all_targets, all_masks = subtract_motor_residuals(motor_signals, all_targets, all_masks)
#
#     # If staging only, stop here
#     if stage_only:
#         return
#
#
#     pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if len(pseudo_ids) > 1 else str(pseudo_ids[0])
#     probe_str = 'merged_probes' if (isinstance(probe_name, list) and len(probe_name) > 1) else probe_name
#
#
#     filenames = []
#     for i in range(len(data_epoch)):
#
#         # NEW: skip non-ROI regions to save time
#         if roi_set is not None and actual_regions is not None:
#             region_name = actual_regions[i][0] if isinstance(actual_regions[i], (list, tuple)) else str(actual_regions[i])
#             if region_name not in roi_set:
#                 continue
#         trialwise_mean_fr = None
#         fr_t0, fr_t1 = time_window  # e.g. (-0.6, -0.1)
#
#         if binsize is None:
#             # data_epoch[i] is (n_trials_subgroup, n_units) of spike COUNTS in that window
#             try:
#                 # mean spikes per unit per second, per trial
#                 trialwise_mean_fr = data_epoch[i].mean(axis=1) / (fr_t1 - fr_t0)
#             except Exception as e:
#                 if debug:
#                     print(f"[DEBUG] FR computation failed for region {actual_regions[i]}: {e}")
#                 trialwise_mean_fr = None
#         else:
#             # binsize != None: data_epoch[i] is lagged predictor matrices, not raw counts
#             trialwise_mean_fr = None
#         # Apply mask to targets
#         if isinstance(all_targets[i][0], list):
#             targets_masked = [
#                 [t[m_] for m_ in np.squeeze(np.where(m))] for t, m in zip(all_targets[i], all_masks[i])
#             ]
#         else:
#             targets_masked = [t[m] for t, m in zip(all_targets[i], all_masks[i])]
#
#         # Apply mask to ephys data
#         if isinstance(data_epoch[0], list):
#             data_masked = [
#                 [data_epoch[i][m_] for m_ in np.squeeze(np.where(m))] for m in all_masks[i]
#             ]
#         else:
#             data_masked = [data_epoch[i][m] for m in all_masks[i]]
#
#         # Fit
#         fit_results = fit_target(
#             all_data=data_masked,
#             all_targets=targets_masked,
#             all_trials=all_trials[i],
#             n_runs=n_runs,
#             all_neurometrics=all_neurometrics[i],
#             pseudo_ids=pseudo_ids,
#             base_rng_seed=str2int(session_id + '_'.join(actual_regions[i])),
#             group_label=group_label,
#             debug=True,
#             min_trials_subgroup=10,
#         )
#
#         # Add the mask to fit results (if enabled in config)
#         for fit_result in fit_results:
#             fit_result['mask'] = all_masks[i] if config['save_predictions'] else None
#
#         # Save
#         region_str = config['regions'] if (config['regions'] == 'all_regions') or (
#                 config['regions'] in config['region_defaults'].keys()) else '_'.join(actual_regions[i])
#
#         filename = output_dir.joinpath(
#             subject, session_id,
#             f'{region_str}_{probe_str}_pseudo_ids_{pseudo_str}_{group_label}.pkl'
#         )
#         filename.parent.mkdir(parents=True, exist_ok=True)
#
#         outdict = {
#             "fit": fit_results,
#             "subject": subject,
#             "eid": session_id,
#             "probe": probe_str,
#             "region": actual_regions[i],
#             "N_units": n_units[i],
#
#             # NEW: save true cluster IDs used for decoding (matches spikes['clusters'])
#             "cluster_ids": cluster_ids_list[i],
#
#             # OPTIONAL: keep uuids if you want (may be None)
#             "cluster_uuids": cluster_uuids_list[i],
#
#             "group_label": group_label,
#         }
#
#         with open(filename, "wb") as fw:
#             pickle.dump(outdict, fw)
#
#         filenames.append(filename)
#
#         if debug:
#             print(f"[DEBUG] {session_id}: saved {filename}")
#
#     return filenames
#

def fit_session_widefield(
        one, session_id, subject, output_dir, pseudo_ids=None, hemisphere=("left", "right"), target='pLeft',
        align_event='stimOn_times', min_rt=0.08, max_rt=None, frame_window=(-2, -2), model='optBay', n_runs=10,
        compute_neurometrics=False, stage_only=False, old_data=False
):

    """
    Fit a single session for widefield data.

    Parameters
    ----------
    one: one.api.ONE object
     ONE instance connected to database that data should be loaded from
    session_id: str
     Database UUID of the session uuid to run the decoding on
    subject: str
     Nickname of the mouse
    output_dir: str, pathlib.Path or None
     Directory in which the results are saved, will be created if it doesn't exist. Note that the function will reuse
     previously computed behavioural models if you are using the same path as for a previous run. Decoding results
     will be overwritten though.
    pseudo_ids: None or list of int
     List of sessions / pseudo sessions to decode, -1 represents decoding of the actual session, integers > 0 indicate
     pseudo_session ids.
    hemisphere: str or tuple of str
     Which hemisphere(s) to decode from {'left', 'right', ('left', 'right')}
    target: str
     Target to be decoded, options are {pLeft, prior, choice, feedback, signcont}, default is pLeft,
     meaning the prior probability that the stimulus  will appear on the left side
    align_event: str
     Event to which we align the time window, default is stimOn_times (stimulus onset). Options are
     {"firstMovement_times", "goCue_times", "stimOn_times", "feedback_times"}
    min_rt: float or None
        Minimum admissible reaction time in seconds for a trial to be included. Default is 0.08. If None, don't apply.
    max_rt: float or None
        Maximum admissible reaction time in seconds for a trial to be included. Default is None. If None, don't apply.
    frame_window: tuple of int
     Window in which neural activity is considered, in frames relative to align_event, default is (-2, -2) i.e. only a
     single frame is considered
    model: str
     Model to be decoded, options are {optBay, actKernel, stimKernel, oracle}, default is optBay
    n_runs: int
     Number of times to repeat full nested cross validation with different folds
    compute_neurometrics: bool
     Whether to compute neurometric shift and slopes (cf. Fig 3 of the paper)
    stage_only: bool
     If true, only download all required data, don't perform the actual decoding
    old_data: False or str
     Only used for sanity check, if false, use updated way of loading data from ONE. If str it should be a path
     to local copies of the previously used version of the data.

    Returns
    -------
    list
     List of paths to the results files
    """

    # Check some inputs
    pseudo_ids, output_dir = check_inputs(model, pseudo_ids, target, output_dir, config, logger, compute_neurometrics)

    # Load trials data
    sl = SessionLoader(one, eid=session_id)
    sl.load_trials()
    trials_mask = compute_mask(sl.trials, align_event=align_event, min_rt=min_rt, max_rt=max_rt, n_trials_crop_end=1)
    # _, trials_mask = load_trials_and_mask(
    #     one=one, eid=session_id, sess_loader=sl, min_rt=min_rt, max_rt=max_rt,
    #     min_trial_len=None, max_trial_len=None,
    #     exclude_nochoice=True, exclude_unbiased=False,
    # )
    if sum(trials_mask) <= config['min_trials']:
        raise ValueError(f"Session {session_id} has {sum(trials_mask)} good trials, less than {config['min_trials']}.")

    # Prepare widefield data
    if old_data is False:
        data_epoch, actual_regions = prepare_widefield(
            one, session_id, regions=config['regions'], align_times=sl.trials[align_event].values,
            frame_window=frame_window, hemisphere=hemisphere, stage_only=stage_only
        )
    else:
        data_epoch, actual_regions = prepare_widefield_old(old_data, hemisphere=hemisphere, regions=config['regions'],
                                                           align_event=align_event, frame_window=frame_window)

    n_pseudo_sets = 1 if actual_regions is None else len(actual_regions)

    # Compute or load behavior targets
    all_trials, all_targets, all_masks, all_neurometrics = prepare_behavior(
        session_id, subject, sl.trials, trials_mask, pseudo_ids=pseudo_ids, n_pseudo_sets=n_pseudo_sets,
        output_dir=output_dir, model=model, target=target, compute_neurometrics=compute_neurometrics)

    # If we are only staging data, we are done here
    if stage_only:
        return

    # Strings for saving
    pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if len(pseudo_ids) > 1 else str(pseudo_ids[0])
    hemi_str = 'both_hemispheres' if isinstance(hemisphere, tuple) or isinstance(hemisphere, list) else hemisphere

    # Fit data per region
    filenames = []
    for i in range(len(data_epoch)):

        # Apply mask to targets
        if isinstance(all_targets[i][0], list):
            targets_masked = [
                [t[m_] for m_ in np.squeeze(np.where(m))] for t, m in zip(all_targets[i], all_masks[i])
            ]
        else:
            targets_masked = [t[m] for t, m in zip(all_targets[i], all_masks[i])]

        # Apply mask to ephys data
        if isinstance(data_epoch[0], list):
            data_masked = [
                [data_epoch[i][m_] for m_ in np.squeeze(np.where(m))] for m in all_masks[i]
            ]
        else:
            data_masked = [data_epoch[i][m] for m in all_masks[i]]

        # Fit
        fit_results = fit_target(
            all_data=data_masked,
            all_targets=targets_masked,
            all_trials=all_trials[i],
            n_runs=n_runs,
            all_neurometrics=all_neurometrics[i],
            pseudo_ids=pseudo_ids,
            base_rng_seed=str2int(session_id + '_'.join(actual_regions[i])),
        )

        # Add the mask to fit results
        for fit_result in fit_results:
            fit_result['mask'] = all_masks[i] if config['save_predictions'] else None

        # Create output paths and save
        region_str = config['regions'] if (config['regions'] == 'all_regions') or (
                config['regions'] in config['region_defaults'].keys()) else '_'.join(actual_regions[i])
        filename = output_dir.joinpath(subject, session_id, f'{region_str}_{hemi_str}_pseudo_ids_{pseudo_str}.pkl')
        filename.parent.mkdir(parents=True, exist_ok=True)

        outdict = {
            "fit": fit_results,
            "subject": subject,
            "eid": session_id,
            "hemisphere": hemisphere,
            "region": actual_regions[0],
            "N_units": data_epoch[0].shape[1],
        }

        with open(filename, "wb") as fw:
            pickle.dump(outdict, fw)
        filenames.append(filename)
    return filenames


def fit_session_pupil(
        one, session_id, subject, output_dir, pseudo_ids=None, target='pLeft', align_event='stimOn_times',
        min_rt=0.08, max_rt=None, time_window=(-0.6, -0.1), model='optBay', n_runs=10, stage_only=False
):
    """
    Fit pupil tracking data to behavior (instead of neural activity)

    Parameters
    ----------
    one: one.api.ONE object
     ONE instance connected to database that data should be loaded from
    session_id: str
     Database UUID of the session uuid to run the decoding on
    subject: str
     Nickname of the mouse
    output_dir: str, pathlib.Path or None
     Directory in which the results are saved, will be created if it doesn't exist. Note that the function will reuse
     previously computed behavioural models if you are using the same path as for a previous run. Decoding results
     will be overwritten though.
    pseudo_ids: None or list of int
     List of sessions / pseudo sessions to decode, -1 represents decoding of the actual session, integers > 0 indicate
     pseudo_session ids.
    target: str
     Target to be decoded, options are {pLeft, prior, choice, feedback, signcont}, default is pLeft,
     meaning the prior probability that the stimulus  will appear on the left side
    align_event: str
     Event to which we align the time window, default is stimOn_times (stimulus onset). Options are
     {"firstMovement_times", "goCue_times", "stimOn_times", "feedback_times"}
    min_rt: float or None
        Minimum admissible reaction time in seconds for a trial to be included. Default is 0.08. If None, don't apply.
    max_rt: float or None
        Maximum admissible reaction time in seconds for a trial to be included. Default is None. If None, don't apply.
    time_window: tuple of float
     Time window in which pupil movement is considered, relative to align_event, default is (-0.6, -0.1)
    model: str
     Model to be decoded, options are {optBay, actKernel, stimKernel, oracle}, default is optBay
    n_runs: int
     Number of times to repeat full nested cross validation with different folds
    stage_only: bool
     If true, only download all required data, don't perform the actual decoding

    Returns
    -------
    list
     List of paths to the results files
    """
    # Check some inputs
    pseudo_ids, output_dir = check_inputs(model, pseudo_ids, target, output_dir, config, logger)

    # Load trials data
    sl = SessionLoader(one, eid=session_id)
    sl.load_trials()
    _, trials_mask = load_trials_and_mask(
        one=one, eid=session_id, sess_loader=sl, min_rt=min_rt, max_rt=max_rt,
        min_trial_len=None, max_trial_len=None,
        exclude_nochoice=True, exclude_unbiased=False,
    )
    if sum(trials_mask) <= config['min_trials']:
        raise ValueError(f"Session {session_id} has {sum(trials_mask)} good trials, less than {config['min_trials']}.")

    # Compute or load behavior targets
    all_trials, all_targets, all_masks, all_neurometrics = prepare_behavior(
        session_id, subject, sl.trials, trials_mask, pseudo_ids=pseudo_ids, n_pseudo_sets=1, output_dir=output_dir,
        model=model, target=target)

    # Load the pupil data
    pupil_data = prepare_pupil(one, session_id=session_id, time_window=time_window, align_event=align_event)

    if stage_only:
        return

    # For trials where there was no pupil data recording (start/end), add these to the trials_mask
    # `all_masks` is a list returned from prepare_behavior(), and generally has an entry for each region we are decoding
    # from. Here we're only decoding from the pupil, so len(all_masks) = 1.
    all_masks[0] = [a & ~np.any(np.isnan(pupil_data), axis=1) for a in all_masks[0]]

    # Apply mask to targets
    targets_masked = [t[m] for t, m in zip(all_targets[0], all_masks[0])]

    # Apply mask to ephys data
    pupil_masked = [pupil_data[m] for m in all_masks[0]]

    # Fit
    fit_results = fit_target(
        all_data=pupil_masked,
        all_targets=targets_masked,
        all_trials=all_trials[0],
        n_runs=n_runs,
        all_neurometrics=all_neurometrics[0],
        pseudo_ids=pseudo_ids,
        base_rng_seed=str2int(session_id),
    )

    # Create output paths and save
    pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if len(pseudo_ids) > 1 else str(pseudo_ids[0])
    filename = output_dir.joinpath(subject, session_id, f'pupil_pseudo_ids_{pseudo_str}.pkl')
    filename.parent.mkdir(parents=True, exist_ok=True)

    outdict = {
        "fit": fit_results,
        "subject": subject,
        "eid": session_id,
    }
    with open(filename, "wb") as fw:
        pickle.dump(outdict, fw)

    return filename


def fit_session_motor(
        one, session_id, subject, output_dir, pseudo_ids=None, target='pLeft', align_event='stimOn_times',
        min_rt=0.08, max_rt=None, time_window=(-0.6, -0.1), model='optBay', n_runs=10, stage_only=False
):
    """
    Fit movement tracking data to behavior (instead of neural actvity)

    Parameters
    ----------
    one: one.api.ONE object
     ONE instance connected to database that data should be loaded from
    session_id: str
     Database UUID of the session uuid to run the decoding on
    subject: str
     Nickname of the mouse
    output_dir: str, pathlib.Path or None
     Directory in which the results are saved, will be created if it doesn't exist. Note that the function will reuse
     previously computed behavioural models if you are using the same path as for a previous run. Decoding results
     will be overwritten though.
    pseudo_ids: None or list of int
     List of sessions / pseudo sessions to decode, -1 represents decoding of the actual session, integers > 0 indicate
     pseudo_session ids.
    target: str
     Target to be decoded, options are {pLeft, prior, choice, feedback, signcont}, default is pLeft,
     meaning the prior probability that the stimulus  will appear on the left side
    align_event: str
     Event to which we align the time window, default is stimOn_times (stimulus onset). Options are
     {"firstMovement_times", "goCue_times", "stimOn_times", "feedback_times"}
    min_rt: float or None
        Minimum admissible reaction time in seconds for a trial to be included. Default is 0.08. If None, don't apply.
    max_rt: float or None
        Maximum admissible reaction time in seconds for a trial to be included. Default is None. If None, don't apply.
    time_window: tuple of float
     Time window in which movement is considered, relative to align_event, default is (-0.6, -0.1)
    model: str
     Model to be decoded, options are {optBay, actKernel, stimKernel, oracle}, default is optBay
    n_runs: int
     Number of times to repeat full nested cross validation with different folds
    stage_only: bool
     If true, only download all required data, don't perform the actual decoding

    Returns
    -------
    list
     List of paths to the results files
    """

    # Check some inputs
    pseudo_ids, output_dir = check_inputs(model, pseudo_ids, target, output_dir, config, logger)

    # Load trials data
    sl = SessionLoader(one, eid=session_id)
    sl.load_trials()
    _, trials_mask = load_trials_and_mask(
        one=one, eid=session_id, sess_loader=sl, min_rt=min_rt, max_rt=max_rt,
        min_trial_len=None, max_trial_len=None,
        exclude_nochoice=True, exclude_unbiased=False,
    )
    if sum(trials_mask) <= config['min_trials']:
        raise ValueError(f"Session {session_id} has {sum(trials_mask)} good trials, less than {config['min_trials']}.")

    # Compute or load behavior targets
    all_trials, all_targets, all_masks, all_neurometrics = prepare_behavior(
        session_id, subject, sl.trials, trials_mask, pseudo_ids=pseudo_ids, n_pseudo_sets=1,
        output_dir=output_dir, model=model, target=target)

    # Load the motor data
    motor_data = prepare_motor(one, session_id=session_id, time_window=time_window, align_event=align_event)

    if stage_only:
        return

    # For trials where there was no pupil data recording (start/end), add these to the trials_mask
    # `all_masks` is a list returned from prepare_behavior(), and generally has an entry for each region we are decoding
    # from. Here we're only decoding from pose estimation traces, so len(all_masks) = 1.
    all_masks[0] = [a & ~np.any(np.isnan(motor_data), axis=1) for a in all_masks[0]]

    # Apply mask to targets
    targets_masked = [t[m] for t, m in zip(all_targets[0], all_masks[0])]

    # Apply mask to motor data
    motor_masked = [motor_data[m] for m in all_masks[0]]

    # Fit
    fit_results = fit_target(
        all_data=motor_masked,
        all_targets=targets_masked,
        all_trials=all_trials[0],
        n_runs=n_runs,
        all_neurometrics=all_neurometrics[0],
        pseudo_ids=pseudo_ids,
        base_rng_seed=str2int(session_id),
    )

    # Create output paths and save
    pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if len(pseudo_ids) > 1 else str(pseudo_ids[0])
    filename = output_dir.joinpath(subject, session_id, f'motor_pseudo_ids_{pseudo_str}.pkl')
    filename.parent.mkdir(parents=True, exist_ok=True)

    outdict = {
        "fit": fit_results,
        "subject": subject,
        "eid": session_id,
    }
    with open(filename, "wb") as fw:
        pickle.dump(outdict, fw)

    return filename





def ensure_2d_trials(Xs, ys):
    """
    Force each trial X to be 2D: (n_samples_per_trial, n_features)
    Force each trial y to be 1D and aligned with X.
    """
    import numpy as np

    Xs2, ys2 = [], []
    for x, y in zip(Xs, ys):
        x = np.asarray(x)
        y = np.asarray(y)

        if x.ndim == 1:
            x = x[None, :]          # (1, n_features)
        elif x.ndim != 2:
            raise ValueError(f"Trial X must be 1D or 2D, got shape {x.shape}")

        y = y.reshape(-1)

        # broadcast y if one value per trial but multiple samples
        if y.size == 1 and x.shape[0] > 1:
            y = np.repeat(y, x.shape[0])

        if y.size != x.shape[0]:
            raise ValueError(
                f"Trial y length {y.size} != X samples {x.shape[0]}"
            )

        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            continue

        Xs2.append(x.astype(np.float64, copy=False))
        ys2.append(y.astype(np.float64, copy=False))

    return Xs2, ys2
import numpy as np
import pickle

def fit_session_ephys(
        one, session_id, subject, probe_name, output_dir, pseudo_ids=None,
        target='pLeft', align_event='stimOn_times',
        min_rt=0.08, max_rt=None,
        time_window=(-0.6, -0.1),
        binsize=None, n_bins_lag=None, n_bins=None,
        model='optBay', n_runs=10,
        compute_neurometrics=False, motor_residuals=False,
        stage_only=False,
        # New
        trials_df=None,
        trial_mask=None,
        group_label="session",
        debug=False,
        roi_set=None,
):


    pseudo_ids, output_dir = check_inputs(
        model, pseudo_ids, target, output_dir, config, logger, compute_neurometrics, motor_residuals
    )

    # Ensure Path-like output_dir
    try:
        _ = output_dir.joinpath
    except Exception:
        from pathlib import Path
        output_dir = Path(output_dir)

    sl = SessionLoader(one=one, eid=session_id)
    sl.load_trials()

    # If user provides external trials_df, replace sl.trials
    if trials_df is not None:
        sl.trials = trials_df.copy()

    # Keep a full copy for behavior computations
    trials_full = sl.trials.copy()

    # Your code currently forces these to None (keep that behavior)
    min_rt = None
    max_rt = None

    # Base mask (choice/QC etc.)
    _, base_mask = load_trials_and_mask(
        one=one, eid=session_id, sess_loader=sl, min_rt=min_rt, max_rt=max_rt,
        min_trial_len=None, max_trial_len=None,
        exclude_nochoice=True, exclude_unbiased=False,
    )

    # Apply optional user-provided mask (IMPORTANT:
    # for "fit full session", just pass trial_mask=None)
    trials_mask_full = base_mask.copy()
    if trial_mask is not None:
        m = np.asarray(trial_mask)
        if m.dtype != bool:
            idx = m.astype(int)
            m2 = np.zeros(len(sl.trials), dtype=bool)
            m2[idx] = True
            m = m2
        if len(m) != len(sl.trials):
            raise ValueError(f"trial_mask length {len(m)} != n_trials {len(sl.trials)}")
        trials_mask_full = trials_mask_full & m

    if int(np.sum(trials_mask_full)) <= config['min_trials']:
        raise ValueError(
            f"Session {session_id} has {int(np.sum(trials_mask_full))} good trials after masking, "
            f"less than {config['min_trials']}."
        )

    # THIS IS THE CRITICAL THING WE NEED FOR POST-HOC SUBGROUPING
    keep_idx_full = np.flatnonzero(trials_mask_full).astype(int)

    # Subset sl.trials for decoding
    sl.trials = trials_full.loc[trials_mask_full].reset_index(drop=True)
    trials_mask = np.ones(len(sl.trials), dtype=bool)

    align_times = sl.trials[align_event].to_numpy()
    intervals = np.vstack([align_times + time_window[0], align_times + time_window[1]]).T

    if not np.all(np.isfinite(intervals)):
        bad = np.where(~np.isfinite(intervals).all(axis=1))[0][:10]
        raise ValueError(f"Non-finite intervals in selected trials (up to 10): {bad}")

    # wheel target support unchanged
    if target in ['wheel-speed', 'wheel-velocity']:
        if binsize is None:
            raise ValueError(f"If target is wheel-speed or wheel-velocity, binsize cannot be None")

        sl.trials, trials_mask = add_target_to_trials(
            session_loader=sl, target=target, intervals=intervals, binsize=binsize,
            interval_len=time_window[1] - time_window[0], mask=trials_mask
        )
        if int(np.sum(trials_mask)) <= config['min_trials']:
            raise ValueError(
                f"Session {session_id} has {int(np.sum(trials_mask))} good trials after wheel target masking, "
                f"less than {config['min_trials']}."
            )
        align_times = sl.trials.loc[trials_mask, align_event].to_numpy()
        intervals = np.vstack([align_times + time_window[0], align_times + time_window[1]]).T

    # Ephys prep
    data_epoch, actual_regions, n_units, cluster_uuids_list, cluster_ids_list = prepare_ephys(
        one, session_id, probe_name, config['regions'], intervals,
        binsize=binsize, n_bins_lag=n_bins_lag, n_bins=n_bins,
        qc=config['unit_qc'], min_units=config['min_units'], stage_only=stage_only,
    )

    # ROI early skip (unchanged)
    if roi_set is not None and actual_regions is not None:
        present = {r[0] if isinstance(r, (list, tuple)) else str(r) for r in actual_regions}
        if len(present.intersection(set(roi_set))) == 0:
            return []

    n_pseudo_sets = 1 if actual_regions is None else len(actual_regions)

    # Behavior prep: still uses keep_idx to slice FULL -> selected
    all_trials, all_targets, all_masks, all_neurometrics = prepare_behavior(
        session_id, subject,
        trials_full,
        trials_mask_full,
        pseudo_ids=pseudo_ids,
        n_pseudo_sets=n_pseudo_sets,
        output_dir=output_dir,
        model=model,
        target=target,
        compute_neurometrics=compute_neurometrics,
        keep_idx=keep_idx_full,
    )

    if motor_residuals:
        motor_signals = prepare_motor(one, session_id, time_window=time_window)
        all_targets, all_masks = subtract_motor_residuals(motor_signals, all_targets, all_masks)

    if stage_only:
        return

    pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if len(pseudo_ids) > 1 else str(pseudo_ids[0])
    probe_str = 'merged_probes' if (isinstance(probe_name, list) and len(probe_name) > 1) else probe_name

    filenames = []
    for i in range(len(data_epoch)):

        # optional ROI filtering
        if roi_set is not None and actual_regions is not None:
            region_name = actual_regions[i][0] if isinstance(actual_regions[i], (list, tuple)) else str(actual_regions[i])
            if region_name not in roi_set:
                continue

        # apply masks (unchanged)
        if isinstance(all_targets[i][0], list):
            targets_masked = [[t[m_] for m_ in np.squeeze(np.where(m))] for t, m in zip(all_targets[i], all_masks[i])]
        else:
            targets_masked = [t[m] for t, m in zip(all_targets[i], all_masks[i])]

        if isinstance(data_epoch[0], list):
            data_masked = [[data_epoch[i][m_] for m_ in np.squeeze(np.where(m))] for m in all_masks[i]]
        else:
            data_masked = [data_epoch[i][m] for m in all_masks[i]]

        # fit (unchanged flow: fit_target -> decode_cv)
        fit_results = fit_target(
            all_data=data_masked,
            all_targets=targets_masked,
            all_trials=all_trials[i],
            n_runs=n_runs,
            all_neurometrics=all_neurometrics[i],
            pseudo_ids=pseudo_ids,
            base_rng_seed=str2int(session_id + '_'.join(actual_regions[i])),
            group_label=group_label,
            debug=bool(debug),
            min_trials_subgroup=10,
        )

        for fit_result in fit_results:
            fit_result['mask'] = all_masks[i] if config['save_predictions'] else None

        region_str = config['regions'] if (config['regions'] == 'all_regions') or (
                config['regions'] in config['region_defaults'].keys()) else '_'.join(actual_regions[i])

        filename = output_dir.joinpath(
            subject, session_id,
            f'{region_str}_{probe_str}_pseudo_ids_{pseudo_str}_{group_label}.pkl'
        )
        filename.parent.mkdir(parents=True, exist_ok=True)

        outdict = {
            "fit": fit_results,
            "subject": subject,
            "eid": session_id,
            "probe": probe_str,
            "region": actual_regions[i],
            "N_units": n_units[i],
            "cluster_ids": cluster_ids_list[i],
            "cluster_uuids": cluster_uuids_list[i],
            "group_label": group_label,

            # >>> NEW: mapping FULL trial indices -> decoded trial order
            "keep_idx_full": keep_idx_full,
        }

        with open(filename, "wb") as fw:
            pickle.dump(outdict, fw)

        filenames.append(filename)

    return filenames


def fit_target(
        all_data, all_targets, all_trials, n_runs, all_neurometrics=None, pseudo_ids=None,
        base_rng_seed=0,
        # -------- NEW knobs ----------
        group_label="session",
        debug=False,
        min_trials_subgroup=10,
):
    """
    Same behavior as original fit_target, plus:
      - enforces min_trials_subgroup at the TRIAL level (default = 10)
      - sets decode_cv context so any debug prints identify group/pseudo/run/seed
    """
    if pseudo_ids is None:
        pseudo_ids = [-1]
    if not all_neurometrics:
        all_neurometrics = [None] * len(all_targets)

    fit_results = []

    for targets, data, trials, neurometrics, pseudo_id in zip(
            all_targets, all_data, all_trials, all_neurometrics, pseudo_ids):

        n_trials_here = len(trials)
        if n_trials_here < min_trials_subgroup:
            if debug:
                print(f"[fit_target] SKIP ctx={_ctx_str()} pseudo_id={pseudo_id} "
                      f"group={group_label} n_trials={n_trials_here} < {min_trials_subgroup}")
            continue

        for i_run in range(n_runs):
            if pseudo_id == -1:
                rng_seed = int(base_rng_seed) + i_run
            else:
                rng_seed = int(base_rng_seed) + int(pseudo_id) * int(n_runs) + i_run

            # set context for decode_cv
            set_decode_cv_context(
                group_label=group_label,
                pseudo_id=int(pseudo_id),
                run_id=int(i_run),
                rng_seed=int(rng_seed),
            )

            fit_result = decode_cv(
                ys=targets,
                Xs=data,
                estimator=config['estimator'],
                estimator_kwargs=config['estimator_kwargs'],
                hyperparam_grid=config['hparam_grid'],
                save_binned=False,
                save_predictions=config['save_predictions'],
                shuffle=config['shuffle'],
                balanced_weight=config['balanced_weighting'],
                rng_seed=rng_seed,
                use_cv_sklearn_method=config.get('use_native_sklearn_for_hyperparam_estimation', False),
                outer_cv=True,
                n_folds=5,
                verbose=bool(debug),
            )

            fit_result["trials_df"] = trials
            fit_result["pseudo_id"] = pseudo_id
            fit_result["run_id"] = i_run
            fit_result["group_label"] = group_label

            if neurometrics:
                fit_result["full_neurometric"], fit_result["fold_neurometric"] = get_neurometric_parameters(
                    fit_result, trialsdf=neurometrics, compute_on_each_fold=config['compute_neuro_on_each_fold']
                )
            else:
                fit_result["full_neurometric"] = None
                fit_result["fold_neurometric"] = None

            fit_results.append(fit_result)

    return fit_results




import numpy as np
import sklearn.linear_model as sklm
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight


_DECODE_CV_CONTEXT = {}

def set_decode_cv_context(**kwargs):
    global _DECODE_CV_CONTEXT
    _DECODE_CV_CONTEXT = dict(kwargs)

def _ctx_str():
    if not _DECODE_CV_CONTEXT:
        return "{}"
    return "{" + ", ".join([f"{k}={v}" for k, v in _DECODE_CV_CONTEXT.items()]) + "}"


def _safe_concat_y(y_list):
    # y_list is list of arrays (per trial). Original code uses concatenate(axis=0).
    # Works for both scalar-per-trial ([array([y])...]) and multi-bin-per-trial.
    return np.concatenate(y_list, axis=0)


def logisticreg_criteria(ys_list, min_unique_counts=2):
    """
    ys_list is list of per-trial arrays. We judge classes after stacking.
    """
    y = _safe_concat_y(ys_list).astype(float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return False
    uniq, cnt = np.unique(y, return_counts=True)
    return (len(uniq) == 2) and (np.min(cnt) >= min_unique_counts)


# def sample_folds(ys, get_kfold, isfoldsat, max_iter=100):
#     """
#     Same as their sample_folds: keep resampling KFold splits until every fold satisfies criteria.
#     ys: list of arrays (per trial)
#     """
#     sample_count = 0
#     ysatisfy = [False]
#
#     while not np.all(np.array(ysatisfy)):
#         if sample_count >= max_iter:
#             raise ValueError(f"[decode_cv] Could not sample satisfactory folds after {max_iter} tries ctx={_ctx_str()}")
#         sample_count += 1
#
#         outer_kfold = get_kfold()
#         fold_iter = [(train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(outer_kfold)]
#
#         # check each fold's *test* y has required properties
#         ysatisfy = []
#         for _, test_idxs in fold_iter:
#             y_test_list = [ys[i] for i in test_idxs]
#             ysatisfy.append(isfoldsat(y_test_list))
#
#     return sample_count, outer_kfold, fold_iter

#
# def decode_cv(
#     ys, Xs,
#     estimator, estimator_kwargs,
#     balanced_weight=False,
#     hyperparam_grid=None,
#     test_prop=0.2,
#     n_folds=5,
#     save_binned=False,
#     save_predictions=True,
#     verbose=False,
#     shuffle=True,
#     outer_cv=True,
#     rng_seed=None,
#     use_cv_sklearn_method=False,
#     min_trials=10,              #MIN TRIALS per subgroup
#     max_sample_folds_iter=200,   # a bit higher because small subgroups can be annoying
# ):
#     """
#     Like original prior_localization.decode_cv:
#       - uses per-trial folds (never splits within a trial)
#       - (optionally) resamples folds until they satisfy criteria (esp. logistic)
#     Adapted for subgroups:
#       - enforces min_trials
#       - caps n_folds to feasible values
#
#     """
#
#     # NOTE: you must have format_data_for_decoding in scope (same as original)
#     ys, Xs = format_data_for_decoding(ys, Xs)
#
#     n_trials = len(Xs)
#     if n_trials < min_trials:
#         raise ValueError(f"[decode_cv] Too few trials for subgroup: n_trials={n_trials} < {min_trials} ctx={_ctx_str()}")
#
#     # Seed control
#     rng = np.random.RandomState(int(rng_seed)) if rng_seed is not None else np.random.RandomState()
#     indices = np.arange(n_trials)
#
#     # scoring function
#     scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score
#
#     # containers
#     scores_test, scores_train = [], []
#     idxes_test, idxes_train = [], []
#     weights, intercepts, best_params = [], [], []
#     predictions = [None for _ in range(n_trials)]
#     predictions_to_save = [None for _ in range(n_trials)]
#
#     # hyperparam grid handling (same assumption as original: single key)
#     if hyperparam_grid is None or len(hyperparam_grid) == 0:
#         grid_key = None
#         grid_vals = None
#     else:
#         grid_key = list(hyperparam_grid.keys())[0]
#         grid_vals = list(hyperparam_grid[grid_key])
#
#     # stack trials into sample rows (bins)
#     def stack_trials(X_list, y_list):
#         X = np.vstack(X_list)
#         y = np.concatenate(y_list, axis=0)
#         X = np.asarray(X, dtype=np.float64)
#         y = np.asarray(y, dtype=np.float64)
#         return X, y
#
#     def dbg(msg):
#         if verbose:
#             print(msg)
#
#     #build outer folds like their code (sample until criteria satisfied)
#     if outer_cv:
#         # cap folds to available trials (and at least 2)
#         n_splits_outer = min(n_folds, n_trials)
#         if n_splits_outer < 2:
#             raise ValueError(f"[decode_cv] Too few trials for outer CV: n_trials={n_trials} ctx={_ctx_str()}")
#
#         def get_kfold_outer():
#             # IMPORTANT: use a fresh random_state each attempt so resampling actually changes
#             rs = rng.randint(0, 2**31 - 1)
#             return KFold(n_splits=n_splits_outer, shuffle=shuffle, random_state=rs).split(indices)
#
#         if estimator == sklm.LogisticRegression:
#             # overall dataset must have both classes with >=2 examples per class
#             if not logisticreg_criteria(ys, min_unique_counts=2):
#                 raise ValueError(f"[decode_cv] LogisticRegression requires 2 classes with >=2 examples each ctx={_ctx_str()}")
#             isysat_outer = lambda y_test_list: logisticreg_criteria(y_test_list, min_unique_counts=1)
#         else:
#             isysat_outer = lambda y_test_list: True
#
#         sample_count, _, outer_iter = sample_folds(
#             ys, get_kfold_outer, isysat_outer, max_iter=max_sample_folds_iter
#         )
#         if sample_count > 1:
#             dbg(f"[decode_cv] sampled outer folds {sample_count} times (subgroup) ctx={_ctx_str()}")
#
#     else:
#         tr, te = train_test_split(
#             indices,
#             test_size=test_prop,
#             shuffle=shuffle,
#             random_state=rng.randint(0, 2**31 - 1),
#         )
#         outer_iter = [(tr, te)]
#
#     # main outer loop
#     for fold_i, (train_idxs_outer, test_idxs_outer) in enumerate(outer_iter):
#
#         X_train_list = [Xs[i] for i in train_idxs_outer]
#         y_train_list = [ys[i] for i in train_idxs_outer]
#         X_test_list  = [Xs[i] for i in test_idxs_outer]
#         y_test_list  = [ys[i] for i in test_idxs_outer]
#
#         # inner CV to choose hyperparam (their nested CV idea)
#         best_param = {}
#
#         if (grid_key is not None) and (not use_cv_sklearn_method):
#             idx_inner = np.arange(len(train_idxs_outer))
#             n_trials_inner = len(idx_inner)
#
#             # cap inner folds too
#             n_splits_inner = min(n_folds, n_trials_inner)
#             if n_splits_inner < 2:
#                 best_val = grid_vals[0]
#             else:
#                 def get_kfold_inner():
#                     rs = rng.randint(0, 2**31 - 1)
#                     return KFold(n_splits=n_splits_inner, shuffle=shuffle, random_state=rs).split(idx_inner)
#
#                 if estimator == sklm.LogisticRegression:
#                     if not logisticreg_criteria(y_train_list, min_unique_counts=2):
#                         # can happen in tiny subgroup after outer split
#                         # fall back to first value (or you can skip fold)
#                         best_val = grid_vals[0]
#                     else:
#                         isysat_inner = lambda y_test_list_inner: logisticreg_criteria(y_test_list_inner, min_unique_counts=1)
#                         sample_count_in, _, inner_iter = sample_folds(
#                             y_train_list, get_kfold_inner, isysat_inner, max_iter=max_sample_folds_iter
#                         )
#                         if sample_count_in > 1:
#                             dbg(f"[decode_cv] sampled inner folds {sample_count_in} times ctx={_ctx_str()}")
#                 else:
#                     # regression: always valid folds
#                     inner_iter = [(tr_i, te_i) for _, (tr_i, te_i) in enumerate(get_kfold_inner())]
#
#                 scores_grid = np.full((len(inner_iter), len(grid_vals)), np.nan, dtype=float)
#
#                 for ifold, (tr_i, te_i) in enumerate(inner_iter):
#                     Xtr, ytr = stack_trials([X_train_list[i] for i in tr_i], [y_train_list[i] for i in tr_i])
#                     Xte, yte = stack_trials([X_train_list[i] for i in te_i], [y_train_list[i] for i in te_i])
#
#
#                     sw = None
#                     if balanced_weight and estimator == sklm.LogisticRegression:
#                         sw = compute_sample_weight("balanced", y=ytr)
#
#                     for ia, val in enumerate(grid_vals):
#                         try:
#                             kw = dict(estimator_kwargs)
#                             kw[grid_key] = val
#                             model_inner = estimator(**kw)
#                             with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
#                                 model_inner.fit(Xtr, ytr, sample_weight=sw)
#                                 yhat = model_inner.predict(Xte)
#                             scores_grid[ifold, ia] = scoring_f(yte, yhat)
#                         except Exception:
#                             # treat failed fit as very poor
#                             scores_grid[ifold, ia] = -np.inf
#
#                 # choose best hyperparam by average inner score
#                 mean_scores = np.nanmean(scores_grid, axis=0)
#                 best_val = grid_vals[int(np.argmax(mean_scores))]
#
#             best_param = {grid_key: best_val}
#
#         # outer fit with best param
#         X_train_array, y_train_array = stack_trials(X_train_list, y_train_list)
#         X_test_array,  y_test_array  = stack_trials(X_test_list,  y_test_list)
#
#         sw_train = None
#         if balanced_weight and estimator == sklm.LogisticRegression:
#             sw_train = compute_sample_weight("balanced", y=y_train_array)
#
#         kw = dict(estimator_kwargs)
#         kw.update(best_param)
#         model = estimator(**kw)
#
#         try:
#             with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
#                 model.fit(X_train_array, y_train_array, sample_weight=sw_train)
#         except Exception:
#             # skip this fold rather than crashing whole subgroup
#             continue
#
#         # train/test scores
#         with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
#             y_pred_train = model.predict(X_train_array)
#             y_pred_test  = model.predict(X_test_array)
#
#         scores_train.append(scoring_f(y_train_array, y_pred_train))
#         scores_test.append(scoring_f(y_test_array, y_pred_test))
#
#         # save per-trial predictions (keep trial boundaries)
#         for i_fold_local, i_global in enumerate(test_idxs_outer):
#             with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
#                 pred_bins = model.predict(X_test_list[i_fold_local])
#             predictions[i_global] = pred_bins
#
#
#             if isinstance(model, sklm.LogisticRegression):
#                 with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
#                     prob = model.predict_proba(X_test_list[i_fold_local])[:, 1]
#                 predictions_to_save[i_global] = prob
#             else:
#                 predictions_to_save[i_global] = pred_bins
#
#         idxes_test.append(test_idxs_outer)
#         idxes_train.append(train_idxs_outer)
#
#         if save_predictions:
#             weights.append(getattr(model, "coef_", None))
#             intercepts.append(getattr(model, "intercept_", None))
#             best_params.append(best_param)
#
#
#     if not any(p is not None for p in predictions):
#         raise ValueError(f"[decode_cv] No successful folds produced predictions ctx={_ctx_str()}")
#
#     ys_true_full = np.concatenate([ys[i] for i in range(n_trials) if predictions[i] is not None], axis=0)
#     ys_pred_full = np.concatenate([predictions[i] for i in range(n_trials) if predictions[i] is not None], axis=0)
#
#     outdict = {}
#     outdict["scores_test_full"] = scoring_f(ys_true_full, ys_pred_full)
#     outdict["scores_train"] = scores_train
#     outdict["scores_test"] = scores_test
#     outdict["Rsquared_test_full"] = r2_score(ys_true_full, ys_pred_full)
#
#     if estimator == sklm.LogisticRegression:
#         outdict["acc_test_full"] = accuracy_score(ys_true_full, ys_pred_full)
#         outdict["balanced_acc_test_full"] = balanced_accuracy_score(ys_true_full, ys_pred_full)
#
#     outdict["weights"] = weights if save_predictions else None
#     outdict["intercepts"] = intercepts if save_predictions else None
#     outdict["target"] = ys
#     outdict["predictions_test"] = predictions_to_save
#     outdict["regressors"] = Xs if save_binned else None
#     outdict["idxes_test"] = idxes_test if save_predictions else None
#     outdict["idxes_train"] = idxes_train if save_predictions else None
#     outdict["best_params"] = best_params if save_predictions else None
#     outdict["n_folds"] = n_folds
#
#     if hasattr(model, "classes_"):
#         outdict["classes_"] = model.classes_
#
#     return outdict
#
#
# def decode_cv(ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
#               n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
#               rng_seed=None, use_cv_sklearn_method=False):
#     """
#     Regresses binned neural activity against a target, using a provided sklearn estimator.
#
#     Parameters
#     ----------
#     ys : list of arrays or np.ndarray or pandas.Series
#         targets; if list, each entry is an array of targets for one trial. if 1D numpy array, each
#         entry is treated as a single scalar for one trial. if pd.Series, trial number is the index
#         and teh value is the target.
#     Xs : list of arrays or np.ndarray
#         predictors; if list, each entry is an array of neural activity for one trial. if 2D numpy
#         array, each row is treated as a single vector of activity for one trial, i.e. the array is
#         of shape (n_trials, n_neurons)
#     estimator : sklearn.linear_model object
#         estimator from sklearn which provides .fit, .score, and .predict methods. CV estimators
#         are NOT SUPPORTED. Must be a normal estimator, which is internally wrapped with
#         GridSearchCV
#     estimator_kwargs : dict
#         additional arguments for sklearn estimator
#     balanced_weight : bool
#         balanced weighting to target
#     hyperparam_grid : dict
#         key indicates hyperparameter to grid search over, and value is an array of nodes on the
#         grid. See sklearn.model_selection.GridSearchCV : param_grid for more specs.
#         Defaults to None, which means no hyperparameter estimation or GridSearchCV use.
#     test_prop : float
#         proportion of data to hold out as the test set after running hyperparameter tuning; only
#         used if `outer_cv=False`
#     n_folds : int
#         Number of folds for cross-validation during hyperparameter tuning; only used if
#         `outer_cv=True`
#     save_binned : bool
#         True to put the regressors Xs into the output dictionary.
#         Can cause file bloat if saving outputs.
#         Note: this function does not actually save any files!
#     save_predictions : bool
#         True to put the model predictions into the output dictionary.
#         Can cause file bloat if saving outputs.
#         Note: this function does not actually save any files!
#     shuffle : bool
#         True for interleaved cross-validation, False for contiguous blocks
#     outer_cv: bool
#         Perform outer cross validation such that the testing spans the entire dataset
#     rng_seed : int
#         control data splits
#     verbose : bool
#         Whether you want to hear about the function's life, how things are going, and what the
#         neighbor down the street said to it the other day.
#
#     Returns
#     -------
#     dict
#         Dictionary of fitting outputs including:
#             - Regression score (from estimator)
#             - Decoding coefficients
#             - Decoding intercept
#             - Per-trial target values (copy of tvec)
#             - Per-trial predictions from model
#             - Input regressors (optional, see Xs argument)
#
#     """
#
#     # transform target data into standard format: list of np.ndarrays
#     ys, Xs = format_data_for_decoding(ys, Xs)
#
#     # initialize containers to save outputs
#     n_trials = len(Xs)
#     bins_per_trial = len(Xs[0])
#     scores_test, scores_train = [], []
#     idxes_test, idxes_train = [], []
#     weights, intercepts, best_params = [], [], []
#     predictions = [None for _ in range(n_trials)]
#     predictions_to_save = [None for _ in range(n_trials)]  # different for logistic regression
#
#     # split the dataset in two parts, train and test
#     # when shuffle=False, the method will take the end of the dataset to create the test set
#     if rng_seed is not None:
#         np.random.seed(rng_seed)
#     indices = np.arange(n_trials)
#     if outer_cv:
#         # create kfold function to loop over
#         get_kfold = lambda: KFold(n_folds if not use_cv_sklearn_method else 50, shuffle=shuffle).split(indices)
#         # define function to evaluate whether folds are satisfactory
#         if estimator == sklm.LogisticRegression:
#             # folds must be chosen such that 2 classes are present in each fold, with minimum 2 examples per class
#             assert logisticreg_criteria(ys)
#             isysat = lambda ys: logisticreg_criteria(ys, min_unique_counts=2)
#         else:
#             isysat = lambda ys: True
#         sample_count, _, outer_kfold_iter = sample_folds(ys, get_kfold, isysat)
#         if sample_count > 1:
#             print(f'sampled outer folds {sample_count} times to ensure enough targets')
#     else:
#         outer_kfold = iter([train_test_split(indices, test_size=test_prop, shuffle=shuffle)])
#         outer_kfold_iter = [(train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(outer_kfold)]
#
#     # scoring function; use R2 for linear regression, accuracy for logistic regression
#     scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score
#
#     # Select either the GridSearchCV estimator for a normal estimator, or use the native estimator
#     # in the case of CV-type estimators
#     if estimator == sklm.RidgeCV or estimator == sklm.LassoCV or estimator == sklm.LogisticRegressionCV:
#         raise NotImplementedError("the code does not support a CV-type estimator.")
#     else:
#         # loop over outer folds
#         for train_idxs_outer, test_idxs_outer in outer_kfold_iter:
#             # outer fold data split
#             # X_train = np.vstack([Xs[i] for i in train_idxs])
#             # y_train = np.concatenate([ys[i] for i in train_idxs], axis=0)
#             # X_test = np.vstack([Xs[i] for i in test_idxs])
#             # y_test = np.concatenate([ys[i] for i in test_idxs], axis=0)
#             X_train = [Xs[i] for i in train_idxs_outer]
#             y_train = [ys[i] for i in train_idxs_outer]
#             X_test = [Xs[i] for i in test_idxs_outer]
#             y_test = [ys[i] for i in test_idxs_outer]
#
#             key = list(hyperparam_grid.keys())[0]  # TODO: make this more robust
#
#             if not use_cv_sklearn_method:
#
#                 """NOTE
#                 This section of the code implements a modified nested-cross validation procedure. When decoding signals
#                 with multiple samples per trial -- such as the wheel -- we need to create folds that do not put
#                 samples from the same trial into different folds.
#                 """
#
#                 # now loop over inner folds
#                 idx_inner = np.arange(len(X_train))
#
#                 get_kfold_inner = lambda: KFold(n_splits=n_folds, shuffle=shuffle).split(idx_inner)
#
#                 # produce inner_fold_iter
#                 if estimator == sklm.LogisticRegression:
#                     # make sure data has at least 2 examples per class
#                     assert logisticreg_criteria(y_train, min_unique_counts=2)
#                     # folds must be chosen such that 2 classes are present in each fold, with min 1 example per class
#                     isysat_inner = lambda ys: logisticreg_criteria(ys, min_unique_counts=1)
#                 else:
#                     isysat_inner = lambda ys: True
#                 sample_count, _, inner_kfold_iter = sample_folds(y_train, get_kfold_inner, isysat_inner)
#                 if sample_count > 1:
#                     print(f'sampled inner folds {sample_count} times to ensure enough targets')
#
#                 r2s = np.zeros([n_folds, len(hyperparam_grid[key])])
#                 inner_predictions = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
#                 inner_targets = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
#                 for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold_iter):
#
#                     # inner fold data split
#                     X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
#                     y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
#                     X_test_inner = np.vstack([X_train[i] for i in test_idxs_inner])
#                     y_test_inner = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)
#
#                     for i_alpha, alpha in enumerate(hyperparam_grid[key]):
#
#                         # compute weight for each training sample if requested
#                         sample_weight = compute_sample_weight("balanced", y=y_train_inner) if balanced_weight else None
#
#                         # initialize model
#                         model_inner = estimator(**{**estimator_kwargs, key: alpha})
#                         # fit model
#                         model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)
#                         # evaluate model
#                         pred_test_inner = model_inner.predict(X_test_inner)
#                         # record predictions and targets to check for nans later
#                         inner_predictions[test_idxs_inner, i_alpha] = np.mean(pred_test_inner)
#                         inner_targets[test_idxs_inner, i_alpha] = np.mean(y_test_inner)
#                         # record score
#                         r2s[ifold, i_alpha] = scoring_f(y_test_inner, pred_test_inner)
#
#                 assert np.all(~np.isnan(inner_predictions))
#                 assert np.all(~np.isnan(inner_targets))
#
#                 # select model with best hyperparameter value evaluated on inner-fold test data;
#                 # refit/evaluate on all inner-fold data
#                 r2s_avg = r2s.mean(axis=0)
#
#                 X_train_array = np.vstack(X_train)
#                 y_train_array = np.concatenate(y_train, axis=0)
#
#                 # compute weight for each training sample if requested
#                 sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None
#
#                 # initialize model
#                 best_alpha = hyperparam_grid[key][np.argmax(r2s_avg)]
#                 model = estimator(**{**estimator_kwargs, key: best_alpha})
#                 # fit model
#                 model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
#             else:
#                 if estimator not in [Ridge, Lasso]:
#                     raise NotImplementedError("This case is not implemented")
#                 model = (
#                     RidgeCV(alphas=hyperparam_grid[key])
#                     if estimator == Ridge
#                     else LassoCV(alphas=hyperparam_grid[key])
#                 )
#                 X_train_array = np.vstack(X_train)
#                 y_train_array = np.concatenate(y_train, axis=0)
#                 sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None
#
#                 model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
#                 best_alpha = model.alpha_
#
#             # evalute model on train data
#             y_pred_train = model.predict(X_train_array)
#             scores_train.append(scoring_f(y_train_array, y_pred_train))
#
#             # evaluate model on test data
#             y_true = np.concatenate(y_test, axis=0)
#             y_pred = model.predict(np.vstack(X_test))
#             if isinstance(model, sklm.LogisticRegression):
#                 y_pred_probs = model.predict_proba(np.vstack(X_test))[:, 1]  # probability of class 1
#             else:
#                 y_pred_probs = None
#             scores_test.append(scoring_f(y_true, y_pred))
#
#             # save the raw prediction in the case of linear and the predicted probabilities when
#             # working with logitistic regression
#             for i_fold, i_global in enumerate(test_idxs_outer):
#                 if bins_per_trial == 1:
#                     # we already computed these estimates, take from above
#                     predictions[i_global] = np.array([y_pred[i_fold]])
#                     if isinstance(model, sklm.LogisticRegression):
#                         predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])
#                     else:
#                         predictions_to_save[i_global] = np.array([y_pred[i_fold]])
#                 else:
#                     # we already computed these above, but after all trials were stacked; recompute per-trial
#                     predictions[i_global] = model.predict(X_test[i_fold])
#                     if isinstance(model, sklm.LogisticRegression):
#                         predictions_to_save[i_global] = model.predict_proba(X_test[i_fold])[:, 1]
#                     else:
#                         predictions_to_save[i_global] = predictions[i_global]
#
#             # save out other data of interest
#             idxes_test.append(test_idxs_outer)
#             idxes_train.append(train_idxs_outer)
#             weights.append(model.coef_)
#             if model.fit_intercept:
#                 intercepts.append(model.intercept_)
#             else:
#                 intercepts.append(None)
#             best_params.append({key: best_alpha})
#
#     ys_true_full = np.concatenate(ys, axis=0)
#     ys_pred_full = np.concatenate(predictions, axis=0)
#     outdict = dict()
#     outdict["scores_test_full"] = scoring_f(ys_true_full, ys_pred_full)
#     outdict["scores_train"] = scores_train
#     outdict["scores_test"] = scores_test
#     outdict["Rsquared_test_full"] = r2_score(ys_true_full, ys_pred_full)
#     if estimator == sklm.LogisticRegression:
#         outdict["acc_test_full"] = accuracy_score(ys_true_full, ys_pred_full)
#         outdict["balanced_acc_test_full"] = balanced_accuracy_score(
#             ys_true_full, ys_pred_full
#         )
#     outdict["weights"] = weights if save_predictions else None
#     outdict["intercepts"] = intercepts if save_predictions else None
#     outdict["target"] = ys
#     outdict["predictions_test"] = predictions_to_save
#     outdict["regressors"] = Xs if save_binned else None
#     outdict["idxes_test"] = idxes_test if save_predictions else None
#     outdict["idxes_train"] = idxes_train if save_predictions else None
#     outdict["best_params"] = best_params if save_predictions else None
#     outdict["n_folds"] = n_folds
#     if hasattr(model, "classes_"):
#         outdict["classes_"] = model.classes_
#
#     # logging
#     if verbose:
#         # verbose output
#         if outer_cv:
#             print("Performance is only described for last outer fold \n")
#         print(
#             "Possible regularization parameters over {} validation sets:".format(
#                 n_folds
#             )
#         )
#         print("{}: {}".format(list(hyperparam_grid.keys())[0], hyperparam_grid))
#         print("\nBest parameters found over {} validation sets:".format(n_folds))
#         print(model.best_params_)
#         print("\nAverage scores over {} validation sets:".format(n_folds))
#         means = model.cv_results_["mean_test_score"]
#         stds = model.cv_results_["std_test_score"]
#         for mean, std, params in zip(means, stds, model.cv_results_["params"]):
#             print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
#         print("\n", "Detailed scores on {} validation sets:".format(n_folds))
#         for i_fold in range(n_folds):
#             tscore_fold = list(
#                 np.round(model.cv_results_["split{}_test_score".format(int(i_fold))], 3)
#             )
#             print("perf on fold {}: {}".format(int(i_fold), tscore_fold))
#
#         print("\n", "Detailed classification report:", "\n")
#         print("The model is trained on the full (train + validation) set.")
#
#     return outdict
#

def sample_folds(ys, get_kfold, isfoldsat, max_iter=100):
    """Sample a set of folds and ensure each fold satisfies user-defined criteria.

    Parameters
    ----------
    ys : array-like
        array of indices to split into folds
    get_kfold : callable
        callable function that returns a generator object
    isfoldsat : callable
        callable function that takes an array as input and returns a bool denoting is criteria are satisfied
    max_iter : int, optional
        maximum number of attempts to split folds

    Returns
    -------
    tuple
        - (int) number of samples required to satisfy fold criteria
        - (generator) fold generator
        - (list) list of tuples (train_idxs, test_idxs), one tuple for each fold

    """
    sample_count = 0
    ysatisfy = [False]
    while not np.all(np.array(ysatisfy)):
        assert sample_count < max_iter
        sample_count += 1
        outer_kfold = get_kfold()
        fold_iter = [(train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(outer_kfold)]
        ysatisfy = [isfoldsat(np.concatenate([ys[i] for i in t_idxs], axis=0)) for t_idxs, _ in fold_iter]

    return sample_count, outer_kfold, fold_iter


#---------------------------------------------------------------------------------------------------------------
def decode_cv(
    ys, Xs, estimator, estimator_kwargs,
    balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
    n_folds=5, save_binned=False, save_predictions=True,
    verbose=False, shuffle=True, outer_cv=True,
    rng_seed=None, use_cv_sklearn_method=False
):
    """
    Regresses binned neural activity against a target, using a provided sklearn estimator.

    Parameters
    ----------
    ys : list of arrays or np.ndarray or pandas.Series
        targets; if list, each entry is an array of targets for one trial. if 1D numpy array, each
        entry is treated as a single scalar for one trial. if pd.Series, trial number is the index
        and the value is the target.
    Xs : list of arrays or np.ndarray
        predictors; if list, each entry is an array of neural activity for one trial. if 2D numpy
        array, each row is treated as a single vector of activity for one trial, i.e. the array is
        of shape (n_trials, n_neurons)
    estimator : sklearn.linear_model object
        estimator from sklearn which provides .fit, .score, and .predict methods. CV estimators
        are NOT SUPPORTED. Must be a normal estimator, which is internally wrapped with
        GridSearchCV (or your manual nested CV below).
    estimator_kwargs : dict
        additional arguments for sklearn estimator
    balanced_weight : bool
        balanced weighting to target
    hyperparam_grid : dict
        key indicates hyperparameter to grid search over, and value is an array of nodes on the
        grid. See sklearn.model_selection.GridSearchCV : param_grid for more specs.
        Defaults to None, which means no hyperparameter estimation or GridSearchCV use.
    test_prop : float
        proportion of data to hold out as the test set after running hyperparameter tuning; only
        used if `outer_cv=False`
    n_folds : int
        Number of folds for cross-validation during hyperparameter tuning; only used if
        `outer_cv=True`
    save_binned : bool
        True to put the regressors Xs into the output dictionary.
        Can cause file bloat if saving outputs.
        Note: this function does not actually save any files!
    save_predictions : bool
        True to put the model predictions into the output dictionary.
        Can cause file bloat if saving outputs.
        Note: this function does not actually save any files!
    shuffle : bool
        True for interleaved cross-validation, False for contiguous blocks
    outer_cv: bool
        Perform outer cross validation such that the testing spans the entire dataset
    rng_seed : int
        control data splits
    verbose : bool
        Whether you want to hear about the function's life, how things are going, and what the
        neighbor down the street said to it the other day.

    Returns
    -------
    dict
        Dictionary of fitting outputs including:
            - Regression score (from estimator)
            - Decoding coefficients
            - Decoding intercept
            - Per-trial target values (copy of tvec)
            - Per-trial predictions from model
            - Input regressors (optional, see Xs argument)
    """

    # -------------------------
    # transform target data into standard format: list of np.ndarrays
    # -------------------------
    ys, Xs = format_data_for_decoding(ys, Xs)

    # initialize containers to save outputs
    n_trials = len(Xs)
    bins_per_trial = len(Xs[0])
    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []
    predictions = [None for _ in range(n_trials)]
    predictions_to_save = [None for _ in range(n_trials)]  # different for logistic regression

    # split the dataset in two parts, train and test
    # when shuffle=False, the method will take the end of the dataset to create the test set
    if rng_seed is not None:
        np.random.seed(rng_seed)
    indices = np.arange(n_trials)

    if outer_cv:
        # create kfold function to loop over
        get_kfold = lambda: KFold(n_folds if not use_cv_sklearn_method else 50, shuffle=shuffle).split(indices)

        # define function to evaluate whether folds are satisfactory
        if estimator == sklm.LogisticRegression:
            # folds must be chosen such that 2 classes are present in each fold, with minimum 2 examples per class
            assert logisticreg_criteria(ys)
            isysat = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=2)
        else:
            isysat = lambda ys_: True

        sample_count, _, outer_kfold_iter = sample_folds(ys, get_kfold, isysat)
        if sample_count > 1:
            print(f"sampled outer folds {sample_count} times to ensure enough targets")
    else:
        outer_kfold = iter([train_test_split(indices, test_size=test_prop, shuffle=shuffle)])
        outer_kfold_iter = [(train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(outer_kfold)]

    # scoring function; use R2 for linear regression, accuracy for logistic regression
    scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score

    # Select either the GridSearchCV estimator for a normal estimator, or use the native estimator
    # in the case of CV-type estimators
    if estimator == sklm.RidgeCV or estimator == sklm.LassoCV or estimator == sklm.LogisticRegressionCV:
        raise NotImplementedError("the code does not support a CV-type estimator.")
    else:
        # loop over outer folds
        for train_idxs_outer, test_idxs_outer in outer_kfold_iter:
            # outer fold data split
            X_train = [Xs[i] for i in train_idxs_outer]
            y_train = [ys[i] for i in train_idxs_outer]
            X_test  = [Xs[i] for i in test_idxs_outer]
            y_test  = [ys[i] for i in test_idxs_outer]

            # hyperparam key (e.g. "alpha")
            if hyperparam_grid is None or len(hyperparam_grid) == 0:
                raise ValueError("hyperparam_grid must be provided (e.g., {'alpha': [...]})")
            key = list(hyperparam_grid.keys())[0]  # TODO: make this more robust

            # we'll store these for verbose printing (only meaningful for manual nested CV)
            r2s = None
            r2s_avg = None

            if not use_cv_sklearn_method:
                """
                NOTE
                This section implements a modified nested-cross validation procedure.
                When decoding signals with multiple samples per trial -- such as the wheel --
                we need to create folds that do not put samples from the same trial into different folds.
                """

                # now loop over inner folds
                idx_inner = np.arange(len(X_train))
                get_kfold_inner = lambda: KFold(n_splits=n_folds, shuffle=shuffle).split(idx_inner)

                # produce inner_fold_iter
                if estimator == sklm.LogisticRegression:
                    assert logisticreg_criteria(y_train, min_unique_counts=2)
                    isysat_inner = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=1)
                else:
                    isysat_inner = lambda ys_: True

                sample_count, _, inner_kfold_iter = sample_folds(y_train, get_kfold_inner, isysat_inner)
                if sample_count > 1:
                    print(f"sampled inner folds {sample_count} times to ensure enough targets")

                r2s = np.zeros([n_folds, len(hyperparam_grid[key])])
                inner_predictions = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
                inner_targets     = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan

                for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold_iter):
                    # inner fold data split (stack trials to sample-level)
                    X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
                    y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
                    X_test_inner  = np.vstack([X_train[i] for i in test_idxs_inner])
                    y_test_inner  = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)

                    for i_alpha, alpha in enumerate(hyperparam_grid[key]):
                        sample_weight = (
                            compute_sample_weight("balanced", y=y_train_inner)
                            if balanced_weight else None
                        )

                        model_inner = estimator(**{**estimator_kwargs, key: alpha})
                        model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)

                        pred_test_inner = model_inner.predict(X_test_inner)

                        # record predictions and targets to check for nans later
                        inner_predictions[test_idxs_inner, i_alpha] = np.mean(pred_test_inner)
                        inner_targets[test_idxs_inner, i_alpha]     = np.mean(y_test_inner)

                        r2s[ifold, i_alpha] = scoring_f(y_test_inner, pred_test_inner)

                assert np.all(~np.isnan(inner_predictions))
                assert np.all(~np.isnan(inner_targets))

                # select model with best hyperparameter averaged over inner folds
                r2s_avg = r2s.mean(axis=0)

                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)

                sample_weight = (
                    compute_sample_weight("balanced", y=y_train_array)
                    if balanced_weight else None
                )

                best_alpha = hyperparam_grid[key][np.argmax(r2s_avg)]
                model = estimator(**{**estimator_kwargs, key: best_alpha})
                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)

            else:
                # sklearn CV method path (note: this does NOT have model.best_params_ / model.cv_results_)
                if estimator not in [Ridge, Lasso]:
                    raise NotImplementedError("This case is not implemented")
                model = (
                    RidgeCV(alphas=hyperparam_grid[key])
                    if estimator == Ridge
                    else LassoCV(alphas=hyperparam_grid[key])
                )

                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)

                sample_weight = (
                    compute_sample_weight("balanced", y=y_train_array)
                    if balanced_weight else None
                )

                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
                best_alpha = model.alpha_

            # evaluate model on train data
            y_pred_train = model.predict(X_train_array)
            scores_train.append(scoring_f(y_train_array, y_pred_train))

            # evaluate model on test data
            y_true = np.concatenate(y_test, axis=0)
            y_pred = model.predict(np.vstack(X_test))

            if isinstance(model, sklm.LogisticRegression):
                y_pred_probs = model.predict_proba(np.vstack(X_test))[:, 1]
            else:
                y_pred_probs = None

            scores_test.append(scoring_f(y_true, y_pred))

            # save per-trial predictions[param] (probability for logistic)
            for i_fold, i_global in enumerate(test_idxs_outer):
                if bins_per_trial == 1:
                    predictions[i_global] = np.array([y_pred[i_fold]])
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])
                    else:
                        predictions_to_save[i_global] = np.array([y_pred[i_fold]])
                else:
                    predictions[i_global] = model.predict(X_test[i_fold])
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = model.predict_proba(X_test[i_fold])[:, 1]
                    else:
                        predictions_to_save[i_global] = predictions[i_global]

            # save out other data of interest
            idxes_test.append(test_idxs_outer)
            idxes_train.append(train_idxs_outer)
            weights.append(model.coef_)
            if getattr(model, "fit_intercept", False):
                intercepts.append(model.intercept_)
            else:
                intercepts.append(None)
            best_params.append({key: float(best_alpha)})

            # -------------------------
            # logging (FIXED: no best_params_ assumption)
            # -------------------------
            if verbose:
                if outer_cv:
                    print("Performance is only described for last outer fold \n")

                print(f"Possible regularization parameters over {n_folds} validation sets:")
                print(f"{key}: {hyperparam_grid}")

                print(f"\nBest parameters found over {n_folds} validation sets:")
                # Always safe because we compute best_alpha in BOTH branches
                print({key: float(best_alpha)})

                # If you used the manual nested CV branch, we can also print avg scores
                if (r2s_avg is not None) and (r2s is not None):
                    print(f"\nAverage scores over {n_folds} validation sets:")
                    for a, s in zip(hyperparam_grid[key], r2s_avg):
                        print(f"{s:0.3f} for {key}={a}")

                    print(f"\nDetailed scores on {n_folds} validation sets:")
                    for i_fold in range(n_folds):
                        tscore_fold = list(np.round(r2s[i_fold, :], 3))
                        print(f"perf on fold {i_fold}: {tscore_fold}")

                # If using RidgeCV/LassoCV, sklearn doesn't expose cv_results_ like GridSearchCV
                elif hasattr(model, "alpha_"):
                    print("\n(skipping cv_results_ details; RidgeCV/LassoCV does not provide GridSearchCV-style cv_results_)")

                print("\nThe model is trained on the full (train + validation) set.")

    # build output dict
    ys_true_full = np.concatenate(ys, axis=0)
    ys_pred_full = np.concatenate(predictions, axis=0)

    outdict = dict()
    outdict["scores_test_full"] = scoring_f(ys_true_full, ys_pred_full)
    outdict["scores_train"] = scores_train
    outdict["scores_test"] = scores_test
    outdict["Rsquared_test_full"] = r2_score(ys_true_full, ys_pred_full)

    if estimator == sklm.LogisticRegression:
        outdict["acc_test_full"] = accuracy_score(ys_true_full, ys_pred_full)
        outdict["balanced_acc_test_full"] = balanced_accuracy_score(ys_true_full, ys_pred_full)

    outdict["weights"] = weights if save_predictions else None
    outdict["intercepts"] = intercepts if save_predictions else None
    outdict["target"] = ys
    outdict["predictions_test"] = predictions_to_save
    outdict["regressors"] = Xs if save_binned else None
    outdict["idxes_test"] = idxes_test if save_predictions else None
    outdict["idxes_train"] = idxes_train if save_predictions else None
    outdict["best_params"] = best_params if save_predictions else None
    outdict["n_folds"] = n_folds
    if hasattr(model, "classes_"):
        outdict["classes_"] = model.classes_

    return outdict

