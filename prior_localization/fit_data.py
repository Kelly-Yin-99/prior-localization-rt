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
# Load and check configuration file
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


def fit_session_ephys(
        one, session_id, subject, probe_name, output_dir, pseudo_ids=None, target='pLeft', align_event='stimOn_times',
        min_rt=None , max_rt=None, time_window=(-0.6, -0.1), binsize=None, n_bins_lag=None, n_bins=None, model='optBay',
        n_runs=10, compute_neurometrics=False, motor_residuals=False, stage_only=False,trials_df=None,trial_mask=None,
        group_label="session",debug=False,

):
    """
    Fits a single session for ephys data.
    Parameters
    ----------
    one: one.api.ONE object
     ONE instance connected to database that data should be loaded from
    session_id: str
     Database UUID of the session uuid to run the decoding on
    subject: str
     Nickname of the mouse
    probe_name: str or list of str
     Probe name(s), if list of probe names, the probes, the data of both probes will be merged for decoding
    output_dir: str, pathlib.Path or None
     Directory in which the results are saved, will be created if it doesn't exist. Note that the function will reuse
     previously computed behavioural models if you are using the same path as for a previous run. Decoding results
     will be overwritten though.
    pseudo_ids: None or list of int
     List of sessions / pseudo sessions to decode, -1 represents decoding of the actual session, integers > 0 indicate
     pseudo_session ids.
    target: str
     Target to be decoded, options are {pLeft, prior, choice, feedback, signcont, 'wheel-speed', 'wheel-velocity'},
     default is pLeft, meaning the prior probability that the stimulus  will appear on the left side
    align_event: str
     Event to which we align the time window, default is stimOn_times (stimulus onset). Options are
     {"firstMovement_times", "goCue_times", "stimOn_times", "feedback_times"}
    min_rt: float or None
        Minimum admissible reaction time in seconds for a trial to be included. Default is 0.08. If None, don't apply.
    max_rt: float or None
        Maximum admissible reaction time in seconds for a trial to be included. Default is None. If None, don't apply.
    time_window: tuple of float
     Time window in which neural activity is considered, relative to align_event, default is (-0.6, -0.1)
    binsize : float or None
     if None, sum spikes in time_window for decoding; if float, split time window into smaller bins
    n_bins_lag : int or None
     number of lagged timepoints (includes zero lag) for decoding wheel and DLC targets
    n_bins : int or None
     number of bins; should be computable from intervals and binsize, but there are occasional rounding errors
    model: str
     Model to be decoded, options are {optBay, actKernel, stimKernel, oracle}, default is optBay
    n_runs: int
     Number of times to repeat full nested cross validation with different folds
    compute_neurometrics: bool
     Whether to compute neurometric shift and slopes (cf. Fig 3 of the paper)
    motor_residuals: bool
     Whether ot compute the motor residual before performing neural decoding. This argument is used to study embodiment
     corresponding to figure 2f, default is False
    stage_only: bool
     If true, only download all required data, don't perform the actual decoding

    Returns
    -------
    list
     List of paths to the results files
    """

    # Check some inputs
    pseudo_ids, output_dir = check_inputs(
        model, pseudo_ids, target, output_dir, config, logger, compute_neurometrics, motor_residuals
    )

    # Load trials data and compute mask
    sl = SessionLoader(one=one, eid=session_id)
    sl.load_trials()
    _, trials_mask = load_trials_and_mask(
        one=one, eid=session_id, sess_loader=sl, min_rt=min_rt, max_rt=max_rt,
        min_trial_len=None, max_trial_len=None,
        exclude_nochoice=True, exclude_unbiased=False,
    )
    intervals = np.vstack([sl.trials[align_event] + time_window[0], sl.trials[align_event] + time_window[1]]).T
    if target in ['wheel-speed', 'wheel-velocity']:
        # add behavior signal to df and update trials mask to reflect trials with signal issues
        if binsize is None:
            raise ValueError(f"If target is wheel-speed or wheel-velocity, binsize cannot be None")
        sl.trials, trials_mask = add_target_to_trials(
            session_loader=sl, target=target, intervals=intervals, binsize=binsize,
            interval_len=time_window[1] - time_window[0], mask=trials_mask)

    if sum(trials_mask) <= config['min_trials']:
        raise ValueError(f"Session {session_id} has {sum(trials_mask)} good trials, less than {config['min_trials']}.")

    # Prepare ephys data
    data_epoch, actual_regions, n_units, cluster_ids = prepare_ephys(
        one, session_id, probe_name, config['regions'], intervals,
        binsize=binsize, n_bins_lag=n_bins_lag, n_bins=n_bins,
        qc=config['unit_qc'], min_units=config['min_units'], stage_only=stage_only,
    )
    n_pseudo_sets = 1 if actual_regions is None else len(actual_regions)

    # Compute or load behavior targets
    all_trials, all_targets, all_masks, all_neurometrics = prepare_behavior(
        session_id, subject, sl.trials, trials_mask, pseudo_ids=pseudo_ids, n_pseudo_sets=n_pseudo_sets,
        output_dir=output_dir, model=model, target=target, compute_neurometrics=compute_neurometrics)

    # Remove the motor residuals from the targets if indicated
    if motor_residuals:
        motor_signals = prepare_motor(one, session_id, time_window=time_window)
        all_targets, all_masks = subtract_motor_residuals(motor_signals, all_targets, all_masks)

    # If we are only staging data, we are done here
    if stage_only:
        return

    # Create strings for saving
    pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if len(pseudo_ids) > 1 else str(pseudo_ids[0])
    probe_str = 'merged_probes' if (isinstance(probe_name, list) and len(probe_name) > 1) else probe_name

    # Fit per region
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
        filename = output_dir.joinpath(subject, session_id, f'{region_str}_{probe_str}_pseudo_ids_{pseudo_str}.pkl')
        filename.parent.mkdir(parents=True, exist_ok=True)

        outdict = {
            "fit": fit_results,
            "subject": subject,
            "eid": session_id,
            "probe": probe_str,
            "region": actual_regions[i],
            "N_units": n_units[i],
            "cluster_uuids": cluster_ids[i],
        }
        with open(filename, "wb") as fw:
            pickle.dump(outdict, fw)
        filenames.append(filename)
    return filenames





def fit_session_ephys(
        one, session_id, subject, probe_name, output_dir, pseudo_ids=None,
        target='pLeft', align_event='stimOn_times',
        min_rt=0.08, max_rt=None,
        time_window=(-0.6, -0.1),
        binsize=None, n_bins_lag=None, n_bins=None,
        model='optBay', n_runs=10,
        compute_neurometrics=False, motor_residuals=False,
        stage_only=False,
        # ===================== NEW =====================
        trials_df=None,
        trial_mask=None,
        group_label="session",
        debug=False,
        # ==============================================
):
    """
    Fits a single session for ephys data.

    NEW additions:
    - trials_df: external trials dataframe (built from ALF trials + your wheel-based RT column)
    - trial_mask: boolean mask or int indices selecting subset trials (fast/normal/slow)
    - group_label: appended to filenames so different RT groups don't overwrite
    - debug: print progress and shapes to identify where failures occur

    RT filtering:
    - We disable built-in RT filtering by setting min_rt/max_rt = None.
      (So your fast <0.08 and slow >1.25 trials remain available.)
    """

    # ----------------------------
    # Check some inputs
    # ----------------------------
    pseudo_ids, output_dir = check_inputs(
        model, pseudo_ids, target, output_dir, config, logger, compute_neurometrics, motor_residuals
    )

    # Ensure Path-like output_dir (check_inputs usually returns Path, but keep safe)
    # If output_dir is a str, it should still have joinpath if converted to Path.
    try:
        _ = output_dir.joinpath
    except Exception:
        from pathlib import Path
        output_dir = Path(output_dir)

    # ----------------------------
    # Load trials
    # ----------------------------
    sl = SessionLoader(one=one, eid=session_id)
    sl.load_trials()

    if debug:
        print(f"[DEBUG] {session_id}: loaded sl.trials rows = {len(sl.trials)}")

    # If user provides external trials_df, replace sl.trials
    if trials_df is not None:
        sl.trials = trials_df.copy()
        if debug:
            print(f"[DEBUG] {session_id}: replaced trials with trials_df rows = {len(sl.trials)}")

    # IMPORTANT: disable their RT filter unless caller explicitly wants it
    # You said you do NOT want min/max RT requirement in your case.
    min_rt = None
    max_rt = None

    # ----------------------------
    # Compute base mask (choice/QC etc.)
    # ----------------------------
    _, trials_mask = load_trials_and_mask(
        one=one, eid=session_id, sess_loader=sl, min_rt=min_rt, max_rt=max_rt,
        min_trial_len=None, max_trial_len=None,
        exclude_nochoice=True, exclude_unbiased=False,
    )

    # Apply your group mask (fast/normal/slow) on top of base mask
    if trial_mask is not None:
        m = np.asarray(trial_mask)

        # allow boolean mask or indices
        if m.dtype != bool:
            idx = m.astype(int)
            m2 = np.zeros(len(sl.trials), dtype=bool)
            m2[idx] = True
            m = m2

        if len(m) != len(sl.trials):
            raise ValueError(f"trial_mask length {len(m)} != n_trials {len(sl.trials)}")

        trials_mask = trials_mask & m

    min_trials_group = 10
    n_good = int(np.sum(trials_mask))

    if debug:
        print(f"[DEBUG] {session_id} group={group_label}: n_trials after mask = {n_good}")

    if n_good < min_trials_group:
        if debug:
            print(f"[DEBUG] {session_id} group={group_label}: too few trials (<{min_trials_group}), skipping")
        return []

    if debug:
        print(f"[DEBUG] {session_id}: trials_mask sum after AND = {int(np.sum(trials_mask))}")

    if int(np.sum(trials_mask)) <= config['min_trials']:
        raise ValueError(
            f"Session {session_id} has {int(np.sum(trials_mask))} good trials after masking, "
            f"less than {config['min_trials']}."
        )

    # ----------------------------
    # Build intervals for *masked* trials only
    # ----------------------------
    # align_times = sl.trials.loc[trials_mask, align_event].to_numpy()
    # intervals = np.vstack([align_times + time_window[0], align_times + time_window[1]]).T

    # align_times_all = sl.trials[align_event].to_numpy()
    # intervals = np.vstack([align_times_all + time_window[0], align_times_all + time_window[1]]).T
    #
    # if debug:
    #     print(f"[DEBUG] {session_id}: intervals shape (ALL trials) = {intervals.shape}")

    # ----------------------------
    # SUBSET TO GROUP TRIALS (CRITICAL FIX)
    # ----------------------------
    # After this point, everything (ephys, behavior, targets) runs on the subgroup only.
    sl.trials = sl.trials.loc[trials_mask].reset_index(drop=True)
    trials_mask = np.ones(len(sl.trials), dtype=bool)

    align_times = sl.trials[align_event].to_numpy()
    intervals = np.vstack([
        align_times + time_window[0],
        align_times + time_window[1]
    ]).T

    if debug:
        print(f"[DEBUG] {session_id}: subgroup trials = {len(sl.trials)}")
        print(f"[DEBUG] {session_id}: intervals shape (SUBGROUP) = {intervals.shape}")

    # safety
    if not np.all(np.isfinite(intervals)):
        bad = np.where(~np.isfinite(intervals).all(axis=1))[0][:10]
        raise ValueError(f"Non-finite intervals in subgroup (showing up to 10 rows): {bad}")

    # if debug:
    #     print(f"[DEBUG] {session_id}: intervals shape = {intervals.shape}")

    # If decoding wheel-based target, add behavior signal and update mask
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

        # Rebuild intervals again after trials_mask changed
        align_times = sl.trials.loc[trials_mask, align_event].to_numpy()
        intervals = np.vstack([align_times + time_window[0], align_times + time_window[1]]).T

        if debug:
            print(f"[DEBUG] {session_id}: intervals shape after wheel target masking = {intervals.shape}")

    # ----------------------------
    # Prepare ephys data
    # ----------------------------
    data_epoch, actual_regions, n_units, cluster_ids = prepare_ephys(
        one, session_id, probe_name, config['regions'], intervals,
        binsize=binsize, n_bins_lag=n_bins_lag, n_bins=n_bins,
        qc=config['unit_qc'], min_units=config['min_units'], stage_only=stage_only,
    )

    if debug:
        print(f"[DEBUG] {session_id}: actual_regions = {actual_regions}")
        print(f"[DEBUG] {session_id}: n_units = {n_units}")

    n_pseudo_sets = 1 if actual_regions is None else len(actual_regions)

    if debug:
        try:
            print(f"[DEBUG] {session_id}: n_regions={len(data_epoch)} n_pseudo_sets={n_pseudo_sets}")
            print(f"[DEBUG] {session_id}: actual_regions is None? {actual_regions is None}")
            if isinstance(data_epoch, list) and len(data_epoch) > 0:
                print(f"[DEBUG] {session_id}: data_epoch[0] type={type(data_epoch[0])}")
        except Exception as _e:
            print("[DEBUG] could not summarize data_epoch:", _e)

    # ----------------------------
    # Compute or load behavior targets
    # ----------------------------
    all_trials, all_targets, all_masks, all_neurometrics = prepare_behavior(
        session_id, subject, sl.trials, trials_mask,
        pseudo_ids=pseudo_ids, n_pseudo_sets=n_pseudo_sets,
        output_dir=output_dir, model=model, target=target,
        compute_neurometrics=compute_neurometrics
    )

    if debug:
        try:
            print(
                f"[DEBUG] {session_id}: prepare_behavior returned lens:",
                len(all_trials), len(all_targets), len(all_masks), len(all_neurometrics)
            )
        except Exception as _e:
            print("[DEBUG] could not summarize behavior outputs:", _e)

    # Remove motor residuals if indicated
    if motor_residuals:
        motor_signals = prepare_motor(one, session_id, time_window=time_window)
        all_targets, all_masks = subtract_motor_residuals(motor_signals, all_targets, all_masks)

    # If staging only, stop here
    if stage_only:
        return

    # ----------------------------
    # Create strings for saving
    # ----------------------------
    pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if len(pseudo_ids) > 1 else str(pseudo_ids[0])
    probe_str = 'merged_probes' if (isinstance(probe_name, list) and len(probe_name) > 1) else probe_name

    # ----------------------------
    # Fit per region
    # ----------------------------
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

        # Add the mask to fit results (if enabled in config)
        for fit_result in fit_results:
            fit_result['mask'] = all_masks[i] if config['save_predictions'] else None

        # Save
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
            "cluster_uuids": cluster_ids[i],
            "group_label": group_label,
        }

        with open(filename, "wb") as fw:
            pickle.dump(outdict, fw)

        filenames.append(filename)

        if debug:
            print(f"[DEBUG] {session_id}: saved {filename}")

    return filenames


def fit_session_ephys(
        one, session_id, subject, probe_name, output_dir, pseudo_ids=None,
        target='pLeft', align_event='stimOn_times',
        min_rt=0.08, max_rt=None,
        time_window=(-0.6, -0.1),
        binsize=None, n_bins_lag=None, n_bins=None,
        model='optBay', n_runs=10,
        compute_neurometrics=False, motor_residuals=False,
        stage_only=False,
        # ===================== NEW =====================
        trials_df=None,
        trial_mask=None,
        group_label="session",
        debug=False,
        # ==============================================
):
    """
    Fits a single session for ephys data, patched to support decoding on trial subgroups
    (e.g., fast/normal/slow within a session) WITHOUT breaking the behavior pipeline.

    Key change vs your current version:
    - Compute behavior targets/masks in FULL trial space (e.g., 425 trials), because
      compute_beh_target/compute_target_mask may expect full session structure.
    - Then slice behavior outputs down to subgroup trials using keep_idx.
    - Ephys is also computed only on subgroup trials (intervals built on subgroup).

    Requires:
    - prepare_behavior patched to accept keep_idx AND to avoid compute_target_mask length mismatch
      (the version we discussed that uses np.isfinite(target) and slices outputs by keep_idx).
    """

    # ----------------------------
    # Check some inputs
    # ----------------------------
    pseudo_ids, output_dir = check_inputs(
        model, pseudo_ids, target, output_dir, config, logger, compute_neurometrics, motor_residuals
    )

    # Ensure Path-like output_dir
    try:
        _ = output_dir.joinpath
    except Exception:
        from pathlib import Path
        output_dir = Path(output_dir)

    # ----------------------------
    # Load trials
    # ----------------------------
    sl = SessionLoader(one=one, eid=session_id)
    sl.load_trials()

    if debug:
        print(f"[DEBUG] {session_id}: loaded sl.trials rows = {len(sl.trials)}")

    # If user provides external trials_df, replace sl.trials
    # if trials_df is not None:
    #     sl.trials = trials_df.copy()
    #     if debug:
    #         print(f"[DEBUG] {session_id}: replaced trials with trials_df rows = {len(sl.trials)}")

    # If user provides external trials_df, DO NOT replace sl.trials (avoid revision mismatch).
    # Only copy over your computed columns.
    if trials_df is not None:
        if len(trials_df) != len(sl.trials):
            raise ValueError(
                f"trials_df length {len(trials_df)} != SessionLoader trials length {len(sl.trials)}. "
                "This likely indicates a revision/collection mismatch."
            )
        sl.trials = sl.trials.copy()
        for col in ["first_movement_onset_times", "reaction_time"]:
            if col in trials_df.columns:
                sl.trials[col] = np.asarray(trials_df[col])

    # Keep a full copy for behavior computations (IMPORTANT)
    trials_full = sl.trials.copy()

    # Disable their RT filter unless caller explicitly wants it
    min_rt = None
    max_rt = None

    # ----------------------------
    # Compute base mask (choice/QC etc.)
    # ----------------------------
    _, base_mask = load_trials_and_mask(
        one=one, eid=session_id, sess_loader=sl, min_rt=min_rt, max_rt=max_rt,
        min_trial_len=None, max_trial_len=None,
        exclude_nochoice=True, exclude_unbiased=False,
    )

    # ----------------------------
    # Apply your group mask (fast/normal/slow) on top of base mask
    # ----------------------------
    trials_mask_full = base_mask.copy()

    if trial_mask is not None:
        m = np.asarray(trial_mask)

        # allow boolean mask or indices
        if m.dtype != bool:
            idx = m.astype(int)
            m2 = np.zeros(len(sl.trials), dtype=bool)
            m2[idx] = True
            m = m2

        if len(m) != len(sl.trials):
            raise ValueError(f"trial_mask length {len(m)} != n_trials {len(sl.trials)}")

        trials_mask_full = trials_mask_full & m

    min_trials_group = 10
    n_good = int(np.sum(trials_mask_full))

    if debug:
        print(f"[DEBUG] {session_id} group={group_label}: n_trials after mask = {n_good}")
        print(f"[DEBUG] {session_id}: trials_mask sum after AND = {int(np.sum(trials_mask_full))}")

    if n_good < min_trials_group:
        if debug:
            print(f"[DEBUG] {session_id} group={group_label}: too few trials (<{min_trials_group}), skipping")
        return []

    if int(np.sum(trials_mask_full)) <= config['min_trials']:
        raise ValueError(
            f"Session {session_id} has {int(np.sum(trials_mask_full))} good trials after masking, "
            f"less than {config['min_trials']}."
        )

    # Indices of subgroup trials in FULL trial space
    keep_idx = np.flatnonzero(trials_mask_full)

    # ----------------------------
    # Compute / load behavior targets in FULL space, then slice to subgroup
    # ----------------------------
    # NOTE: n_pseudo_sets depends on actual_regions, but that's ephys-dependent.
    # In the original code they set n_pseudo_sets = len(actual_regions).
    # We can temporarily set it to 1, and later replicate behavior outputs if needed
    # OR (recommended) compute ephys first to know actual_regions, but ephys needs intervals
    # which needs subgroup. That's OK: n_pseudo_sets is only used to repeat pseudosession sets;
    # behavior generation does not depend on region identity.
    #
    # Practical approach: compute ephys first (subgroup), determine n_pseudo_sets, then
    # call prepare_behavior in full-space with that n_pseudo_sets, slicing to subgroup via keep_idx.
    #
    # So we delay prepare_behavior until after prepare_ephys, but we will still pass trials_full and trials_mask_full.

    # ----------------------------
    # SUBSET TRIALS for ephys & intervals (subgroup only)
    # ----------------------------
    sl.trials = trials_full.loc[trials_mask_full].reset_index(drop=True)
    trials_mask = np.ones(len(sl.trials), dtype=bool)

    align_times = sl.trials[align_event].to_numpy()
    intervals = np.vstack([
        align_times + time_window[0],
        align_times + time_window[1]
    ]).T

    if debug:
        print(f"[DEBUG] {session_id}: subgroup trials = {len(sl.trials)}")
        print(f"[DEBUG] {session_id}: intervals shape (SUBGROUP) = {intervals.shape}")

    if not np.all(np.isfinite(intervals)):
        bad = np.where(~np.isfinite(intervals).all(axis=1))[0][:10]
        raise ValueError(f"Non-finite intervals in subgroup (showing up to 10 rows): {bad}")

    # If decoding wheel-based target, add behavior signal and update mask (subgroup space)
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

        # Rebuild intervals again after trials_mask changed
        align_times = sl.trials.loc[trials_mask, align_event].to_numpy()
        intervals = np.vstack([align_times + time_window[0], align_times + time_window[1]]).T

        if debug:
            print(f"[DEBUG] {session_id}: intervals shape after wheel target masking = {intervals.shape}")

    # ----------------------------
    # Prepare ephys data (subgroup)
    # ----------------------------
    data_epoch, actual_regions, n_units, cluster_ids = prepare_ephys(
        one, session_id, probe_name, config['regions'], intervals,
        binsize=binsize, n_bins_lag=n_bins_lag, n_bins=n_bins,
        qc=config['unit_qc'], min_units=config['min_units'], stage_only=stage_only,
    )

    if debug:
        print(f"[DEBUG] {session_id}: actual_regions = {actual_regions}")
        print(f"[DEBUG] {session_id}: n_units = {n_units}")

    n_pseudo_sets = 1 if actual_regions is None else len(actual_regions)

    if debug:
        try:
            print(f"[DEBUG] {session_id}: n_regions={len(data_epoch)} n_pseudo_sets={n_pseudo_sets}")
            print(f"[DEBUG] {session_id}: actual_regions is None? {actual_regions is None}")
            if isinstance(data_epoch, list) and len(data_epoch) > 0:
                print(f"[DEBUG] {session_id}: data_epoch[0] type={type(data_epoch[0])}")
        except Exception as _e:
            print("[DEBUG] could not summarize data_epoch:", _e)

    # ----------------------------
    # NOW compute behavior targets/masks in FULL space, slice to subgroup via keep_idx
    # ----------------------------
    all_trials, all_targets, all_masks, all_neurometrics = prepare_behavior(
        session_id, subject,
        trials_full,                 # FULL (e.g., 425)
        trials_mask_full,            # FULL mask incl. subgroup (True only on subgroup trials)
        pseudo_ids=pseudo_ids,
        n_pseudo_sets=n_pseudo_sets,
        output_dir=output_dir,
        model=model,
        target=target,
        compute_neurometrics=compute_neurometrics,
        keep_idx=keep_idx,           # <-- NEW, slices to subgroup length (e.g., 47)
    )

    if debug:
        try:
            # show that behavior outputs are subgroup-length
            ex_lens_t = [len(all_targets[0][k]) for k in range(min(3, len(all_targets[0])))]
            ex_lens_m = [len(all_masks[0][k]) for k in range(min(3, len(all_masks[0])))]
            print(f"[DEBUG] {session_id}: behavior lens example targets={ex_lens_t} masks={ex_lens_m}")
            print(
                f"[DEBUG] {session_id}: prepare_behavior returned lens:",
                len(all_trials), len(all_targets), len(all_masks), len(all_neurometrics)
            )
        except Exception as _e:
            print("[DEBUG] could not summarize behavior outputs:", _e)

    # Remove motor residuals if indicated
    if motor_residuals:
        motor_signals = prepare_motor(one, session_id, time_window=time_window)
        all_targets, all_masks = subtract_motor_residuals(motor_signals, all_targets, all_masks)

    # If staging only, stop here
    if stage_only:
        return

    # ----------------------------
    # Create strings for saving
    # ----------------------------
    pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if len(pseudo_ids) > 1 else str(pseudo_ids[0])
    probe_str = 'merged_probes' if (isinstance(probe_name, list) and len(probe_name) > 1) else probe_name

    # ----------------------------
    # Fit per region
    # ----------------------------
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

        # Add the mask to fit results (if enabled in config)
        for fit_result in fit_results:
            fit_result['mask'] = all_masks[i] if config['save_predictions'] else None

        # Save
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
            "cluster_uuids": cluster_ids[i],
            "group_label": group_label,
        }

        with open(filename, "wb") as fw:
            pickle.dump(outdict, fw)

        filenames.append(filename)

        if debug:
            print(f"[DEBUG] {session_id}: saved {filename}")

    return filenames


def fit_session_ephys(
        one, session_id, subject, probe_name, output_dir, pseudo_ids=None,
        target='pLeft', align_event='stimOn_times',
        min_rt=0.08, max_rt=None,
        time_window=(-0.6, -0.1),
        binsize=None, n_bins_lag=None, n_bins=None,
        model='optBay', n_runs=10,
        compute_neurometrics=False, motor_residuals=False,
        stage_only=False,
        # ===================== NEW =====================
        trials_df=None,
        trial_mask=None,
        group_label="session",
        debug=False,
        # ==============================================
):
    """
    Fits a single session for ephys data, patched to support decoding on trial subgroups
    (e.g., fast/normal/slow within a session) WITHOUT breaking the behavior pipeline.

    Key change vs your current version:
    - Compute behavior targets/masks in FULL trial space (e.g., 425 trials), because
      compute_beh_target/compute_target_mask may expect full session structure.
    - Then slice behavior outputs down to subgroup trials using keep_idx.
    - Ephys is also computed only on subgroup trials (intervals built on subgroup).

    Requires:
    - prepare_behavior patched to accept keep_idx AND to avoid compute_target_mask length mismatch
      (the version we discussed that uses np.isfinite(target) and slices outputs by keep_idx).
    """

    # ----------------------------
    # Check some inputs
    # ----------------------------
    pseudo_ids, output_dir = check_inputs(
        model, pseudo_ids, target, output_dir, config, logger, compute_neurometrics, motor_residuals
    )

    # Ensure Path-like output_dir
    try:
        _ = output_dir.joinpath
    except Exception:
        from pathlib import Path
        output_dir = Path(output_dir)

    # ----------------------------
    # Load trials
    # ----------------------------
    sl = SessionLoader(one=one, eid=session_id)
    sl.load_trials()

    if debug:
        print(f"[DEBUG] {session_id}: loaded sl.trials rows = {len(sl.trials)}")

    # If user provides external trials_df, replace sl.trials
    if trials_df is not None:
        sl.trials = trials_df.copy()
        if debug:
            print(f"[DEBUG] {session_id}: replaced trials with trials_df rows = {len(sl.trials)}")

    # Keep a full copy for behavior computations (IMPORTANT)
    trials_full = sl.trials.copy()

    # Disable their RT filter unless caller explicitly wants it
    min_rt = None
    max_rt = None

    # ----------------------------
    # Compute base mask (choice/QC etc.)
    # ----------------------------
    _, base_mask = load_trials_and_mask(
        one=one, eid=session_id, sess_loader=sl, min_rt=min_rt, max_rt=max_rt,
        min_trial_len=None, max_trial_len=None,
        exclude_nochoice=True, exclude_unbiased=False,
    )

    # ----------------------------
    # Apply your group mask (fast/normal/slow) on top of base mask
    # ----------------------------
    trials_mask_full = base_mask.copy()

    if trial_mask is not None:
        m = np.asarray(trial_mask)

        # allow boolean mask or indices
        if m.dtype != bool:
            idx = m.astype(int)
            m2 = np.zeros(len(sl.trials), dtype=bool)
            m2[idx] = True
            m = m2

        if len(m) != len(sl.trials):
            raise ValueError(f"trial_mask length {len(m)} != n_trials {len(sl.trials)}")

        trials_mask_full = trials_mask_full & m

    min_trials_group = 10
    n_good = int(np.sum(trials_mask_full))

    if debug:
        print(f"[DEBUG] {session_id} group={group_label}: n_trials after mask = {n_good}")
        print(f"[DEBUG] {session_id}: trials_mask sum after AND = {int(np.sum(trials_mask_full))}")

    if n_good < min_trials_group:
        if debug:
            print(f"[DEBUG] {session_id} group={group_label}: too few trials (<{min_trials_group}), skipping")
        return []

    if int(np.sum(trials_mask_full)) <= config['min_trials']:
        raise ValueError(
            f"Session {session_id} has {int(np.sum(trials_mask_full))} good trials after masking, "
            f"less than {config['min_trials']}."
        )

    # Indices of subgroup trials in FULL trial space
    keep_idx = np.flatnonzero(trials_mask_full)

    # ----------------------------
    # Compute / load behavior targets in FULL space, then slice to subgroup
    # ----------------------------
    # NOTE: n_pseudo_sets depends on actual_regions, but that's ephys-dependent.
    # In the original code they set n_pseudo_sets = len(actual_regions).
    # We can temporarily set it to 1, and later replicate behavior outputs if needed
    # OR (recommended) compute ephys first to know actual_regions, but ephys needs intervals
    # which needs subgroup. That's OK: n_pseudo_sets is only used to repeat pseudosession sets;
    # behavior generation does not depend on region identity.
    #
    # Practical approach: compute ephys first (subgroup), determine n_pseudo_sets, then
    # call prepare_behavior in full-space with that n_pseudo_sets, slicing to subgroup via keep_idx.
    #
    # So we delay prepare_behavior until after prepare_ephys, but we will still pass trials_full and trials_mask_full.

    # ----------------------------
    # SUBSET TRIALS for ephys & intervals (subgroup only)
    # ----------------------------
    sl.trials = trials_full.loc[trials_mask_full].reset_index(drop=True)
    trials_mask = np.ones(len(sl.trials), dtype=bool)

    align_times = sl.trials[align_event].to_numpy()
    intervals = np.vstack([
        align_times + time_window[0],
        align_times + time_window[1]
    ]).T

    if debug:
        print(f"[DEBUG] {session_id}: subgroup trials = {len(sl.trials)}")
        print(f"[DEBUG] {session_id}: intervals shape (SUBGROUP) = {intervals.shape}")

    if not np.all(np.isfinite(intervals)):
        bad = np.where(~np.isfinite(intervals).all(axis=1))[0][:10]
        raise ValueError(f"Non-finite intervals in subgroup (showing up to 10 rows): {bad}")

    # If decoding wheel-based target, add behavior signal and update mask (subgroup space)
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

        # Rebuild intervals again after trials_mask changed
        align_times = sl.trials.loc[trials_mask, align_event].to_numpy()
        intervals = np.vstack([align_times + time_window[0], align_times + time_window[1]]).T

        if debug:
            print(f"[DEBUG] {session_id}: intervals shape after wheel target masking = {intervals.shape}")

    # ----------------------------
    # Prepare ephys data (subgroup)
    # ----------------------------
    data_epoch, actual_regions, n_units, cluster_ids = prepare_ephys(
        one, session_id, probe_name, config['regions'], intervals,
        binsize=binsize, n_bins_lag=n_bins_lag, n_bins=n_bins,
        qc=config['unit_qc'], min_units=config['min_units'], stage_only=stage_only,
    )

    if debug:
        print(f"[DEBUG] {session_id}: actual_regions = {actual_regions}")
        print(f"[DEBUG] {session_id}: n_units = {n_units}")

    n_pseudo_sets = 1 if actual_regions is None else len(actual_regions)

    if debug:
        try:
            print(f"[DEBUG] {session_id}: n_regions={len(data_epoch)} n_pseudo_sets={n_pseudo_sets}")
            print(f"[DEBUG] {session_id}: actual_regions is None? {actual_regions is None}")
            if isinstance(data_epoch, list) and len(data_epoch) > 0:
                print(f"[DEBUG] {session_id}: data_epoch[0] type={type(data_epoch[0])}")
        except Exception as _e:
            print("[DEBUG] could not summarize data_epoch:", _e)

    # ----------------------------
    # NOW compute behavior targets/masks in FULL space, slice to subgroup via keep_idx
    # ----------------------------
    all_trials, all_targets, all_masks, all_neurometrics = prepare_behavior(
        session_id, subject,
        trials_full,                 # FULL (e.g., 425)
        trials_mask_full,            # FULL mask incl. subgroup (True only on subgroup trials)
        pseudo_ids=pseudo_ids,
        n_pseudo_sets=n_pseudo_sets,
        output_dir=output_dir,
        model=model,
        target=target,
        compute_neurometrics=compute_neurometrics,
        keep_idx=keep_idx,           # <-- NEW, slices to subgroup length (e.g., 47)
    )

    if debug:
        try:
            # show that behavior outputs are subgroup-length
            ex_lens_t = [len(all_targets[0][k]) for k in range(min(3, len(all_targets[0])))]
            ex_lens_m = [len(all_masks[0][k]) for k in range(min(3, len(all_masks[0])))]
            print(f"[DEBUG] {session_id}: behavior lens example targets={ex_lens_t} masks={ex_lens_m}")
            print(
                f"[DEBUG] {session_id}: prepare_behavior returned lens:",
                len(all_trials), len(all_targets), len(all_masks), len(all_neurometrics)
            )
        except Exception as _e:
            print("[DEBUG] could not summarize behavior outputs:", _e)

    # Remove motor residuals if indicated
    if motor_residuals:
        motor_signals = prepare_motor(one, session_id, time_window=time_window)
        all_targets, all_masks = subtract_motor_residuals(motor_signals, all_targets, all_masks)

    # If staging only, stop here
    if stage_only:
        return

    # ----------------------------
    # Create strings for saving
    # ----------------------------
    pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if len(pseudo_ids) > 1 else str(pseudo_ids[0])
    probe_str = 'merged_probes' if (isinstance(probe_name, list) and len(probe_name) > 1) else probe_name

    # ----------------------------
    # Fit per region
    # ----------------------------
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

        # Add the mask to fit results (if enabled in config)
        for fit_result in fit_results:
            fit_result['mask'] = all_masks[i] if config['save_predictions'] else None

        # Save
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
            "cluster_uuids": cluster_ids[i],
            "group_label": group_label,
        }

        with open(filename, "wb") as fw:
            pickle.dump(outdict, fw)

        filenames.append(filename)

        if debug:
            print(f"[DEBUG] {session_id}: saved {filename}")

    return filenames





def fit_session_ephys(
        one, session_id, subject, probe_name, output_dir, pseudo_ids=None,
        target='pLeft', align_event='stimOn_times',
        min_rt=0.08, max_rt=None,
        time_window=(-0.6, -0.1),
        binsize=None, n_bins_lag=None, n_bins=None,
        model='optBay', n_runs=10,
        compute_neurometrics=False, motor_residuals=False,
        stage_only=False,
        # ===================== NEW =====================
        trials_df=None,
        trial_mask=None,
        group_label="session",
        debug=False,
        roi_set=None,   # <-- NEW: set like {"MOp","MOs","ACAd","ORBvl"} or None
        # ==============================================
):
    """
    Fits a single session for ephys data, patched to support decoding on trial subgroups
    (e.g., fast/normal/slow within a session) WITHOUT breaking the behavior pipeline.

    Additional patch:
    - If roi_set is provided:
        * Early skip sessions that have NO ROI regions
        * Only decode regions that are in roi_set
    """

    # ----------------------------
    # Check some inputs
    # ----------------------------
    pseudo_ids, output_dir = check_inputs(
        model, pseudo_ids, target, output_dir, config, logger, compute_neurometrics, motor_residuals
    )

    # Ensure Path-like output_dir
    try:
        _ = output_dir.joinpath
    except Exception:
        from pathlib import Path
        output_dir = Path(output_dir)

    # ----------------------------
    # Load trials
    # ----------------------------
    sl = SessionLoader(one=one, eid=session_id)
    sl.load_trials()

    if debug:
        print(f"[DEBUG] {session_id}: loaded sl.trials rows = {len(sl.trials)}")

    # If user provides external trials_df, replace sl.trials
    if trials_df is not None:
        sl.trials = trials_df.copy()
        if debug:
            print(f"[DEBUG] {session_id}: replaced trials with trials_df rows = {len(sl.trials)}")

    # Keep a full copy for behavior computations (IMPORTANT)
    trials_full = sl.trials.copy()

    # Disable their RT filter unless caller explicitly wants it
    min_rt = None
    max_rt = None

    # ----------------------------
    # Compute base mask (choice/QC etc.)
    # ----------------------------
    _, base_mask = load_trials_and_mask(
        one=one, eid=session_id, sess_loader=sl, min_rt=min_rt, max_rt=max_rt,
        min_trial_len=None, max_trial_len=None,
        exclude_nochoice=True, exclude_unbiased=False,
    )

    # ----------------------------
    # Apply your group mask (fast/normal/slow) on top of base mask
    # ----------------------------
    trials_mask_full = base_mask.copy()

    if trial_mask is not None:
        m = np.asarray(trial_mask)

        # allow boolean mask or indices
        if m.dtype != bool:
            idx = m.astype(int)
            m2 = np.zeros(len(sl.trials), dtype=bool)
            m2[idx] = True
            m = m2

        if len(m) != len(sl.trials):
            raise ValueError(f"trial_mask length {len(m)} != n_trials {len(sl.trials)}")

        trials_mask_full = trials_mask_full & m

    min_trials_group = 10
    n_good = int(np.sum(trials_mask_full))

    if debug:
        print(f"[DEBUG] {session_id} group={group_label}: n_trials after mask = {n_good}")
        print(f"[DEBUG] {session_id}: trials_mask sum after AND = {int(np.sum(trials_mask_full))}")

    if n_good < min_trials_group:
        if debug:
            print(f"[DEBUG] {session_id} group={group_label}: too few trials (<{min_trials_group}), skipping")
        return []

    if int(np.sum(trials_mask_full)) <= config['min_trials']:
        raise ValueError(
            f"Session {session_id} has {int(np.sum(trials_mask_full))} good trials after masking, "
            f"less than {config['min_trials']}."
        )

    # Indices of subgroup trials in FULL trial space
    keep_idx = np.flatnonzero(trials_mask_full)

    # ----------------------------
    # SUBSET TRIALS for ephys & intervals (subgroup only)
    # ----------------------------
    sl.trials = trials_full.loc[trials_mask_full].reset_index(drop=True)
    trials_mask = np.ones(len(sl.trials), dtype=bool)

    align_times = sl.trials[align_event].to_numpy()
    intervals = np.vstack([
        align_times + time_window[0],
        align_times + time_window[1]
    ]).T

    if debug:
        print(f"[DEBUG] {session_id}: subgroup trials = {len(sl.trials)}")
        print(f"[DEBUG] {session_id}: intervals shape (SUBGROUP) = {intervals.shape}")

    if not np.all(np.isfinite(intervals)):
        bad = np.where(~np.isfinite(intervals).all(axis=1))[0][:10]
        raise ValueError(f"Non-finite intervals in subgroup (showing up to 10 rows): {bad}")

    # If decoding wheel-based target, add behavior signal and update mask (subgroup space)
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

        # Rebuild intervals again after trials_mask changed
        align_times = sl.trials.loc[trials_mask, align_event].to_numpy()
        intervals = np.vstack([align_times + time_window[0], align_times + time_window[1]]).T

        if debug:
            print(f"[DEBUG] {session_id}: intervals shape after wheel target masking = {intervals.shape}")

    # ----------------------------
    # Prepare ephys data (subgroup)
    # ----------------------------
    data_epoch, actual_regions, n_units, cluster_ids = prepare_ephys(
        one, session_id, probe_name, config['regions'], intervals,
        binsize=binsize, n_bins_lag=n_bins_lag, n_bins=n_bins,
        qc=config['unit_qc'], min_units=config['min_units'], stage_only=stage_only,
    )

    if debug:
        print(f"[DEBUG] {session_id}: actual_regions = {actual_regions}")
        print(f"[DEBUG] {session_id}: n_units = {n_units}")

    # ----------------------------
    # NEW: ROI early skip BEFORE expensive prepare_behavior
    # ----------------------------
    if roi_set is not None and actual_regions is not None:
        present = {r[0] if isinstance(r, (list, tuple)) else str(r) for r in actual_regions}
        if len(present.intersection(set(roi_set))) == 0:
            if debug:
                print(f"[DEBUG] {session_id} group={group_label}: no ROI regions present -> SKIP prepare_behavior")
            return []

    n_pseudo_sets = 1 if actual_regions is None else len(actual_regions)

    if debug:
        try:
            print(f"[DEBUG] {session_id}: n_regions={len(data_epoch)} n_pseudo_sets={n_pseudo_sets}")
            print(f"[DEBUG] {session_id}: actual_regions is None? {actual_regions is None}")
            if isinstance(data_epoch, list) and len(data_epoch) > 0:
                print(f"[DEBUG] {session_id}: data_epoch[0] type={type(data_epoch[0])}")
        except Exception as _e:
            print("[DEBUG] could not summarize data_epoch:", _e)

    # ----------------------------
    # NOW compute behavior targets/masks in FULL space, slice to subgroup via keep_idx
    # ----------------------------
    all_trials, all_targets, all_masks, all_neurometrics = prepare_behavior(
        session_id, subject,
        trials_full,                 # FULL
        trials_mask_full,            # FULL mask incl subgroup
        pseudo_ids=pseudo_ids,
        n_pseudo_sets=n_pseudo_sets,
        output_dir=output_dir,
        model=model,
        target=target,
        compute_neurometrics=compute_neurometrics,
        keep_idx=keep_idx,           # slices to subgroup length
    )

    if debug:
        try:
            ex_lens_t = [len(all_targets[0][k]) for k in range(min(3, len(all_targets[0])))]
            ex_lens_m = [len(all_masks[0][k]) for k in range(min(3, len(all_masks[0])))]
            print(f"[DEBUG] {session_id}: behavior lens example targets={ex_lens_t} masks={ex_lens_m}")
            print(
                f"[DEBUG] {session_id}: prepare_behavior returned lens:",
                len(all_trials), len(all_targets), len(all_masks), len(all_neurometrics)
            )
        except Exception as _e:
            print("[DEBUG] could not summarize behavior outputs:", _e)

    # Remove motor residuals if indicated
    if motor_residuals:
        motor_signals = prepare_motor(one, session_id, time_window=time_window)
        all_targets, all_masks = subtract_motor_residuals(motor_signals, all_targets, all_masks)

    # If staging only, stop here
    if stage_only:
        return

    # ----------------------------
    # Create strings for saving
    # ----------------------------
    pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if len(pseudo_ids) > 1 else str(pseudo_ids[0])
    probe_str = 'merged_probes' if (isinstance(probe_name, list) and len(probe_name) > 1) else probe_name

    # ----------------------------
    # Fit per region
    # ----------------------------
    filenames = []
    for i in range(len(data_epoch)):

        # NEW: skip non-ROI regions to save time
        if roi_set is not None and actual_regions is not None:
            region_name = actual_regions[i][0] if isinstance(actual_regions[i], (list, tuple)) else str(actual_regions[i])
            if region_name not in roi_set:
                continue

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

        # Add the mask to fit results (if enabled in config)
        for fit_result in fit_results:
            fit_result['mask'] = all_masks[i] if config['save_predictions'] else None

        # Save
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
            "cluster_uuids": cluster_ids[i],
            "group_label": group_label,
        }

        with open(filename, "wb") as fw:
            pickle.dump(outdict, fw)

        filenames.append(filename)

        if debug:
            print(f"[DEBUG] {session_id}: saved {filename}")

    return filenames

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


def fit_target(
        all_data, all_targets, all_trials, n_runs, all_neurometrics=None, pseudo_ids=None,
        base_rng_seed=0
):
    """
    Fits data (neural, motor, etc) to behavior targets.

    Parameters
    ----------
    all_data : list of np.ndarray
        List of neural or other data, each element is a (n_trials, n_units) array with the averaged neural activity
    all_targets : list of np.ndarray
        List of behavior targets, each element is a (n_trials,) array with the behavior targets for one (pseudo)session
    all_trials : list of pd.DataFrames
        List of trial information, each element is a pd.DataFrame with the trial information for one (pseudo)session
    n_runs: int
        Number of times to repeat full nested cross validation with different folds
    all_neurometrics : list of pd.DataFrames or None
        List of neurometrics, each element is a pd.DataFrame with the neurometrics for one (pseudo)session.
        If None, don't compute neurometrics. Default is None
    pseudo_ids : list of int or None
        List of pseudo session ids, -1 indicates the actual session. If None, run only on actual session.
        Default is None.
    base_rng_seed : int
        seed that will be added to run- and pseudo_id-specific seeds
    """

    # Loop over (pseudo) sessions and then over runs
    if pseudo_ids is None:
        pseudo_ids = [-1]
    if not all_neurometrics:
        all_neurometrics = [None] * len(all_targets)
    fit_results = []
    for targets, data, trials, neurometrics, pseudo_id in zip(
            all_targets, all_data, all_trials, all_neurometrics, pseudo_ids):
        # run decoders
        for i_run in range(n_runs):
            # set seed for reproducibility
            if pseudo_id == -1:
                rng_seed = base_rng_seed + i_run
            else:
                rng_seed = base_rng_seed + pseudo_id * n_runs + i_run
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
                use_cv_sklearn_method=config['use_native_sklearn_for_hyperparam_estimation'],


            )

            fit_result["trials_df"] = trials
            fit_result["pseudo_id"] = pseudo_id
            fit_result["run_id"] = i_run

            if neurometrics:
                fit_result["full_neurometric"], fit_result["fold_neurometric"] = get_neurometric_parameters(
                    fit_result, trialsdf=neurometrics, compute_on_each_fold=config['compute_neuro_on_each_fold']
                )
            else:
                fit_result["full_neurometric"] = None
                fit_result["fold_neurometric"] = None

            fit_results.append(fit_result)

    return fit_results

def fit_target(
        all_data, all_targets, all_trials, n_runs, all_neurometrics=None, pseudo_ids=None,
        base_rng_seed=0
):
    """
    Fits data (neural, motor, etc) to behavior targets.
    Patched to:
      - pass debug_tag into decode_cv for tracing RuntimeWarnings (overflow/div0/invalid)
      - enforce min_trials=10 for subgroup decoding
    """

    # Loop over (pseudo) sessions and then over runs
    if pseudo_ids is None:
        pseudo_ids = [-1]

    if not all_neurometrics:
        all_neurometrics = [None] * len(all_targets)

    fit_results = []

    for targets, data, trials, neurometrics, pseudo_id in zip(
            all_targets, all_data, all_trials, all_neurometrics, pseudo_ids
    ):

        # Try to infer group label for better debug prints
        group_label = None
        try:
            # if trials is a DataFrame and you added "group_label" column
            if hasattr(trials, "columns") and ("group_label" in trials.columns):
                # take first non-null, else fallback
                vals = trials["group_label"].dropna().astype(str).unique()
                group_label = vals[0] if len(vals) else None
        except Exception:
            group_label = None

        # As a fallback, if caller attached attribute (rare)
        if group_label is None:
            group_label = getattr(trials, "group_label", None)

        if group_label is None:
            group_label = "unknown_group"

        # run decoders
        for i_run in range(n_runs):

            # set seed for reproducibility
            if pseudo_id == -1:
                rng_seed = base_rng_seed + i_run
            else:
                rng_seed = base_rng_seed + pseudo_id * n_runs + i_run

            # Helpful tag that will be printed by decode_cv when warnings happen
            # (region is not known here; region is already baked into base_rng_seed upstream)
            debug_tag = f"group={group_label} pseudo={pseudo_id} run={i_run} seed={rng_seed}"

            fit_result = decode_cv(
                ys=targets,
                Xs=data,
                estimator=config['estimator'],
                estimator_kwargs=config['estimator_kwargs'],
                hyperparam_grid=config.get('hparam_grid', None),
                save_binned=False,
                save_predictions=config['save_predictions'],
                shuffle=config['shuffle'],
                balanced_weight=config['balanced_weighting'],
                rng_seed=rng_seed,
                use_cv_sklearn_method=config['use_native_sklearn_for_hyperparam_estimation'],

                # -------- NEW args supported by your patched decode_cv --------
                min_trials=10,
                debug_tag=debug_tag,
                debug_warnings=True,
                sanitize_X=True,
                standardize_X=True,
            )

            # keep original metadata additions
            fit_result["trials_df"] = trials
            fit_result["pseudo_id"] = pseudo_id
            fit_result["run_id"] = i_run
            fit_result["debug_tag"] = debug_tag  # optional, but useful to store

            if neurometrics:
                fit_result["full_neurometric"], fit_result["fold_neurometric"] = get_neurometric_parameters(
                    fit_result, trialsdf=neurometrics, compute_on_each_fold=config['compute_neuro_on_each_fold']
                )
            else:
                fit_result["full_neurometric"] = None
                fit_result["fold_neurometric"] = None

            fit_results.append(fit_result)

    return fit_results

def decode_cv(ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
              n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
              rng_seed=None, use_cv_sklearn_method=False):
    """
    Regresses binned neural activity against a target, using a provided sklearn estimator.

    Parameters
    ----------
    ys : list of arrays or np.ndarray or pandas.Series
        targets; if list, each entry is an array of targets for one trial. if 1D numpy array, each
        entry is treated as a single scalar for one trial. if pd.Series, trial number is the index
        and teh value is the target.
    Xs : list of arrays or np.ndarray
        predictors; if list, each entry is an array of neural activity for one trial. if 2D numpy
        array, each row is treated as a single vector of activity for one trial, i.e. the array is
        of shape (n_trials, n_neurons)
    estimator : sklearn.linear_model object
        estimator from sklearn which provides .fit, .score, and .predict methods. CV estimators
        are NOT SUPPORTED. Must be a normal estimator, which is internally wrapped with
        GridSearchCV
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

    # transform target data into standard format: list of np.ndarrays
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
            isysat = lambda ys: logisticreg_criteria(ys, min_unique_counts=2)
        else:
            isysat = lambda ys: True
        sample_count, _, outer_kfold_iter = sample_folds(ys, get_kfold, isysat)
        if sample_count > 1:
            print(f'sampled outer folds {sample_count} times to ensure enough targets')
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
            # X_train = np.vstack([Xs[i] for i in train_idxs])
            # y_train = np.concatenate([ys[i] for i in train_idxs], axis=0)
            # X_test = np.vstack([Xs[i] for i in test_idxs])
            # y_test = np.concatenate([ys[i] for i in test_idxs], axis=0)
            X_train = [Xs[i] for i in train_idxs_outer]
            y_train = [ys[i] for i in train_idxs_outer]
            X_test = [Xs[i] for i in test_idxs_outer]
            y_test = [ys[i] for i in test_idxs_outer]

            key = list(hyperparam_grid.keys())[0]  # TODO: make this more robust

            if not use_cv_sklearn_method:

                """NOTE
                This section of the code implements a modified nested-cross validation procedure. When decoding signals
                with multiple samples per trial -- such as the wheel -- we need to create folds that do not put
                samples from the same trial into different folds.
                """

                # now loop over inner folds
                idx_inner = np.arange(len(X_train))

                get_kfold_inner = lambda: KFold(n_splits=n_folds, shuffle=shuffle).split(idx_inner)

                # produce inner_fold_iter
                if estimator == sklm.LogisticRegression:
                    # make sure data has at least 2 examples per class
                    assert logisticreg_criteria(y_train, min_unique_counts=2)
                    # folds must be chosen such that 2 classes are present in each fold, with min 1 example per class
                    isysat_inner = lambda ys: logisticreg_criteria(ys, min_unique_counts=1)
                else:
                    isysat_inner = lambda ys: True
                sample_count, _, inner_kfold_iter = sample_folds(y_train, get_kfold_inner, isysat_inner)
                if sample_count > 1:
                    print(f'sampled inner folds {sample_count} times to ensure enough targets')

                r2s = np.zeros([n_folds, len(hyperparam_grid[key])])
                inner_predictions = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
                inner_targets = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
                for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold_iter):

                    # inner fold data split
                    X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
                    y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
                    X_test_inner = np.vstack([X_train[i] for i in test_idxs_inner])
                    y_test_inner = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)

                    for i_alpha, alpha in enumerate(hyperparam_grid[key]):

                        # compute weight for each training sample if requested
                        sample_weight = compute_sample_weight("balanced", y=y_train_inner) if balanced_weight else None

                        # initialize model
                        model_inner = estimator(**{**estimator_kwargs, key: alpha})
                        # fit model
                        model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)
                        # evaluate model
                        pred_test_inner = model_inner.predict(X_test_inner)
                        # record predictions and targets to check for nans later
                        inner_predictions[test_idxs_inner, i_alpha] = np.mean(pred_test_inner)
                        inner_targets[test_idxs_inner, i_alpha] = np.mean(y_test_inner)
                        # record score
                        r2s[ifold, i_alpha] = scoring_f(y_test_inner, pred_test_inner)

                assert np.all(~np.isnan(inner_predictions))
                assert np.all(~np.isnan(inner_targets))

                # select model with best hyperparameter value evaluated on inner-fold test data;
                # refit/evaluate on all inner-fold data
                r2s_avg = r2s.mean(axis=0)

                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)

                # compute weight for each training sample if requested
                sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None

                # initialize model
                best_alpha = hyperparam_grid[key][np.argmax(r2s_avg)]
                model = estimator(**{**estimator_kwargs, key: best_alpha})
                # fit model
                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
            else:
                if estimator not in [Ridge, Lasso]:
                    raise NotImplementedError("This case is not implemented")
                model = (
                    RidgeCV(alphas=hyperparam_grid[key])
                    if estimator == Ridge
                    else LassoCV(alphas=hyperparam_grid[key])
                )
                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)
                sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None

                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
                best_alpha = model.alpha_

            # evalute model on train data
            y_pred_train = model.predict(X_train_array)
            scores_train.append(scoring_f(y_train_array, y_pred_train))

            # evaluate model on test data
            y_true = np.concatenate(y_test, axis=0)
            y_pred = model.predict(np.vstack(X_test))
            if isinstance(model, sklm.LogisticRegression):
                y_pred_probs = model.predict_proba(np.vstack(X_test))[:, 1]  # probability of class 1
            else:
                y_pred_probs = None
            scores_test.append(scoring_f(y_true, y_pred))

            # save the raw prediction in the case of linear and the predicted probabilities when
            # working with logitistic regression
            for i_fold, i_global in enumerate(test_idxs_outer):
                if bins_per_trial == 1:
                    # we already computed these estimates, take from above
                    predictions[i_global] = np.array([y_pred[i_fold]])
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])
                    else:
                        predictions_to_save[i_global] = np.array([y_pred[i_fold]])
                else:
                    # we already computed these above, but after all trials were stacked; recompute per-trial
                    predictions[i_global] = model.predict(X_test[i_fold])
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = model.predict_proba(X_test[i_fold])[:, 1]
                    else:
                        predictions_to_save[i_global] = predictions[i_global]

            # save out other data of interest
            idxes_test.append(test_idxs_outer)
            idxes_train.append(train_idxs_outer)
            weights.append(model.coef_)
            if model.fit_intercept:
                intercepts.append(model.intercept_)
            else:
                intercepts.append(None)
            best_params.append({key: best_alpha})

    ys_true_full = np.concatenate(ys, axis=0)
    ys_pred_full = np.concatenate(predictions, axis=0)
    outdict = dict()
    outdict["scores_test_full"] = scoring_f(ys_true_full, ys_pred_full)
    outdict["scores_train"] = scores_train
    outdict["scores_test"] = scores_test
    outdict["Rsquared_test_full"] = r2_score(ys_true_full, ys_pred_full)
    if estimator == sklm.LogisticRegression:
        outdict["acc_test_full"] = accuracy_score(ys_true_full, ys_pred_full)
        outdict["balanced_acc_test_full"] = balanced_accuracy_score(
            ys_true_full, ys_pred_full
        )
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

    # logging
    if verbose:
        # verbose output
        if outer_cv:
            print("Performance is only described for last outer fold \n")
        print(
            "Possible regularization parameters over {} validation sets:".format(
                n_folds
            )
        )
        print("{}: {}".format(list(hyperparam_grid.keys())[0], hyperparam_grid))
        print("\nBest parameters found over {} validation sets:".format(n_folds))
        print(model.best_params_)
        print("\nAverage scores over {} validation sets:".format(n_folds))
        means = model.cv_results_["mean_test_score"]
        stds = model.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, model.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print("\n", "Detailed scores on {} validation sets:".format(n_folds))
        for i_fold in range(n_folds):
            tscore_fold = list(
                np.round(model.cv_results_["split{}_test_score".format(int(i_fold))], 3)
            )
            print("perf on fold {}: {}".format(int(i_fold), tscore_fold))

        print("\n", "Detailed classification report:", "\n")
        print("The model is trained on the full (train + validation) set.")

    return outdict


def decode_cv(ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
              n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
              rng_seed=None, use_cv_sklearn_method=False):
    """
    Regresses binned neural activity against a target, using a provided sklearn estimator.
    (Patched: adaptive outer CV folds + safe inner CV folds.)
    """

    # transform target data into standard format: list of np.ndarrays
    ys, Xs = format_data_for_decoding(ys, Xs)
    finite_trials = []
    for i in range(len(Xs)):
        Xi = np.asarray(Xs[i])
        yi = np.asarray(ys[i])
        finite_trials.append(np.all(np.isfinite(Xi)) and np.all(np.isfinite(yi)))

    finite_trials = np.array(finite_trials, dtype=bool)

    if not np.all(finite_trials):
        Xs = [x for x, ok in zip(Xs, finite_trials) if ok]
        ys = [y for y, ok in zip(ys, finite_trials) if ok]

    # enforce at least 10 trials AFTER cleaning
    if len(Xs) < 10:
        raise ValueError(f"[decode_cv] Too few finite trials after cleaning: n={len(Xs)}")

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
        # -------------------------------
        # PATCH 1: adaptive OUTER CV folds
        # -------------------------------
        def _choose_outer_splits(n: int) -> int:
            # Your rule:
            # < 50 trials  -> 5-fold
            # 50-100       -> 10-fold
            # > 100        -> 50-fold
            if n < 50:
                desired = 5
            elif n <= 100:
                desired = 10
            else:
                desired = 50
            # KFold requires 2 <= n_splits <= n_trials
            return max(2, min(desired, n))

        outer_splits = _choose_outer_splits(n_trials)
        if verbose:
            print(f"[decode_cv] n_trials={n_trials} -> outer_splits={outer_splits} (use_cv_sklearn_method={use_cv_sklearn_method})")

        # create kfold function to loop over
        get_kfold = lambda: KFold(n_splits=outer_splits, shuffle=shuffle).split(indices)

        # define function to evaluate whether folds are satisfactory
        if estimator == sklm.LogisticRegression:
            # folds must be chosen such that 2 classes are present in each fold, with minimum 2 examples per class
            assert logisticreg_criteria(ys)
            isysat = lambda ys: logisticreg_criteria(ys, min_unique_counts=2)
        else:
            isysat = lambda ys: True

        sample_count, _, outer_kfold_iter = sample_folds(ys, get_kfold, isysat)
        if sample_count > 1:
            print(f'sampled outer folds {sample_count} times to ensure enough targets')
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
            X_test = [Xs[i] for i in test_idxs_outer]
            y_test = [ys[i] for i in test_idxs_outer]

            key = list(hyperparam_grid.keys())[0]  # TODO: make this more robust

            if not use_cv_sklearn_method:

                """NOTE
                This section of the code implements a modified nested-cross validation procedure. When decoding signals
                with multiple samples per trial -- such as the wheel -- we need to create folds that do not put
                samples from the same trial into different folds.
                """

                # now loop over inner folds
                idx_inner = np.arange(len(X_train))

                # -------------------------------
                # PATCH 2: safe INNER CV folds
                # -------------------------------
                inner_splits = max(2, min(n_folds, len(idx_inner)))
                get_kfold_inner = lambda: KFold(n_splits=inner_splits, shuffle=shuffle).split(idx_inner)

                if verbose:
                    print(f"[decode_cv] inner_splits={inner_splits} (n_train_trials={len(idx_inner)})")

                # produce inner_fold_iter
                if estimator == sklm.LogisticRegression:
                    # make sure data has at least 2 examples per class
                    assert logisticreg_criteria(y_train, min_unique_counts=2)
                    # folds must be chosen such that 2 classes are present in each fold, with min 1 example per class
                    isysat_inner = lambda ys: logisticreg_criteria(ys, min_unique_counts=1)
                else:
                    isysat_inner = lambda ys: True

                sample_count, _, inner_kfold_iter = sample_folds(y_train, get_kfold_inner, isysat_inner)
                if sample_count > 1:
                    print(f'sampled inner folds {sample_count} times to ensure enough targets')

                # NOTE: r2s first dimension must match the actual number of inner folds
                r2s = np.zeros([inner_splits, len(hyperparam_grid[key])])
                inner_predictions = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
                inner_targets = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan

                for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold_iter):

                    # inner fold data split
                    X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
                    y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
                    X_test_inner = np.vstack([X_train[i] for i in test_idxs_inner])
                    y_test_inner = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)

                    for i_alpha, alpha in enumerate(hyperparam_grid[key]):

                        # compute weight for each training sample if requested
                        sample_weight = compute_sample_weight("balanced", y=y_train_inner) if balanced_weight else None

                        # initialize model
                        model_inner = estimator(**{**estimator_kwargs, key: alpha})
                        # fit model
                        model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)
                        # evaluate model
                        pred_test_inner = model_inner.predict(X_test_inner)
                        # record predictions and targets to check for nans later
                        inner_predictions[test_idxs_inner, i_alpha] = np.mean(pred_test_inner)
                        inner_targets[test_idxs_inner, i_alpha] = np.mean(y_test_inner)
                        # record score
                        r2s[ifold, i_alpha] = scoring_f(y_test_inner, pred_test_inner)

                assert np.all(~np.isnan(inner_predictions))
                assert np.all(~np.isnan(inner_targets))

                # select model with best hyperparameter value evaluated on inner-fold test data;
                # refit/evaluate on all inner-fold data
                r2s_avg = r2s.mean(axis=0)

                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)
                # ------------------------------

                # compute weight for each training sample if requested
                sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None

                # initialize model
                best_alpha = hyperparam_grid[key][np.argmax(r2s_avg)]
                model = estimator(**{**estimator_kwargs, key: best_alpha})
                # fit model
                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)

            else:
                if estimator not in [Ridge, Lasso]:
                    raise NotImplementedError("This case is not implemented")
                model = (
                    RidgeCV(alphas=hyperparam_grid[key])
                    if estimator == Ridge
                    else LassoCV(alphas=hyperparam_grid[key])
                )
                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)
                sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None

                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
                best_alpha = model.alpha_

            # evalute model on train data
            y_pred_train = model.predict(X_train_array)
            scores_train.append(scoring_f(y_train_array, y_pred_train))

            # evaluate model on test data
            y_true = np.concatenate(y_test, axis=0)
            y_pred = model.predict(np.vstack(X_test))
            if isinstance(model, sklm.LogisticRegression):
                y_pred_probs = model.predict_proba(np.vstack(X_test))[:, 1]  # probability of class 1
            else:
                y_pred_probs = None
            scores_test.append(scoring_f(y_true, y_pred))

            # save the raw prediction in the case of linear and the predicted probabilities when
            # working with logitistic regression
            for i_fold, i_global in enumerate(test_idxs_outer):
                if bins_per_trial == 1:
                    # we already computed these estimates, take from above
                    predictions[i_global] = np.array([y_pred[i_fold]])
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])
                    else:
                        predictions_to_save[i_global] = np.array([y_pred[i_fold]])
                else:
                    # we already computed these above, but after all trials were stacked; recompute per-trial
                    predictions[i_global] = model.predict(X_test[i_fold])
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = model.predict_proba(X_test[i_fold])[:, 1]
                    else:
                        predictions_to_save[i_global] = predictions[i_global]

            # save out other data of interest
            idxes_test.append(test_idxs_outer)
            idxes_train.append(train_idxs_outer)
            weights.append(model.coef_)
            if model.fit_intercept:
                intercepts.append(model.intercept_)
            else:
                intercepts.append(None)
            best_params.append({key: best_alpha})

    ys_true_full = np.concatenate(ys, axis=0)
    ys_pred_full = np.concatenate(predictions, axis=0)
    outdict = dict()
    outdict["scores_test_full"] = scoring_f(ys_true_full, ys_pred_full)
    outdict["scores_train"] = scores_train
    outdict["scores_test"] = scores_test
    outdict["Rsquared_test_full"] = r2_score(ys_true_full, ys_pred_full)
    if estimator == sklm.LogisticRegression:
        outdict["acc_test_full"] = accuracy_score(ys_true_full, ys_pred_full)
        outdict["balanced_acc_test_full"] = balanced_accuracy_score(
            ys_true_full, ys_pred_full
        )
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

    # logging
    if verbose:
        # verbose output
        if outer_cv:
            print("Performance is only described for last outer fold \n")
        print(
            "Possible regularization parameters over {} validation sets:".format(
                n_folds
            )
        )
        print("{}: {}".format(list(hyperparam_grid.keys())[0], hyperparam_grid))
        print("\nBest parameters found over {} validation sets:".format(n_folds))
        print(model.best_params_)
        print("\nAverage scores over {} validation sets:".format(n_folds))
        means = model.cv_results_["mean_test_score"]
        stds = model.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, model.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print("\n", "Detailed scores on {} validation sets:".format(n_folds))
        for i_fold in range(n_folds):
            tscore_fold = list(
                np.round(model.cv_results_["split{}_test_score".format(int(i_fold))], 3)
            )
            print("perf on fold {}: {}".format(int(i_fold), tscore_fold))

        print("\n", "Detailed classification report:", "\n")
        print("The model is trained on the full (train + validation) set.")

    return outdict




# def decode_cv(
#     ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
#     n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
#     rng_seed=None, use_cv_sklearn_method=False
# ):
#     """
#     Regresses binned neural activity against a target, using a provided sklearn estimator.
#
#     PATCHED:
#     - Robustly sanitizes regressor matrices to prevent NaN/Inf and zero-variance columns from causing
#       sklearn matmul overflow/divide-by-zero/invalid warnings.
#     """
#
#     # transform target data into standard format: list of np.ndarrays
#     ys, Xs = format_data_for_decoding(ys, Xs)
#
#     # -------------------------------
#     # NEW: helper to sanitize X
#     # -------------------------------
#     def _sanitize_X(X, eps=1e-12):
#         """
#         Ensure X is finite and well-conditioned for linear models:
#         - cast to float64
#         - replace NaN/Inf with 0
#         - drop near-zero-variance columns (var <= eps)
#         Returns (X_sanitized, keep_cols_mask)
#         """
#         X = np.asarray(X, dtype=np.float64)
#         X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
#
#         var = X.var(axis=0)
#         keep = var > eps
#         keep &= np.all(np.isfinite(X), axis=0)
#
#         if keep.sum() == 0:
#             raise ValueError(
#                 "[decode_cv] All features removed after sanitization "
#                 "(matrix is all invalid or zero-variance)."
#             )
#         return X[:, keep], keep
#
#     def _apply_keep(X, keep_cols):
#         """Apply keep_cols to X with safe casting and nan_to_num."""
#         X = np.asarray(X, dtype=np.float64)
#         X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
#         return X[:, keep_cols]
#
#     # filter out trials with non-finite X or y at the TRIAL level (keeps existing behavior)
#     finite_trials = []
#     for i in range(len(Xs)):
#         Xi = np.asarray(Xs[i])
#         yi = np.asarray(ys[i])
#         finite_trials.append(np.all(np.isfinite(Xi)) and np.all(np.isfinite(yi)))
#     finite_trials = np.array(finite_trials, dtype=bool)
#
#     if not np.all(finite_trials):
#         Xs = [x for x, ok in zip(Xs, finite_trials) if ok]
#         ys = [y for y, ok in zip(ys, finite_trials) if ok]
#
#     # enforce at least 10 trials AFTER cleaning
#     if len(Xs) < 10:
#         raise ValueError(f"[decode_cv] Too few finite trials after cleaning: n={len(Xs)}")
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
#     if rng_seed is not None:
#         np.random.seed(rng_seed)
#     indices = np.arange(n_trials)
#
#     if outer_cv:
#         # -------------------------------
#         # adaptive OUTER CV folds
#         # -------------------------------
#         def _choose_outer_splits(n: int) -> int:
#             if n < 50:
#                 desired = 5
#             elif n <= 100:
#                 desired = 10
#             else:
#                 desired = 50
#             return max(2, min(desired, n))
#
#         outer_splits = _choose_outer_splits(n_trials)
#         if verbose:
#             print(f"[decode_cv] n_trials={n_trials} -> outer_splits={outer_splits} (use_cv_sklearn_method={use_cv_sklearn_method})")
#
#         get_kfold = lambda: KFold(n_splits=outer_splits, shuffle=shuffle).split(indices)
#
#         if estimator == sklm.LogisticRegression:
#             assert logisticreg_criteria(ys)
#             isysat = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=2)
#         else:
#             isysat = lambda ys_: True
#
#         sample_count, _, outer_kfold_iter = sample_folds(ys, get_kfold, isysat)
#         if sample_count > 1:
#             print(f"sampled outer folds {sample_count} times to ensure enough targets")
#     else:
#         outer_kfold = iter([train_test_split(indices, test_size=test_prop, shuffle=shuffle)])
#         outer_kfold_iter = [(train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(outer_kfold)]
#
#     scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score
#
#     if estimator in [sklm.RidgeCV, sklm.LassoCV, sklm.LogisticRegressionCV]:
#         raise NotImplementedError("the code does not support a CV-type estimator.")
#     else:
#         for train_idxs_outer, test_idxs_outer in outer_kfold_iter:
#
#             X_train = [Xs[i] for i in train_idxs_outer]
#             y_train = [ys[i] for i in train_idxs_outer]
#             X_test = [Xs[i] for i in test_idxs_outer]
#             y_test = [ys[i] for i in test_idxs_outer]
#
#             key = list(hyperparam_grid.keys())[0]
#
#             if not use_cv_sklearn_method:
#                 # inner folds
#                 idx_inner = np.arange(len(X_train))
#                 inner_splits = max(2, min(n_folds, len(idx_inner)))
#                 get_kfold_inner = lambda: KFold(n_splits=inner_splits, shuffle=shuffle).split(idx_inner)
#
#                 if verbose:
#                     print(f"[decode_cv] inner_splits={inner_splits} (n_train_trials={len(idx_inner)})")
#
#                 if estimator == sklm.LogisticRegression:
#                     assert logisticreg_criteria(y_train, min_unique_counts=2)
#                     isysat_inner = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=1)
#                 else:
#                     isysat_inner = lambda ys_: True
#
#                 sample_count, _, inner_kfold_iter = sample_folds(y_train, get_kfold_inner, isysat_inner)
#                 if sample_count > 1:
#                     print(f"sampled inner folds {sample_count} times to ensure enough targets")
#
#                 r2s = np.zeros([inner_splits, len(hyperparam_grid[key])])
#                 inner_predictions = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
#                 inner_targets = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
#
#                 for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold_iter):
#
#                     X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
#                     y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
#                     X_test_inner = np.vstack([X_train[i] for i in test_idxs_inner])
#                     y_test_inner = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)
#
#                     # -------- NEW: sanitize inner fold matrices --------
#                     X_train_inner, keep_cols_inner = _sanitize_X(X_train_inner)
#                     X_test_inner = _apply_keep(X_test_inner, keep_cols_inner)
#
#                     for i_alpha, alpha in enumerate(hyperparam_grid[key]):
#
#                         sample_weight = compute_sample_weight("balanced", y=y_train_inner) if balanced_weight else None
#
#                         model_inner = estimator(**{**estimator_kwargs, key: alpha})
#                         model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)
#
#                         pred_test_inner = model_inner.predict(X_test_inner)
#
#                         inner_predictions[test_idxs_inner, i_alpha] = np.mean(pred_test_inner)
#                         inner_targets[test_idxs_inner, i_alpha] = np.mean(y_test_inner)
#                         r2s[ifold, i_alpha] = scoring_f(y_test_inner, pred_test_inner)
#
#                 assert np.all(~np.isnan(inner_predictions))
#                 assert np.all(~np.isnan(inner_targets))
#
#                 r2s_avg = r2s.mean(axis=0)
#
#                 X_train_array = np.vstack(X_train)
#                 y_train_array = np.concatenate(y_train, axis=0)
#
#                 # -------- NEW: sanitize outer refit matrix --------
#                 X_train_array, keep_cols = _sanitize_X(X_train_array)
#
#                 sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None
#
#                 best_alpha = hyperparam_grid[key][np.argmax(r2s_avg)]
#                 model = estimator(**{**estimator_kwargs, key: best_alpha})
#                 model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
#
#             else:
#                 if estimator not in [Ridge, Lasso]:
#                     raise NotImplementedError("This case is not implemented")
#
#                 model = (
#                     RidgeCV(alphas=hyperparam_grid[key])
#                     if estimator == Ridge
#                     else LassoCV(alphas=hyperparam_grid[key])
#                 )
#
#                 X_train_array = np.vstack(X_train)
#                 y_train_array = np.concatenate(y_train, axis=0)
#
#                 # -------- NEW: sanitize outer refit matrix --------
#                 X_train_array, keep_cols = _sanitize_X(X_train_array)
#
#                 sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None
#                 model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
#                 best_alpha = model.alpha_
#
#             # evaluate model on train data
#             y_pred_train = model.predict(X_train_array)
#             scores_train.append(scoring_f(y_train_array, y_pred_train))
#
#             # evaluate model on test data
#             y_true = np.concatenate(y_test, axis=0)
#
#             X_test_array = np.vstack(X_test)
#             X_test_array = _apply_keep(X_test_array, keep_cols)
#
#             y_pred = model.predict(X_test_array)
#
#             if isinstance(model, sklm.LogisticRegression):
#                 y_pred_probs = model.predict_proba(X_test_array)[:, 1]
#             else:
#                 y_pred_probs = None
#
#             scores_test.append(scoring_f(y_true, y_pred))
#
#             # save per-trial predictions
#             for i_fold, i_global in enumerate(test_idxs_outer):
#                 if bins_per_trial == 1:
#                     predictions[i_global] = np.array([y_pred[i_fold]])
#                     if isinstance(model, sklm.LogisticRegression):
#                         predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])
#                     else:
#                         predictions_to_save[i_global] = np.array([y_pred[i_fold]])
#                 else:
#                     Xt = np.asarray(X_test[i_fold], dtype=np.float64)
#                     Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)[:, keep_cols]
#                     predictions[i_global] = model.predict(Xt)
#                     if isinstance(model, sklm.LogisticRegression):
#                         predictions_to_save[i_global] = model.predict_proba(Xt)[:, 1]
#                     else:
#                         predictions_to_save[i_global] = predictions[i_global]
#
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
#
#     outdict = dict()
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
#     if hasattr(model, "classes_"):
#         outdict["classes_"] = model.classes_
#
#     if verbose:
#         if outer_cv:
#             print("Performance is only described for last outer fold \n")
#
#     return outdict


# def decode_cv(
#     ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
#     n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
#     rng_seed=None, use_cv_sklearn_method=False
# ):
#     """
#     Regresses binned neural activity against a target, using a provided sklearn estimator.
#
#     PATCHED:
#     - Robustly sanitizes regressor matrices to prevent NaN/Inf and zero-variance columns.
#     - Fold-wise standardization (z-score) + clipping to prevent extreme magnitudes causing overflow in matmul.
#     - Adaptive outer CV folds + safe inner CV folds.
#     """
#
#     # transform target data into standard format: list of np.ndarrays
#     ys, Xs = format_data_for_decoding(ys, Xs)
#
#     # -------------------------------
#     # Helpers: sanitize + standardize
#     # -------------------------------
#     def _sanitize_X(X, eps=1e-12):
#         """
#         Ensure X is finite and remove near-zero-variance columns.
#         Returns (X_sanitized, keep_cols_mask).
#         """
#         X = np.asarray(X, dtype=np.float64)
#         X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
#
#         # Drop constant/near-constant features
#         var = X.var(axis=0)
#         keep = var > eps
#
#         # Safety: keep only finite columns
#         keep &= np.all(np.isfinite(X), axis=0)
#
#         if keep.sum() == 0:
#             raise ValueError(
#                 "[decode_cv] All features removed after sanitization "
#                 "(matrix invalid or zero-variance)."
#             )
#
#         return X[:, keep], keep
#
#     def _apply_keep(X, keep_cols):
#         X = np.asarray(X, dtype=np.float64)
#         X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
#         return X[:, keep_cols]
#
#     def _standardize_fit(X, eps=1e-12):
#         mu = X.mean(axis=0)
#         sd = X.std(axis=0)
#         sd = np.maximum(sd, eps)
#         return mu, sd
#
#     def _standardize_apply(X, mu, sd, clip=20.0):
#         Z = (X - mu) / sd
#         return np.clip(Z, -clip, clip)
#
#     # ------------------------------------
#     # Filter out trials with non-finite X/y
#     # ------------------------------------
#     finite_trials = []
#     for i in range(len(Xs)):
#         Xi = np.asarray(Xs[i])
#         yi = np.asarray(ys[i])
#         finite_trials.append(np.all(np.isfinite(Xi)) and np.all(np.isfinite(yi)))
#     finite_trials = np.array(finite_trials, dtype=bool)
#
#     if not np.all(finite_trials):
#         Xs = [x for x, ok in zip(Xs, finite_trials) if ok]
#         ys = [y for y, ok in zip(ys, finite_trials) if ok]
#
#     if len(Xs) < 10:
#         raise ValueError(f"[decode_cv] Too few finite trials after cleaning: n={len(Xs)}")
#
#     # initialize containers to save outputs
#     n_trials = len(Xs)
#     bins_per_trial = len(Xs[0])
#     scores_test, scores_train = [], []
#     idxes_test, idxes_train = [], []
#     weights, intercepts, best_params = [], [], []
#     predictions = [None for _ in range(n_trials)]
#     predictions_to_save = [None for _ in range(n_trials)]
#
#     # split the dataset in two parts, train and test
#     if rng_seed is not None:
#         np.random.seed(rng_seed)
#     indices = np.arange(n_trials)
#
#     # ------------------------------------
#     # Outer CV folds
#     # ------------------------------------
#     if outer_cv:
#         def _choose_outer_splits(n: int) -> int:
#             if n < 50:
#                 desired = 5
#             elif n <= 100:
#                 desired = 10
#             else:
#                 desired = 50
#             return max(2, min(desired, n))
#
#         outer_splits = _choose_outer_splits(n_trials)
#         if verbose:
#             print(f"[decode_cv] n_trials={n_trials} -> outer_splits={outer_splits}")
#
#         get_kfold = lambda: KFold(n_splits=outer_splits, shuffle=shuffle).split(indices)
#
#         if estimator == sklm.LogisticRegression:
#             assert logisticreg_criteria(ys)
#             isysat = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=2)
#         else:
#             isysat = lambda ys_: True
#
#         sample_count, _, outer_kfold_iter = sample_folds(ys, get_kfold, isysat)
#         if sample_count > 1:
#             print(f"sampled outer folds {sample_count} times to ensure enough targets")
#     else:
#         outer_kfold = iter([train_test_split(indices, test_size=test_prop, shuffle=shuffle)])
#         outer_kfold_iter = [(train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(outer_kfold)]
#
#     scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score
#
#     if estimator in [sklm.RidgeCV, sklm.LassoCV, sklm.LogisticRegressionCV]:
#         raise NotImplementedError("decode_cv does not support CV-type estimators.")
#     else:
#         for train_idxs_outer, test_idxs_outer in outer_kfold_iter:
#
#             X_train = [Xs[i] for i in train_idxs_outer]
#             y_train = [ys[i] for i in train_idxs_outer]
#             X_test = [Xs[i] for i in test_idxs_outer]
#             y_test = [ys[i] for i in test_idxs_outer]
#
#             if hyperparam_grid is None or len(hyperparam_grid) == 0:
#                 raise ValueError(
#                     "[decode_cv] hyperparam_grid is empty. For Ridge you should pass e.g. "
#                     "{'alpha':[0.1,1,10,100,1000,10000]} (also set in config.yaml)."
#                 )
#
#             key = list(hyperparam_grid.keys())[0]
#
#             if not use_cv_sklearn_method:
#                 # -------------------------------
#                 # Inner CV folds (safe)
#                 # -------------------------------
#                 idx_inner = np.arange(len(X_train))
#                 inner_splits = max(2, min(n_folds, len(idx_inner)))
#                 get_kfold_inner = lambda: KFold(n_splits=inner_splits, shuffle=shuffle).split(idx_inner)
#
#                 if verbose:
#                     print(f"[decode_cv] inner_splits={inner_splits} (n_train_trials={len(idx_inner)})")
#
#                 if estimator == sklm.LogisticRegression:
#                     assert logisticreg_criteria(y_train, min_unique_counts=2)
#                     isysat_inner = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=1)
#                 else:
#                     isysat_inner = lambda ys_: True
#
#                 sample_count, _, inner_kfold_iter = sample_folds(y_train, get_kfold_inner, isysat_inner)
#                 if sample_count > 1:
#                     print(f"sampled inner folds {sample_count} times to ensure enough targets")
#
#                 r2s = np.zeros([inner_splits, len(hyperparam_grid[key])])
#                 inner_predictions = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
#                 inner_targets = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
#
#                 for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold_iter):
#
#                     X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
#                     y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
#                     X_test_inner = np.vstack([X_train[i] for i in test_idxs_inner])
#                     y_test_inner = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)
#
#                     # sanitize + standardize (inner)
#                     X_train_inner, keep_cols_inner = _sanitize_X(X_train_inner)
#                     X_test_inner = _apply_keep(X_test_inner, keep_cols_inner)
#
#                     mu_i, sd_i = _standardize_fit(X_train_inner)
#                     X_train_inner = _standardize_apply(X_train_inner, mu_i, sd_i)
#                     X_test_inner = _standardize_apply(X_test_inner, mu_i, sd_i)
#
#                     for i_alpha, alpha in enumerate(hyperparam_grid[key]):
#
#                         sample_weight = compute_sample_weight("balanced", y=y_train_inner) if balanced_weight else None
#                         model_inner = estimator(**{**estimator_kwargs, key: alpha})
#                         model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)
#
#                         pred_test_inner = model_inner.predict(X_test_inner)
#
#                         inner_predictions[test_idxs_inner, i_alpha] = np.mean(pred_test_inner)
#                         inner_targets[test_idxs_inner, i_alpha] = np.mean(y_test_inner)
#                         r2s[ifold, i_alpha] = scoring_f(y_test_inner, pred_test_inner)
#
#                 assert np.all(~np.isnan(inner_predictions))
#                 assert np.all(~np.isnan(inner_targets))
#
#                 r2s_avg = r2s.mean(axis=0)
#
#                 # -------------------------------
#                 # Refit on full outer-train set with best alpha
#                 # -------------------------------
#                 X_train_array = np.vstack(X_train)
#                 y_train_array = np.concatenate(y_train, axis=0)
#
#                 X_train_array, keep_cols = _sanitize_X(X_train_array)
#                 mu, sd = _standardize_fit(X_train_array)
#                 X_train_array = _standardize_apply(X_train_array, mu, sd)
#
#                 sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None
#                 best_alpha = hyperparam_grid[key][np.argmax(r2s_avg)]
#
#                 model = estimator(**{**estimator_kwargs, key: best_alpha})
#                 model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
#
#             else:
#                 # sklearn CV-type path not supported in this project’s design
#                 raise NotImplementedError("use_cv_sklearn_method=True path not implemented in this patched version.")
#
#             # eval on train
#             y_pred_train = model.predict(X_train_array)
#             scores_train.append(scoring_f(y_train_array, y_pred_train))
#
#             # eval on test
#             y_true = np.concatenate(y_test, axis=0)
#             X_test_array = np.vstack(X_test)
#             X_test_array = _apply_keep(X_test_array, keep_cols)
#             X_test_array = _standardize_apply(X_test_array, mu, sd)
#             y_pred = model.predict(X_test_array)
#
#             if isinstance(model, sklm.LogisticRegression):
#                 y_pred_probs = model.predict_proba(X_test_array)[:, 1]
#             else:
#                 y_pred_probs = None
#
#             scores_test.append(scoring_f(y_true, y_pred))
#
#             # save per-trial predictions
#             for i_fold, i_global in enumerate(test_idxs_outer):
#                 if bins_per_trial == 1:
#                     predictions[i_global] = np.array([y_pred[i_fold]])
#                     if isinstance(model, sklm.LogisticRegression):
#                         predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])
#                     else:
#                         predictions_to_save[i_global] = np.array([y_pred[i_fold]])
#                 else:
#                     Xt = np.asarray(X_test[i_fold], dtype=np.float64)
#                     Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)[:, keep_cols]
#                     Xt = _standardize_apply(Xt, mu, sd)
#                     predictions[i_global] = model.predict(Xt)
#                     if isinstance(model, sklm.LogisticRegression):
#                         predictions_to_save[i_global] = model.predict_proba(Xt)[:, 1]
#                     else:
#                         predictions_to_save[i_global] = predictions[i_global]
#
#             idxes_test.append(test_idxs_outer)
#             idxes_train.append(train_idxs_outer)
#             weights.append(model.coef_)
#             intercepts.append(model.intercept_ if model.fit_intercept else None)
#             best_params.append({key: best_alpha})
#
#     ys_true_full = np.concatenate(ys, axis=0)
#     ys_pred_full = np.concatenate(predictions, axis=0)
#
#     outdict = dict()
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
#     if hasattr(model, "classes_"):
#         outdict["classes_"] = model.classes_
#
#     return outdict
#



def decode_cv(
    ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
    n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
    rng_seed=None, use_cv_sklearn_method=False
):
    """
    Regresses binned neural activity against a target, using a provided sklearn estimator.

    PATCHED:
    - Robustly sanitizes regressor matrices to prevent NaN/Inf and zero-variance columns from causing
      sklearn matmul overflow/divide-by-zero/invalid warnings.
    """

    # transform target data into standard format: list of np.ndarrays
    ys, Xs = format_data_for_decoding(ys, Xs)

    # -------------------------------
    # NEW: helper to sanitize X
    # -------------------------------
    def _sanitize_X(X, eps=1e-12):
        """
        Ensure X is finite and well-conditioned for linear models:
        - cast to float64
        - replace NaN/Inf with 0
        - drop near-zero-variance columns (var <= eps)
        Returns (X_sanitized, keep_cols_mask)
        """
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        var = X.var(axis=0)
        keep = var > eps
        keep &= np.all(np.isfinite(X), axis=0)

        if keep.sum() == 0:
            raise ValueError(
                "[decode_cv] All features removed after sanitization "
                "(matrix is all invalid or zero-variance)."
            )
        return X[:, keep], keep

    def _apply_keep(X, keep_cols):
        """Apply keep_cols to X with safe casting and nan_to_num."""
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X[:, keep_cols]

    # filter out trials with non-finite X or y at the TRIAL level (keeps existing behavior)
    finite_trials = []
    for i in range(len(Xs)):
        Xi = np.asarray(Xs[i])
        yi = np.asarray(ys[i])
        finite_trials.append(np.all(np.isfinite(Xi)) and np.all(np.isfinite(yi)))
    finite_trials = np.array(finite_trials, dtype=bool)

    if not np.all(finite_trials):
        Xs = [x for x, ok in zip(Xs, finite_trials) if ok]
        ys = [y for y, ok in zip(ys, finite_trials) if ok]

    # enforce at least 10 trials AFTER cleaning
    if len(Xs) < 10:
        raise ValueError(f"[decode_cv] Too few finite trials after cleaning: n={len(Xs)}")

    # initialize containers to save outputs
    n_trials = len(Xs)
    bins_per_trial = len(Xs[0])
    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []
    predictions = [None for _ in range(n_trials)]
    predictions_to_save = [None for _ in range(n_trials)]  # different for logistic regression

    # split the dataset in two parts, train and test
    if rng_seed is not None:
        np.random.seed(rng_seed)
    indices = np.arange(n_trials)

    if outer_cv:
        # -------------------------------
        # adaptive OUTER CV folds
        # -------------------------------
        def _choose_outer_splits(n: int) -> int:
            if n < 50:
                desired = 5
            elif n <= 100:
                desired = 10
            else:
                desired = 50
            return max(2, min(desired, n))

        outer_splits = _choose_outer_splits(n_trials)
        if verbose:
            print(f"[decode_cv] n_trials={n_trials} -> outer_splits={outer_splits} (use_cv_sklearn_method={use_cv_sklearn_method})")

        get_kfold = lambda: KFold(n_splits=outer_splits, shuffle=shuffle).split(indices)

        if estimator == sklm.LogisticRegression:
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

    scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score

    if estimator in [sklm.RidgeCV, sklm.LassoCV, sklm.LogisticRegressionCV]:
        raise NotImplementedError("the code does not support a CV-type estimator.")
    else:
        for train_idxs_outer, test_idxs_outer in outer_kfold_iter:

            X_train = [Xs[i] for i in train_idxs_outer]
            y_train = [ys[i] for i in train_idxs_outer]
            X_test = [Xs[i] for i in test_idxs_outer]
            y_test = [ys[i] for i in test_idxs_outer]

            key = list(hyperparam_grid.keys())[0]

            if not use_cv_sklearn_method:
                # inner folds
                idx_inner = np.arange(len(X_train))
                inner_splits = max(2, min(n_folds, len(idx_inner)))
                get_kfold_inner = lambda: KFold(n_splits=inner_splits, shuffle=shuffle).split(idx_inner)

                if verbose:
                    print(f"[decode_cv] inner_splits={inner_splits} (n_train_trials={len(idx_inner)})")

                if estimator == sklm.LogisticRegression:
                    assert logisticreg_criteria(y_train, min_unique_counts=2)
                    isysat_inner = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=1)
                else:
                    isysat_inner = lambda ys_: True

                sample_count, _, inner_kfold_iter = sample_folds(y_train, get_kfold_inner, isysat_inner)
                if sample_count > 1:
                    print(f"sampled inner folds {sample_count} times to ensure enough targets")

                r2s = np.zeros([inner_splits, len(hyperparam_grid[key])])
                inner_predictions = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
                inner_targets = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan

                for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold_iter):

                    X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
                    y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
                    X_test_inner = np.vstack([X_train[i] for i in test_idxs_inner])
                    y_test_inner = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)

                    # -------- NEW: sanitize inner fold matrices --------
                    X_train_inner, keep_cols_inner = _sanitize_X(X_train_inner)
                    X_test_inner = _apply_keep(X_test_inner, keep_cols_inner)

                    for i_alpha, alpha in enumerate(hyperparam_grid[key]):

                        sample_weight = compute_sample_weight("balanced", y=y_train_inner) if balanced_weight else None

                        model_inner = estimator(**{**estimator_kwargs, key: alpha})
                        model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)

                        pred_test_inner = model_inner.predict(X_test_inner)

                        inner_predictions[test_idxs_inner, i_alpha] = np.mean(pred_test_inner)
                        inner_targets[test_idxs_inner, i_alpha] = np.mean(y_test_inner)
                        r2s[ifold, i_alpha] = scoring_f(y_test_inner, pred_test_inner)

                assert np.all(~np.isnan(inner_predictions))
                assert np.all(~np.isnan(inner_targets))

                r2s_avg = r2s.mean(axis=0)

                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)

                # -------- NEW: sanitize outer refit matrix --------
                X_train_array, keep_cols = _sanitize_X(X_train_array)

                sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None

                best_alpha = hyperparam_grid[key][np.argmax(r2s_avg)]
                model = estimator(**{**estimator_kwargs, key: best_alpha})
                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)

            else:
                if estimator not in [Ridge, Lasso]:
                    raise NotImplementedError("This case is not implemented")

                model = (
                    RidgeCV(alphas=hyperparam_grid[key])
                    if estimator == Ridge
                    else LassoCV(alphas=hyperparam_grid[key])
                )

                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)

                # -------- NEW: sanitize outer refit matrix --------
                X_train_array, keep_cols = _sanitize_X(X_train_array)

                sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None
                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
                best_alpha = model.alpha_

            # evaluate model on train data
            y_pred_train = model.predict(X_train_array)
            scores_train.append(scoring_f(y_train_array, y_pred_train))

            # evaluate model on test data
            y_true = np.concatenate(y_test, axis=0)

            X_test_array = np.vstack(X_test)
            X_test_array = _apply_keep(X_test_array, keep_cols)

            y_pred = model.predict(X_test_array)

            if isinstance(model, sklm.LogisticRegression):
                y_pred_probs = model.predict_proba(X_test_array)[:, 1]
            else:
                y_pred_probs = None

            scores_test.append(scoring_f(y_true, y_pred))

            # save per-trial predictions
            for i_fold, i_global in enumerate(test_idxs_outer):
                if bins_per_trial == 1:
                    predictions[i_global] = np.array([y_pred[i_fold]])
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])
                    else:
                        predictions_to_save[i_global] = np.array([y_pred[i_fold]])
                else:
                    Xt = np.asarray(X_test[i_fold], dtype=np.float64)
                    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)[:, keep_cols]
                    predictions[i_global] = model.predict(Xt)
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = model.predict_proba(Xt)[:, 1]
                    else:
                        predictions_to_save[i_global] = predictions[i_global]

            idxes_test.append(test_idxs_outer)
            idxes_train.append(train_idxs_outer)
            weights.append(model.coef_)
            if model.fit_intercept:
                intercepts.append(model.intercept_)
            else:
                intercepts.append(None)
            best_params.append({key: best_alpha})

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

    if verbose:
        if outer_cv:
            print("Performance is only described for last outer fold \n")

    return outdict





def decode_cv(
    ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
    n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
    rng_seed=None, use_cv_sklearn_method=False
):
    """
    Regresses binned neural activity against a target, using a provided sklearn estimator.

    Patched to eliminate sklearn RuntimeWarnings:
      - overflow/divide-by-zero/invalid encountered in matmul

    Key fixes:
      1) Only use class-balanced sample weights for *classification* (LogisticRegression).
         Using compute_sample_weight("balanced", y=continuous) can explode weights -> huge coef_ -> overflow.
      2) Sanitize NaN/Inf values in X (replace with 0) and drop near-zero-variance columns per fold.
      3) Standardize X per fold using train statistics (mean/std), improving conditioning.
      4) Use safe numbers of folds when trial counts are small.

    Notes:
      - This function supports both:
          * Xs as list of (n_bins, n_features) arrays per trial
          * Xs as 2D array (n_trials, n_features) which format_data_for_decoding converts to list
      - hyperparam_grid is expected to have ONE key (e.g. {"alpha": [...]}).
    """

    import numpy as np
    from sklearn import linear_model as sklm
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.linear_model import RidgeCV, Ridge, Lasso, LassoCV
    from sklearn.utils.class_weight import compute_sample_weight

    # assumes these exist in your module (as in the original repo)
    # - format_data_for_decoding
    # - logisticreg_criteria
    # - sample_folds

    # -------------------------------
    # Standardize inputs
    # -------------------------------
    ys, Xs = format_data_for_decoding(ys, Xs)

    if hyperparam_grid is None:
        hyperparam_grid = {}

    if len(hyperparam_grid) == 0:
        raise ValueError(
            "decode_cv: hyperparam_grid is empty. For Ridge/Lasso you should provide e.g.\n"
            "  hyperparam_grid={'alpha': [0.1, 1, 10, 100, 1000]}\n"
            "Otherwise alpha selection may be unstable."
        )

    # -------------------------------
    # Helpers: sanitize + standardize
    # -------------------------------
    def _sanitize_X(X, eps=1e-12):
        """
        Ensure X is finite and usable:
          - cast float64
          - replace NaN/Inf with 0
          - drop near-zero-variance columns (var <= eps)
        Returns (X_sanitized, keep_cols_mask)
        """
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # variance per feature
        var = X.var(axis=0)
        keep = (var > eps) & np.all(np.isfinite(X), axis=0)

        if keep.sum() == 0:
            raise ValueError(
                "[decode_cv] All features removed after sanitization "
                "(matrix is all invalid or zero-variance)."
            )
        return X[:, keep], keep

    def _apply_keep(X, keep_cols):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X[:, keep_cols]

    def _standardize_train_test(Xtr, Xte, eps=1e-12):
        """
        Z-score using train statistics only.
        """
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd = np.where(sd < eps, 1.0, sd)
        return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd

    # -------------------------------
    # Remove non-finite trials at trial level
    # -------------------------------
    finite_trials = []
    for i in range(len(Xs)):
        Xi = np.asarray(Xs[i])
        yi = np.asarray(ys[i])
        finite_trials.append(np.all(np.isfinite(Xi)) and np.all(np.isfinite(yi)))
    finite_trials = np.array(finite_trials, dtype=bool)

    if not np.all(finite_trials):
        Xs = [x for x, ok in zip(Xs, finite_trials) if ok]
        ys = [y for y, ok in zip(ys, finite_trials) if ok]

    if len(Xs) < 10:
        raise ValueError(f"[decode_cv] Too few finite trials after cleaning: n={len(Xs)}")

    # -------------------------------
    # Initialize containers
    # -------------------------------
    n_trials = len(Xs)
    bins_per_trial = len(Xs[0])

    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []

    predictions = [None for _ in range(n_trials)]
    predictions_to_save = [None for _ in range(n_trials)]

    # -------------------------------
    # CV split setup
    # -------------------------------
    if rng_seed is not None:
        np.random.seed(rng_seed)
    indices = np.arange(n_trials)

    # scoring function
    scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score

    # IMPORTANT FIX: only compute "balanced" weights for classification
    use_balanced = bool(balanced_weight) and (estimator == sklm.LogisticRegression)

    if outer_cv:
        # choose splits robustly
        def _choose_outer_splits(n):
            # original code used 50 folds when use_cv_sklearn_method=True; keep a safe variant
            if n < 50:
                return 5
            if n <= 100:
                return 10
            # large n: 50 is ok but cap at n
            return 50

        outer_splits = max(2, min(_choose_outer_splits(n_trials), n_trials))

        get_kfold = lambda: KFold(n_splits=outer_splits, shuffle=shuffle).split(indices)

        # fold criteria for logistic regression
        if estimator == sklm.LogisticRegression:
            assert logisticreg_criteria(ys)
            isysat = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=2)
        else:
            isysat = lambda ys_: True

        sample_count, _, outer_kfold_iter = sample_folds(ys, get_kfold, isysat)
        if sample_count > 1 and verbose:
            print(f"sampled outer folds {sample_count} times to ensure enough targets")
    else:
        outer_kfold = iter([train_test_split(indices, test_size=test_prop, shuffle=shuffle)])
        outer_kfold_iter = [(train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(outer_kfold)]

    # -------------------------------
    # Main loop
    # -------------------------------
    if estimator in [sklm.RidgeCV, sklm.LassoCV, sklm.LogisticRegressionCV]:
        raise NotImplementedError("decode_cv does not support CV-type estimators as input.")
    else:
        for train_idxs_outer, test_idxs_outer in outer_kfold_iter:

            X_train = [Xs[i] for i in train_idxs_outer]
            y_train = [ys[i] for i in train_idxs_outer]
            X_test = [Xs[i] for i in test_idxs_outer]
            y_test = [ys[i] for i in test_idxs_outer]

            key = list(hyperparam_grid.keys())[0]  # repo assumes single key
            grid_vals = hyperparam_grid[key]

            # ---------------------------
            # Hyperparam selection
            # ---------------------------
            if not use_cv_sklearn_method:
                # inner folds over *trials* (not samples) so no leakage across bins-in-trial
                idx_inner = np.arange(len(X_train))
                inner_splits = max(2, min(n_folds, len(idx_inner)))
                get_kfold_inner = lambda: KFold(n_splits=inner_splits, shuffle=shuffle).split(idx_inner)

                if estimator == sklm.LogisticRegression:
                    assert logisticreg_criteria(y_train, min_unique_counts=2)
                    isysat_inner = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=1)
                else:
                    isysat_inner = lambda ys_: True

                sample_count, _, inner_kfold_iter = sample_folds(y_train, get_kfold_inner, isysat_inner)
                if sample_count > 1 and verbose:
                    print(f"sampled inner folds {sample_count} times to ensure enough targets")

                r2s = np.zeros([inner_splits, len(grid_vals)])

                # inner CV
                for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold_iter):
                    X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
                    y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
                    X_test_inner = np.vstack([X_train[i] for i in test_idxs_inner])
                    y_test_inner = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)

                    # sanitize and standardize using train stats
                    X_train_inner, keep_cols_inner = _sanitize_X(X_train_inner)
                    X_test_inner = _apply_keep(X_test_inner, keep_cols_inner)
                    X_train_inner, X_test_inner, _, _ = _standardize_train_test(X_train_inner, X_test_inner)

                    for i_val, val in enumerate(grid_vals):
                        # balanced weights ONLY for classification
                        sample_weight = compute_sample_weight("balanced", y=y_train_inner) if use_balanced else None

                        model_inner = estimator(**{**estimator_kwargs, key: val})
                        model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)

                        pred_test_inner = model_inner.predict(X_test_inner)
                        r2s[ifold, i_val] = scoring_f(y_test_inner, pred_test_inner)

                best_val = grid_vals[int(np.argmax(r2s.mean(axis=0)))]

                # refit on full outer-train
                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)

                X_train_array, keep_cols = _sanitize_X(X_train_array)
                X_train_array, _, mu, sd = _standardize_train_test(X_train_array, X_train_array)

                sample_weight = compute_sample_weight("balanced", y=y_train_array) if use_balanced else None

                model = estimator(**{**estimator_kwargs, key: best_val})
                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)

            else:
                # native CV estimators (fast) for Ridge/Lasso only
                if estimator not in [Ridge, Lasso]:
                    raise NotImplementedError("use_cv_sklearn_method=True is only implemented for Ridge/Lasso.")

                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)

                X_train_array, keep_cols = _sanitize_X(X_train_array)
                X_train_array, _, mu, sd = _standardize_train_test(X_train_array, X_train_array)

                model = RidgeCV(alphas=grid_vals) if estimator == Ridge else LassoCV(alphas=grid_vals)

                # balanced weights ONLY for classification (won't apply here)
                sample_weight = compute_sample_weight("balanced", y=y_train_array) if use_balanced else None
                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)

                best_val = getattr(model, "alpha_", None)

            # ---------------------------
            # Evaluate train
            # ---------------------------
            y_pred_train = model.predict(X_train_array)
            scores_train.append(scoring_f(y_train_array, y_pred_train))

            # ---------------------------
            # Evaluate test
            # ---------------------------
            y_true = np.concatenate(y_test, axis=0)

            X_test_array = np.vstack(X_test)
            X_test_array = _apply_keep(X_test_array, keep_cols)
            X_test_array = (X_test_array - mu) / sd

            y_pred = model.predict(X_test_array)

            if isinstance(model, sklm.LogisticRegression):
                y_pred_probs = model.predict_proba(X_test_array)[:, 1]
            else:
                y_pred_probs = None

            scores_test.append(scoring_f(y_true, y_pred))

            # ---------------------------
            # Save per-trial predictions
            # ---------------------------
            for i_fold, i_global in enumerate(test_idxs_outer):
                if bins_per_trial == 1:
                    predictions[i_global] = np.array([y_pred[i_fold]])
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])
                    else:
                        predictions_to_save[i_global] = np.array([y_pred[i_fold]])
                else:
                    Xt = np.asarray(X_test[i_fold], dtype=np.float64)
                    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)[:, keep_cols]
                    Xt = (Xt - mu) / sd
                    predictions[i_global] = model.predict(Xt)
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = model.predict_proba(Xt)[:, 1]
                    else:
                        predictions_to_save[i_global] = predictions[i_global]

            # bookkeeping
            idxes_test.append(test_idxs_outer)
            idxes_train.append(train_idxs_outer)
            weights.append(model.coef_)
            intercepts.append(model.intercept_ if getattr(model, "fit_intercept", False) else None)
            best_params.append({key: best_val})

    # -------------------------------
    # Final full-score computation
    # -------------------------------
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


import numpy as np
import sklearn.linear_model as sklm
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.linear_model import RidgeCV, Ridge, Lasso, LassoCV
from sklearn.utils.class_weight import compute_sample_weight

# assumes these already exist in your namespace (same as original code):
# - format_data_for_decoding
# - logisticreg_criteria
# - sample_folds
#
#
# def decode_cv(
#     ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
#     n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
#     rng_seed=None, use_cv_sklearn_method=False
# ):
#     """
#     ORIGINAL decode_cv with minimal robustness patches for subsession decoding:
#
#     PATCH 1: sanitize + z-score X using TRAIN stats (per outer fold)
#     PATCH 2: do NOT use 50 outer folds when use_cv_sklearn_method=True; use min(n_folds, n_trials)
#
#     This keeps the same workflow and remains fast with RidgeCV/LassoCV.
#     """
#
#     # ----------------------------
#     # helpers (PATCH 1)
#     # ----------------------------
#     def _nan_to_num(X):
#         X = np.asarray(X, dtype=np.float64)
#         return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
#
#     def _standardize_train_test(X_train, X_test, eps=1e-12):
#         # z-score using train statistics only
#         mu = X_train.mean(axis=0)
#         sd = X_train.std(axis=0)
#         sd = np.where(sd < eps, 1.0, sd)  # protect near-constant cols
#         return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd
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
#     if rng_seed is not None:
#         np.random.seed(rng_seed)
#     indices = np.arange(n_trials)
#
#     if outer_cv:
#         # PATCH 2: don't do 50 outer folds for small subsession groups
#         n_splits_outer = max(2, min(n_folds, n_trials))
#         get_kfold = lambda: KFold(n_splits=n_splits_outer, shuffle=shuffle).split(indices)
#
#         # define function to evaluate whether folds are satisfactory
#         if estimator == sklm.LogisticRegression:
#             assert logisticreg_criteria(ys)
#             isysat = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=2)
#         else:
#             isysat = lambda ys_: True
#
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
#     if estimator in [sklm.RidgeCV, sklm.LassoCV, sklm.LogisticRegressionCV]:
#         raise NotImplementedError("the code does not support a CV-type estimator.")
#     else:
#         for train_idxs_outer, test_idxs_outer in outer_kfold_iter:
#             X_train = [Xs[i] for i in train_idxs_outer]
#             y_train = [ys[i] for i in train_idxs_outer]
#             X_test  = [Xs[i] for i in test_idxs_outer]
#             y_test  = [ys[i] for i in test_idxs_outer]
#
#             key = list(hyperparam_grid.keys())[0]
#
#             if not use_cv_sklearn_method:
#                 """
#                 Modified nested CV (original logic), but with PATCH 1 applied inside:
#                 - sanitize + standardize X in each inner fold and in final refit.
#                 """
#                 idx_inner = np.arange(len(X_train))
#                 n_splits_inner = max(2, min(n_folds, len(idx_inner)))
#                 get_kfold_inner = lambda: KFold(n_splits=n_splits_inner, shuffle=shuffle).split(idx_inner)
#
#                 if estimator == sklm.LogisticRegression:
#                     assert logisticreg_criteria(y_train, min_unique_counts=2)
#                     isysat_inner = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=1)
#                 else:
#                     isysat_inner = lambda ys_: True
#
#                 sample_count, _, inner_kfold_iter = sample_folds(y_train, get_kfold_inner, isysat_inner)
#                 if sample_count > 1:
#                     print(f'sampled inner folds {sample_count} times to ensure enough targets')
#
#                 r2s = np.zeros([n_splits_inner, len(hyperparam_grid[key])])
#                 inner_predictions = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
#                 inner_targets     = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
#
#                 for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold_iter):
#                     X_train_inner = _nan_to_num(np.vstack([X_train[i] for i in train_idxs_inner]))
#                     y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
#                     X_test_inner  = _nan_to_num(np.vstack([X_train[i] for i in test_idxs_inner]))
#                     y_test_inner  = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)
#
#                     # PATCH 1: standardize using train stats
#                     X_train_inner, X_test_inner, _, _ = _standardize_train_test(X_train_inner, X_test_inner)
#
#                     for i_alpha, alpha in enumerate(hyperparam_grid[key]):
#                         sample_weight = compute_sample_weight("balanced", y=y_train_inner) if balanced_weight else None
#                         model_inner = estimator(**{**estimator_kwargs, key: alpha})
#                         model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)
#                         pred_test_inner = model_inner.predict(X_test_inner)
#
#                         inner_predictions[test_idxs_inner, i_alpha] = np.mean(pred_test_inner)
#                         inner_targets[test_idxs_inner, i_alpha]     = np.mean(y_test_inner)
#                         r2s[ifold, i_alpha] = scoring_f(y_test_inner, pred_test_inner)
#
#                 assert np.all(~np.isnan(inner_predictions))
#                 assert np.all(~np.isnan(inner_targets))
#
#                 r2s_avg = r2s.mean(axis=0)
#
#                 X_train_array = _nan_to_num(np.vstack(X_train))
#                 y_train_array = np.concatenate(y_train, axis=0)
#
#                 # PATCH 1: standardize final refit on train
#                 X_train_array, _, mu, sd = _standardize_train_test(X_train_array, X_train_array)
#
#                 sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None
#
#                 best_alpha = hyperparam_grid[key][np.argmax(r2s_avg)]
#                 model = estimator(**{**estimator_kwargs, key: best_alpha})
#                 model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
#
#             else:
#                 # FAST path: RidgeCV/LassoCV (your desired setting)
#                 if estimator not in [Ridge, Lasso]:
#                     raise NotImplementedError("This case is not implemented")
#
#                 model = RidgeCV(alphas=hyperparam_grid[key]) if estimator == Ridge else LassoCV(alphas=hyperparam_grid[key])
#
#                 X_train_array = _nan_to_num(np.vstack(X_train))
#                 y_train_array = np.concatenate(y_train, axis=0)
#
#                 # PATCH 1: standardize train
#                 X_train_array, _, mu, sd = _standardize_train_test(X_train_array, X_train_array)
#
#                 sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None
#                 model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
#                 best_alpha = model.alpha_
#
#             # evaluate model on train data (train is already standardized)
#             y_pred_train = model.predict(X_train_array)
#             scores_train.append(scoring_f(y_train_array, y_pred_train))
#
#             # evaluate model on test data (PATCH 1: standardize using train stats)
#             y_true = np.concatenate(y_test, axis=0)
#             X_test_array = _nan_to_num(np.vstack(X_test))
#             X_test_array = (X_test_array - mu) / sd
#             y_pred = model.predict(X_test_array)
#
#             if isinstance(model, sklm.LogisticRegression):
#                 y_pred_probs = model.predict_proba(X_test_array)[:, 1]
#             else:
#                 y_pred_probs = None
#             scores_test.append(scoring_f(y_true, y_pred))
#
#             # save per-trial predictions
#             for i_fold, i_global in enumerate(test_idxs_outer):
#                 if bins_per_trial == 1:
#                     predictions[i_global] = np.array([y_pred[i_fold]])
#                     if isinstance(model, sklm.LogisticRegression):
#                         predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])
#                     else:
#                         predictions_to_save[i_global] = np.array([y_pred[i_fold]])
#                 else:
#                     Xt = _nan_to_num(X_test[i_fold])
#                     Xt = (Xt - mu) / sd
#                     predictions[i_global] = model.predict(Xt)
#                     if isinstance(model, sklm.LogisticRegression):
#                         predictions_to_save[i_global] = model.predict_proba(Xt)[:, 1]
#                     else:
#                         predictions_to_save[i_global] = predictions[i_global]
#
#             idxes_test.append(test_idxs_outer)
#             idxes_train.append(train_idxs_outer)
#             weights.append(model.coef_)
#             intercepts.append(model.intercept_ if getattr(model, "fit_intercept", False) else None)
#             best_params.append({key: best_alpha})
#
#     ys_true_full = np.concatenate(ys, axis=0)
#     ys_pred_full = np.concatenate(predictions, axis=0)
#
#     outdict = dict()
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
#     if hasattr(model, "classes_"):
#         outdict["classes_"] = model.classes_
#
#     if verbose and outer_cv:
#         print("Performance is only described for last outer fold \n")
#
#     return outdict



def decode_cv(ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
              n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
              rng_seed=None, use_cv_sklearn_method=False):

    ys, Xs = format_data_for_decoding(ys, Xs)

    n_trials = len(Xs)
    bins_per_trial = len(Xs[0])
    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []
    predictions = [None for _ in range(n_trials)]
    predictions_to_save = [None for _ in range(n_trials)]

    if rng_seed is not None:
        np.random.seed(rng_seed)
    indices = np.arange(n_trials)

    if outer_cv:
        # PATCH: never do 50 outer folds; use at most n_folds and at most n_trials
        n_splits_outer = max(2, min(n_folds, n_trials))
        get_kfold = lambda: KFold(n_splits=n_splits_outer, shuffle=shuffle).split(indices)

        if estimator == sklm.LogisticRegression:
            assert logisticreg_criteria(ys)
            isysat = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=2)
        else:
            isysat = lambda ys_: True

        sample_count, _, outer_kfold_iter = sample_folds(ys, get_kfold, isysat)
        if sample_count > 1:
            print(f'sampled outer folds {sample_count} times to ensure enough targets')
    else:
        outer_kfold = iter([train_test_split(indices, test_size=test_prop, shuffle=shuffle)])
        outer_kfold_iter = [(train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(outer_kfold)]

    scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score

    if estimator in [sklm.RidgeCV, sklm.LassoCV, sklm.LogisticRegressionCV]:
        raise NotImplementedError("the code does not support a CV-type estimator.")
    else:
        for train_idxs_outer, test_idxs_outer in outer_kfold_iter:
            X_train = [Xs[i] for i in train_idxs_outer]
            y_train = [ys[i] for i in train_idxs_outer]
            X_test  = [Xs[i] for i in test_idxs_outer]
            y_test  = [ys[i] for i in test_idxs_outer]

            key = list(hyperparam_grid.keys())[0]

            if not use_cv_sklearn_method:
                idx_inner = np.arange(len(X_train))
                n_splits_inner = max(2, min(n_folds, len(idx_inner)))
                get_kfold_inner = lambda: KFold(n_splits=n_splits_inner, shuffle=shuffle).split(idx_inner)

                if estimator == sklm.LogisticRegression:
                    assert logisticreg_criteria(y_train, min_unique_counts=2)
                    isysat_inner = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=1)
                else:
                    isysat_inner = lambda ys_: True

                sample_count, _, inner_kfold_iter = sample_folds(y_train, get_kfold_inner, isysat_inner)
                if sample_count > 1:
                    print(f'sampled inner folds {sample_count} times to ensure enough targets')

                r2s = np.zeros([n_splits_inner, len(hyperparam_grid[key])])
                inner_predictions = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
                inner_targets = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan

                for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold_iter):
                    X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
                    y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
                    X_test_inner  = np.vstack([X_train[i] for i in test_idxs_inner])
                    y_test_inner  = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)

                    # PATCH: sanitize + guard
                    X_train_inner = np.nan_to_num(X_train_inner, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
                    X_test_inner  = np.nan_to_num(X_test_inner,  nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
                    max_abs = float(np.max(np.abs(X_train_inner)))
                    if max_abs > 1e6:
                        raise RuntimeError(f"[decode_cv] insane X magnitude in inner fold (max_abs={max_abs})")

                    for i_alpha, alpha in enumerate(hyperparam_grid[key]):
                        sample_weight = compute_sample_weight("balanced", y=y_train_inner) if balanced_weight else None
                        model_inner = estimator(**{**estimator_kwargs, key: alpha})
                        model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)
                        pred_test_inner = model_inner.predict(X_test_inner)

                        inner_predictions[test_idxs_inner, i_alpha] = np.mean(pred_test_inner)
                        inner_targets[test_idxs_inner, i_alpha] = np.mean(y_test_inner)
                        r2s[ifold, i_alpha] = scoring_f(y_test_inner, pred_test_inner)

                assert np.all(~np.isnan(inner_predictions))
                assert np.all(~np.isnan(inner_targets))

                r2s_avg = r2s.mean(axis=0)

                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)

                # PATCH: sanitize + guard
                X_train_array = np.nan_to_num(X_train_array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
                max_abs = float(np.max(np.abs(X_train_array)))
                if max_abs > 1e6:
                    raise RuntimeError(f"[decode_cv] insane X magnitude in outer refit (max_abs={max_abs})")

                sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None
                best_alpha = hyperparam_grid[key][np.argmax(r2s_avg)]
                model = estimator(**{**estimator_kwargs, key: best_alpha})
                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)

            else:
                if estimator not in [Ridge, Lasso]:
                    raise NotImplementedError("This case is not implemented")
                model = RidgeCV(alphas=hyperparam_grid[key]) if estimator == Ridge else LassoCV(alphas=hyperparam_grid[key])

                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)

                # PATCH: sanitize + guard
                X_train_array = np.nan_to_num(X_train_array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
                max_abs = float(np.max(np.abs(X_train_array)))
                if max_abs > 1e6:
                    raise RuntimeError(f"[decode_cv] insane X magnitude in RidgeCV/LassoCV refit (max_abs={max_abs})")

                sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None
                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
                best_alpha = model.alpha_

            y_pred_train = model.predict(X_train_array)
            scores_train.append(scoring_f(y_train_array, y_pred_train))

            y_true = np.concatenate(y_test, axis=0)
            X_test_array = np.vstack(X_test)
            X_test_array = np.nan_to_num(X_test_array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
            y_pred = model.predict(X_test_array)

            if isinstance(model, sklm.LogisticRegression):
                y_pred_probs = model.predict_proba(X_test_array)[:, 1]
            else:
                y_pred_probs = None
            scores_test.append(scoring_f(y_true, y_pred))

            for i_fold, i_global in enumerate(test_idxs_outer):
                if bins_per_trial == 1:
                    predictions[i_global] = np.array([y_pred[i_fold]])
                    predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]]) if isinstance(model, sklm.LogisticRegression) else np.array([y_pred[i_fold]])
                else:
                    Xt = np.nan_to_num(X_test[i_fold], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
                    predictions[i_global] = model.predict(Xt)
                    predictions_to_save[i_global] = model.predict_proba(Xt)[:, 1] if isinstance(model, sklm.LogisticRegression) else predictions[i_global]

            idxes_test.append(test_idxs_outer)
            idxes_train.append(train_idxs_outer)
            weights.append(model.coef_)
            intercepts.append(model.intercept_ if model.fit_intercept else None)
            best_params.append({key: best_alpha})

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




def decode_cv(
    ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
    n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
    rng_seed=None, use_cv_sklearn_method=False
):
    """
    Regresses binned neural activity against a target, using a provided sklearn estimator.

    Patched to eliminate sklearn RuntimeWarnings:
      - overflow/divide-by-zero/invalid encountered in matmul

    Key fixes:
      1) Only use class-balanced sample weights for *classification* (LogisticRegression).
         Using compute_sample_weight("balanced", y=continuous) can explode weights -> huge coef_ -> overflow.
      2) Sanitize NaN/Inf values in X (replace with 0) and drop near-zero-variance columns per fold.
      3) Standardize X per fold using train statistics (mean/std), improving conditioning.
      4) Use safe numbers of folds when trial counts are small.

    Notes:
      - This function supports both:
          * Xs as list of (n_bins, n_features) arrays per trial
          * Xs as 2D array (n_trials, n_features) which format_data_for_decoding converts to list
      - hyperparam_grid is expected to have ONE key (e.g. {"alpha": [...]}).
    """

    import numpy as np
    from sklearn import linear_model as sklm
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.linear_model import RidgeCV, Ridge, Lasso, LassoCV
    from sklearn.utils.class_weight import compute_sample_weight

    # assumes these exist in your module (as in the original repo)
    # - format_data_for_decoding
    # - logisticreg_criteria
    # - sample_folds

    # -------------------------------
    # Standardize inputs
    # -------------------------------
    ys, Xs = format_data_for_decoding(ys, Xs)

    if hyperparam_grid is None:
        hyperparam_grid = {}

    if len(hyperparam_grid) == 0:
        raise ValueError(
            "decode_cv: hyperparam_grid is empty. For Ridge/Lasso you should provide e.g.\n"
            "  hyperparam_grid={'alpha': [0.1, 1, 10, 100, 1000]}\n"
            "Otherwise alpha selection may be unstable."
        )

    # -------------------------------
    # Helpers: sanitize + standardize
    # -------------------------------
    def _sanitize_X(X, eps=1e-12):
        """
        Ensure X is finite and usable:
          - cast float64
          - replace NaN/Inf with 0
          - drop near-zero-variance columns (var <= eps)
        Returns (X_sanitized, keep_cols_mask)
        """
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # variance per feature
        var = X.var(axis=0)
        keep = (var > eps) & np.all(np.isfinite(X), axis=0)

        if keep.sum() == 0:
            raise ValueError(
                "[decode_cv] All features removed after sanitization "
                "(matrix is all invalid or zero-variance)."
            )
        return X[:, keep], keep

    def _apply_keep(X, keep_cols):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X[:, keep_cols]

    def _standardize_train_test(Xtr, Xte, eps=1e-12):
        """
        Z-score using train statistics only.
        """
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd = np.where(sd < eps, 1.0, sd)
        return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd

    # -------------------------------
    # Remove non-finite trials at trial level
    # -------------------------------
    finite_trials = []
    for i in range(len(Xs)):
        Xi = np.asarray(Xs[i])
        yi = np.asarray(ys[i])
        finite_trials.append(np.all(np.isfinite(Xi)) and np.all(np.isfinite(yi)))
    finite_trials = np.array(finite_trials, dtype=bool)

    if not np.all(finite_trials):
        Xs = [x for x, ok in zip(Xs, finite_trials) if ok]
        ys = [y for y, ok in zip(ys, finite_trials) if ok]

    if len(Xs) < 10:
        raise ValueError(f"[decode_cv] Too few finite trials after cleaning: n={len(Xs)}")

    # -------------------------------
    # Initialize containers
    # -------------------------------
    n_trials = len(Xs)
    bins_per_trial = len(Xs[0])

    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []

    predictions = [None for _ in range(n_trials)]
    predictions_to_save = [None for _ in range(n_trials)]

    # -------------------------------
    # CV split setup
    # -------------------------------
    if rng_seed is not None:
        np.random.seed(rng_seed)
    indices = np.arange(n_trials)

    # scoring function
    scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score

    # IMPORTANT FIX: only compute "balanced" weights for classification
    use_balanced = bool(balanced_weight) and (estimator == sklm.LogisticRegression)

    if outer_cv:
        # choose splits robustly
        def _choose_outer_splits(n):
            # original code used 50 folds when use_cv_sklearn_method=True; keep a safe variant
            if n < 50:
                return 5
            if n <= 100:
                return 10
            # large n: 50 is ok but cap at n
            return 50

        outer_splits = max(2, min(_choose_outer_splits(n_trials), n_trials))

        get_kfold = lambda: KFold(n_splits=outer_splits, shuffle=shuffle).split(indices)

        # fold criteria for logistic regression
        if estimator == sklm.LogisticRegression:
            assert logisticreg_criteria(ys)
            isysat = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=2)
        else:
            isysat = lambda ys_: True

        sample_count, _, outer_kfold_iter = sample_folds(ys, get_kfold, isysat)
        if sample_count > 1 and verbose:
            print(f"sampled outer folds {sample_count} times to ensure enough targets")
    else:
        outer_kfold = iter([train_test_split(indices, test_size=test_prop, shuffle=shuffle)])
        outer_kfold_iter = [(train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(outer_kfold)]

    # -------------------------------
    # Main loop
    # -------------------------------
    if estimator in [sklm.RidgeCV, sklm.LassoCV, sklm.LogisticRegressionCV]:
        raise NotImplementedError("decode_cv does not support CV-type estimators as input.")
    else:
        for train_idxs_outer, test_idxs_outer in outer_kfold_iter:

            X_train = [Xs[i] for i in train_idxs_outer]
            y_train = [ys[i] for i in train_idxs_outer]
            X_test = [Xs[i] for i in test_idxs_outer]
            y_test = [ys[i] for i in test_idxs_outer]

            key = list(hyperparam_grid.keys())[0]  # repo assumes single key
            grid_vals = hyperparam_grid[key]

            # ---------------------------
            # Hyperparam selection
            # ---------------------------
            if not use_cv_sklearn_method:
                # inner folds over *trials* (not samples) so no leakage across bins-in-trial
                idx_inner = np.arange(len(X_train))
                inner_splits = max(2, min(n_folds, len(idx_inner)))
                get_kfold_inner = lambda: KFold(n_splits=inner_splits, shuffle=shuffle).split(idx_inner)

                if estimator == sklm.LogisticRegression:
                    assert logisticreg_criteria(y_train, min_unique_counts=2)
                    isysat_inner = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=1)
                else:
                    isysat_inner = lambda ys_: True

                sample_count, _, inner_kfold_iter = sample_folds(y_train, get_kfold_inner, isysat_inner)
                if sample_count > 1 and verbose:
                    print(f"sampled inner folds {sample_count} times to ensure enough targets")

                r2s = np.zeros([inner_splits, len(grid_vals)])

                # inner CV
                for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold_iter):
                    X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
                    y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
                    X_test_inner = np.vstack([X_train[i] for i in test_idxs_inner])
                    y_test_inner = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)

                    # sanitize and standardize using train stats
                    X_train_inner, keep_cols_inner = _sanitize_X(X_train_inner)
                    X_test_inner = _apply_keep(X_test_inner, keep_cols_inner)
                    X_train_inner, X_test_inner, _, _ = _standardize_train_test(X_train_inner, X_test_inner)

                    for i_val, val in enumerate(grid_vals):
                        # balanced weights ONLY for classification
                        sample_weight = compute_sample_weight("balanced", y=y_train_inner) if use_balanced else None

                        model_inner = estimator(**{**estimator_kwargs, key: val})
                        model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)

                        pred_test_inner = model_inner.predict(X_test_inner)
                        r2s[ifold, i_val] = scoring_f(y_test_inner, pred_test_inner)

                best_val = grid_vals[int(np.argmax(r2s.mean(axis=0)))]

                # refit on full outer-train
                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)

                X_train_array, keep_cols = _sanitize_X(X_train_array)
                X_train_array, _, mu, sd = _standardize_train_test(X_train_array, X_train_array)

                sample_weight = compute_sample_weight("balanced", y=y_train_array) if use_balanced else None

                model = estimator(**{**estimator_kwargs, key: best_val})
                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)

            else:
                # native CV estimators (fast) for Ridge/Lasso only
                if estimator not in [Ridge, Lasso]:
                    raise NotImplementedError("use_cv_sklearn_method=True is only implemented for Ridge/Lasso.")

                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)

                X_train_array, keep_cols = _sanitize_X(X_train_array)
                X_train_array, _, mu, sd = _standardize_train_test(X_train_array, X_train_array)

                model = RidgeCV(alphas=grid_vals) if estimator == Ridge else LassoCV(alphas=grid_vals)

                # balanced weights ONLY for classification (won't apply here)
                sample_weight = compute_sample_weight("balanced", y=y_train_array) if use_balanced else None
                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)

                best_val = getattr(model, "alpha_", None)

            # ---------------------------
            # Evaluate train
            # ---------------------------
            y_pred_train = model.predict(X_train_array)
            scores_train.append(scoring_f(y_train_array, y_pred_train))

            # ---------------------------
            # Evaluate test
            # ---------------------------
            y_true = np.concatenate(y_test, axis=0)

            X_test_array = np.vstack(X_test)
            X_test_array = _apply_keep(X_test_array, keep_cols)
            X_test_array = (X_test_array - mu) / sd

            y_pred = model.predict(X_test_array)

            if isinstance(model, sklm.LogisticRegression):
                y_pred_probs = model.predict_proba(X_test_array)[:, 1]
            else:
                y_pred_probs = None

            scores_test.append(scoring_f(y_true, y_pred))

            # ---------------------------
            # Save per-trial predictions
            # ---------------------------
            for i_fold, i_global in enumerate(test_idxs_outer):
                if bins_per_trial == 1:
                    predictions[i_global] = np.array([y_pred[i_fold]])
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])
                    else:
                        predictions_to_save[i_global] = np.array([y_pred[i_fold]])
                else:
                    Xt = np.asarray(X_test[i_fold], dtype=np.float64)
                    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)[:, keep_cols]
                    Xt = (Xt - mu) / sd
                    predictions[i_global] = model.predict(Xt)
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = model.predict_proba(Xt)[:, 1]
                    else:
                        predictions_to_save[i_global] = predictions[i_global]

            # bookkeeping
            idxes_test.append(test_idxs_outer)
            idxes_train.append(train_idxs_outer)
            weights.append(model.coef_)
            intercepts.append(model.intercept_ if getattr(model, "fit_intercept", False) else None)
            best_params.append({key: best_val})

    # -------------------------------
    # Final full-score computation
    # -------------------------------
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


def decode_cv(
    ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
    n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
    rng_seed=None, use_cv_sklearn_method=False
):
    """
    Regresses binned neural activity against a target, using a provided sklearn estimator.

    Patched:
      - hides sklearn/numpy RuntimeWarnings (overflow/div0/invalid in matmul)
      - prints *where* the numerical issue occurs (outer fold / inner fold / alpha / stage)
      - skips the bad outer fold and continues

    Keeps the EXACT SAME function signature.
    """

    import numpy as np
    import warnings
    from sklearn import linear_model as sklm
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.linear_model import RidgeCV, Ridge, Lasso, LassoCV
    from sklearn.utils.class_weight import compute_sample_weight

    # assumes these exist (same as repo)
    # - format_data_for_decoding
    # - logisticreg_criteria
    # - sample_folds

    # -------------------------------
    # Silence the noisy RuntimeWarnings globally inside this function
    # -------------------------------
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="overflow encountered in matmul")
    warnings.filterwarnings("ignore", message="invalid value encountered in matmul")
    warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
    warnings.filterwarnings("ignore", message="overflow encountered")
    warnings.filterwarnings("ignore", message="invalid value encountered")
    warnings.filterwarnings("ignore", message="divide by zero encountered")

    # -------------------------------
    # Helpers
    # -------------------------------
    def _x_stats(name, X):
        X = np.asarray(X, dtype=np.float64)
        finite = np.isfinite(X)
        if not finite.any():
            return f"{name}: shape={X.shape}, ALL NON-FINITE"
        Xf = X[finite]
        stds = np.nanstd(X, axis=0) if X.ndim == 2 else np.array([np.nanstd(X)])
        return (
            f"{name}: shape={X.shape}, max_abs={float(np.nanmax(np.abs(Xf))):.3e}, "
            f"any_nan={bool(np.isnan(X).any())}, any_inf={bool(np.isinf(X).any())}, "
            f"std_min={float(np.nanmin(stds)):.3e}, std_max={float(np.nanmax(stds)):.3e}"
        )

    def _sanitize_X(X, eps=1e-12):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        var = X.var(axis=0)
        keep = (var > eps) & np.all(np.isfinite(X), axis=0)
        if keep.sum() == 0:
            raise FloatingPointError("[decode_cv] all features removed (invalid/zero-variance).")
        return X[:, keep], keep

    def _apply_keep(X, keep_cols):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X[:, keep_cols]

    def _standardize_train_test(Xtr, Xte, eps=1e-12):
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd = np.where(sd < eps, 1.0, sd)
        return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd

    # -------------------------------
    # Standardize inputs
    # -------------------------------
    ys, Xs = format_data_for_decoding(ys, Xs)

    if hyperparam_grid is None or len(hyperparam_grid) == 0:
        raise ValueError("decode_cv: hyperparam_grid is empty")

    # remove non-finite trials at trial-level
    finite_trials = []
    for i in range(len(Xs)):
        Xi = np.asarray(Xs[i])
        yi = np.asarray(ys[i])
        finite_trials.append(np.all(np.isfinite(Xi)) and np.all(np.isfinite(yi)))
    finite_trials = np.array(finite_trials, dtype=bool)
    if not np.all(finite_trials):
        Xs = [x for x, ok in zip(Xs, finite_trials) if ok]
        ys = [y for y, ok in zip(ys, finite_trials) if ok]
    if len(Xs) < 10:
        raise ValueError(f"[decode_cv] Too few trials after cleaning: {len(Xs)}")

    n_trials = len(Xs)
    bins_per_trial = len(Xs[0])

    # outputs
    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []
    predictions = [None for _ in range(n_trials)]
    predictions_to_save = [None for _ in range(n_trials)]

    # RNG
    if rng_seed is not None:
        np.random.seed(rng_seed)
    indices = np.arange(n_trials)

    scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score

    # IMPORTANT: only balanced weights for classification
    use_balanced = bool(balanced_weight) and (estimator == sklm.LogisticRegression)

    # -------------------------------
    # Outer CV splits
    # -------------------------------
    if outer_cv:
        # Keep your original spirit but avoid silly small folds
        outer_splits = max(2, min((50 if use_cv_sklearn_method else n_folds), n_trials))
        get_kfold = lambda: KFold(n_splits=outer_splits, shuffle=shuffle).split(indices)

        if estimator == sklm.LogisticRegression:
            assert logisticreg_criteria(ys)
            isysat = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=2)
        else:
            isysat = lambda ys_: True

        _, _, outer_kfold_iter = sample_folds(ys, get_kfold, isysat)
    else:
        outer_kfold = iter([train_test_split(indices, test_size=test_prop, shuffle=shuffle)])
        outer_kfold_iter = [(train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(outer_kfold)]

    if estimator in [sklm.RidgeCV, sklm.LassoCV, sklm.LogisticRegressionCV]:
        raise NotImplementedError("decode_cv does not support CV-type estimators as input.")

    key = list(hyperparam_grid.keys())[0]
    grid_vals = hyperparam_grid[key]

    # -------------------------------
    # Main loop
    # -------------------------------
    for outer_fold_id, (train_idxs_outer, test_idxs_outer) in enumerate(outer_kfold_iter):
        try:
            X_train = [Xs[i] for i in train_idxs_outer]
            y_train = [ys[i] for i in train_idxs_outer]
            X_test = [Xs[i] for i in test_idxs_outer]
            y_test = [ys[i] for i in test_idxs_outer]

            if not use_cv_sklearn_method:
                # inner folds over trials
                idx_inner = np.arange(len(X_train))
                inner_splits = max(2, min(n_folds, len(idx_inner)))
                get_kfold_inner = lambda: KFold(n_splits=inner_splits, shuffle=shuffle).split(idx_inner)

                if estimator == sklm.LogisticRegression:
                    assert logisticreg_criteria(y_train, min_unique_counts=2)
                    isysat_inner = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=1)
                else:
                    isysat_inner = lambda ys_: True

                _, _, inner_kfold_iter = sample_folds(y_train, get_kfold_inner, isysat_inner)

                r2s = np.zeros([inner_splits, len(grid_vals)])

                for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold_iter):
                    X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
                    y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
                    X_test_inner = np.vstack([X_train[i] for i in test_idxs_inner])
                    y_test_inner = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)

                    # sanitize + standardize
                    X_train_inner, keep_cols_inner = _sanitize_X(X_train_inner)
                    X_test_inner = _apply_keep(X_test_inner, keep_cols_inner)
                    X_train_inner, X_test_inner, _, _ = _standardize_train_test(X_train_inner, X_test_inner)

                    for i_val, val in enumerate(grid_vals):
                        sample_weight = compute_sample_weight("balanced", y=y_train_inner) if use_balanced else None
                        model_inner = estimator(**{**estimator_kwargs, key: val})

                        # catch floating blowups as exceptions (no warning spam)
                        with np.errstate(over="raise", invalid="raise", divide="raise"):
                            model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)
                            pred = model_inner.predict(X_test_inner)

                        r2s[ifold, i_val] = scoring_f(y_test_inner, pred)

                best_val = grid_vals[int(np.argmax(r2s.mean(axis=0)))]

                # refit on outer-train
                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)
                X_train_array, keep_cols = _sanitize_X(X_train_array)
                X_train_array, _, mu, sd = _standardize_train_test(X_train_array, X_train_array)

                sample_weight = compute_sample_weight("balanced", y=y_train_array) if use_balanced else None
                model = estimator(**{**estimator_kwargs, key: best_val})

                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    model.fit(X_train_array, y_train_array, sample_weight=sample_weight)

            else:
                # fast branch: RidgeCV/LassoCV
                if estimator not in [Ridge, Lasso]:
                    raise NotImplementedError("use_cv_sklearn_method=True only implemented for Ridge/Lasso")

                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)
                X_train_array, keep_cols = _sanitize_X(X_train_array)
                X_train_array, _, mu, sd = _standardize_train_test(X_train_array, X_train_array)

                model = RidgeCV(alphas=grid_vals) if estimator == Ridge else LassoCV(alphas=grid_vals)
                sample_weight = compute_sample_weight("balanced", y=y_train_array) if use_balanced else None

                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    model.fit(X_train_array, y_train_array, sample_weight=sample_weight)

                best_val = getattr(model, "alpha_", None)

            # evaluate train
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                y_pred_train = model.predict(X_train_array)
            scores_train.append(scoring_f(y_train_array, y_pred_train))

            # evaluate test
            y_true = np.concatenate(y_test, axis=0)
            X_test_array = np.vstack(X_test)
            X_test_array = _apply_keep(X_test_array, keep_cols)
            X_test_array = (X_test_array - mu) / sd

            with np.errstate(over="raise", invalid="raise", divide="raise"):
                y_pred = model.predict(X_test_array)

            if isinstance(model, sklm.LogisticRegression):
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    y_pred_probs = model.predict_proba(X_test_array)[:, 1]
            else:
                y_pred_probs = None

            scores_test.append(scoring_f(y_true, y_pred))

            # save per-trial predictions
            for i_fold, i_global in enumerate(test_idxs_outer):
                if bins_per_trial == 1:
                    predictions[i_global] = np.array([y_pred[i_fold]])
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])
                    else:
                        predictions_to_save[i_global] = np.array([y_pred[i_fold]])
                else:
                    Xt = np.asarray(X_test[i_fold], dtype=np.float64)
                    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)[:, keep_cols]
                    Xt = (Xt - mu) / sd
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        predictions[i_global] = model.predict(Xt)
                    if isinstance(model, sklm.LogisticRegression):
                        with np.errstate(over="raise", invalid="raise", divide="raise"):
                            predictions_to_save[i_global] = model.predict_proba(Xt)[:, 1]
                    else:
                        predictions_to_save[i_global] = predictions[i_global]

            idxes_test.append(test_idxs_outer)
            idxes_train.append(train_idxs_outer)
            weights.append(model.coef_)
            intercepts.append(model.intercept_ if getattr(model, "fit_intercept", False) else None)
            best_params.append({key: best_val})

        except (FloatingPointError, OverflowError, ValueError) as e:
            # PRINT ONLY ONE CLEAN BLOCK (no spam)
            print("\n================= DECODE_CV NUMERIC ERROR =================")
            print(f"Outer fold: {outer_fold_id}")
            print("Exception:", repr(e))

            # try to report what we have
            try:
                print(_x_stats("X_train_array", X_train_array))
            except Exception:
                print("X_train_array: not available")
            try:
                print(_x_stats("X_test_array", X_test_array))
            except Exception:
                print("X_test_array: not available")

            # also show which alpha key exists
            try:
                print(f"grid key: {key}, n_grid={len(grid_vals)}")
            except Exception:
                pass

            print("===========================================================\n")
            # skip this outer fold
            continue

    # if everything failed, crash clearly
    if all(p is None for p in predictions):
        raise RuntimeError("decode_cv: All outer folds failed with numeric errors; see printed debug blocks.")

    # compute full-score using available predictions only
    ok = [p is not None for p in predictions]
    ys_true_full = np.concatenate([ys[i] for i, good in enumerate(ok) if good], axis=0)
    ys_pred_full = np.concatenate([predictions[i] for i, good in enumerate(ok) if good], axis=0)

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
    if "model" in locals() and hasattr(model, "classes_"):
        outdict["classes_"] = model.classes_

    return outdict





def decode_cv(
    ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
    n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
    rng_seed=None, use_cv_sklearn_method=False
):
    """
    Same signature as repo.

    Patch goals:
      1) Prevent numpy/sklearn RuntimeWarnings from becoming hard FloatingPointError crashes
         (common if np.seterr(all="raise") is set somewhere else).
      2) Print where it happens (outer fold + stage).
      3) Avoid pathological outer-fold counts when n_trials is small (e.g. 47 trials).
    """

    import numpy as np
    import warnings
    from sklearn import linear_model as sklm
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.linear_model import RidgeCV, Ridge, Lasso, LassoCV
    from sklearn.utils.class_weight import compute_sample_weight

    # assumes these exist (same as repo)
    # - format_data_for_decoding
    # - logisticreg_criteria
    # - sample_folds

    # -------------------------------
    # Helper: compact stats print
    # -------------------------------
    def _x_stats(name, X):
        X = np.asarray(X, dtype=np.float64)
        finite = np.isfinite(X)
        if not finite.any():
            return f"{name}: shape={X.shape}, ALL NON-FINITE"
        Xf = X[finite]
        stds = np.nanstd(X, axis=0) if X.ndim == 2 else np.array([np.nanstd(X)])
        return (
            f"{name}: shape={X.shape}, max_abs={float(np.nanmax(np.abs(Xf))):.3e}, "
            f"any_nan={bool(np.isnan(X).any())}, any_inf={bool(np.isinf(X).any())}, "
            f"std_min={float(np.nanmin(stds)):.3e}, std_max={float(np.nanmax(stds)):.3e}"
        )

    def _sanitize_X(X, eps=1e-12):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        var = X.var(axis=0)
        keep = (var > eps) & np.all(np.isfinite(X), axis=0)
        if keep.sum() == 0:
            raise ValueError("[decode_cv] All features removed after sanitization (zero-variance/invalid).")
        return X[:, keep], keep

    def _apply_keep(X, keep_cols):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X[:, keep_cols]

    def _standardize_train_test(Xtr, Xte, eps=1e-12):
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd = np.where(sd < eps, 1.0, sd)
        return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd

    # -------------------------------
    # IMPORTANT: locally suppress numeric warnings becoming exceptions
    # -------------------------------
    old_seterr = np.seterr(all="ignore")  # overrides any global np.seterr(all="raise")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    try:
        # -------------------------------
        # Standardize inputs
        # -------------------------------
        ys, Xs = format_data_for_decoding(ys, Xs)

        if hyperparam_grid is None or len(hyperparam_grid) == 0:
            raise ValueError("decode_cv: hyperparam_grid is empty")

        # remove trial-level non-finite
        finite_trials = []
        for i in range(len(Xs)):
            Xi = np.asarray(Xs[i])
            yi = np.asarray(ys[i])
            finite_trials.append(np.all(np.isfinite(Xi)) and np.all(np.isfinite(yi)))
        finite_trials = np.array(finite_trials, dtype=bool)
        if not np.all(finite_trials):
            Xs = [x for x, ok in zip(Xs, finite_trials) if ok]
            ys = [y for y, ok in zip(ys, finite_trials) if ok]

        if len(Xs) < 10:
            raise ValueError(f"[decode_cv] Too few trials after cleaning: {len(Xs)}")

        n_trials = len(Xs)
        bins_per_trial = len(Xs[0])

        scores_test, scores_train = [], []
        idxes_test, idxes_train = [], []
        weights, intercepts, best_params = [], [], []
        predictions = [None for _ in range(n_trials)]
        predictions_to_save = [None for _ in range(n_trials)]

        if rng_seed is not None:
            np.random.seed(rng_seed)
        indices = np.arange(n_trials)

        scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score
        use_balanced = bool(balanced_weight) and (estimator == sklm.LogisticRegression)

        # -------------------------------
        # Outer CV splits
        # -------------------------------
        if outer_cv:
            # CRITICAL: for small n_trials (like 47), do NOT do huge outer folds.
            # Always cap to <= n_folds.
            outer_splits = max(2, min(n_folds, n_trials))
            get_kfold = lambda: KFold(n_splits=outer_splits, shuffle=shuffle).split(indices)

            if estimator == sklm.LogisticRegression:
                assert logisticreg_criteria(ys)
                isysat = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=2)
            else:
                isysat = lambda ys_: True

            _, _, outer_kfold_iter = sample_folds(ys, get_kfold, isysat)
        else:
            outer_kfold = iter([train_test_split(indices, test_size=test_prop, shuffle=shuffle)])
            outer_kfold_iter = [(train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(outer_kfold)]

        if estimator in [sklm.RidgeCV, sklm.LassoCV, sklm.LogisticRegressionCV]:
            raise NotImplementedError("decode_cv does not support CV-type estimators as input.")

        key = list(hyperparam_grid.keys())[0]
        grid_vals = hyperparam_grid[key]

        # -------------------------------
        # Main loop
        # -------------------------------
        for outer_fold_id, (train_idxs_outer, test_idxs_outer) in enumerate(outer_kfold_iter):
            stage = "start"
            try:
                X_train = [Xs[i] for i in train_idxs_outer]
                y_train = [ys[i] for i in train_idxs_outer]
                X_test = [Xs[i] for i in test_idxs_outer]
                y_test = [ys[i] for i in test_idxs_outer]

                # We keep your logic: two modes for hyperparam selection
                if not use_cv_sklearn_method:
                    # inner CV over trials
                    idx_inner = np.arange(len(X_train))
                    inner_splits = max(2, min(n_folds, len(idx_inner)))
                    get_kfold_inner = lambda: KFold(n_splits=inner_splits, shuffle=shuffle).split(idx_inner)

                    if estimator == sklm.LogisticRegression:
                        assert logisticreg_criteria(y_train, min_unique_counts=2)
                        isysat_inner = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=1)
                    else:
                        isysat_inner = lambda ys_: True

                    _, _, inner_kfold_iter = sample_folds(y_train, get_kfold_inner, isysat_inner)
                    r2s = np.zeros([inner_splits, len(grid_vals)])

                    for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold_iter):
                        stage = f"inner_fold_{ifold}_stack"
                        X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
                        y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
                        X_test_inner = np.vstack([X_train[i] for i in test_idxs_inner])
                        y_test_inner = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)

                        stage = f"inner_fold_{ifold}_sanitize"
                        X_train_inner, keep_cols_inner = _sanitize_X(X_train_inner)
                        X_test_inner = _apply_keep(X_test_inner, keep_cols_inner)
                        X_train_inner, X_test_inner, _, _ = _standardize_train_test(X_train_inner, X_test_inner)

                        for i_val, val in enumerate(grid_vals):
                            stage = f"inner_fold_{ifold}_fit_alpha_{val}"
                            sample_weight = compute_sample_weight("balanced", y=y_train_inner) if use_balanced else None
                            model_inner = estimator(**{**estimator_kwargs, key: val})
                            model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)
                            pred = model_inner.predict(X_test_inner)
                            r2s[ifold, i_val] = scoring_f(y_test_inner, pred)

                    best_val = grid_vals[int(np.argmax(r2s.mean(axis=0)))]

                    stage = "outer_refit_stack"
                    X_train_array = np.vstack(X_train)
                    y_train_array = np.concatenate(y_train, axis=0)

                    stage = "outer_refit_sanitize"
                    X_train_array, keep_cols = _sanitize_X(X_train_array)
                    X_train_array, _, mu, sd = _standardize_train_test(X_train_array, X_train_array)

                    stage = f"outer_refit_fit_alpha_{best_val}"
                    sample_weight = compute_sample_weight("balanced", y=y_train_array) if use_balanced else None
                    model = estimator(**{**estimator_kwargs, key: best_val})
                    model.fit(X_train_array, y_train_array, sample_weight=sample_weight)

                else:
                    # RidgeCV/LassoCV path
                    if estimator not in [Ridge, Lasso]:
                        raise NotImplementedError("use_cv_sklearn_method=True only implemented for Ridge/Lasso")

                    stage = "outer_refit_stack"
                    X_train_array = np.vstack(X_train)
                    y_train_array = np.concatenate(y_train, axis=0)

                    stage = "outer_refit_sanitize"
                    X_train_array, keep_cols = _sanitize_X(X_train_array)
                    X_train_array, _, mu, sd = _standardize_train_test(X_train_array, X_train_array)

                    stage = "outer_refit_fit_nativeCV"
                    model = RidgeCV(alphas=grid_vals) if estimator == Ridge else LassoCV(alphas=grid_vals)
                    sample_weight = compute_sample_weight("balanced", y=y_train_array) if use_balanced else None
                    model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
                    best_val = getattr(model, "alpha_", None)

                # evaluate train
                stage = "predict_train"
                y_pred_train = model.predict(X_train_array)
                scores_train.append(scoring_f(y_train_array, y_pred_train))

                # evaluate test
                stage = "predict_test_stack"
                y_true = np.concatenate(y_test, axis=0)
                X_test_array = np.vstack(X_test)
                X_test_array = _apply_keep(X_test_array, keep_cols)
                X_test_array = (X_test_array - mu) / sd

                stage = "predict_test"
                y_pred = model.predict(X_test_array)
                if isinstance(model, sklm.LogisticRegression):
                    y_pred_probs = model.predict_proba(X_test_array)[:, 1]
                else:
                    y_pred_probs = None

                scores_test.append(scoring_f(y_true, y_pred))

                # save per-trial predictions
                stage = "save_predictions"
                for i_fold, i_global in enumerate(test_idxs_outer):
                    if bins_per_trial == 1:
                        predictions[i_global] = np.array([y_pred[i_fold]])
                        if isinstance(model, sklm.LogisticRegression):
                            predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])
                        else:
                            predictions_to_save[i_global] = np.array([y_pred[i_fold]])
                    else:
                        Xt = np.asarray(X_test[i_fold], dtype=np.float64)
                        Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)[:, keep_cols]
                        Xt = (Xt - mu) / sd
                        predictions[i_global] = model.predict(Xt)
                        if isinstance(model, sklm.LogisticRegression):
                            predictions_to_save[i_global] = model.predict_proba(Xt)[:, 1]
                        else:
                            predictions_to_save[i_global] = predictions[i_global]

                idxes_test.append(test_idxs_outer)
                idxes_train.append(train_idxs_outer)
                weights.append(model.coef_)
                intercepts.append(model.intercept_ if getattr(model, "fit_intercept", False) else None)
                best_params.append({key: best_val})

            except Exception as e:
                # single clean block, then skip this outer fold
                print("\n================= DECODE_CV NUMERIC/CRASH DEBUG =================")
                print(f"Outer fold: {outer_fold_id} / stage: {stage}")
                print("Exception:", repr(e))
                try:
                    print(_x_stats("X_train_array", X_train_array))
                except Exception:
                    print("X_train_array: not available")
                try:
                    print(_x_stats("X_test_array", X_test_array))
                except Exception:
                    print("X_test_array: not available")
                print(f"grid key: {key}, n_grid={len(grid_vals)}")
                print("===========================================================\n")
                continue

        if all(p is None for p in predictions):
            raise RuntimeError("decode_cv: all outer folds failed; see debug blocks above.")

        ok = [p is not None for p in predictions]
        ys_true_full = np.concatenate([ys[i] for i, good in enumerate(ok) if good], axis=0)
        ys_pred_full = np.concatenate([predictions[i] for i, good in enumerate(ok) if good], axis=0)

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
        if "model" in locals() and hasattr(model, "classes_"):
            outdict["classes_"] = model.classes_

        return outdict

    finally:
        # restore numpy error handling
        np.seterr(**old_seterr)


def decode_cv(
    ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
    n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
    rng_seed=None, use_cv_sklearn_method=False
):
    """
    Drop-in replacement for Findling et al decode_cv.

    - Silences sklearn / numpy RuntimeWarnings (overflow/div0/invalid)
    - Converts real numeric blowups into controlled debug printouts
    - Skips only the bad outer fold instead of killing the run
    - Prints whether warnings are still happening

    DOES NOT change the function signature or outputs.
    """

    import numpy as np
    import warnings
    from sklearn import linear_model as sklm
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.linear_model import RidgeCV, Ridge, Lasso, LassoCV
    from sklearn.utils.class_weight import compute_sample_weight

    # Silence sklearn spam
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # ---------------- helpers ----------------
    def _x_stats(name, X):
        X = np.asarray(X, dtype=np.float64)
        if X.size == 0:
            return f"{name}: EMPTY"
        return (
            f"{name}: shape={X.shape}, "
            f"max_abs={np.nanmax(np.abs(X)):.3e}, "
            f"any_nan={np.isnan(X).any()}, "
            f"any_inf={np.isinf(X).any()}"
        )

    def _sanitize_X(X, eps=1e-12):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        var = X.var(axis=0)
        keep = var > eps
        if keep.sum() == 0:
            raise FloatingPointError("All features zero/invalid")
        return X[:, keep], keep

    def _apply_keep(X, keep):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X[:, keep]

    def _standardize(Xtr, Xte):
        mu = Xtr.mean(0)
        sd = Xtr.std(0)
        sd[sd < 1e-12] = 1
        return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd

    # ---------------- prepare data ----------------
    ys, Xs = format_data_for_decoding(ys, Xs)

    if hyperparam_grid is None or len(hyperparam_grid) == 0:
        raise ValueError("hyperparam_grid must not be empty")

    # Remove non-finite trials
    good = []
    for x, y in zip(Xs, ys):
        good.append(np.isfinite(x).all() and np.isfinite(y).all())
    Xs = [x for x, g in zip(Xs, good) if g]
    ys = [y for y, g in zip(ys, good) if g]

    n_trials = len(Xs)
    bins_per_trial = len(Xs[0])

    indices = np.arange(n_trials)
    if rng_seed is not None:
        np.random.seed(rng_seed)

    scoring_f = balanced_accuracy_score if estimator == sklm.LogisticRegression else r2_score
    use_balanced = balanced_weight and estimator == sklm.LogisticRegression

    # ---------------- CV ----------------
    if outer_cv:
        outer_splits = min(50 if use_cv_sklearn_method else n_folds, n_trials)
        outer_splits = max(2, outer_splits)
        get_kfold = lambda: KFold(outer_splits, shuffle=shuffle).split(indices)
        _, _, outer_folds = sample_folds(ys, get_kfold, lambda y: True)
    else:
        outer_folds = [(train_test_split(indices, test_size=test_prop, shuffle=shuffle))]

    key = list(hyperparam_grid.keys())[0]
    grid_vals = hyperparam_grid[key]

    scores_train, scores_test = [], []
    predictions = [None] * n_trials
    predictions_to_save = [None] * n_trials
    weights, intercepts, best_params = [], [], []
    idxes_train, idxes_test = [], []

    # ---------------- MAIN LOOP ----------------
    for outer_fold, (train_idx, test_idx) in enumerate(outer_folds):

        try:
            X_train = [Xs[i] for i in train_idx]
            y_train = [ys[i] for i in train_idx]
            X_test = [Xs[i] for i in test_idx]
            y_test = [ys[i] for i in test_idx]

            # -------- hyperparam --------
            best_score = -np.inf
            best_val = None

            for val in grid_vals:
                Xtr = np.vstack(X_train)
                ytr = np.concatenate(y_train)
                Xte = np.vstack(X_test)
                yte = np.concatenate(y_test)

                Xtr, keep = _sanitize_X(Xtr)
                Xte = _apply_keep(Xte, keep)
                Xtr, Xte, mu, sd = _standardize(Xtr, Xte)

                model = estimator(**{**estimator_kwargs, key: val})

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", RuntimeWarning)
                    model.fit(Xtr, ytr)
                    if w:
                        print(f"[decode_cv] outer {outer_fold} alpha={val} warnings:", w[0].message)

                pred = model.predict(Xte)
                sc = scoring_f(yte, pred)

                if sc > best_score:
                    best_score = sc
                    best_val = val

            # -------- refit best --------
            Xtr = np.vstack(X_train)
            ytr = np.concatenate(y_train)
            Xte = np.vstack(X_test)
            yte = np.concatenate(y_test)

            Xtr, keep = _sanitize_X(Xtr)
            Xte = _apply_keep(Xte, keep)
            Xtr, Xte, mu, sd = _standardize(Xtr, Xte)

            model = estimator(**{**estimator_kwargs, key: best_val})

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", RuntimeWarning)
                model.fit(Xtr, ytr)
                if w:
                    print(f"[decode_cv] outer {outer_fold} REFIT warnings:", w[0].message)

            ypred = model.predict(Xte)

            scores_train.append(scoring_f(ytr, model.predict(Xtr)))
            scores_test.append(scoring_f(yte, ypred))

            for i, gi in enumerate(test_idx):
                predictions[gi] = np.array([ypred[i]])
                predictions_to_save[gi] = predictions[gi]

            weights.append(model.coef_)
            intercepts.append(model.intercept_)
            best_params.append({key: best_val})
            idxes_train.append(train_idx)
            idxes_test.append(test_idx)

        except Exception as e:
            print("\n==== DECODE_CV FAILURE ====")
            print("outer fold", outer_fold, "error:", e)
            try:
                print(_x_stats("Xtr", Xtr))
                print(_x_stats("Xte", Xte))
            except:
                pass
            print("==========================\n")
            continue

    ys_true = np.concatenate([ys[i] for i in range(n_trials) if predictions[i] is not None])
    ys_pred = np.concatenate([predictions[i] for i in range(n_trials) if predictions[i] is not None])

    out = dict()
    out["scores_test_full"] = scoring_f(ys_true, ys_pred)
    out["scores_train"] = scores_train
    out["scores_test"] = scores_test
    out["weights"] = weights
    out["intercepts"] = intercepts
    out["best_params"] = best_params
    out["predictions_test"] = predictions_to_save
    out["idxes_test"] = idxes_test
    out["idxes_train"] = idxes_train

    return out



def decode_cv(
    ys, Xs, estimator, estimator_kwargs,
    balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
    n_folds=5, save_binned=False, save_predictions=True, verbose=False,
    shuffle=True, outer_cv=True, rng_seed=None, use_cv_sklearn_method=False,
    min_trials=10,  # <-- NEW: subgroup minimum
):
    import numpy as np
    import sklearn.linear_model as sklm
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.linear_model import RidgeCV, Ridge, Lasso, LassoCV
    from sklearn.utils.class_weight import compute_sample_weight

    # transform target data into standard format: list of np.ndarrays
    ys, Xs = format_data_for_decoding(ys, Xs)

    # basic guard for subgroups
    n_trials = len(Xs)
    if n_trials < min_trials:
        # Return a "safe empty" result that won't crash downstream
        return {
            "scores_test_full": np.nan,
            "scores_train": [],
            "scores_test": [],
            "Rsquared_test_full": np.nan,
            "acc_test_full": None,
            "balanced_acc_test_full": None,
            "weights": None,
            "intercepts": None,
            "target": ys,
            "predictions_test": [None for _ in range(n_trials)],
            "regressors": Xs if save_binned else None,
            "idxes_test": None,
            "idxes_train": None,
            "best_params": None,
            "n_folds": n_folds,
        }

    bins_per_trial = len(Xs[0])

    # containers
    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []
    predictions = [None for _ in range(n_trials)]
    predictions_to_save = [None for _ in range(n_trials)]

    if rng_seed is not None:
        np.random.seed(rng_seed)
    indices = np.arange(n_trials)

    # scoring
    scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score

    # Determine whether we do hyperparam search
    do_grid = hyperparam_grid is not None and len(hyperparam_grid) > 0
    if do_grid:
        key = list(hyperparam_grid.keys())[0]  # e.g., "alpha"
        grid = hyperparam_grid[key]
        if len(grid) == 0:
            do_grid = False

    # Outer CV splits: avoid n_splits > n_trials
    if outer_cv:
        # In their code they optionally use 50 folds; that's disastrous for small subgroups.
        # For subgroup decoding, cap splits at min(n_folds, n_trials) with floor 2.
        outer_splits = max(2, min(n_folds, n_trials))
        get_kfold = lambda: KFold(n_splits=outer_splits, shuffle=shuffle).split(indices)

        # For logistic regression, keep their fold-sampling criteria; for Ridge/Lasso, no need.
        if estimator == sklm.LogisticRegression:
            assert logisticreg_criteria(ys)
            isysat = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=2)
            _, _, outer_kfold_iter = sample_folds(ys, get_kfold, isysat)
        else:
            outer_kfold_iter = [(train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(get_kfold())]
    else:
        outer_kfold = iter([train_test_split(indices, test_size=test_prop, shuffle=shuffle)])
        outer_kfold_iter = [(train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(outer_kfold)]

    # Disallow CV-type estimators as in original
    if estimator in [sklm.RidgeCV, sklm.LassoCV, sklm.LogisticRegressionCV]:
        raise NotImplementedError("the code does not support a CV-type estimator.")

    # loop outer folds
    for train_idxs_outer, test_idxs_outer in outer_kfold_iter:
        # Avoid pathological folds (esp. when subgroup is small)
        if len(test_idxs_outer) < 2 or len(train_idxs_outer) < 2:
            continue

        X_train = [Xs[i] for i in train_idxs_outer]
        y_train = [ys[i] for i in train_idxs_outer]
        X_test = [Xs[i] for i in test_idxs_outer]
        y_test = [ys[i] for i in test_idxs_outer]

        # If no hyperparam grid: fit once
        if not do_grid:
            X_train_array = np.vstack(X_train)
            y_train_array = np.concatenate(y_train, axis=0)

            sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None

            model = estimator(**estimator_kwargs)
            model.fit(X_train_array, y_train_array, sample_weight=sample_weight)

            best_alpha = None
            best_param = {}

        else:
            # Nested CV hyperparam selection (same structure as their original)
            idx_inner = np.arange(len(X_train))
            inner_splits = max(2, min(n_folds, len(X_train)))
            get_kfold_inner = lambda: KFold(n_splits=inner_splits, shuffle=shuffle).split(idx_inner)

            if estimator == sklm.LogisticRegression:
                assert logisticreg_criteria(y_train, min_unique_counts=2)
                isysat_inner = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=1)
                _, _, inner_kfold_iter = sample_folds(y_train, get_kfold_inner, isysat_inner)
            else:
                inner_kfold_iter = [(tr, te) for _, (tr, te) in enumerate(get_kfold_inner())]

            # Evaluate alphas
            r2s = np.full([inner_splits, len(grid)], np.nan)

            for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold_iter):
                if len(test_idxs_inner) < 2 or len(train_idxs_inner) < 2:
                    continue

                X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
                y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
                X_test_inner = np.vstack([X_train[i] for i in test_idxs_inner])
                y_test_inner = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)

                for i_alpha, alpha in enumerate(grid):
                    sample_weight = compute_sample_weight("balanced", y=y_train_inner) if balanced_weight else None
                    model_inner = estimator(**{**estimator_kwargs, key: alpha})
                    model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)
                    pred_test_inner = model_inner.predict(X_test_inner)

                    sc = scoring_f(y_test_inner, pred_test_inner)
                    if np.isfinite(sc):
                        r2s[ifold, i_alpha] = sc

            r2s_avg = np.nanmean(r2s, axis=0)
            if not np.any(np.isfinite(r2s_avg)):
                # fallback: choose first grid value
                best_alpha = grid[0]
            else:
                best_alpha = grid[int(np.nanargmax(r2s_avg))]

            # Fit best on full outer train
            X_train_array = np.vstack(X_train)
            y_train_array = np.concatenate(y_train, axis=0)
            sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None

            model = estimator(**{**estimator_kwargs, key: best_alpha})
            model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
            best_param = {key: best_alpha}

        # evaluate train
        y_pred_train = model.predict(X_train_array)
        sc_tr = scoring_f(y_train_array, y_pred_train)
        if np.isfinite(sc_tr):
            scores_train.append(sc_tr)

        # evaluate test
        y_true = np.concatenate(y_test, axis=0)
        y_pred = model.predict(np.vstack(X_test))

        sc_te = scoring_f(y_true, y_pred)
        if np.isfinite(sc_te):
            scores_test.append(sc_te)

        # save predictions
        if isinstance(model, sklm.LogisticRegression):
            y_pred_probs = model.predict_proba(np.vstack(X_test))[:, 1]
        else:
            y_pred_probs = None

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

        # save interest
        idxes_test.append(test_idxs_outer)
        idxes_train.append(train_idxs_outer)
        weights.append(model.coef_)
        intercepts.append(model.intercept_ if getattr(model, "fit_intercept", False) else None)
        best_params.append(best_param)

    # Full score (guard against missing preds)
    keep = [i for i in range(n_trials) if predictions[i] is not None]
    if len(keep) < 2:
        ys_true_full = np.concatenate(ys, axis=0)
        ys_pred_full = np.concatenate([p if p is not None else np.array([np.nan]) for p in predictions], axis=0)
    else:
        ys_true_full = np.concatenate([ys[i] for i in keep], axis=0)
        ys_pred_full = np.concatenate([predictions[i] for i in keep], axis=0)

    outdict = dict()
    outdict["scores_test_full"] = scoring_f(ys_true_full, ys_pred_full)
    outdict["scores_train"] = scores_train
    outdict["scores_test"] = scores_test
    outdict["Rsquared_test_full"] = r2_score(ys_true_full, ys_pred_full) if ys_true_full.size > 1 else np.nan

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


def decode_cv(
    ys, Xs, estimator, estimator_kwargs,
    balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
    n_folds=5, save_binned=False, save_predictions=True, verbose=False,
    shuffle=True, outer_cv=True, rng_seed=None, use_cv_sklearn_method=False,
    # --------- NEW (safe defaults) ----------
    min_trials=10,
    debug_tag="",
    debug_warnings=True,
    sanitize_X=True,
    standardize_X=True,
    var_eps=1e-12,
    # ---------------------------------------
):
    """
    Findling-compatible decode_cv, patched for subgroup decoding stability.

    Key behavior change vs your current patched version:
      - If hyperparam_grid is None or {}: DO NOT grid search, just fit one model.
        (This matches their YAML `hparam_grid: {}` behavior.)
    """

    import numpy as np
    import warnings
    import sklearn.linear_model as sklm
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.utils.class_weight import compute_sample_weight

    # ---------------- helpers ----------------
    def _stats(name, X):
        X = np.asarray(X)
        return {
            "name": name,
            "shape": tuple(X.shape),
            "dtype": str(X.dtype),
            "max_abs": float(np.nanmax(np.abs(X))) if X.size else None,
            "any_nan": bool(np.isnan(X).any()) if X.size else False,
            "any_inf": bool(np.isinf(X).any()) if X.size else False,
            "min": float(np.nanmin(X)) if X.size else None,
            "max": float(np.nanmax(X)) if X.size else None,
        }

    def _sanitize_matrix(X):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        # remove near-constant columns
        v = X.var(axis=0)
        keep = v > var_eps
        if keep.sum() == 0:
            return None, None
        return X[:, keep], keep

    def _apply_keep(X, keep):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X[:, keep]

    def _zscore_fit(Xtr):
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd < 1e-12] = 1.0
        return mu, sd

    def _zscore_apply(X, mu, sd):
        return (X - mu) / sd

    # ---------------- format data ----------------
    ys, Xs = format_data_for_decoding(ys, Xs)

    n_trials = len(Xs)
    if n_trials < min_trials:
        # Return an “empty but valid” result dict so upstream doesn’t crash
        return {
            "scores_test_full": np.nan,
            "scores_train": [],
            "scores_test": [],
            "Rsquared_test_full": np.nan,
            "weights": None if save_predictions else None,
            "intercepts": None if save_predictions else None,
            "target": ys,
            "predictions_test": [None] * n_trials,
            "regressors": Xs if save_binned else None,
            "idxes_test": None if save_predictions else None,
            "idxes_train": None if save_predictions else None,
            "best_params": None if save_predictions else None,
            "n_folds": n_folds,
        }

    bins_per_trial = len(Xs[0])

    if rng_seed is not None:
        np.random.seed(rng_seed)

    indices = np.arange(n_trials)

    # scoring function; use R2 for linear regression, balanced acc for logistic regression
    scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score

    # containers
    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []
    predictions = [None for _ in range(n_trials)]
    predictions_to_save = [None for _ in range(n_trials)]

    # outer folds
    if outer_cv:
        n_splits_outer = min(n_folds if not use_cv_sklearn_method else 50, n_trials)
        n_splits_outer = max(2, n_splits_outer)
        outer_kfold_iter = list(KFold(n_splits_outer, shuffle=shuffle).split(indices))
    else:
        train_idxs, test_idxs = train_test_split(indices, test_size=test_prop, shuffle=shuffle)
        outer_kfold_iter = [(train_idxs, test_idxs)]

    # ----------------------------------------------------------
    # IMPORTANT: Handle empty grid (Findling YAML: hparam_grid: {})
    # ----------------------------------------------------------
    do_grid = (hyperparam_grid is not None) and (len(hyperparam_grid) > 0)

    # decide alpha candidates
    if do_grid:
        key = list(hyperparam_grid.keys())[0]
        grid_vals = list(hyperparam_grid[key])
        if len(grid_vals) == 0:
            do_grid = False
    else:
        key = None
        grid_vals = None

    # pick a stable default alpha when no grid
    # (this is the behavior you want to match when YAML uses {})
    default_alpha = estimator_kwargs.get("alpha", 1.0)

    # to reduce solver weirdness on some BLAS builds, force Ridge solver if user didn’t specify
    if estimator in [sklm.Ridge] and "solver" not in estimator_kwargs:
        estimator_kwargs = dict(estimator_kwargs)
        estimator_kwargs["solver"] = "svd"

    # ---------------- MAIN LOOP ----------------
    for outer_fold, (train_idxs_outer, test_idxs_outer) in enumerate(outer_kfold_iter):
        X_train = [Xs[i] for i in train_idxs_outer]
        y_train = [ys[i] for i in train_idxs_outer]
        X_test = [Xs[i] for i in test_idxs_outer]
        y_test = [ys[i] for i in test_idxs_outer]

        # stack
        X_train_array = np.vstack(X_train)
        y_train_array = np.concatenate(y_train, axis=0)
        X_test_array = np.vstack(X_test)
        y_true = np.concatenate(y_test, axis=0)

        # sample weights
        sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None

        # sanitize + standardize (same transform for all alphas on this fold)
        keep = None
        if sanitize_X:
            tmp = _sanitize_matrix(X_train_array)
            if tmp[0] is None:
                continue
            X_train_array, keep = tmp
            X_test_array = _apply_keep(X_test_array, keep)

        if standardize_X:
            mu, sd = _zscore_fit(X_train_array)
            X_train_array = _zscore_apply(X_train_array, mu, sd)
            X_test_array = _zscore_apply(X_test_array, mu, sd)

        # ---------------- choose alpha ----------------
        if not do_grid:
            best_alpha = float(default_alpha)
        else:
            # inner CV on trial-level (like Findling)
            idx_inner = np.arange(len(X_train))
            n_splits_inner = min(n_folds, len(idx_inner))
            n_splits_inner = max(2, n_splits_inner)
            inner_iter = list(KFold(n_splits_inner, shuffle=shuffle).split(idx_inner))

            r2s = np.zeros([len(inner_iter), len(grid_vals)], dtype=float)

            for ifold, (tr_in, te_in) in enumerate(inner_iter):
                X_tr_in = np.vstack([X_train[i] for i in tr_in])
                y_tr_in = np.concatenate([y_train[i] for i in tr_in], axis=0)
                X_te_in = np.vstack([X_train[i] for i in te_in])
                y_te_in = np.concatenate([y_train[i] for i in te_in], axis=0)

                # same sanitize+standardize inside inner folds
                if sanitize_X:
                    tmp2 = _sanitize_matrix(X_tr_in)
                    if tmp2[0] is None:
                        continue
                    X_tr_in, keep2 = tmp2
                    X_te_in = _apply_keep(X_te_in, keep2)
                else:
                    keep2 = None

                if standardize_X:
                    mu2, sd2 = _zscore_fit(X_tr_in)
                    X_tr_in = _zscore_apply(X_tr_in, mu2, sd2)
                    X_te_in = _zscore_apply(X_te_in, mu2, sd2)

                sw_in = compute_sample_weight("balanced", y=y_tr_in) if balanced_weight else None

                for ia, alpha in enumerate(grid_vals):
                    model_inner = estimator(**{**estimator_kwargs, key: float(alpha)})

                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always", RuntimeWarning)
                        model_inner.fit(X_tr_in, y_tr_in, sample_weight=sw_in)
                        if debug_warnings and w:
                            print(
                                f"\n[decode_cv WARNING] outer_fold={outer_fold} inner_fold={ifold} "
                                f"alpha={alpha} model_inner.fit"
                            )
                            print(f"  debug_tag: {debug_tag}")
                            print(f"  warning: {w[0].message}")
                            print(f"  X_train_inner: {_stats('X_train_inner', X_tr_in)}")
                            print(f"  y_train_inner: {_stats('y_train_inner', y_tr_in)}\n")

                    pred = model_inner.predict(X_te_in)
                    r2s[ifold, ia] = scoring_f(y_te_in, pred)

            r2s_avg = np.nanmean(r2s, axis=0)
            best_alpha = float(grid_vals[int(np.nanargmax(r2s_avg))])

        # ---------------- fit final model on outer train ----------------
        model = estimator(**({**estimator_kwargs, **({} if key is None else {key: best_alpha})}))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)
            model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
            if debug_warnings and w:
                print(f"\n[decode_cv WARNING] outer_fold={outer_fold} FINAL_FIT")
                print(f"  debug_tag: {debug_tag}")
                print(f"  warning: {w[0].message}")
                print(f"  X_train: {_stats('X_train', X_train_array)}")
                print(f"  y_train: {_stats('y_train', y_train_array)}\n")

        # train score
        y_pred_train = model.predict(X_train_array)
        scores_train.append(scoring_f(y_train_array, y_pred_train))

        # test score
        y_pred = model.predict(X_test_array)
        scores_test.append(scoring_f(y_true, y_pred))

        # save predictions per trial (Findling-style)
        for i_fold, i_global in enumerate(test_idxs_outer):
            if bins_per_trial == 1:
                predictions[i_global] = np.array([y_pred[i_fold]])
                predictions_to_save[i_global] = np.array([y_pred[i_fold]])
            else:
                # NOTE: here X_test is list-of-trials; recompute per-trial
                # But must apply keep + zscore transform consistently:
                Xt = np.asarray(X_test[i_fold], dtype=np.float64)
                Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
                if keep is not None:
                    Xt = Xt[:, keep]
                if standardize_X:
                    Xt = _zscore_apply(Xt, mu, sd)
                predictions[i_global] = model.predict(Xt)
                predictions_to_save[i_global] = predictions[i_global]

        # save fold metadata
        idxes_test.append(test_idxs_outer)
        idxes_train.append(train_idxs_outer)
        weights.append(model.coef_)
        intercepts.append(model.intercept_ if getattr(model, "fit_intercept", False) else None)
        best_params.append({("alpha" if key is None else key): best_alpha})

    # ---------------- finalize ----------------
    ok = [p is not None for p in predictions]
    ys_true_full = np.concatenate([ys[i] for i in range(n_trials) if ok[i]], axis=0)
    ys_pred_full = np.concatenate([predictions[i] for i in range(n_trials) if ok[i]], axis=0)

    outdict = dict()
    outdict["scores_test_full"] = scoring_f(ys_true_full, ys_pred_full) if ys_true_full.size else np.nan
    outdict["scores_train"] = scores_train
    outdict["scores_test"] = scores_test
    outdict["Rsquared_test_full"] = r2_score(ys_true_full, ys_pred_full) if ys_true_full.size else np.nan

    if estimator == sklm.LogisticRegression:
        outdict["acc_test_full"] = accuracy_score(ys_true_full, ys_pred_full) if ys_true_full.size else np.nan
        outdict["balanced_acc_test_full"] = balanced_accuracy_score(ys_true_full, ys_pred_full) if ys_true_full.size else np.nan

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




def decode_cv(
    ys, Xs, estimator, estimator_kwargs,
    balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
    n_folds=5, save_binned=False, save_predictions=True, verbose=False,
    shuffle=True, outer_cv=True, rng_seed=None, use_cv_sklearn_method=False,
    # --- subgroup safety ---
    min_trials=10,
    debug_tag="",
    warn_print=True,
):
    """
    Findling-style decode_cv with correct handling of hparam_grid: {} (NO grid search).
    Also stabilizes Ridge fits for small subgroup folds.
    """
    import numpy as np
    import warnings
    import sklearn.linear_model as sklm
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.utils.class_weight import compute_sample_weight

    # -------------- helpers --------------
    def _stats(name, X):
        X = np.asarray(X)
        return {
            "name": name,
            "shape": tuple(X.shape),
            "dtype": str(X.dtype),
            "max_abs": float(np.nanmax(np.abs(X))) if X.size else None,
            "any_nan": bool(np.isnan(X).any()) if X.size else False,
            "any_inf": bool(np.isinf(X).any()) if X.size else False,
            "min": float(np.nanmin(X)) if X.size else None,
            "max": float(np.nanmax(X)) if X.size else None,
        }

    def _nan_to_num(X):
        X = np.asarray(X, dtype=np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _standardize_fit(X):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-12] = 1.0
        return mu, sd

    def _standardize_apply(X, mu, sd):
        return (X - mu) / sd

    # -------------- format data --------------
    ys, Xs = format_data_for_decoding(ys, Xs)
    n_trials = len(Xs)

    if n_trials < min_trials:
        return {
            "scores_test_full": np.nan,
            "scores_train": [],
            "scores_test": [],
            "Rsquared_test_full": np.nan,
            "weights": None if save_predictions else None,
            "intercepts": None if save_predictions else None,
            "target": ys,
            "predictions_test": [None] * n_trials,
            "regressors": Xs if save_binned else None,
            "idxes_test": None if save_predictions else None,
            "idxes_train": None if save_predictions else None,
            "best_params": None if save_predictions else None,
            "n_folds": n_folds,
        }

    bins_per_trial = len(Xs[0])
    if rng_seed is not None:
        np.random.seed(rng_seed)

    indices = np.arange(n_trials)

    scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score

    # -------------- outer folds --------------
    if outer_cv:
        n_splits_outer = min(n_folds if not use_cv_sklearn_method else 50, n_trials)
        n_splits_outer = max(2, n_splits_outer)
        outer_kfold_iter = list(KFold(n_splits_outer, shuffle=shuffle).split(indices))
    else:
        tr, te = train_test_split(indices, test_size=test_prop, shuffle=shuffle)
        outer_kfold_iter = [(tr, te)]

    # -------------- IMPORTANT: Findling behavior for {} --------------
    do_grid = (hyperparam_grid is not None) and (len(hyperparam_grid) > 0)

    # If YAML has hparam_grid: {} then hyperparam_grid == {} and do_grid becomes False.
    # That MUST mean: fit once with default alpha.
    if not do_grid:
        grid_key = None
        grid_vals = None
        # Stable default for Ridge if user didn't provide alpha
        default_alpha = float(estimator_kwargs.get("alpha", 1.0))
    else:
        grid_key = list(hyperparam_grid.keys())[0]
        grid_vals = list(hyperparam_grid[grid_key])
        if len(grid_vals) == 0:
            do_grid = False
            grid_key = None
            grid_vals = None
            default_alpha = float(estimator_kwargs.get("alpha", 1.0))

    # Stabilize Ridge
    if estimator == sklm.Ridge and "solver" not in estimator_kwargs:
        estimator_kwargs = dict(estimator_kwargs)
        estimator_kwargs["solver"] = "svd"

    # -------------- containers --------------
    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []
    predictions = [None for _ in range(n_trials)]
    predictions_to_save = [None for _ in range(n_trials)]

    # -------------- main loop --------------
    for outer_fold, (train_idx, test_idx) in enumerate(outer_kfold_iter):

        X_train_list = [Xs[i] for i in train_idx]
        y_train_list = [ys[i] for i in train_idx]
        X_test_list = [Xs[i] for i in test_idx]
        y_test_list = [ys[i] for i in test_idx]

        Xtr = _nan_to_num(np.vstack(X_train_list))
        ytr = np.concatenate(y_train_list, axis=0)
        Xte = _nan_to_num(np.vstack(X_test_list))
        yte = np.concatenate(y_test_list, axis=0)

        # standardize using train only
        mu, sd = _standardize_fit(Xtr)
        Xtr = _standardize_apply(Xtr, mu, sd)
        Xte = _standardize_apply(Xte, mu, sd)

        sample_weight = compute_sample_weight("balanced", y=ytr) if balanced_weight else None

        # -------- choose alpha --------
        if not do_grid:
            best_alpha = default_alpha
        else:
            # inner CV (trial-wise)
            idx_inner = np.arange(len(X_train_list))
            n_splits_inner = min(n_folds, len(idx_inner))
            n_splits_inner = max(2, n_splits_inner)
            inner_iter = list(KFold(n_splits_inner, shuffle=shuffle).split(idx_inner))

            scores_alpha = np.zeros(len(grid_vals), dtype=float)

            # average over inner folds
            fold_scores = np.zeros((len(inner_iter), len(grid_vals)), dtype=float)

            for ifold, (tri, tei) in enumerate(inner_iter):
                X_tr_in = _nan_to_num(np.vstack([X_train_list[i] for i in tri]))
                y_tr_in = np.concatenate([y_train_list[i] for i in tri], axis=0)
                X_te_in = _nan_to_num(np.vstack([X_train_list[i] for i in tei]))
                y_te_in = np.concatenate([y_train_list[i] for i in tei], axis=0)

                mu2, sd2 = _standardize_fit(X_tr_in)
                X_tr_in = _standardize_apply(X_tr_in, mu2, sd2)
                X_te_in = _standardize_apply(X_te_in, mu2, sd2)

                sw_in = compute_sample_weight("balanced", y=y_tr_in) if balanced_weight else None

                for ia, alpha in enumerate(grid_vals):
                    model_inner = estimator(**{**estimator_kwargs, grid_key: float(alpha)})

                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always", RuntimeWarning)
                        model_inner.fit(X_tr_in, y_tr_in, sample_weight=sw_in)
                        if warn_print and w:
                            print(f"\n[decode_cv WARNING] outer_fold={outer_fold} inner_fold={ifold} alpha={alpha} model_inner.fit")
                            print(f"  debug_tag: {debug_tag}")
                            print(f"  warning: {w[0].message}")
                            print(f"  X_train_inner: {_stats('X_train_inner', X_tr_in)}")
                            print(f"  y_train_inner: {_stats('y_train_inner', y_tr_in)}\n")

                    pred = model_inner.predict(X_te_in)
                    fold_scores[ifold, ia] = scoring_f(y_te_in, pred)

            scores_alpha = np.nanmean(fold_scores, axis=0)
            best_alpha = float(grid_vals[int(np.nanargmax(scores_alpha))])

        # -------- fit final model on outer train --------
        if grid_key is None:
            model = estimator(**{**estimator_kwargs, "alpha": float(best_alpha)} if estimator == sklm.Ridge else estimator_kwargs)
        else:
            model = estimator(**{**estimator_kwargs, grid_key: float(best_alpha)})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)
            model.fit(Xtr, ytr, sample_weight=sample_weight)
            if warn_print and w:
                print(f"\n[decode_cv WARNING] outer_fold={outer_fold} FINAL_FIT")
                print(f"  debug_tag: {debug_tag}")
                print(f"  warning: {w[0].message}")
                print(f"  X_train: {_stats('X_train', Xtr)}")
                print(f"  y_train: {_stats('y_train', ytr)}\n")

        # scores
        y_pred_train = model.predict(Xtr)
        scores_train.append(scoring_f(ytr, y_pred_train))

        y_pred = model.predict(Xte)
        scores_test.append(scoring_f(yte, y_pred))

        # per-trial predictions
        for i_fold, i_global in enumerate(test_idx):
            if bins_per_trial == 1:
                predictions[i_global] = np.array([y_pred[i_fold]])
                predictions_to_save[i_global] = np.array([y_pred[i_fold]])
            else:
                Xt = _nan_to_num(X_test_list[i_fold])
                Xt = _standardize_apply(Xt, mu, sd)
                predictions[i_global] = model.predict(Xt)
                predictions_to_save[i_global] = predictions[i_global]

        # save metadata
        idxes_test.append(test_idx)
        idxes_train.append(train_idx)
        weights.append(model.coef_)
        intercepts.append(model.intercept_ if getattr(model, "fit_intercept", False) else None)
        if save_predictions:
            if grid_key is None:
                best_params.append({"alpha": float(best_alpha)})
            else:
                best_params.append({grid_key: float(best_alpha)})

    # finalize full score
    ok = [p is not None for p in predictions]
    ys_true_full = np.concatenate([ys[i] for i in range(n_trials) if ok[i]], axis=0)
    ys_pred_full = np.concatenate([predictions[i] for i in range(n_trials) if ok[i]], axis=0)

    out = dict()
    out["scores_test_full"] = scoring_f(ys_true_full, ys_pred_full) if ys_true_full.size else np.nan
    out["scores_train"] = scores_train
    out["scores_test"] = scores_test
    out["Rsquared_test_full"] = r2_score(ys_true_full, ys_pred_full) if ys_true_full.size else np.nan

    if estimator == sklm.LogisticRegression:
        out["acc_test_full"] = accuracy_score(ys_true_full, ys_pred_full) if ys_true_full.size else np.nan
        out["balanced_acc_test_full"] = balanced_accuracy_score(ys_true_full, ys_pred_full) if ys_true_full.size else np.nan

    out["weights"] = weights if save_predictions else None
    out["intercepts"] = intercepts if save_predictions else None
    out["target"] = ys
    out["predictions_test"] = predictions_to_save
    out["regressors"] = Xs if save_binned else None
    out["idxes_test"] = idxes_test if save_predictions else None
    out["idxes_train"] = idxes_train if save_predictions else None
    out["best_params"] = best_params if save_predictions else None
    out["n_folds"] = n_folds
    if hasattr(model, "classes_"):
        out["classes_"] = model.classes_
    return out


# def fit_target(
#         all_data, all_targets, all_trials, n_runs, all_neurometrics=None, pseudo_ids=None,
#         base_rng_seed=0,
#         # NEW:
#         debug_tag="",
#         force_no_grid_if_empty_yaml=True,
# ):
#     """
#     Patched fit_target:
#       - passes debug_tag into decode_cv so you can see group/pseudo/run/seed
#       - optionally forces hyperparam_grid=None when yaml grid is {} (so decode_cv never enters inner loop)
#     """
#     if pseudo_ids is None:
#         pseudo_ids = [-1]
#     if not all_neurometrics:
#         all_neurometrics = [None] * len(all_targets)
#
#     fit_results = []
#
#     for targets, data, trials, neurometrics, pseudo_id in zip(
#             all_targets, all_data, all_trials, all_neurometrics, pseudo_ids):
#
#         for i_run in range(n_runs):
#             if pseudo_id == -1:
#                 rng_seed = base_rng_seed + i_run
#             else:
#                 rng_seed = base_rng_seed + pseudo_id * n_runs + i_run
#
#             # ---- key line: match Findling YAML behavior ----
#             hgrid = config.get("hparam_grid", None)
#             if force_no_grid_if_empty_yaml and (hgrid is not None) and (len(hgrid) == 0):
#                 hgrid = None  # IMPORTANT: ensures no inner loop at all
#
#             fit_result = decode_cv(
#                 ys=targets,
#                 Xs=data,
#                 estimator=config['estimator'],
#                 estimator_kwargs=config['estimator_kwargs'],
#                 hyperparam_grid=hgrid,
#                 save_binned=False,
#                 save_predictions=config['save_predictions'],
#                 shuffle=config['shuffle'],
#                 balanced_weight=config['balanced_weighting'],
#                 rng_seed=rng_seed,
#                 use_cv_sklearn_method=config['use_native_sklearn_for_hyperparam_estimation'],
#                 # subgroup safety:
#                 min_trials=10,
#                 debug_tag=f"{debug_tag} pseudo={pseudo_id} run={i_run} seed={rng_seed}",
#                 warn_print=True,
#             )
#
#             fit_result["trials_df"] = trials
#             fit_result["pseudo_id"] = pseudo_id
#             fit_result["run_id"] = i_run
#
#             if neurometrics:
#                 fit_result["full_neurometric"], fit_result["fold_neurometric"] = get_neurometric_parameters(
#                     fit_result, trialsdf=neurometrics, compute_on_each_fold=config['compute_neuro_on_each_fold']
#                 )
#             else:
#                 fit_result["full_neurometric"] = None
#                 fit_result["fold_neurometric"] = None
#
#             fit_results.append(fit_result)
#
#     return fit_results
#

# =========================
# decode_cv debug context
# =========================

_DECODE_CV_CONTEXT = {}

def set_decode_cv_context(**kwargs):
    """
    Called by fit_target before decode_cv so decode_cv can print useful tags.
    Example:
        set_decode_cv_context(group_label="fast", pseudo_id=3, run_id=1, rng_seed=123)
    """
    global _DECODE_CV_CONTEXT
    _DECODE_CV_CONTEXT = dict(kwargs)


def decode_cv(
    ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
    n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
    rng_seed=None, use_cv_sklearn_method=False
):
    """
    Drop-in replacement for Findling et al decode_cv, made robust for subgroup decoding.

    Key changes (behaviorally safe):
    - If hyperparam_grid is None or {}, we SKIP hyperparameter tuning (fits once per outer fold).
      This matches their YAML usage: hparam_grid: {}
    - Caps n_folds so that test folds are not size 1 (prevents R^2 undefined + numeric blowups).
    - Skips outer/inner folds that would yield <2 samples (R^2 undefined).
    - Sanitizes X each fold: drops zero-variance features, standardizes, converts non-finite to 0.
    - Catches RuntimeWarnings and prints fold-local debug (with group/pseudo/run/seed tags).

    Signature + output keys kept compatible with their usage.
    """

    import numpy as np
    import warnings

    from sklearn import linear_model as sklm
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.utils.class_weight import compute_sample_weight

    # IMPORTANT: rely on the project helper
    # (this exists in Findling repo)
    ys, Xs = format_data_for_decoding(ys, Xs)

    # ---------------------------
    # small helpers
    # ---------------------------
    def _ctx_str():
        c = dict(_DECODE_CV_CONTEXT) if isinstance(_DECODE_CV_CONTEXT, dict) else {}
        # keep keys stable
        group = c.get("group_label", "unknown_group")
        pseudo = c.get("pseudo_id", "NA")
        run = c.get("run_id", "NA")
        seed = c.get("rng_seed", rng_seed)
        return f"group={group} pseudo={pseudo} run={run} seed={seed}"

    def _arr_stats(name, a):
        a = np.asarray(a)
        out = {
            "name": name,
            "shape": tuple(a.shape),
            "dtype": str(a.dtype),
            "any_nan": bool(np.isnan(a).any()) if np.issubdtype(a.dtype, np.floating) else False,
            "any_inf": bool(np.isinf(a).any()) if np.issubdtype(a.dtype, np.floating) else False,
        }
        if a.size > 0 and np.issubdtype(a.dtype, np.number):
            with np.errstate(all="ignore"):
                out["max_abs"] = float(np.nanmax(np.abs(a)))
                out["min"] = float(np.nanmin(a))
                out["max"] = float(np.nanmax(a))
        return out

    def _sanitize_and_standardize(Xtr, Xte, eps=1e-12):
        """
        - convert to float64
        - replace non-finite with 0
        - drop near-constant columns based on TRAIN
        - z-score using TRAIN mean/std
        """
        Xtr = np.asarray(Xtr, dtype=np.float64)
        Xte = np.asarray(Xte, dtype=np.float64)

        Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
        Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)

        var = Xtr.var(axis=0)
        keep = var > eps
        if int(keep.sum()) == 0:
            raise FloatingPointError("All features are constant/invalid in this fold")

        Xtr = Xtr[:, keep]
        Xte = Xte[:, keep]

        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd < eps] = 1.0

        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        return Xtr, Xte, keep

    def _warn_print(where, outer_fold=None, inner_fold=None, alpha=None, w=None, X=None, y=None):
        if w is None or len(w) == 0:
            return
        msg = str(w[0].message)
        print(f"\n[decode_cv WARNING] outer_fold={outer_fold} inner_fold={inner_fold} alpha={alpha} {where}")
        print(f"  debug_tag: {_ctx_str()}")
        print(f"  warning: {msg}")
        if X is not None:
            print(f"  X: {_arr_stats('X', X)}")
        if y is not None:
            print(f"  y: {_arr_stats('y', y)}")
        print("")

    # ---------------------------
    # prepare data & scoring
    # ---------------------------
    n_trials = len(Xs)
    if n_trials == 0:
        raise ValueError("decode_cv: no trials after format_data_for_decoding")

    bins_per_trial = len(Xs[0]) if isinstance(Xs[0], (list, tuple, np.ndarray)) else 1

    # Remove trials with any non-finite in their per-trial arrays or y
    good = []
    for x, y in zip(Xs, ys):
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        good.append(np.isfinite(x_arr).all() and np.isfinite(y_arr).all())
    Xs = [x for x, g in zip(Xs, good) if g]
    ys = [y for y, g in zip(ys, good) if g]

    n_trials = len(Xs)
    if n_trials < 2:
        raise ValueError(f"decode_cv: too few usable trials after finite-check (n_trials={n_trials})")

    # set RNG
    if rng_seed is not None:
        np.random.seed(rng_seed)

    scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score
    use_balanced = bool(balanced_weight) and (estimator == sklm.LogisticRegression)

    # ---------------------------
    # handle empty hyperparam grid
    # ---------------------------
    do_grid = True
    if hyperparam_grid is None or (isinstance(hyperparam_grid, dict) and len(hyperparam_grid) == 0):
        do_grid = False

    if do_grid:
        # Findling code assumes exactly one key
        if not isinstance(hyperparam_grid, dict) or len(hyperparam_grid) != 1:
            raise ValueError("decode_cv: hyperparam_grid must be a dict with exactly one key")
        key = list(hyperparam_grid.keys())[0]
        grid_vals = list(hyperparam_grid[key])
        if len(grid_vals) == 0:
            do_grid = False

    # ---------------------------
    # build outer folds (trial-level)
    # ---------------------------
    indices = np.arange(n_trials)

    # Cap n_folds to avoid 1-trial test folds (THIS IS THE BIG FIX)
    # Guarantee each test fold has >=2 trials when possible.
    if outer_cv:
        # choose folds based on how many trials we have
        max_reasonable_folds = max(2, n_trials // 2)  # ensures >=2 trials in test fold
        n_folds_eff = int(min(n_folds, max_reasonable_folds))
        if n_folds_eff < 2:
            n_folds_eff = 2

        # If someone turns on use_cv_sklearn_method, their old code used 50 folds (bad for subgroups).
        # We STILL cap it.
        n_outer = n_folds_eff
        outer_kfold_iter = KFold(n_splits=n_outer, shuffle=shuffle).split(indices)

        # For LogisticRegression: ensure fold has both classes, use their helper
        if estimator == sklm.LogisticRegression:
            assert logisticreg_criteria(ys)
            isysat = lambda y_list: logisticreg_criteria(y_list, min_unique_counts=2)
            sample_count, _, outer_kfold_iter = sample_folds(ys, lambda: KFold(n_outer, shuffle=shuffle).split(indices), isysat)
            if sample_count > 1 and verbose:
                print(f"sampled outer folds {sample_count} times to ensure enough targets")

    else:
        train_idxs, test_idxs = train_test_split(indices, test_size=test_prop, shuffle=shuffle, random_state=rng_seed)
        outer_kfold_iter = [(train_idxs, test_idxs)]

    # ---------------------------
    # outputs
    # ---------------------------
    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []
    predictions = [None for _ in range(n_trials)]
    predictions_to_save = [None for _ in range(n_trials)]

    # ---------------------------
    # main outer loop
    # ---------------------------
    for outer_fold, (train_idxs_outer, test_idxs_outer) in enumerate(outer_kfold_iter):

        # trial-level splits
        X_train_trials = [Xs[i] for i in train_idxs_outer]
        y_train_trials = [ys[i] for i in train_idxs_outer]
        X_test_trials = [Xs[i] for i in test_idxs_outer]
        y_test_trials = [ys[i] for i in test_idxs_outer]

        # stack into samples
        X_train_array = np.vstack(X_train_trials)
        y_train_array = np.concatenate(y_train_trials, axis=0)
        X_test_array = np.vstack(X_test_trials)
        y_test_array = np.concatenate(y_test_trials, axis=0)

        # If test has <2 samples, R2 is undefined → skip fold
        if estimator != sklm.LogisticRegression and y_test_array.shape[0] < 2:
            if verbose:
                print(f"[decode_cv] skip outer_fold={outer_fold}: test samples <2 ({y_test_array.shape[0]}) [{_ctx_str()}]")
            continue

        # compute sample_weight if needed
        sample_weight = compute_sample_weight("balanced", y=y_train_array) if use_balanced else None

        # ---------------------------
        # inner loop (grid search) OR direct fit if grid empty
        # ---------------------------
        best_alpha = None

        if do_grid:
            # inner folds on TRIALS (not samples) to avoid leaking within trial
            idx_inner = np.arange(len(X_train_trials))

            # cap inner folds too
            max_inner = max(2, len(idx_inner) // 2) if len(idx_inner) >= 4 else 2
            n_inner = int(min(n_folds, max_inner))
            if n_inner < 2:
                n_inner = 2

            get_kfold_inner = lambda: KFold(n_splits=n_inner, shuffle=shuffle).split(idx_inner)

            if estimator == sklm.LogisticRegression:
                assert logisticreg_criteria(y_train_trials, min_unique_counts=2)
                isysat_inner = lambda y_list: logisticreg_criteria(y_list, min_unique_counts=1)
            else:
                isysat_inner = lambda y_list: True

            sample_count, _, inner_kfold_iter = sample_folds(y_train_trials, get_kfold_inner, isysat_inner)
            if sample_count > 1 and verbose:
                print(f"sampled inner folds {sample_count} times to ensure enough targets")

            # evaluate each alpha
            fold_scores = []
            for alpha in grid_vals:
                per_fold = []
                for inner_fold, (train_i, test_i) in enumerate(inner_kfold_iter):

                    Xtr_i = np.vstack([X_train_trials[i] for i in train_i])
                    ytr_i = np.concatenate([y_train_trials[i] for i in train_i], axis=0)
                    Xte_i = np.vstack([X_train_trials[i] for i in test_i])
                    yte_i = np.concatenate([y_train_trials[i] for i in test_i], axis=0)

                    if estimator != sklm.LogisticRegression and yte_i.shape[0] < 2:
                        continue  # skip invalid inner fold

                    try:
                        Xtr_i, Xte_i, _keep = _sanitize_and_standardize(Xtr_i, Xte_i)
                        model_inner = estimator(**{**estimator_kwargs, key: alpha})

                        with warnings.catch_warnings(record=True) as w:
                            warnings.simplefilter("always", RuntimeWarning)
                            model_inner.fit(Xtr_i, ytr_i, sample_weight=sample_weight)
                            _warn_print(
                                "model_inner.fit",
                                outer_fold=outer_fold, inner_fold=inner_fold, alpha=alpha,
                                w=w, X=Xtr_i, y=ytr_i
                            )

                        pred_i = model_inner.predict(Xte_i)
                        per_fold.append(scoring_f(yte_i, pred_i))

                    except Exception as e:
                        if verbose:
                            print(f"[decode_cv] inner fold error outer={outer_fold} inner={inner_fold} alpha={alpha}: {e} [{_ctx_str()}]")
                        continue

                # mean over valid inner folds
                if len(per_fold) == 0:
                    fold_scores.append(-np.inf)
                else:
                    fold_scores.append(float(np.mean(per_fold)))

            best_alpha = grid_vals[int(np.argmax(fold_scores))]

        # ---------------------------
        # final fit on outer train
        # ---------------------------
        try:
            Xtr, Xte, _keep = _sanitize_and_standardize(X_train_array, X_test_array)

            if do_grid:
                model = estimator(**{**estimator_kwargs, key: best_alpha})
            else:
                # no grid search: just use estimator_kwargs as-is
                model = estimator(**estimator_kwargs)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", RuntimeWarning)
                model.fit(Xtr, y_train_array, sample_weight=sample_weight)
                _warn_print(
                    "FINAL_FIT",
                    outer_fold=outer_fold, inner_fold=None, alpha=best_alpha,
                    w=w, X=Xtr, y=y_train_array
                )

            # predict
            with warnings.catch_warnings(record=True) as w_pred:
                warnings.simplefilter("always", RuntimeWarning)
                y_pred_train = model.predict(Xtr)
                y_pred_test = model.predict(Xte)
                _warn_print(
                    "PREDICT",
                    outer_fold=outer_fold, inner_fold=None, alpha=best_alpha,
                    w=w_pred, X=Xte, y=y_test_array
                )

            # scores
            scores_train.append(scoring_f(y_train_array, y_pred_train))
            scores_test.append(scoring_f(y_test_array, y_pred_test))

            # store per-trial predictions (trial-level container)
            # We do the same logic as Findling’s code: if bins_per_trial==1 store scalar.
            # Otherwise store per-sample prediction for that trial.
            for i_fold, i_global in enumerate(test_idxs_outer):
                if bins_per_trial == 1:
                    predictions[i_global] = np.array([y_pred_test[i_fold]])
                    predictions_to_save[i_global] = predictions[i_global]
                else:
                    # reconstruct per-trial predictions
                    pred_trial = model.predict(np.asarray(X_test_trials[i_fold]))
                    predictions[i_global] = pred_trial
                    predictions_to_save[i_global] = pred_trial

            # save other info
            idxes_test.append(test_idxs_outer)
            idxes_train.append(train_idxs_outer)
            weights.append(getattr(model, "coef_", None))
            intercepts.append(getattr(model, "intercept_", None) if getattr(model, "fit_intercept", False) else None)
            best_params.append({list(hyperparam_grid.keys())[0]: best_alpha} if do_grid else {})

        except Exception as e:
            if verbose:
                print(f"[decode_cv] outer fold failure outer={outer_fold}: {e} [{_ctx_str()}]")
            continue

    # ---------------------------
    # finalize full-score
    # ---------------------------
    # only keep trials where we got predictions
    ok = [i for i in range(n_trials) if predictions[i] is not None]
    if len(ok) == 0:
        raise RuntimeError(f"decode_cv: all outer folds failed/skipped [{_ctx_str()}]")

    ys_true_full = np.concatenate([ys[i] for i in ok], axis=0)
    ys_pred_full = np.concatenate([predictions[i] for i in ok], axis=0)

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

    # for completeness
    try:
        if hasattr(model, "classes_"):
            outdict["classes_"] = model.classes_
    except Exception:
        pass

    return outdict
def fit_target(
        all_data, all_targets, all_trials, n_runs, all_neurometrics=None, pseudo_ids=None,
        base_rng_seed=0,
        # ---------- NEW (optional, safe defaults) ----------
        group_label="unknown_group",
        debug=False,
        min_trials_per_run=10,
):
    """
    Patched fit_target for subgroup runs:
    - sets a decode_cv debug context so warnings identify group/pseudo/run/seed
    - enforces min_trials_per_run (trial-level) before running decode_cv
    - robust to hparam_grid == {} (decode_cv handles it)

    NOTE: This does NOT change external outputs format.
    """

    import numpy as np

    if pseudo_ids is None:
        pseudo_ids = [-1]
    if not all_neurometrics:
        all_neurometrics = [None] * len(all_targets)

    fit_results = []

    # all_targets, all_data, all_trials are lists over pseudo sessions
    for targets, data, trials, neurometrics, pseudo_id in zip(
            all_targets, all_data, all_trials, all_neurometrics, pseudo_ids):

        # -------------------
        # Trial-level sanity
        # -------------------
        # targets and data can be:
        # - list of per-trial arrays (bins_per_trial potentially >1)
        # - or numpy arrays already (rare here)
        n_trials_here = len(trials) if hasattr(trials, "__len__") else None
        if n_trials_here is None:
            raise ValueError("fit_target: trials has no length?")

        if n_trials_here < min_trials_per_run:
            if debug:
                print(f"[fit_target] SKIP pseudo_id={pseudo_id} group={group_label}: n_trials={n_trials_here} < {min_trials_per_run}")
            continue

        # run decoders
        for i_run in range(n_runs):

            # seed for reproducibility (same logic as original)
            if pseudo_id == -1:
                rng_seed = int(base_rng_seed) + i_run
            else:
                rng_seed = int(base_rng_seed) + int(pseudo_id) * int(n_runs) + i_run

            # tell decode_cv who we are (so warnings are meaningful)
            set_decode_cv_context(
                group_label=group_label,
                pseudo_id=int(pseudo_id),
                run_id=int(i_run),
                rng_seed=int(rng_seed),
            )

            # OPTIONAL: skip if too few trials for stable CV
            # (decode_cv also guards fold sizes, but this avoids pointless runs)
            if n_trials_here < max(min_trials_per_run, 2 * 2):
                if debug:
                    print(f"[fit_target] SKIP run (too few trials for CV) pseudo={pseudo_id} group={group_label} n_trials={n_trials_here}")
                continue

            fit_result = decode_cv(
                ys=targets,
                Xs=data,
                estimator=config['estimator'],
                estimator_kwargs=config['estimator_kwargs'],
                hyperparam_grid=config['hparam_grid'],  # may be {} -> decode_cv skips grid search
                save_binned=False,
                save_predictions=config['save_predictions'],
                shuffle=config['shuffle'],
                balanced_weight=config['balanced_weighting'],
                rng_seed=rng_seed,
                use_cv_sklearn_method=config.get('use_native_sklearn_for_hyperparam_estimation', False),
                verbose=bool(debug),
                outer_cv=True,
                n_folds=5,
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
# =========================
# decode_cv debug context
# =========================
_DECODE_CV_CONTEXT = {}

def set_decode_cv_context(**kwargs):
    global _DECODE_CV_CONTEXT
    _DECODE_CV_CONTEXT = dict(kwargs)


def decode_cv(
    ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
    n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
    rng_seed=None, use_cv_sklearn_method=False,
    # -------- NEW knobs (safe defaults) --------
    alpha_floor=1e-2,              # raise to 1e-1 if you still see coef blowups
    max_warn_print=10              # avoid printing 10k warnings
):
    import numpy as np
    import warnings

    from sklearn import linear_model as sklm
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.utils.class_weight import compute_sample_weight

    ys, Xs = format_data_for_decoding(ys, Xs)

    def _ctx_str():
        c = _DECODE_CV_CONTEXT if isinstance(_DECODE_CV_CONTEXT, dict) else {}
        return f"group={c.get('group_label','unknown_group')} pseudo={c.get('pseudo_id','NA')} run={c.get('run_id','NA')} seed={c.get('rng_seed', rng_seed)}"

    def _arr_stats(name, a):
        a = np.asarray(a)
        out = {"name": name, "shape": tuple(a.shape), "dtype": str(a.dtype)}
        if np.issubdtype(a.dtype, np.floating):
            out["any_nan"] = bool(np.isnan(a).any())
            out["any_inf"] = bool(np.isinf(a).any())
        else:
            out["any_nan"] = False
            out["any_inf"] = False
        if a.size > 0 and np.issubdtype(a.dtype, np.number):
            with np.errstate(all="ignore"):
                out["max_abs"] = float(np.nanmax(np.abs(a)))
                out["min"] = float(np.nanmin(a))
                out["max"] = float(np.nanmax(a))
        return out

    warn_print_count = 0
    def _warn_print(where, outer_fold=None, inner_fold=None, alpha=None, w=None, X=None, y=None):
        nonlocal warn_print_count
        if w is None or len(w) == 0:
            return
        if warn_print_count >= max_warn_print:
            return
        warn_print_count += 1
        msg = str(w[0].message)
        print(f"\n[decode_cv WARNING] outer_fold={outer_fold} inner_fold={inner_fold} alpha={alpha} {where}")
        print(f"  debug_tag: {_ctx_str()}")
        print(f"  warning: {msg}")
        if X is not None:
            print(f"  X: {_arr_stats('X', X)}")
        if y is not None:
            print(f"  y: {_arr_stats('y', y)}")
        print("")

    def _sanitize_and_standardize(Xtr, Xte, eps=1e-12):
        Xtr = np.asarray(Xtr, dtype=np.float64)
        Xte = np.asarray(Xte, dtype=np.float64)
        Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
        Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)

        var = Xtr.var(axis=0)
        keep = var > eps
        if int(keep.sum()) == 0:
            raise FloatingPointError("All features are constant/invalid in this fold")
        Xtr = Xtr[:, keep]
        Xte = Xte[:, keep]

        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd < eps] = 1.0
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        return Xtr, Xte

    def _coef_is_finite(model):
        coef = getattr(model, "coef_", None)
        if coef is None:
            return True
        coef = np.asarray(coef)
        return np.isfinite(coef).all()

    # remove trials with non-finite
    good = []
    for x, y in zip(Xs, ys):
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        good.append(np.isfinite(x_arr).all() and np.isfinite(y_arr).all())
    Xs = [x for x, g in zip(Xs, good) if g]
    ys = [y for y, g in zip(ys, good) if g]

    n_trials = len(Xs)
    if n_trials < 2:
        raise ValueError(f"decode_cv: too few usable trials (n_trials={n_trials})")

    bins_per_trial = len(Xs[0]) if isinstance(Xs[0], (list, tuple, np.ndarray)) else 1

    if rng_seed is not None:
        np.random.seed(rng_seed)

    scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score
    use_balanced = bool(balanced_weight) and (estimator == sklm.LogisticRegression)

    # --- handle empty grid ---
    do_grid = not (hyperparam_grid is None or (isinstance(hyperparam_grid, dict) and len(hyperparam_grid) == 0))
    if do_grid:
        if not isinstance(hyperparam_grid, dict) or len(hyperparam_grid) != 1:
            raise ValueError("decode_cv: hyperparam_grid must be a dict with exactly one key")
        key = list(hyperparam_grid.keys())[0]
        grid_vals = [float(a) for a in list(hyperparam_grid[key])]
        if len(grid_vals) == 0:
            do_grid = False
    else:
        key, grid_vals = None, []

    # --- IMPORTANT: alpha floor in p≈n regimes ---
    # Only apply to Ridge/Lasso style regularization
    is_ridge_like = (estimator in [sklm.Ridge, getattr(sklm, "Ridge", sklm.Ridge)])
    if do_grid and is_ridge_like:
        grid_vals = [max(alpha_floor, a) for a in grid_vals]

    indices = np.arange(n_trials)

    # cap folds so test folds aren’t single-trial
    if outer_cv:
        max_reasonable_folds = max(2, n_trials // 2)
        n_outer = int(min(n_folds, max_reasonable_folds))
        outer_kfold_iter = KFold(n_splits=n_outer, shuffle=shuffle).split(indices)
    else:
        tr, te = train_test_split(indices, test_size=test_prop, shuffle=shuffle, random_state=rng_seed)
        outer_kfold_iter = [(tr, te)]

    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []
    predictions = [None for _ in range(n_trials)]
    predictions_to_save = [None for _ in range(n_trials)]

    for outer_fold, (train_idxs_outer, test_idxs_outer) in enumerate(outer_kfold_iter):

        X_train_trials = [Xs[i] for i in train_idxs_outer]
        y_train_trials = [ys[i] for i in train_idxs_outer]
        X_test_trials  = [Xs[i] for i in test_idxs_outer]
        y_test_trials  = [ys[i] for i in test_idxs_outer]

        X_train_array = np.vstack(X_train_trials)
        y_train_array = np.asarray(np.concatenate(y_train_trials, axis=0), dtype=np.float64)
        X_test_array  = np.vstack(X_test_trials)
        y_test_array  = np.asarray(np.concatenate(y_test_trials, axis=0), dtype=np.float64)

        if estimator != sklm.LogisticRegression and y_test_array.shape[0] < 2:
            continue

        sample_weight = compute_sample_weight("balanced", y=y_train_array) if use_balanced else None

        best_alpha = None
        if do_grid:
            idx_inner = np.arange(len(X_train_trials))
            max_inner = max(2, len(idx_inner) // 2) if len(idx_inner) >= 4 else 2
            n_inner = int(min(n_folds, max_inner))
            inner_iter = KFold(n_splits=n_inner, shuffle=shuffle).split(idx_inner)

            alpha_scores = []
            for alpha in grid_vals:
                per_fold = []
                for inner_fold, (tr_i, te_i) in enumerate(inner_iter):
                    Xtr_i = np.vstack([X_train_trials[i] for i in tr_i])
                    ytr_i = np.asarray(np.concatenate([y_train_trials[i] for i in tr_i], axis=0), dtype=np.float64)
                    Xte_i = np.vstack([X_train_trials[i] for i in te_i])
                    yte_i = np.asarray(np.concatenate([y_train_trials[i] for i in te_i], axis=0), dtype=np.float64)

                    if estimator != sklm.LogisticRegression and yte_i.shape[0] < 2:
                        continue

                    try:
                        Xtr_i, Xte_i = _sanitize_and_standardize(Xtr_i, Xte_i)

                        # Force numerically-stable solver for Ridge
                        ek = dict(estimator_kwargs)
                        if is_ridge_like:
                            ek.setdefault("solver", "svd")

                        model_inner = estimator(**{**ek, key: alpha})

                        with warnings.catch_warnings(record=True) as w:
                            warnings.simplefilter("always", RuntimeWarning)
                            model_inner.fit(Xtr_i, ytr_i, sample_weight=sample_weight)
                            _warn_print("model_inner.fit", outer_fold, inner_fold, alpha, w, Xtr_i, ytr_i)

                        # If coefficients blew up, mark invalid
                        if not _coef_is_finite(model_inner):
                            continue

                        pred_i = model_inner.predict(Xte_i)
                        if not np.isfinite(pred_i).all():
                            continue

                        per_fold.append(scoring_f(yte_i, pred_i))
                    except Exception:
                        continue

                alpha_scores.append(np.mean(per_fold) if len(per_fold) else -np.inf)

            best_alpha = grid_vals[int(np.argmax(alpha_scores))]

        # final fit
        try:
            Xtr, Xte = _sanitize_and_standardize(X_train_array, X_test_array)

            ek = dict(estimator_kwargs)
            if is_ridge_like:
                ek.setdefault("solver", "svd")

            if do_grid:
                model = estimator(**{**ek, key: best_alpha})
            else:
                model = estimator(**ek)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", RuntimeWarning)
                model.fit(Xtr, y_train_array, sample_weight=sample_weight)
                _warn_print("FINAL_FIT", outer_fold, None, best_alpha, w, Xtr, y_train_array)

            if not _coef_is_finite(model):
                # skip this fold entirely
                continue

            y_pred_train = model.predict(Xtr)
            y_pred_test  = model.predict(Xte)

            if not (np.isfinite(y_pred_train).all() and np.isfinite(y_pred_test).all()):
                continue

            scores_train.append(scoring_f(y_train_array, y_pred_train))
            scores_test.append(scoring_f(y_test_array, y_pred_test))

            for i_fold, i_global in enumerate(test_idxs_outer):
                if bins_per_trial == 1:
                    predictions[i_global] = np.array([y_pred_test[i_fold]])
                    predictions_to_save[i_global] = predictions[i_global]
                else:
                    pred_trial = model.predict(np.asarray(X_test_trials[i_fold], dtype=np.float64))
                    predictions[i_global] = pred_trial
                    predictions_to_save[i_global] = pred_trial

            idxes_test.append(test_idxs_outer)
            idxes_train.append(train_idxs_outer)
            weights.append(getattr(model, "coef_", None))
            intercepts.append(getattr(model, "intercept_", None) if getattr(model, "fit_intercept", False) else None)
            best_params.append({list(hyperparam_grid.keys())[0]: best_alpha} if do_grid else {})

        except Exception:
            continue

    ok = [i for i in range(n_trials) if predictions[i] is not None]
    if len(ok) == 0:
        raise RuntimeError(f"decode_cv: all folds failed/skipped [{_ctx_str()}]")

    ys_true_full = np.asarray(np.concatenate([ys[i] for i in ok], axis=0), dtype=np.float64)
    ys_pred_full = np.asarray(np.concatenate([predictions[i] for i in ok], axis=0), dtype=np.float64)

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


def decode_cv(
    ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
    n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
    rng_seed=None, use_cv_sklearn_method=False
):
    """
    Regresses binned neural activity against a target, using a provided sklearn estimator.

    PATCHED:
    - Robustly sanitizes regressor matrices to prevent NaN/Inf and zero-variance columns from causing
      sklearn matmul overflow/divide-by-zero/invalid warnings.
    """

    # transform target data into standard format: list of np.ndarrays
    ys, Xs = format_data_for_decoding(ys, Xs)

    # -------------------------------
    # NEW: helper to sanitize X
    # -------------------------------
    def _sanitize_X(X, eps=1e-12):
        """
        Ensure X is finite and well-conditioned for linear models:
        - cast to float64
        - replace NaN/Inf with 0
        - drop near-zero-variance columns (var <= eps)
        Returns (X_sanitized, keep_cols_mask)
        """
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        var = X.var(axis=0)
        keep = var > eps
        keep &= np.all(np.isfinite(X), axis=0)

        if keep.sum() == 0:
            raise ValueError(
                "[decode_cv] All features removed after sanitization "
                "(matrix is all invalid or zero-variance)."
            )
        return X[:, keep], keep

    def _apply_keep(X, keep_cols):
        """Apply keep_cols to X with safe casting and nan_to_num."""
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X[:, keep_cols]

    # filter out trials with non-finite X or y at the TRIAL level (keeps existing behavior)
    finite_trials = []
    for i in range(len(Xs)):
        Xi = np.asarray(Xs[i])
        yi = np.asarray(ys[i])
        finite_trials.append(np.all(np.isfinite(Xi)) and np.all(np.isfinite(yi)))
    finite_trials = np.array(finite_trials, dtype=bool)

    if not np.all(finite_trials):
        Xs = [x for x, ok in zip(Xs, finite_trials) if ok]
        ys = [y for y, ok in zip(ys, finite_trials) if ok]

    # enforce at least 10 trials AFTER cleaning
    if len(Xs) < 10:
        raise ValueError(f"[decode_cv] Too few finite trials after cleaning: n={len(Xs)}")

    # initialize containers to save outputs
    n_trials = len(Xs)
    bins_per_trial = len(Xs[0])
    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []
    predictions = [None for _ in range(n_trials)]
    predictions_to_save = [None for _ in range(n_trials)]  # different for logistic regression

    # split the dataset in two parts, train and test
    if rng_seed is not None:
        np.random.seed(rng_seed)
    indices = np.arange(n_trials)

    if outer_cv:
        # -------------------------------
        # adaptive OUTER CV folds
        # -------------------------------
        def _choose_outer_splits(n: int) -> int:
            if n < 50:
                desired = 5
            elif n <= 100:
                desired = 10
            else:
                desired = 50
            return max(2, min(desired, n))

        outer_splits = _choose_outer_splits(n_trials)
        if verbose:
            print(f"[decode_cv] n_trials={n_trials} -> outer_splits={outer_splits} (use_cv_sklearn_method={use_cv_sklearn_method})")

        get_kfold = lambda: KFold(n_splits=outer_splits, shuffle=shuffle).split(indices)

        if estimator == sklm.LogisticRegression:
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

    scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score

    if estimator in [sklm.RidgeCV, sklm.LassoCV, sklm.LogisticRegressionCV]:
        raise NotImplementedError("the code does not support a CV-type estimator.")
    else:
        for train_idxs_outer, test_idxs_outer in outer_kfold_iter:

            X_train = [Xs[i] for i in train_idxs_outer]
            y_train = [ys[i] for i in train_idxs_outer]
            X_test = [Xs[i] for i in test_idxs_outer]
            y_test = [ys[i] for i in test_idxs_outer]

            key = list(hyperparam_grid.keys())[0]

            if not use_cv_sklearn_method:
                # inner folds
                idx_inner = np.arange(len(X_train))
                inner_splits = max(2, min(n_folds, len(idx_inner)))
                get_kfold_inner = lambda: KFold(n_splits=inner_splits, shuffle=shuffle).split(idx_inner)

                if verbose:
                    print(f"[decode_cv] inner_splits={inner_splits} (n_train_trials={len(idx_inner)})")

                if estimator == sklm.LogisticRegression:
                    assert logisticreg_criteria(y_train, min_unique_counts=2)
                    isysat_inner = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=1)
                else:
                    isysat_inner = lambda ys_: True

                sample_count, _, inner_kfold_iter = sample_folds(y_train, get_kfold_inner, isysat_inner)
                if sample_count > 1:
                    print(f"sampled inner folds {sample_count} times to ensure enough targets")

                r2s = np.zeros([inner_splits, len(hyperparam_grid[key])])
                inner_predictions = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
                inner_targets = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan

                for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold_iter):

                    X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
                    y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
                    X_test_inner = np.vstack([X_train[i] for i in test_idxs_inner])
                    y_test_inner = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)

                    # -------- NEW: sanitize inner fold matrices --------
                    X_train_inner, keep_cols_inner = _sanitize_X(X_train_inner)
                    X_test_inner = _apply_keep(X_test_inner, keep_cols_inner)

                    for i_alpha, alpha in enumerate(hyperparam_grid[key]):

                        sample_weight = compute_sample_weight("balanced", y=y_train_inner) if balanced_weight else None

                        model_inner = estimator(**{**estimator_kwargs, key: alpha})
                        model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)

                        pred_test_inner = model_inner.predict(X_test_inner)

                        inner_predictions[test_idxs_inner, i_alpha] = np.mean(pred_test_inner)
                        inner_targets[test_idxs_inner, i_alpha] = np.mean(y_test_inner)
                        r2s[ifold, i_alpha] = scoring_f(y_test_inner, pred_test_inner)

                assert np.all(~np.isnan(inner_predictions))
                assert np.all(~np.isnan(inner_targets))

                r2s_avg = r2s.mean(axis=0)

                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)

                # -------- NEW: sanitize outer refit matrix --------
                X_train_array, keep_cols = _sanitize_X(X_train_array)

                sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None

                best_alpha = hyperparam_grid[key][np.argmax(r2s_avg)]
                model = estimator(**{**estimator_kwargs, key: best_alpha})
                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)

            else:
                if estimator not in [Ridge, Lasso]:
                    raise NotImplementedError("This case is not implemented")

                model = (
                    RidgeCV(alphas=hyperparam_grid[key])
                    if estimator == Ridge
                    else LassoCV(alphas=hyperparam_grid[key])
                )

                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)

                # -------- NEW: sanitize outer refit matrix --------
                X_train_array, keep_cols = _sanitize_X(X_train_array)

                sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None
                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
                best_alpha = model.alpha_

            # evaluate model on train data
            y_pred_train = model.predict(X_train_array)
            scores_train.append(scoring_f(y_train_array, y_pred_train))

            # evaluate model on test data
            y_true = np.concatenate(y_test, axis=0)

            X_test_array = np.vstack(X_test)
            X_test_array = _apply_keep(X_test_array, keep_cols)

            y_pred = model.predict(X_test_array)

            if isinstance(model, sklm.LogisticRegression):
                y_pred_probs = model.predict_proba(X_test_array)[:, 1]
            else:
                y_pred_probs = None

            scores_test.append(scoring_f(y_true, y_pred))

            # save per-trial predictions
            for i_fold, i_global in enumerate(test_idxs_outer):
                if bins_per_trial == 1:
                    predictions[i_global] = np.array([y_pred[i_fold]])
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])
                    else:
                        predictions_to_save[i_global] = np.array([y_pred[i_fold]])
                else:
                    Xt = np.asarray(X_test[i_fold], dtype=np.float64)
                    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)[:, keep_cols]
                    predictions[i_global] = model.predict(Xt)
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = model.predict_proba(Xt)[:, 1]
                    else:
                        predictions_to_save[i_global] = predictions[i_global]

            idxes_test.append(test_idxs_outer)
            idxes_train.append(train_idxs_outer)
            weights.append(model.coef_)
            if model.fit_intercept:
                intercepts.append(model.intercept_)
            else:
                intercepts.append(None)
            best_params.append({key: best_alpha})

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

    if verbose:
        if outer_cv:
            print("Performance is only described for last outer fold \n")

    return outdict











def decode_cv(
    ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
    n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
    rng_seed=None, use_cv_sklearn_method=False
):
    """
    Regresses binned neural activity against a target, using a provided sklearn estimator.

    PATCHED:
    - Robustly sanitizes regressor matrices to prevent NaN/Inf and zero-variance columns from causing
      sklearn matmul overflow/divide-by-zero/invalid warnings.
    - Suppresses common numerical / convergence warnings during fitting.
    """
    import warnings
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning

    # =========================
    # NEW: hide warnings inside decode_cv
    # =========================
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # silence sklearn "X does not have valid feature names" etc (optional but common)
    warnings.filterwarnings("ignore", message=".*valid feature names.*")

    # If you also want to suppress numpy floating warnings:
    with np.errstate(all="ignore"):

        # transform target data into standard format: list of np.ndarrays
        ys, Xs = format_data_for_decoding(ys, Xs)

        # -------------------------------
        # NEW: helper to sanitize X
        # -------------------------------
        def _sanitize_X(X, eps=1e-12):
            """
            Ensure X is finite and well-conditioned for linear models:
            - cast to float64
            - replace NaN/Inf with 0
            - drop near-zero-variance columns (var <= eps)
            Returns (X_sanitized, keep_cols_mask)
            """
            X = np.asarray(X, dtype=np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            var = X.var(axis=0)
            keep = var > eps
            keep &= np.all(np.isfinite(X), axis=0)

            if keep.sum() == 0:
                raise ValueError(
                    "[decode_cv] All features removed after sanitization "
                    "(matrix is all invalid or zero-variance)."
                )
            return X[:, keep], keep

        def _apply_keep(X, keep_cols):
            """Apply keep_cols to X with safe casting and nan_to_num."""
            X = np.asarray(X, dtype=np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            return X[:, keep_cols]

        # filter out trials with non-finite X or y at the TRIAL level (keeps existing behavior)
        finite_trials = []
        for i in range(len(Xs)):
            Xi = np.asarray(Xs[i])
            yi = np.asarray(ys[i])
            finite_trials.append(np.all(np.isfinite(Xi)) and np.all(np.isfinite(yi)))
        finite_trials = np.array(finite_trials, dtype=bool)

        if not np.all(finite_trials):
            Xs = [x for x, ok in zip(Xs, finite_trials) if ok]
            ys = [y for y, ok in zip(ys, finite_trials) if ok]

        # enforce at least 10 trials AFTER cleaning
        if len(Xs) < 10:
            raise ValueError(f"[decode_cv] Too few finite trials after cleaning: n={len(Xs)}")

        # initialize containers to save outputs
        n_trials = len(Xs)
        bins_per_trial = len(Xs[0])
        scores_test, scores_train = [], []
        idxes_test, idxes_train = [], []
        weights, intercepts, best_params = [], [], []
        predictions = [None for _ in range(n_trials)]
        predictions_to_save = [None for _ in range(n_trials)]  # different for logistic regression

        # split the dataset in two parts, train and test
        if rng_seed is not None:
            np.random.seed(rng_seed)
        indices = np.arange(n_trials)

        if outer_cv:
            # -------------------------------
            # adaptive OUTER CV folds
            # -------------------------------
            def _choose_outer_splits(n: int) -> int:
                if n < 50:
                    desired = 5
                elif n <= 100:
                    desired = 10
                else:
                    desired = 50
                return max(2, min(desired, n))

            outer_splits = _choose_outer_splits(n_trials)
            if verbose:
                print(
                    f"[decode_cv] n_trials={n_trials} -> outer_splits={outer_splits} "
                    f"(use_cv_sklearn_method={use_cv_sklearn_method})"
                )

            get_kfold = lambda: KFold(n_splits=outer_splits, shuffle=shuffle).split(indices)

            if estimator == sklm.LogisticRegression:
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

        scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score

        if estimator in [sklm.RidgeCV, sklm.LassoCV, sklm.LogisticRegressionCV]:
            raise NotImplementedError("the code does not support a CV-type estimator.")
        else:
            for train_idxs_outer, test_idxs_outer in outer_kfold_iter:

                X_train = [Xs[i] for i in train_idxs_outer]
                y_train = [ys[i] for i in train_idxs_outer]
                X_test = [Xs[i] for i in test_idxs_outer]
                y_test = [ys[i] for i in test_idxs_outer]

                key = list(hyperparam_grid.keys())[0]

                if not use_cv_sklearn_method:
                    # inner folds
                    idx_inner = np.arange(len(X_train))
                    inner_splits = max(2, min(n_folds, len(idx_inner)))
                    get_kfold_inner = lambda: KFold(n_splits=inner_splits, shuffle=shuffle).split(idx_inner)

                    if verbose:
                        print(f"[decode_cv] inner_splits={inner_splits} (n_train_trials={len(idx_inner)})")

                    if estimator == sklm.LogisticRegression:
                        assert logisticreg_criteria(y_train, min_unique_counts=2)
                        isysat_inner = lambda ys_: logisticreg_criteria(ys_, min_unique_counts=1)
                    else:
                        isysat_inner = lambda ys_: True

                    sample_count, _, inner_kfold_iter = sample_folds(y_train, get_kfold_inner, isysat_inner)
                    if sample_count > 1:
                        print(f"sampled inner folds {sample_count} times to ensure enough targets")

                    r2s = np.zeros([inner_splits, len(hyperparam_grid[key])])
                    inner_predictions = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
                    inner_targets = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan

                    for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold_iter):

                        X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
                        y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
                        X_test_inner = np.vstack([X_train[i] for i in test_idxs_inner])
                        y_test_inner = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)

                        # -------- NEW: sanitize inner fold matrices --------
                        X_train_inner, keep_cols_inner = _sanitize_X(X_train_inner)
                        X_test_inner = _apply_keep(X_test_inner, keep_cols_inner)

                        for i_alpha, alpha in enumerate(hyperparam_grid[key]):

                            sample_weight = compute_sample_weight("balanced", y=y_train_inner) if balanced_weight else None

                            model_inner = estimator(**{**estimator_kwargs, key: alpha})
                            model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)

                            pred_test_inner = model_inner.predict(X_test_inner)

                            inner_predictions[test_idxs_inner, i_alpha] = np.mean(pred_test_inner)
                            inner_targets[test_idxs_inner, i_alpha] = np.mean(y_test_inner)
                            r2s[ifold, i_alpha] = scoring_f(y_test_inner, pred_test_inner)

                    assert np.all(~np.isnan(inner_predictions))
                    assert np.all(~np.isnan(inner_targets))

                    r2s_avg = r2s.mean(axis=0)

                    X_train_array = np.vstack(X_train)
                    y_train_array = np.concatenate(y_train, axis=0)

                    # -------- NEW: sanitize outer refit matrix --------
                    X_train_array, keep_cols = _sanitize_X(X_train_array)

                    sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None

                    best_alpha = hyperparam_grid[key][np.argmax(r2s_avg)]
                    model = estimator(**{**estimator_kwargs, key: best_alpha})
                    model.fit(X_train_array, y_train_array, sample_weight=sample_weight)

                else:
                    if estimator not in [Ridge, Lasso]:
                        raise NotImplementedError("This case is not implemented")

                    model = (
                        RidgeCV(alphas=hyperparam_grid[key])
                        if estimator == Ridge
                        else LassoCV(alphas=hyperparam_grid[key])
                    )

                    X_train_array = np.vstack(X_train)
                    y_train_array = np.concatenate(y_train, axis=0)

                    # -------- NEW: sanitize outer refit matrix --------
                    X_train_array, keep_cols = _sanitize_X(X_train_array)

                    sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None
                    model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
                    best_alpha = model.alpha_

                # evaluate model on train data
                y_pred_train = model.predict(X_train_array)
                scores_train.append(scoring_f(y_train_array, y_pred_train))

                # evaluate model on test data
                y_true = np.concatenate(y_test, axis=0)

                X_test_array = np.vstack(X_test)
                X_test_array = _apply_keep(X_test_array, keep_cols)

                y_pred = model.predict(X_test_array)

                if isinstance(model, sklm.LogisticRegression):
                    y_pred_probs = model.predict_proba(X_test_array)[:, 1]
                else:
                    y_pred_probs = None

                scores_test.append(scoring_f(y_true, y_pred))

                # save per-trial predictions
                for i_fold, i_global in enumerate(test_idxs_outer):
                    if bins_per_trial == 1:
                        predictions[i_global] = np.array([y_pred[i_fold]])
                        if isinstance(model, sklm.LogisticRegression):
                            predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])
                        else:
                            predictions_to_save[i_global] = np.array([y_pred[i_fold]])
                    else:
                        Xt = np.asarray(X_test[i_fold], dtype=np.float64)
                        Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)[:, keep_cols]
                        predictions[i_global] = model.predict(Xt)
                        if isinstance(model, sklm.LogisticRegression):
                            predictions_to_save[i_global] = model.predict_proba(Xt)[:, 1]
                        else:
                            predictions_to_save[i_global] = predictions[i_global]

                idxes_test.append(test_idxs_outer)
                idxes_train.append(train_idxs_outer)
                weights.append(model.coef_)
                if model.fit_intercept:
                    intercepts.append(model.intercept_)
                else:
                    intercepts.append(None)
                best_params.append({key: best_alpha})

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

        if verbose:
            if outer_cv:
                print("Performance is only described for last outer fold \n")

        return outdict

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
