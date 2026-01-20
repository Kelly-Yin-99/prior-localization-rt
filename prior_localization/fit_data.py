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
        roi_set=None,   # <-- NEW: set like {"MOp","MOs","ACAd","ORBvl"} or None

):
    """
    Fits a single session for ephys data, patched to support decoding on trial subgroups
    (e.g., fast/normal/slow within a session) WITHOUT breaking the behavior pipeline.

    Additional patch:
    - If roi_set is provided:
        * Early skip sessions that have NO ROI regions
        * Only decode regions that are in roi_set
    """


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

    if debug:
        print(f"[DEBUG] {session_id}: loaded sl.trials rows = {len(sl.trials)}")

    # If user provides external trials_df, replace sl.trials
    if trials_df is not None:
        sl.trials = trials_df.copy()
        if debug:
            print(f"[DEBUG] {session_id}: replaced trials with trials_df rows = {len(sl.trials)}")

    # Keep a full copy for behavior computations (IMPORTANT)
    trials_full = sl.trials.copy()

    min_rt = None
    max_rt = None


    # Compute base mask (choice/QC etc.)

    _, base_mask = load_trials_and_mask(
        one=one, eid=session_id, sess_loader=sl, min_rt=min_rt, max_rt=max_rt,
        min_trial_len=None, max_trial_len=None,
        exclude_nochoice=True, exclude_unbiased=False,
    )

    # Apply group mask (fast/normal/slow) on top of base mask

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


    # Prepare ephys data (subgroup)

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


    pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if len(pseudo_ids) > 1 else str(pseudo_ids[0])
    probe_str = 'merged_probes' if (isinstance(probe_name, list) and len(probe_name) > 1) else probe_name


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
            group_label=group_label,
            debug=True,
            min_trials_subgroup=10,
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


def sample_folds(ys, get_kfold, isfoldsat, max_iter=100):
    """
    Same as their sample_folds: keep resampling KFold splits until every fold satisfies criteria.
    ys: list of arrays (per trial)
    """
    sample_count = 0
    ysatisfy = [False]

    while not np.all(np.array(ysatisfy)):
        if sample_count >= max_iter:
            raise ValueError(f"[decode_cv] Could not sample satisfactory folds after {max_iter} tries ctx={_ctx_str()}")
        sample_count += 1

        outer_kfold = get_kfold()
        fold_iter = [(train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(outer_kfold)]

        # check each fold's *test* y has required properties
        ysatisfy = []
        for _, test_idxs in fold_iter:
            y_test_list = [ys[i] for i in test_idxs]
            ysatisfy.append(isfoldsat(y_test_list))

    return sample_count, outer_kfold, fold_iter


def decode_cv(
    ys, Xs,
    estimator, estimator_kwargs,
    balanced_weight=False,
    hyperparam_grid=None,
    test_prop=0.2,
    n_folds=5,
    save_binned=False,
    save_predictions=True,
    verbose=False,
    shuffle=True,
    outer_cv=True,
    rng_seed=None,
    use_cv_sklearn_method=False,
    min_trials=10,              #MIN TRIALS per subgroup
    max_sample_folds_iter=200,   # a bit higher because small subgroups can be annoying
):
    """
    Like original prior_localization.decode_cv:
      - uses per-trial folds (never splits within a trial)
      - (optionally) resamples folds until they satisfy criteria (esp. logistic)
    Adapted for subgroups:
      - enforces min_trials
      - caps n_folds to feasible values

    """

    # NOTE: you must have format_data_for_decoding in scope (same as original)
    ys, Xs = format_data_for_decoding(ys, Xs)

    n_trials = len(Xs)
    if n_trials < min_trials:
        raise ValueError(f"[decode_cv] Too few trials for subgroup: n_trials={n_trials} < {min_trials} ctx={_ctx_str()}")

    # Seed control
    rng = np.random.RandomState(int(rng_seed)) if rng_seed is not None else np.random.RandomState()
    indices = np.arange(n_trials)

    # scoring function
    scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score

    # containers
    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []
    predictions = [None for _ in range(n_trials)]
    predictions_to_save = [None for _ in range(n_trials)]

    # hyperparam grid handling (same assumption as original: single key)
    if hyperparam_grid is None or len(hyperparam_grid) == 0:
        grid_key = None
        grid_vals = None
    else:
        grid_key = list(hyperparam_grid.keys())[0]
        grid_vals = list(hyperparam_grid[grid_key])

    # stack trials into sample rows (bins)
    def stack_trials(X_list, y_list):
        X = np.vstack(X_list)
        y = np.concatenate(y_list, axis=0)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return X, y

    def dbg(msg):
        if verbose:
            print(msg)

    #build outer folds like their code (sample until criteria satisfied)
    if outer_cv:
        # cap folds to available trials (and at least 2)
        n_splits_outer = min(n_folds, n_trials)
        if n_splits_outer < 2:
            raise ValueError(f"[decode_cv] Too few trials for outer CV: n_trials={n_trials} ctx={_ctx_str()}")

        def get_kfold_outer():
            # IMPORTANT: use a fresh random_state each attempt so resampling actually changes
            rs = rng.randint(0, 2**31 - 1)
            return KFold(n_splits=n_splits_outer, shuffle=shuffle, random_state=rs).split(indices)

        if estimator == sklm.LogisticRegression:
            # overall dataset must have both classes with >=2 examples per class
            if not logisticreg_criteria(ys, min_unique_counts=2):
                raise ValueError(f"[decode_cv] LogisticRegression requires 2 classes with >=2 examples each ctx={_ctx_str()}")
            isysat_outer = lambda y_test_list: logisticreg_criteria(y_test_list, min_unique_counts=1)
        else:
            isysat_outer = lambda y_test_list: True

        sample_count, _, outer_iter = sample_folds(
            ys, get_kfold_outer, isysat_outer, max_iter=max_sample_folds_iter
        )
        if sample_count > 1:
            dbg(f"[decode_cv] sampled outer folds {sample_count} times (subgroup) ctx={_ctx_str()}")

    else:
        tr, te = train_test_split(
            indices,
            test_size=test_prop,
            shuffle=shuffle,
            random_state=rng.randint(0, 2**31 - 1),
        )
        outer_iter = [(tr, te)]

    # main outer loop
    for fold_i, (train_idxs_outer, test_idxs_outer) in enumerate(outer_iter):

        X_train_list = [Xs[i] for i in train_idxs_outer]
        y_train_list = [ys[i] for i in train_idxs_outer]
        X_test_list  = [Xs[i] for i in test_idxs_outer]
        y_test_list  = [ys[i] for i in test_idxs_outer]

        # inner CV to choose hyperparam (their nested CV idea)
        best_param = {}

        if (grid_key is not None) and (not use_cv_sklearn_method):
            idx_inner = np.arange(len(train_idxs_outer))
            n_trials_inner = len(idx_inner)

            # cap inner folds too
            n_splits_inner = min(n_folds, n_trials_inner)
            if n_splits_inner < 2:
                best_val = grid_vals[0]
            else:
                def get_kfold_inner():
                    rs = rng.randint(0, 2**31 - 1)
                    return KFold(n_splits=n_splits_inner, shuffle=shuffle, random_state=rs).split(idx_inner)

                if estimator == sklm.LogisticRegression:
                    if not logisticreg_criteria(y_train_list, min_unique_counts=2):
                        # can happen in tiny subgroup after outer split
                        # fall back to first value (or you can skip fold)
                        best_val = grid_vals[0]
                    else:
                        isysat_inner = lambda y_test_list_inner: logisticreg_criteria(y_test_list_inner, min_unique_counts=1)
                        sample_count_in, _, inner_iter = sample_folds(
                            y_train_list, get_kfold_inner, isysat_inner, max_iter=max_sample_folds_iter
                        )
                        if sample_count_in > 1:
                            dbg(f"[decode_cv] sampled inner folds {sample_count_in} times ctx={_ctx_str()}")
                else:
                    # regression: always valid folds
                    inner_iter = [(tr_i, te_i) for _, (tr_i, te_i) in enumerate(get_kfold_inner())]

                scores_grid = np.full((len(inner_iter), len(grid_vals)), np.nan, dtype=float)

                for ifold, (tr_i, te_i) in enumerate(inner_iter):
                    Xtr, ytr = stack_trials([X_train_list[i] for i in tr_i], [y_train_list[i] for i in tr_i])
                    Xte, yte = stack_trials([X_train_list[i] for i in te_i], [y_train_list[i] for i in te_i])


                    sw = None
                    if balanced_weight and estimator == sklm.LogisticRegression:
                        sw = compute_sample_weight("balanced", y=ytr)

                    for ia, val in enumerate(grid_vals):
                        try:
                            kw = dict(estimator_kwargs)
                            kw[grid_key] = val
                            model_inner = estimator(**kw)
                            with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
                                model_inner.fit(Xtr, ytr, sample_weight=sw)
                                yhat = model_inner.predict(Xte)
                            scores_grid[ifold, ia] = scoring_f(yte, yhat)
                        except Exception:
                            # treat failed fit as very poor
                            scores_grid[ifold, ia] = -np.inf

                # choose best hyperparam by average inner score
                mean_scores = np.nanmean(scores_grid, axis=0)
                best_val = grid_vals[int(np.argmax(mean_scores))]

            best_param = {grid_key: best_val}

        # outer fit with best param
        X_train_array, y_train_array = stack_trials(X_train_list, y_train_list)
        X_test_array,  y_test_array  = stack_trials(X_test_list,  y_test_list)

        sw_train = None
        if balanced_weight and estimator == sklm.LogisticRegression:
            sw_train = compute_sample_weight("balanced", y=y_train_array)

        kw = dict(estimator_kwargs)
        kw.update(best_param)
        model = estimator(**kw)

        try:
            with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
                model.fit(X_train_array, y_train_array, sample_weight=sw_train)
        except Exception:
            # skip this fold rather than crashing whole subgroup
            continue

        # train/test scores
        with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
            y_pred_train = model.predict(X_train_array)
            y_pred_test  = model.predict(X_test_array)

        scores_train.append(scoring_f(y_train_array, y_pred_train))
        scores_test.append(scoring_f(y_test_array, y_pred_test))

        # save per-trial predictions (keep trial boundaries)
        for i_fold_local, i_global in enumerate(test_idxs_outer):
            with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
                pred_bins = model.predict(X_test_list[i_fold_local])
            predictions[i_global] = pred_bins


            if isinstance(model, sklm.LogisticRegression):
                with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
                    prob = model.predict_proba(X_test_list[i_fold_local])[:, 1]
                predictions_to_save[i_global] = prob
            else:
                predictions_to_save[i_global] = pred_bins

        idxes_test.append(test_idxs_outer)
        idxes_train.append(train_idxs_outer)

        if save_predictions:
            weights.append(getattr(model, "coef_", None))
            intercepts.append(getattr(model, "intercept_", None))
            best_params.append(best_param)


    if not any(p is not None for p in predictions):
        raise ValueError(f"[decode_cv] No successful folds produced predictions ctx={_ctx_str()}")

    ys_true_full = np.concatenate([ys[i] for i in range(n_trials) if predictions[i] is not None], axis=0)
    ys_pred_full = np.concatenate([predictions[i] for i in range(n_trials) if predictions[i] is not None], axis=0)

    outdict = {}
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
