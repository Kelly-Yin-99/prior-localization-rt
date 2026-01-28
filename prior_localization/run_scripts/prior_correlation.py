import os
import ast
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from joblib import Parallel, delayed

from one.api import ONE
from iblatlas.atlas import AllenAtlas
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from typing import Optional

TXT_PATH = "/Users/changyin/Downloads/session_ids_for_behav_analysis.txt"



DEBUG_N_SESSIONS = 2   # set to None to run all

BASE_DIR = Path(__file__).resolve().parents[1]


OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Regions of interest
ROIS = ["MOp", "MOs", "ACAd", "ORBvl"]

DEBUG_CORR = True  # set False later

def dbg(msg):
    if DEBUG_CORR:
        print(msg)

# ITI window rules
ITI_MAX_DUR = 1.5          # seconds, use first 1.5s of ITI
ITI_SHORT_CUTOFF = 2.0     # if ITI < 2s, end at next_stim - 0.5s
ITI_PRESTIM_GUARD = 0.5    # seconds before next stim

# Ephys criteria
MIN_NEURONS = 10
MIN_NEURON_FR_HZ = 0.1
MIN_TOTAL_ITI_SPIKES = 10_000

# Wheel / RT grouping
EARLY_RT = 0.08
LATE_RT = 1.25

# ATI session inclusion: sessions must have >=401 trials BEFORE dropping NaNs
MIN_TRIALS_SESSION_RAW = 401
# Animal inclusion for ATI: must have >=2 such sessions
MIN_SESSIONS_PER_ANIMAL_FOR_ATI = 2

# Parallelism (set to 1 while debugging)
N_JOBS = 1

# ONE / Alyx (RIS)
atlas = AllenAtlas()
one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international')

# =========================
# UTIL: read session IDs
# =========================
def read_session_ids_from_txt(txt_path: str) -> list[str]:
    session_ids = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids_in_line = ast.literal_eval(line)  # safe parsing of python list
            session_ids.extend(ids_in_line)
    # unique, preserve order
    session_ids = list(dict.fromkeys(session_ids))
    return session_ids


# =========================
# WHEEL / FIRST MOVEMENT
# (your existing logic, kept)
# =========================
def load_wheel_data(eid: str):
    try:
        wheel_data = one.load_object(eid, "wheel", collection="alf")
        return wheel_data["position"], wheel_data["timestamps"]
    except Exception as e:
        print(f"No wheel data for {eid}: {e}")
        return None, None


def calc_wheel_velocity(position, timestamps):
    wheel_velocity = [0.0]
    for widx in range(len(position) - 1):
        time_diff = timestamps[widx + 1] - timestamps[widx]
        if time_diff != 0:
            velocity = (position[widx + 1] - position[widx]) / time_diff
        else:
            velocity = 0.0
        wheel_velocity.append(velocity)
    return wheel_velocity


def calc_trialwise_wheel(position, timestamps, velocity, stimOn_times, feedback_times):
    stimOn_pre_duration = 0.3  # [s]
    total_trial_count = len(stimOn_times)

    trial_position = [[] for _ in range(total_trial_count)]
    trial_timestamps = [[] for _ in range(total_trial_count)]
    trial_velocity = [[] for _ in range(total_trial_count)]

    tridx = 0
    for tsidx in range(len(timestamps)):
        timestamp = timestamps[tsidx]
        while tridx < total_trial_count - 1 and timestamp > stimOn_times[tridx + 1] - stimOn_pre_duration:
            tridx += 1

        if stimOn_times[tridx] - stimOn_pre_duration <= timestamp < feedback_times[tridx]:
            trial_position[tridx].append(position[tsidx])
            trial_timestamps[tridx].append(timestamps[tsidx])
            trial_velocity[tridx].append(velocity[tsidx])

    return trial_position, trial_timestamps, trial_velocity


def calc_movement_onset_times(trial_timestamps, trial_velocity, stimOn_times):
    speed_threshold = 0.5
    duration_threshold = 0.05  # [s]

    movement_onset_times = []
    movement_directions = []
    first_movement_onset_times = np.zeros(len(trial_timestamps))
    last_movement_onset_times = np.zeros(len(trial_timestamps))
    first_movement_directions = np.zeros(len(trial_timestamps))
    last_movement_directions = np.zeros(len(trial_timestamps))

    for tridx in range(len(trial_timestamps)):
        movement_onset_times.append([])
        movement_directions.append([])
        cm_dur = 0.0
        for tpidx in range(len(trial_timestamps[tridx])):
            t = trial_timestamps[tridx][tpidx]
            if tpidx == 0:
                tprev = stimOn_times.iloc[tridx] - 0.3 if hasattr(stimOn_times, "iloc") else stimOn_times[tridx] - 0.3
            cm_dur += (t - tprev)
            if abs(trial_velocity[tridx][tpidx]) > speed_threshold:
                if cm_dur > duration_threshold:
                    movement_onset_times[tridx].append(t)
                    movement_directions[tridx].append(np.sign(trial_velocity[tridx][tpidx]))
                cm_dur = 0.0
            tprev = t

        if len(movement_onset_times[tridx]) == 0:
            first_movement_onset_times[tridx] = np.nan
            last_movement_onset_times[tridx] = np.nan
            first_movement_directions[tridx] = 0
            last_movement_directions[tridx] = 0
        else:
            first_movement_onset_times[tridx] = movement_onset_times[tridx][0]
            last_movement_onset_times[tridx] = movement_onset_times[tridx][-1]
            first_movement_directions[tridx] = movement_directions[tridx][0]
            last_movement_directions[tridx] = movement_directions[tridx][-1]

    return (movement_onset_times,
            first_movement_onset_times,
            last_movement_onset_times,
            first_movement_directions,
            last_movement_directions)


# =========================
# BEHAVIOR: per-session trials
# =========================
def load_trials_with_wheel(eid: str) -> Optional[pd.DataFrame]:

    """
    Returns a per-trial dataframe for one session including:
    - stimOn_times, feedback_times, probabilityLeft, etc. from IBL trials
    - wheel-derived first_movement_onset_times
    - subject/sex/age/eid metadata
    - trial_index (original order)
    """
    try:
        sl = SessionLoader(eid=eid, one=one)
        sl.load_trials()
        trials = sl.trials
        if trials is None or len(trials) == 0:
            print(f"[behav] skip {eid}: no trials")
            return None

        n_trials_raw = len(trials)
        if n_trials_raw < MIN_TRIALS_SESSION_RAW:
            print(f"[behav] skip {eid}: n_trials_raw={n_trials_raw} < {MIN_TRIALS_SESSION_RAW}")
            return None

        # Metadata (subject, sex, age)
        session_data = one.alyx.rest("sessions", "read", id=eid)
        subject = session_data["subject"]
        subject_data = one.alyx.rest("subjects", "list", nickname=subject)[0]

        start_time_date = datetime.strptime(session_data["start_time"][:10], "%Y-%m-%d")
        birth_date = datetime.strptime(subject_data["birth_date"], "%Y-%m-%d")
        age_days = (start_time_date - birth_date).days
        sex = subject_data.get("sex", None)

        trials = trials.copy()
        trials["eid"] = eid
        trials["subject"] = subject
        trials["sex"] = sex
        trials["age"] = age_days
        trials["n_trials_raw"] = n_trials_raw
        trials["trial_index"] = np.arange(len(trials), dtype=int)

        # Wheel data
        wheel_position, wheel_timestamps = load_wheel_data(eid)
        if wheel_position is None:
            print(f"[behav] skip {eid}: missing wheel")
            return None

        wheel_velocity = calc_wheel_velocity(wheel_position, wheel_timestamps)

        trial_position, trial_timestamps, trial_velocity = calc_trialwise_wheel(
            wheel_position, wheel_timestamps, wheel_velocity,
            trials["stimOn_times"], trials["feedback_times"]
        )

        (movement_onset_times,
         first_movement_onset_times,
         last_movement_onset_times,
         first_movement_directions,
         last_movement_directions) = calc_movement_onset_times(
            trial_timestamps, trial_velocity, trials["stimOn_times"]
        )

        trials["wheel_position"] = trial_position
        trials["wheel_timestamps"] = trial_timestamps
        trials["wheel_velocity"] = trial_velocity
        trials["first_movement_directions"] = first_movement_directions
        trials["last_movement_directions"] = last_movement_directions
        trials["movement_onset_times"] = movement_onset_times
        trials["first_movement_onset_times"] = first_movement_onset_times
        trials["last_movement_onset_times"] = last_movement_onset_times

        df = pd.DataFrame(trials)

        # RT / groups (can be NaN if movement/stim missing)
        df["reaction_time"] = df["first_movement_onset_times"] - df["stimOn_times"]
        df["rt_group"] = np.where(df["reaction_time"] < EARLY_RT, "early",
                          np.where(df["reaction_time"] > LATE_RT, "late", "normal"))

        return df

    except Exception as e:
        print(f"[behav] failed {eid}: {e}")
        return None


def build_behavior_dataset(eids: list[str]) -> pd.DataFrame:
    dfs = []
    for eid in eids:
        df = load_trials_with_wheel(eid)
        if df is not None and len(df) > 0:
            dfs.append(df)
            print(f"[behav] OK {eid} trials={len(df)} subj={df['subject'].iloc[0]}")
    if not dfs:
        return pd.DataFrame()
    big_df = pd.concat(dfs, ignore_index=True)
    return big_df


# =========================
# ATI CLEANING + SUBJECT ATI TABLE
# =========================
def make_ati_clean_trials(big_df: pd.DataFrame) -> pd.DataFrame:
    """
    ATI rules (your spec):
    - session had >=401 trials BEFORE dropping NaNs (already enforced in load)
    - drop trials with missing stimOn or missing first movement
    - remove last 40 trials of each session (by trial_index)
    """
    df = big_df.copy()

    # Drop NaNs required for RT
    df = df.dropna(subset=["stimOn_times", "first_movement_onset_times", "reaction_time", "probabilityLeft"])

    # Remove last 40 trials PER SESSION using trial_index
    # Keep trials with trial_index <= (max_index - 40)
    max_idx = df.groupby("eid")["trial_index"].transform("max")
    df = df[df["trial_index"] <= (max_idx - 40)]

    # Recompute group after filtering (safe)
    df["rt_group"] = np.where(df["reaction_time"] < EARLY_RT, "early",
                      np.where(df["reaction_time"] > LATE_RT, "late", "normal"))

    return df


def make_subject_ati_table(ati_trials: pd.DataFrame) -> pd.DataFrame:
    """
    Animal ATI:
    ATI = (N_early - N_late) / N_total across all included sessions of that animal
    Include animals with >=2 sessions (after ATI trial cleaning).
    Also report sex, age (mean age across sessions), and number of sessions.
    """
    # sessions per subject
    sess_counts = ati_trials.groupby("subject")["eid"].nunique()
    keep_subjects = sess_counts[sess_counts >= MIN_SESSIONS_PER_ANIMAL_FOR_ATI].index
    df = ati_trials[ati_trials["subject"].isin(keep_subjects)].copy()

    # Count early/late/total per subject
    g = df.groupby("subject")
    n_total = g.size()
    n_early = g.apply(lambda x: (x["rt_group"] == "early").sum())
    n_late = g.apply(lambda x: (x["rt_group"] == "late").sum())

    ati = (n_early - n_late) / n_total

    out = pd.DataFrame({
        "subject": ati.index,
        "ATI": ati.values,
        "n_trials_total": n_total.values,
        "n_early": n_early.values,
        "n_late": n_late.values,
        "n_sessions": df.groupby("subject")["eid"].nunique().reindex(ati.index).values,
        "sex": df.groupby("subject")["sex"].first().reindex(ati.index).values,
        "age_mean_days": df.groupby("subject")["age"].mean().reindex(ati.index).values,
    })

    return out.sort_values("ATI", ascending=False).reset_index(drop=True)


# =========================
# ITI WINDOWS FROM TRIAL TABLE
# =========================
def compute_iti_windows_for_trials(trials_df: pd.DataFrame):
    """
    Returns arrays:
      starts, ends, valid_mask
    where valid_mask indicates trials with a valid ITI window.
    Uses:
      start = feedback_times[t]
      next_stim = stimOn_times[t+1]
      ITI_len = next_stim - feedback
      end =
        if ITI_len < 2: min(feedback+1.5, next_stim-0.5)
        else: feedback+1.5
    Requires end > start.
    """
    fb = trials_df["feedback_times"].to_numpy(dtype=float)
    stim = trials_df["stimOn_times"].to_numpy(dtype=float)

    n = len(trials_df)
    starts = np.full(n, np.nan, dtype=float)
    ends = np.full(n, np.nan, dtype=float)

    # last trial has no "next stim"
    next_stim = np.roll(stim, -1)
    next_stim[-1] = np.nan

    starts[:] = fb
    iti_len = next_stim - fb

    # default end: fb + 1.5
    ends[:] = fb + ITI_MAX_DUR

    short_mask = iti_len < ITI_SHORT_CUTOFF
    # for short ITI: also guard pre-stim
    guarded_end = next_stim - ITI_PRESTIM_GUARD
    ends[short_mask] = np.minimum(ends[short_mask], guarded_end[short_mask])

    valid = np.isfinite(starts) & np.isfinite(ends) & (ends > starts)

    return starts, ends, valid


# =========================
# EPHYS: ROI selection and ITI spike counting fast
# =========================
def list_pids_for_session(eid: str) -> list[str]:
    """
    A session can have multiple probe insertions (pids).
    We'll consider all pids returned by Alyx.
    """
    ins = one.alyx.rest("insertions", "list", session=eid)
    pids = [x["id"] for x in ins]
    return pids


def load_good_clusters_by_roi(pid: str):
    """
    Loads spike sorting and returns:
      spikes_times, spikes_clusters, clusters_df
      roi_to_cluster_ids dict
    """
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    clusters_df = pd.DataFrame({
        "cluster_id": clusters["cluster_id"],
        "label": clusters.get("label", np.zeros_like(clusters["cluster_id"])),
        "acronym": clusters["acronym"],
    })

    good_cluster_ids = set(clusters_df.loc[clusters_df["label"] >= 0, "cluster_id"].tolist())

    roi_to_cluster_ids = {roi: [] for roi in ROIS}
    for cid, acr in zip(clusters_df["cluster_id"].values, clusters_df["acronym"].values):
        if cid not in good_cluster_ids:
            continue
        for roi in ROIS:
            if isinstance(acr, str) and acr.startswith(roi):
                roi_to_cluster_ids[roi].append(cid)
                break

    return spikes["times"], spikes["clusters"], roi_to_cluster_ids


def filter_clusters_by_task_fr(spike_times, spike_clusters, cluster_ids, task_start, task_end):
    """
    Keep clusters with firing rate >= MIN_NEURON_FR_HZ during [task_start, task_end]
    """
    if len(cluster_ids) == 0:
        return np.array([], dtype=int)

    cluster_ids = np.array(cluster_ids, dtype=int)

    # spikes in task window
    mask_t = (spike_times >= task_start) & (spike_times <= task_end)
    sc = spike_clusters[mask_t]

    dur = float(task_end - task_start)
    counts = np.array([(sc == cid).sum() for cid in cluster_ids], dtype=float)
    fr = counts / dur

    keep = cluster_ids[fr >= MIN_NEURON_FR_HZ]
    return keep


def count_spikes_in_intervals_fast(spike_times_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    For sorted spike_times, count spikes in each [start, end) interval using searchsorted.
    Returns counts per interval (len=number of intervals).
    """
    left = np.searchsorted(spike_times_sorted, starts, side="left")
    right = np.searchsorted(spike_times_sorted, ends, side="left")
    return (right - left).astype(int)


def compute_pid_roi_correlations(eid: str, pid: str, ati_trials_session: pd.DataFrame):
    """
    Compute correlations for one session+pid, for each ROI:
      - overall (all trials)
      - early/normal/late groups
    Returns rows for two outputs.
    """
    rows_all = []
    rows_groups = []

    # Need at least some trials
    if ati_trials_session is None or len(ati_trials_session) < 5:
        return rows_all, rows_groups

    # ITI windows from these trials (ATI-cleaned trials, as you requested)
    starts, ends, valid_iti = compute_iti_windows_for_trials(ati_trials_session)
    if valid_iti.sum() < 10:
        return rows_all, rows_groups

    # Prior vector and RT group
    prior = ati_trials_session["probabilityLeft"].to_numpy(dtype=float)
    rt_group = ati_trials_session["rt_group"].astype(str).to_numpy()
    subj = ati_trials_session["subject"].iloc[0]
    sex = ati_trials_session["sex"].iloc[0]
    age = ati_trials_session["age"].iloc[0]

    # Load spikes + ROI clusters
    try:
        spike_times, spike_clusters, roi_to_cluster_ids = load_good_clusters_by_roi(pid)
        roi_counts = {k: len(v) for k, v in roi_to_cluster_ids.items()}
        dbg(f"[dbg] eid={eid} pid={pid}: ROI cluster counts (pre-FR filter) {roi_counts}")
    except Exception as e:
        print(f"[ephys] fail load pid={pid} eid={eid}: {e}")
        return rows_all, rows_groups

    # Sort spikes once for fast counting (we'll subset per ROI)
    spike_times = np.asarray(spike_times, dtype=float)
    spike_clusters = np.asarray(spike_clusters, dtype=int)

    # Task window for FR filtering (use session trial span)
    task_start = float(np.nanmin(ati_trials_session["stimOn_times"].to_numpy()) - 2.0)
    task_end = float(np.nanmax(ati_trials_session["response_times"].to_numpy()) + 2.0) if "response_times" in ati_trials_session.columns else float(np.nanmax(ati_trials_session["feedback_times"].to_numpy()) + 2.0)

    for roi in ROIS:
        cluster_ids = roi_to_cluster_ids.get(roi, [])
        if len(cluster_ids) == 0:
            continue

        # FR filter (>=0.1 Hz)
        valid_clusters = filter_clusters_by_task_fr(
            spike_times, spike_clusters, cluster_ids, task_start, task_end
        )
        n_neurons = int(len(valid_clusters))
        if n_neurons < MIN_NEURONS:
            continue

        # ROI spikes only
        mask_roi = np.isin(spike_clusters, valid_clusters)
        roi_spike_times = np.sort(spike_times[mask_roi])

        # Total ITI spikes threshold (across valid ITI windows)
        total_iti_spikes = int(count_spikes_in_intervals_fast(
            roi_spike_times,
            starts[valid_iti],
            ends[valid_iti]
        ).sum())

        if total_iti_spikes < MIN_TOTAL_ITI_SPIKES:
            continue

        # Per-trial population FR[t] (Hz):
        # counts per trial / (duration * n_neurons)
        trial_counts = np.full(len(starts), np.nan, dtype=float)
        trial_counts[valid_iti] = count_spikes_in_intervals_fast(
            roi_spike_times,
            starts[valid_iti],
            ends[valid_iti]
        )
        durations = (ends - starts).astype(float)
        fr = trial_counts / (durations * float(n_neurons))

        # Only trials with valid ITI and finite prior and fr
        base_mask = valid_iti & np.isfinite(prior) & np.isfinite(fr)
        if base_mask.sum() < 10:
            continue

        # Overall correlation
        try:
            if np.nanstd(prior[base_mask]) == 0 or np.nanstd(fr[base_mask]) == 0:
                r_all = np.nan
            else:
                r_all = pearsonr(prior[base_mask], fr[base_mask]).statistic
        except Exception:
            r_all = np.nan

        rows_all.append({
            "eid": eid,
            "pid": pid,
            "region": roi,
            "subject": subj,
            "sex": sex,
            "age_days": age,
            "n_neurons": n_neurons,
            "total_iti_spikes": total_iti_spikes,
            "n_trials_used": int(base_mask.sum()),
            "pearson_r_all": r_all,
        })

        # Group correlations
        for gname in ["early", "normal", "late"]:
            m = base_mask & (rt_group == gname)
            if m.sum() < 10:
                r_g = np.nan
                n_used = int(m.sum())
            else:
                try:
                    if np.nanstd(prior[m]) == 0 or np.nanstd(fr[m]) == 0:
                        r_g = np.nan
                    else:
                        r_g = pearsonr(prior[m], fr[m]).statistic
                except Exception:
                    r_g = np.nan
                n_used = int(m.sum())

            rows_groups.append({
                "eid": eid,
                "pid": pid,
                "region": roi,
                "subject": subj,
                "sex": sex,
                "age_days": age,
                "n_neurons": n_neurons,
                "total_iti_spikes": total_iti_spikes,
                "group": gname,
                "n_trials_used_group": n_used,
                "pearson_r_group": r_g,
            })

    return rows_all, rows_groups


# =========================
# MAIN PIPELINE
# =========================
def save_df(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main():
    # 1) Read EIDs from txt
    eids = read_session_ids_from_txt(TXT_PATH)
    print(f"Found {len(eids)} session IDs in txt")

    if DEBUG_N_SESSIONS is not None:
        eids = eids[:DEBUG_N_SESSIONS]
        print(f"DEBUG: running only first {len(eids)} sessions")

    # 2) Build behavior dataset (WHOLE behavioral dataset output #1)
    big_df = build_behavior_dataset(eids)
    if big_df.empty:
        print("No behavioral data produced. Exiting.")
        return

    # Save behavioral dataset
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    behav_path = OUT_DIR / f"behavior_trials_with_wheel_{ts}.csv"
    save_df(big_df, behav_path)

    # 3) Make ATI-clean trials + subject ATI table (output #2)
    ati_trials = make_ati_clean_trials(big_df)

    # Enforce subject >=2 sessions after cleaning
    subj_ati = make_subject_ati_table(ati_trials)

    ati_trials_path = OUT_DIR / f"ati_clean_trials_{ts}.csv"
    subj_ati_path = OUT_DIR / f"subject_ati_table_{ts}.csv"
    save_df(ati_trials, ati_trials_path)
    save_df(subj_ati, subj_ati_path)

    # Keep only subjects that survive ATI filtering for downstream correlation files
    keep_subjects = set(subj_ati["subject"].tolist())
    ati_trials = ati_trials[ati_trials["subject"].isin(keep_subjects)].copy()

    # 4) Correlations for each session/pid/ROI using ATI-clean trials
    # Group by session
    eid_to_trials = {eid: df for eid, df in ati_trials.groupby("eid", sort=False)}

    def process_one_eid(eid: str):
        trials_session = eid_to_trials.get(eid, None)
        if trials_session is None or len(trials_session) < 10:
            return [], []

        # list pids (multiple probes possible)
        pids = list_pids_for_session(eid)
        dbg(f"[dbg] eid={eid}: found {len(pids)} pids")
        if len(pids) == 0:
            dbg(f"[dbg][STOP] eid={eid}: no pids returned. Check ONE auth / Alyx access.")
            return [], []

        all_rows = []
        grp_rows = []
        for pid in pids:
            r_all, r_grp = compute_pid_roi_correlations(eid, pid, trials_session)
            all_rows.extend(r_all)
            grp_rows.extend(r_grp)

        print(f"[corr] eid={eid} pids={len(pids)} rows_all={len(all_rows)} rows_grp={len(grp_rows)}")
        return all_rows, grp_rows

    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_one_eid)(eid) for eid in eid_to_trials.keys()
    )

    rows_all = []
    rows_grp = []
    for ra, rg in results:
        rows_all.extend(ra)
        rows_grp.extend(rg)

    corr_all_df = pd.DataFrame(rows_all)
    corr_grp_df = pd.DataFrame(rows_grp)

    corr_all_path = OUT_DIR / f"iti_prior_corr_session_pid_roi_alltrials_{ts}.csv"
    corr_grp_path = OUT_DIR / f"iti_prior_corr_session_pid_roi_rtgroups_{ts}.csv"
    save_df(corr_all_df, corr_all_path)
    save_df(corr_grp_df, corr_grp_path)

    print("\nDONE.")
    print(f"Behavior trials: {behav_path}")
    print(f"ATI clean trials: {ati_trials_path}")
    print(f"Subject ATI table: {subj_ati_path}")
    print(f"Corr all trials: {corr_all_path}")
    print(f"Corr RT groups: {corr_grp_path}")


if __name__ == "__main__":
    main()
