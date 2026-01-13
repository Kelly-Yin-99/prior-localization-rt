import ast
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from one.api import ONE
from iblatlas.atlas import AllenAtlas
from brainbox.io.one import SessionLoader, SpikeSortingLoader


# =========================
# USER CONFIG
# =========================


TXT_PATH = "/home/wg-yin/session_ids_for_behav_analysis.txt"  # <-- change if needed



# Output directory on RIS
OUT_DIR = Path("/storage1/fs1/hiratani/Active/shared/ibl_space/derived/prior_iti_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


DEBUG_N_SESSIONS = None  # None to run all

# ROIs (collapsed by prefix)
ROIS = ["MOp", "MOs", "ACAd", "ORBvl"]

# Session inclusion (raw trials)
MIN_TRIALS_SESSION_RAW = 401

# ITI rule
ITI_MAX_DUR = 1.5
ITI_SHORT_CUTOFF = 2.0
ITI_PRESTIM_GUARD = 0.5

# Ephys QC
MIN_NEURONS = 10
MIN_NEURON_FR_HZ = 0.1
MIN_TOTAL_ITI_SPIKES = 10_000

# RT grouping
EARLY_RT = 0.08
LATE_RT = 1.25

# ATI / session-level filtering
REMOVE_LAST_N_TRIALS = 40
MIN_SESSIONS_PER_SUBJECT_ATI = 2

# Debug printing
VERBOSE_SKIP_REASONS = True


def log(msg: str):
    print(msg)


def skiplog(msg: str):
    if VERBOSE_SKIP_REASONS:
        print(msg)


# =========================
# ONE / Atlas
# =========================
atlas = AllenAtlas()
ONE.setup(
    base_url="https://openalyx.internationalbrainlab.org",
    silent=True,
    cache_dir=Path("/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org"),
)
one = ONE(
    base_url="https://openalyx.internationalbrainlab.org",
    username=os.getenv("ALYX_LOGIN"),
    password=os.getenv("ALYX_PASSWORD"),
    silent=True,
)



# =========================
# UTIL: read session IDs
# =========================
def read_session_ids_from_txt(txt_path: str) -> List[str]:
    session_ids: List[str] = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids_in_line = ast.literal_eval(line)
            session_ids.extend(ids_in_line)
    return list(dict.fromkeys(session_ids))


# =========================
# WHEEL / FIRST MOVEMENT
# =========================
def load_wheel_data(eid: str):
    try:
        wheel_data = one.load_object(eid, "wheel", collection="alf")
        return wheel_data["position"], wheel_data["timestamps"]
    except Exception as e:
        skiplog(f"[wheel][skip] eid={eid}: no wheel data: {e}")
        return None, None


def calc_wheel_velocity(position, timestamps):
    wheel_velocity = [0.0]
    for widx in range(len(position) - 1):
        dt = timestamps[widx + 1] - timestamps[widx]
        wheel_velocity.append((position[widx + 1] - position[widx]) / dt if dt != 0 else 0.0)
    return wheel_velocity


def calc_trialwise_wheel(position, timestamps, velocity, stimOn_times, feedback_times):
    stimOn_pre_duration = 0.3
    n_trials = len(stimOn_times)

    trial_position = [[] for _ in range(n_trials)]
    trial_timestamps = [[] for _ in range(n_trials)]
    trial_velocity = [[] for _ in range(n_trials)]

    tridx = 0
    for tsidx, t in enumerate(timestamps):
        while tridx < n_trials - 1 and t > stimOn_times[tridx + 1] - stimOn_pre_duration:
            tridx += 1

        if stimOn_times[tridx] - stimOn_pre_duration <= t < feedback_times[tridx]:
            trial_position[tridx].append(position[tsidx])
            trial_timestamps[tridx].append(timestamps[tsidx])
            trial_velocity[tridx].append(velocity[tsidx])

    return trial_position, trial_timestamps, trial_velocity


def calc_movement_onset_times(trial_timestamps, trial_velocity, stimOn_times):
    speed_threshold = 0.5
    duration_threshold = 0.05

    first_movement_onset_times = np.full(len(trial_timestamps), np.nan, dtype=float)

    for tridx in range(len(trial_timestamps)):
        cm_dur = 0.0
        tprev = (stimOn_times.iloc[tridx] - 0.3) if hasattr(stimOn_times, "iloc") else (stimOn_times[tridx] - 0.3)

        for tpidx in range(len(trial_timestamps[tridx])):
            t = trial_timestamps[tridx][tpidx]
            cm_dur += (t - tprev)

            if abs(trial_velocity[tridx][tpidx]) > speed_threshold:
                if cm_dur > duration_threshold:
                    first_movement_onset_times[tridx] = t
                    break
                cm_dur = 0.0

            tprev = t

    return first_movement_onset_times


# =========================
# BEHAVIOR: load session + compute wheel RT
# =========================
def load_trials_with_wheel(eid: str) -> Optional[pd.DataFrame]:
    try:
        sl = SessionLoader(eid=eid, one=one)
        sl.load_trials()
        trials = sl.trials
        if trials is None or len(trials) == 0:
            skiplog(f"[behav][skip] eid={eid}: no trials")
            return None

        n_trials_raw = len(trials)
        if n_trials_raw < MIN_TRIALS_SESSION_RAW:
            skiplog(f"[behav][skip] eid={eid}: n_trials_raw={n_trials_raw} < {MIN_TRIALS_SESSION_RAW}")
            return None

        session_data = one.alyx.rest("sessions", "read", id=eid)
        subject = session_data["subject"]
        subject_data = one.alyx.rest("subjects", "list", nickname=subject)[0]
        sex = subject_data.get("sex", None)

        # age in days
        start_time_date = datetime.strptime(session_data["start_time"][:10], "%Y-%m-%d")
        birth_date = datetime.strptime(subject_data["birth_date"], "%Y-%m-%d")
        age_days = (start_time_date - birth_date).days

        trials = trials.copy()
        trials["eid"] = eid
        trials["subject"] = subject
        trials["sex"] = sex
        trials["age_days"] = age_days
        trials["n_trials_raw"] = n_trials_raw
        trials["trial_index"] = np.arange(len(trials), dtype=int)

        # wheel → first movement
        wheel_position, wheel_timestamps = load_wheel_data(eid)
        if wheel_position is None:
            return None

        wheel_velocity = calc_wheel_velocity(wheel_position, wheel_timestamps)
        trial_position, trial_timestamps, trial_velocity = calc_trialwise_wheel(
            wheel_position, wheel_timestamps, wheel_velocity,
            trials["stimOn_times"], trials["feedback_times"]
        )

        first_movement_onset_times = calc_movement_onset_times(
            trial_timestamps, trial_velocity, trials["stimOn_times"]
        )

        trials["first_movement_onset_times"] = first_movement_onset_times
        df = pd.DataFrame(trials)

        # RT is defined from your wheel first movement
        df["reaction_time"] = df["first_movement_onset_times"] - df["stimOn_times"]
        df["rt_group"] = np.where(df["reaction_time"] < EARLY_RT, "early",
                          np.where(df["reaction_time"] > LATE_RT, "late", "normal"))

        return df

    except Exception as e:
        skiplog(f"[behav][skip] eid={eid}: failed load: {e}")
        return None


def build_behavior_dataset(eids: List[str]) -> pd.DataFrame:
    dfs = []
    for eid in eids:
        df = load_trials_with_wheel(eid)
        if df is not None and len(df) > 0:
            dfs.append(df)
            log(f"[behav] OK eid={eid} trials={len(df)} subj={df['subject'].iloc[0]}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# =========================
# ITI windows
# =========================
def compute_iti_windows_for_trials(trials_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fb = trials_df["feedback_times"].to_numpy(dtype=float)
    stim = trials_df["stimOn_times"].to_numpy(dtype=float)

    next_stim = np.roll(stim, -1)
    next_stim[-1] = np.nan

    starts = fb
    ends = fb + ITI_MAX_DUR

    iti_len = next_stim - fb
    short_mask = iti_len < ITI_SHORT_CUTOFF
    guarded_end = next_stim - ITI_PRESTIM_GUARD
    ends[short_mask] = np.minimum(ends[short_mask], guarded_end[short_mask])

    valid = np.isfinite(starts) & np.isfinite(ends) & (ends > starts)
    return starts, ends, valid


# =========================
# EPHYS HELPERS
# =========================
def list_pids_for_session(eid: str) -> List[str]:
    ins = one.alyx.rest("insertions", "list", session=eid)
    return [x["id"] for x in ins]


def load_clusters_by_roi_for_pid(pid: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[int]]]:
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    labels = clusters.get("label", np.zeros_like(clusters["cluster_id"]))
    good_cluster_ids = set(clusters["cluster_id"][labels >= 0].astype(int).tolist())

    roi_to_cluster_ids = {roi: [] for roi in ROIS}
    for cid, acr in zip(clusters["cluster_id"], clusters["acronym"]):
        cid = int(cid)
        if cid not in good_cluster_ids:
            continue
        if not isinstance(acr, str):
            continue
        for roi in ROIS:
            if acr.startswith(roi):
                roi_to_cluster_ids[roi].append(cid)
                break

    return np.asarray(spikes["times"], float), np.asarray(spikes["clusters"], int), roi_to_cluster_ids


def filter_clusters_by_task_fr(spike_times, spike_clusters, cluster_ids, task_start, task_end) -> np.ndarray:
    if len(cluster_ids) == 0:
        return np.array([], dtype=int)

    cluster_ids = np.array(cluster_ids, dtype=int)
    mask_t = (spike_times >= task_start) & (spike_times <= task_end)
    sc = spike_clusters[mask_t]
    dur = float(task_end - task_start)

    counts = np.array([(sc == cid).sum() for cid in cluster_ids], dtype=float)
    fr = counts / dur
    return cluster_ids[fr >= MIN_NEURON_FR_HZ]


def count_spikes_in_intervals_fast(spike_times_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    left = np.searchsorted(spike_times_sorted, starts, side="left")
    right = np.searchsorted(spike_times_sorted, ends, side="left")
    return (right - left).astype(int)


# =========================
# CORRELATION METRICS (per pid per roi)
# =========================
def compute_pid_roi_metrics(
    eid: str,
    pid: str,
    trials_session: pd.DataFrame,
    label: str,
) -> Tuple[Dict[str, dict], List[str]]:
    """
    Returns:
      - roi_metrics: {roi: metrics_dict}
      - skip_msgs: list of skip strings (printed by caller)
    """
    skip_msgs: List[str] = []
    roi_metrics: Dict[str, dict] = {}

    # For correlation we need stimOn + first_movement, and also probabilityLeft + feedback_times
    need_cols = ["stimOn_times", "first_movement_onset_times", "feedback_times", "probabilityLeft"]
    missing = [c for c in need_cols if c not in trials_session.columns]
    if missing:
        skip_msgs.append(f"[corr:{label}][skip] eid={eid} pid={pid}: missing columns {missing}")
        return roi_metrics, skip_msgs

    # You asked: behavioral data only drops stimOn/first_movement when None.
    # We'll enforce that here for the correlation view.
    df = trials_session.dropna(subset=["stimOn_times", "first_movement_onset_times"]).copy()

    # correlation needs probabilityLeft to exist
    df = df.dropna(subset=["probabilityLeft", "feedback_times"]).copy()

    if len(df) < 20:
        skip_msgs.append(f"[corr:{label}][skip] eid={eid} pid={pid}: too few trials after dropna (n={len(df)})")
        return roi_metrics, skip_msgs

    df["reaction_time"] = df["first_movement_onset_times"] - df["stimOn_times"]
    df["rt_group"] = np.where(df["reaction_time"] < EARLY_RT, "early",
                      np.where(df["reaction_time"] > LATE_RT, "late", "normal"))

    starts, ends, valid_iti = compute_iti_windows_for_trials(df)
    if valid_iti.sum() < 10:
        skip_msgs.append(f"[corr:{label}][skip] eid={eid} pid={pid}: valid ITI trials < 10")
        return roi_metrics, skip_msgs

    prior = df["probabilityLeft"].to_numpy(dtype=float)
    rt_group = df["rt_group"].astype(str).to_numpy()

    # task bounds for neuron FR filter
    task_start = float(np.nanmin(df["stimOn_times"].to_numpy()) - 2.0)
    task_end = float(np.nanmax(df["feedback_times"].to_numpy()) + 2.0)

    # load spikes
    try:
        spike_times, spike_clusters, roi_to_cluster_ids = load_clusters_by_roi_for_pid(pid)
    except Exception as e:
        skip_msgs.append(f"[corr:{label}][skip] eid={eid} pid={pid}: spike load failed: {e}")
        return roi_metrics, skip_msgs

    any_roi_passed = False

    for roi in ROIS:
        cluster_ids = roi_to_cluster_ids.get(roi, [])
        if len(cluster_ids) == 0:
            skip_msgs.append(f"[corr:{label}][skip] eid={eid} pid={pid} roi={roi}: 0 clusters (ROI match)")
            continue

        valid_clusters = filter_clusters_by_task_fr(spike_times, spike_clusters, cluster_ids, task_start, task_end)
        n_neurons = int(len(valid_clusters))
        if n_neurons < MIN_NEURONS:
            skip_msgs.append(f"[corr:{label}][skip] eid={eid} pid={pid} roi={roi}: n_neurons={n_neurons} < {MIN_NEURONS}")
            continue

        mask_roi = np.isin(spike_clusters, valid_clusters)
        roi_spike_times = np.sort(spike_times[mask_roi])

        # spike counts per ITI interval
        trial_counts = np.full(len(starts), np.nan, dtype=float)
        trial_counts[valid_iti] = count_spikes_in_intervals_fast(
            roi_spike_times, starts[valid_iti], ends[valid_iti]
        )
        total_iti_spikes = int(np.nansum(trial_counts[valid_iti]))
        if total_iti_spikes < MIN_TOTAL_ITI_SPIKES:
            skip_msgs.append(f"[corr:{label}][skip] eid={eid} pid={pid} roi={roi}: total_iti_spikes={total_iti_spikes} < {MIN_TOTAL_ITI_SPIKES}")
            continue

        durations = (ends - starts).astype(float)
        fr = trial_counts / (durations * float(n_neurons))

        base_mask = valid_iti & np.isfinite(prior) & np.isfinite(fr)
        n_trials_used = int(base_mask.sum())
        if n_trials_used < 10:
            skip_msgs.append(f"[corr:{label}][skip] eid={eid} pid={pid} roi={roi}: n_trials_used={n_trials_used} < 10")
            continue

        # Pearson r all
        if np.nanstd(prior[base_mask]) == 0 or np.nanstd(fr[base_mask]) == 0:
            r_all = np.nan
            skip_msgs.append(f"[corr:{label}][skip] eid={eid} pid={pid} roi={roi}: zero variance (prior or FR) -> r=nan")
        else:
            r_all = float(pearsonr(prior[base_mask], fr[base_mask]).statistic)

        # "correlation for blocks": we store block means (mean FR per prior value)
        pvals = prior[base_mask]
        frvals = fr[base_mask]
        block_means = {}
        for pv in sorted(set(pvals.tolist())):
            m = (pvals == pv)
            block_means[str(pv)] = float(np.nanmean(frvals[m])) if m.any() else np.nan

        # also store r within each RT group (optional but useful)
        r_by_group = {}
        n_by_group = {}
        for gname in ["early", "normal", "late"]:
            m = base_mask & (rt_group == gname)
            n_by_group[gname] = int(m.sum())
            if m.sum() < 10 or np.nanstd(prior[m]) == 0 or np.nanstd(fr[m]) == 0:
                r_by_group[gname] = np.nan
            else:
                r_by_group[gname] = float(pearsonr(prior[m], fr[m]).statistic)

        roi_metrics[roi] = {
            "pid": pid,
            "n_neurons": n_neurons,
            "total_iti_spikes": total_iti_spikes,
            "iti_trial_count": int(valid_iti.sum()),
            "n_trials_used": n_trials_used,
            "pearson_r_all": r_all,
            "block_mean_FR": block_means,             # mean FR per block prior value
            "pearson_r_by_rt_group": r_by_group,      # optional
            "n_trials_by_rt_group": n_by_group,       # optional
        }

        any_roi_passed = True

    if not any_roi_passed:
        skip_msgs.append(f"[corr:{label}][skip] eid={eid} pid={pid}: no ROIs passed QC (neurons/spikes/trials)")

    return roi_metrics, skip_msgs


# =========================
# OUTPUT STRUCTURE BUILDER
# =========================
def init_corr_container() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    region -> session_id -> session_entry
    session_entry:
      {
        "eid": ..., "subject": ..., "sex": ..., "age_days": ...,
        "pids": { pid: { ... per pid metrics ... } }
      }
    """
    return {roi: {} for roi in ROIS}


def upsert_session_pid(
    container: Dict[str, Dict[str, Any]],
    roi: str,
    eid: str,
    subject: str,
    sex: Any,
    age_days: int,
    pid: str,
    pid_metrics_for_roi: dict,
):
    if eid not in container[roi]:
        container[roi][eid] = {
            "eid": eid,
            "subject": subject,
            "sex": sex,
            "age_days": age_days,
            "pids": {}
        }
    container[roi][eid]["pids"][pid] = pid_metrics_for_roi


# =========================
# ATI: subject-level table
# =========================
def compute_subject_ati_and_keep_subjects(df_for_ati: pd.DataFrame) -> Tuple[pd.DataFrame, set]:
    """
    df_for_ati should already have:
      - stimOn + first_movement present (since RT depends on them)
      - probabilityLeft present (not strictly needed for ATI but consistent)
    Keep subjects with >=2 sessions.
    """
    if df_for_ati.empty:
        return pd.DataFrame(), set()

    df = df_for_ati.copy()
    df["reaction_time"] = df["first_movement_onset_times"] - df["stimOn_times"]
    df["rt_group"] = np.where(df["reaction_time"] < EARLY_RT, "early",
                      np.where(df["reaction_time"] > LATE_RT, "late", "normal"))

    sess_per_subj = df.groupby("subject")["eid"].nunique()
    keep_subjects = set(sess_per_subj[sess_per_subj >= MIN_SESSIONS_PER_SUBJECT_ATI].index.tolist())
    df = df[df["subject"].isin(keep_subjects)].copy()

    if df.empty:
        return pd.DataFrame(), set()

    subj = df["subject"]
    n_total = subj.groupby(subj).size()
    n_early = df["rt_group"].eq("early").groupby(subj).sum()
    n_late = df["rt_group"].eq("late").groupby(subj).sum()
    ati = (n_early - n_late) / n_total

    session_ids = (
        df.groupby("subject", sort=False)["eid"]
        .agg(lambda s: list(dict.fromkeys(s.tolist())))
    )

    out = pd.DataFrame({
        "subject": ati.index.to_list(),
        "ATI": ati.to_numpy(),
        "n_trials_total": n_total.reindex(ati.index).to_numpy(),
        "n_early": n_early.reindex(ati.index).to_numpy(),
        "n_late": n_late.reindex(ati.index).to_numpy(),
        "n_sessions": df.groupby("subject")["eid"].nunique().reindex(ati.index).to_numpy(),
        "session_ids_json": session_ids.reindex(ati.index).apply(json.dumps).to_numpy(),
    }).sort_values("ATI", ascending=False).reset_index(drop=True)

    return out, keep_subjects


# =========================
# MAIN
# =========================
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    eids = read_session_ids_from_txt(TXT_PATH)
    log(f"Found {len(eids)} session IDs in txt")
    if DEBUG_N_SESSIONS is not None:
        eids = eids[:DEBUG_N_SESSIONS]
        log(f"DEBUG: running only first {len(eids)} sessions")

    # 1) Combined behavior (no aggressive dropping at save time)
    big_df = build_behavior_dataset(eids)
    if big_df.empty:
        log("No behavioral data produced. Exiting.")
        return

    behav_csv = OUT_DIR / f"behavior_combined_{ts}.csv"
    big_df.to_csv(behav_csv, index=False)
    log(f"Saved: {behav_csv}")

    # 2) Prepare filtered views
    # You asked: for behavioral data filtering, only drop trials where stimOn or first movement is None
    base_behav = big_df.dropna(subset=["stimOn_times", "first_movement_onset_times"]).copy()

    # trial-level correlations use ALL sessions (raw>=401 already), no last40, no subject>=2
    trialcorr_df = base_behav.dropna(subset=["probabilityLeft", "feedback_times"]).copy()

    # session-level/ATI correlations:
    # - remove last 40 trials
    # - then keep subjects with >=2 sessions
    sessioncorr_df = base_behav.copy()
    max_idx = sessioncorr_df.groupby("eid")["trial_index"].transform("max")
    sessioncorr_df = sessioncorr_df[sessioncorr_df["trial_index"] <= (max_idx - REMOVE_LAST_N_TRIALS)].copy()
    sessioncorr_df = sessioncorr_df.dropna(subset=["probabilityLeft", "feedback_times"]).copy()

    # ATI + keep subjects>=2 (based on sessioncorr_df)
    subj_ati, keep_subjects = compute_subject_ati_and_keep_subjects(sessioncorr_df)
    ati_csv = OUT_DIR / f"ATI_subject_table_{ts}.csv"
    subj_ati.to_csv(ati_csv, index=False)
    log(f"Saved: {ati_csv}")

    # session-level correlations only for those subjects
    sessioncorr_df = sessioncorr_df[sessioncorr_df["subject"].isin(keep_subjects)].copy()

    # 3) Build two combined correlation containers
    trial_corr_container = init_corr_container()
    session_corr_container = init_corr_container()

    # Also keep skip logs in one file
    skiplog_path = OUT_DIR / f"corr_skip_reasons_{ts}.txt"
    skip_lines: List[str] = []

    def process_dataset(label: str, df: pd.DataFrame, container: Dict[str, Dict[str, Any]]):
        if df.empty:
            log(f"[corr:{label}] dataset empty, skipping.")
            return

        eid_groups = list(df.groupby("eid", sort=False))
        log(f"[corr:{label}] sessions to process: {len(eid_groups)}")

        for eid, df_sess in eid_groups:
            subject = str(df_sess["subject"].iloc[0])
            sex = df_sess["sex"].iloc[0]
            age_days = int(df_sess["age_days"].iloc[0])

            try:
                pids = list_pids_for_session(eid)
            except Exception as e:
                msg = f"[corr:{label}][skip] eid={eid}: list_pids failed: {e}"
                skip_lines.append(msg)
                skiplog(msg)
                continue

            if len(pids) == 0:
                msg = f"[corr:{label}][skip] eid={eid}: no pids/insertions"
                skip_lines.append(msg)
                skiplog(msg)
                continue

            for pid in pids:
                roi_metrics, msgs = compute_pid_roi_metrics(eid=eid, pid=pid, trials_session=df_sess, label=label)
                for m in msgs:
                    skip_lines.append(m)
                    skiplog(m)

                # write into container: region -> session -> pid
                for roi, met in roi_metrics.items():
                    upsert_session_pid(
                        container=container,
                        roi=roi,
                        eid=eid,
                        subject=subject,
                        sex=sex,
                        age_days=age_days,
                        pid=pid,
                        pid_metrics_for_roi=met
                    )

    # A) trial-level combined correlation file
    process_dataset(label="trial_level", df=trialcorr_df, container=trial_corr_container)

    # B) session-level (ATI-subject filtered) combined correlation file
    process_dataset(label="session_level", df=sessioncorr_df, container=session_corr_container)

    # 4) Save correlation containers (single file each, not CSV)
    trial_corr_path = OUT_DIR / f"corr_trial_level_all_sessions_{ts}.pkl"
    session_corr_path = OUT_DIR / f"corr_session_level_all_sessions_{ts}.pkl"

    with open(trial_corr_path, "wb") as f:
        pickle.dump(trial_corr_container, f)
    log(f"Saved: {trial_corr_path}")

    with open(session_corr_path, "wb") as f:
        pickle.dump(session_corr_container, f)
    log(f"Saved: {session_corr_path}")

    # 5) Save skip reasons
    with open(skiplog_path, "w") as f:
        for line in skip_lines:
            f.write(line + "\n")
    log(f"Saved: {skiplog_path}")

    log("\nDONE.")
    log(f"Behavior: {behav_csv}")
    log(f"ATI table: {ati_csv}")
    log(f"Trial corr (single file): {trial_corr_path}")
    log(f"Session corr (single file): {session_corr_path}")
    log(f"Skip reasons: {skiplog_path}")


if __name__ == "__main__":
    main()
