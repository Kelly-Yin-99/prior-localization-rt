import numpy as np

def load_wheel_data(one, eid):
    try:
        wheel_data = one.load_object(eid, 'wheel', collection='alf')
        wheel_position = wheel_data['position']
        wheel_timestamps = wheel_data['timestamps']
        return wheel_position, wheel_timestamps
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
    movement_onset_counts = np.zeros(len(trial_timestamps))

    for tridx in range(len(trial_timestamps)):
        movement_onset_times.append([])
        movement_directions.append([])
        cm_dur = 0.0
        for tpidx in range(len(trial_timestamps[tridx])):
            t = trial_timestamps[tridx][tpidx]
            if tpidx == 0:
                # stimOn_times could be pandas series or numpy array
                tprev = stimOn_times.iloc[tridx] - 0.3 if hasattr(stimOn_times, "iloc") else stimOn_times[tridx] - 0.3
            cm_dur += (t - tprev)
            if abs(trial_velocity[tridx][tpidx]) > speed_threshold:
                if cm_dur > duration_threshold:
                    movement_onset_times[tridx].append(t)
                    movement_directions[tridx].append(np.sign(trial_velocity[tridx][tpidx]))
                cm_dur = 0.0
            tprev = t

        movement_onset_counts[tridx] = len(movement_onset_times[tridx])
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
