
import numpy as np
import pandas as pd
from one.api import ONE

from prior_localization.my_rt import (
    load_wheel_data, calc_wheel_velocity, calc_trialwise_wheel, calc_movement_onset_times
)

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
    trial_pos, trial_ts, trial_vel = calc_trialwise_wheel(
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



from pathlib import Path
import json
from one.api import ONE


from prior_localization.fit_data import fit_session_ephys


def run_one_eid(eid: str, out_root: Path, n_pseudo: int = 100, debug: bool = True):
    one = ONE()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # EXACTLY how they do it in the original example
    probe_name = one.eid2pid(eid)[1]

    ses = one.alyx.rest("sessions", "read", id=eid)
    subject = ses["subject"]

    df, n_raw_trials = compute_trials_with_my_rt(one, eid)

    # YOUR QC: >=401 trials BEFORE dropna
    if n_raw_trials < 401:
        raise RuntimeError(f"Skip {eid}: only {n_raw_trials} trials (<401) before dropna")

    masks, meta = make_fast_normal_slow_masks(df)
    (out_root / f"{eid}_rt_group_counts.json").write_text(json.dumps(meta, indent=2))

    if debug:
        print(f"[DEBUG] {eid}: raw_trials={n_raw_trials} meta={meta}")
        print(f"[DEBUG] {eid}: probe_name={probe_name}")

    pseudo_ids = [-1] + list(range(1, n_pseudo + 1))

    results = {}
    for group_label in ["fast", "normal", "slow"]:
        n_group = int(masks[group_label].sum())
        print(f"  -> {eid} group={group_label} n={n_group}")

        group_dir = out_root / eid / group_label
        group_dir.mkdir(parents=True, exist_ok=True)

        res = fit_session_ephys(
            one=one,
            session_id=eid,
            subject=subject,
            probe_name=probe_name,     # keep original probe behavior
            output_dir=group_dir,      # Path is safest
            pseudo_ids=pseudo_ids,
            target="pLeft",
            align_event="stimOn_times",
            time_window=(-0.6, -0.1),  # match original default example
            model="optBay",
            n_runs=2,                  # change to your desired repeats

            # disable their RT filtering inside the function
            min_rt=None,
            max_rt=None,

            # NEW PATCHED ARGS
            trials_df=df,
            trial_mask=masks[group_label],
            group_label=group_label,
            debug=debug,
        )
        results[group_label] = res

    return results, meta


def main():
    out_root = Path("./prior_localization_rt_groups_output")

    eids = [
       # "d85c454e-8737-4cba-b6ad-b2339429d99b",
        '56956777-dca5-468c-87cb-78150432cc57'

    ]

    for eid in eids:
        print(f"\n=== Running eid={eid} ===")
        try:
            results, meta = run_one_eid(eid, out_root=out_root, n_pseudo=100, debug=True)
            print("DONE", eid, meta)
        except Exception as e:
            print("FAILED", eid, e)


if __name__ == "__main__":
    main()
