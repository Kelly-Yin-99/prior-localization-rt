from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- 1) point to your saved pickle ---
pkl_path = Path(
    "prior_localization_rt_groups_output/56956777-dca5-468c-87cb-78150432cc57/normal/"
    "NYU-11/56956777-dca5-468c-87cb-78150432cc57/"
    "VPM_merged_probes_pseudo_ids_-1_100_normal.pkl"
)

# --- 2) load ---
with open(pkl_path, "rb") as f:
    decoding_results = pickle.load(f)

# --- 3) inspect keys / metadata ---
print(f"decoding keys are {decoding_results.keys()}")
print(
    f"decoding session: {decoding_results['eid']}\n"
    f"decoding subject: {decoding_results['subject']}\n"
    f"decoding region: {decoding_results['region']}\n"
    f"number of units in region: {decoding_results['N_units']}"
)
print(f"number of decodings performed: {len(decoding_results['fit'])}")

# --- 4) grab predictions/targets for the REAL session (pseudo_id == -1) ---
predictions = np.asarray(
    [fit["predictions_test"] for fit in decoding_results["fit"] if fit["pseudo_id"] == -1]
).squeeze()

target = np.asarray(
    [fit["target"] for fit in decoding_results["fit"] if fit["pseudo_id"] == -1]
).squeeze()

# --- 5) plot run 0 (your example assumes n_runs=2 so predictions[0, :] exists) ---
plt.figure(figsize=(12, 4))
plt.plot(predictions[0, :], label="decoded prior")
plt.plot(target[0, :], label="target")
plt.legend()
plt.xlabel("trial")
plt.ylabel("pLeft")
plt.show()

# --- 6) compute uncorrected / pseudo / corrected R^2 ---
pseudo_Rsquared = []
Rsquared = None

for fit in decoding_results["fit"]:
    if fit["pseudo_id"] == -1:
        Rsquared = fit["scores_test_full"]
    else:
        pseudo_Rsquared.append(fit["scores_test_full"])

print(f"Uncorrected session Rsquared: {Rsquared}")
print(f"Average pseudosession Rsquared*: {np.mean(pseudo_Rsquared)}")
print(f"Corrected Rsquared: {Rsquared - np.mean(pseudo_Rsquared)}")
