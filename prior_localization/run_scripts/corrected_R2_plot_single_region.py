
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
PKL_PATH = Path("prior_localization_groups_output/roi_summary_MOp.pkl")

with open(PKL_PATH, "rb") as f:
    rows = pickle.load(f)

df = pd.DataFrame(rows)

ROI = "MOp"
groups = ["fast", "normal", "slow"]

df = df[df["region"] == ROI].copy()
df = df[np.isfinite(df["R2_corrected"])]

print("N per group:")
print(df.groupby("group").size())


plt.figure(figsize=(5, 4))

plt.boxplot(
    [df[df.group == g]["R2_corrected"] for g in groups],
    labels=groups,
    showfliers=False
)

plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.ylabel("Corrected R² (session − pseudo)")
plt.title("Corrected prior decoding")

plt.tight_layout()
plt.show()
