
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
# PKL_PATH = Path("prior_localization_groups_output/roi_summary_MOp.pkl")
#
# with open(PKL_PATH, "rb") as f:
#     rows = pickle.load(f)
#
# df = pd.DataFrame(rows)
#
# ROI = "MOp"
# groups = ["fast", "normal", "slow"]
#
# df = df[df["region"] == ROI].copy()
# df = df[np.isfinite(df["R2_corrected"])]
#
# print("N per group:")
# print(df.groupby("group").size())
#
#
# plt.figure(figsize=(5, 4))
#
# plt.boxplot(
#     [df[df.group == g]["R2_corrected"] for g in groups],
#     labels=groups,
#     showfliers=False
# )
#
# plt.axhline(0, color="k", linestyle="--", linewidth=1)
# plt.ylabel("Corrected R² (session − pseudo)")
# plt.title("Corrected prior decoding")
#
# plt.tight_layout()
# plt.show()

from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PKL_PATH = Path("prior_localization_groups_output/roi_summary_MOp.pkl")

with open(PKL_PATH, "rb") as f:
    rows = pickle.load(f)

df = pd.DataFrame(rows)

ROI = "MOp"
groups = ["fast", "normal", "slow"]

df = df[df["region"] == ROI].copy()

# numeric + finite
for col in ["R2_corrected", "r_real", "r_pseudo_mean"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# corrected Pearson r (difference)
if "r_real" in df.columns and "r_pseudo_mean" in df.columns:
    df["r_corrected"] = df["r_real"] - df["r_pseudo_mean"]
else:
    raise RuntimeError("Need columns 'r_real' and 'r_pseudo_mean' in the PKL to compute corrected Pearson r.")

# keep finite for each metric when plotting
df_r2 = df[np.isfinite(df["R2_corrected"])].copy()
df_r  = df[np.isfinite(df["r_corrected"])].copy()

print("N per group (R2_corrected):")
print(df_r2.groupby("group").size())

print("\nN per group (r_corrected):")
print(df_r.groupby("group").size())

# -----------------------
# Plot 1: Corrected R^2
# -----------------------
plt.figure(figsize=(5, 4))
plt.boxplot(
    [df_r2[df_r2.group == g]["R2_corrected"] for g in groups],
    labels=groups,
    showfliers=False
)
plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.ylabel("Corrected R² (session − pseudo)")
plt.title(f"{ROI}: Corrected prior decoding (R²)")
plt.tight_layout()
plt.show()

# -----------------------
# Plot 2: Corrected Pearson r (difference)
# -----------------------
plt.figure(figsize=(5, 4))
plt.boxplot(
    [df_r[df_r.group == g]["r_corrected"] for g in groups],
    labels=groups,
    showfliers=False
)
plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.ylabel("Corrected Pearson r (real − pseudo mean)")
plt.title(f"{ROI}: Corrected prior decoding (Pearson r)")
plt.tight_layout()
plt.show()
