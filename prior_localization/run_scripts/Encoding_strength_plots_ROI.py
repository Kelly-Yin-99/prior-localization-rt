import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PKL_PATH = os.path.expanduser("~/Documents/roi_summary_A_minus_B.pkl")
GROUP_ORDER = ["fast", "normal", "slow"]
MIN_UNITS = 5

# Group colors
PALETTE = {"fast": "#d73027", "normal": "#4575b4", "slow": "#1a9850"}

CLIP_ABS_UNCORR = None
CLIP_ABS_PSEUDO = None
CLIP_ABS_CORR = None


with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

df = pd.DataFrame(data)
print("\nColumns in roi_summary.pkl:")
for c in df.columns:
    print(" -", c)

# Basic QC
for col in ["R2_uncorrected", "R2_pseudo_mean", "N_units", "group", "region"]:
    if col not in df.columns:
        raise RuntimeError(f"Missing column '{col}' in roi_summary.pkl rows.")

df = df[np.isfinite(df["R2_uncorrected"])]
df = df[np.isfinite(df["R2_pseudo_mean"])]
df = df[np.isfinite(df["N_units"])]
df = df[df["N_units"] >= MIN_UNITS].copy()


df["R2_corrected"] = df["R2_uncorrected"] - df["R2_pseudo_mean"]

df = df[np.isfinite(df["R2_corrected"])].copy()

df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
regions = sorted(df["region"].unique().tolist())

if len(regions) == 0:
    raise RuntimeError("No regions found after filtering. Check MIN_UNITS or input file.")

# Optional clipping to reduce extreme outliers for visualization only
if CLIP_ABS_UNCORR is not None:
    df = df[np.abs(df["R2_uncorrected"]) <= float(CLIP_ABS_UNCORR)].copy()

if CLIP_ABS_PSEUDO is not None:
    df = df[np.abs(df["R2_pseudo_mean"]) <= float(CLIP_ABS_PSEUDO)].copy()

if CLIP_ABS_CORR is not None:
    df = df[np.abs(df["R2_corrected"]) <= float(CLIP_ABS_CORR)].copy()


def add_n_labels(ax, sub_df, y_col):
    y_top = sub_df[y_col].max()
    if not np.isfinite(y_top):
        y_top = 0.0
    y_n = y_top + 0.04

    for j, g in enumerate(GROUP_ORDER):
        n = int((sub_df["group"] == g).sum())
        ax.text(j, y_n, f"n={n}", ha="center", va="bottom", fontsize=10)

def draw_panel(ax, sub, y_col, title):
    sns.boxplot(
        data=sub,
        x="group",
        y=y_col,
        hue="group",              # avoid seaborn palette deprecation warning
        order=GROUP_ORDER,
        hue_order=GROUP_ORDER,
        palette=PALETTE,
        width=0.6,
        fliersize=0,
        ax=ax,
        legend=False,
    )
    sns.stripplot(
        data=sub,
        x="group",
        y=y_col,
        order=GROUP_ORDER,
        color="black",
        alpha=0.55,
        jitter=True,
        size=4,
        ax=ax,
    )
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    add_n_labels(ax, sub, y_col)


n_regions = len(regions)
ncols = 2 if n_regions > 1 else 1
nrows = int(np.ceil(n_regions / ncols))

fig1, axes1 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.8 * ncols, 5.0 * nrows), sharey=True)
axes1 = np.array(axes1).reshape(-1)

for ax_i, region in enumerate(regions):
    ax = axes1[ax_i]
    sub = df[df["region"] == region].copy()
    draw_panel(ax, sub, "R2_uncorrected", region)

for k in range(n_regions, len(axes1)):
    axes1[k].axis("off")

fig1.supylabel("Uncorrected R² (real sessions)", fontsize=14)
title1 = "Uncorrected prior decoding by RT group and ROI"
if CLIP_ABS_UNCORR is not None:
    title1 += f" (|R2_uncorrected| ≤ {CLIP_ABS_UNCORR})"
fig1.suptitle(title1, fontsize=16, y=0.995)
plt.tight_layout()
plt.show()


fig2, axes2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.8 * ncols, 5.0 * nrows), sharey=True)
axes2 = np.array(axes2).reshape(-1)

for ax_i, region in enumerate(regions):
    ax = axes2[ax_i]
    sub = df[df["region"] == region].copy()
    draw_panel(ax, sub, "R2_pseudo_mean", region)

for k in range(n_regions, len(axes2)):
    axes2[k].axis("off")

fig2.supylabel("Pseudo R² (mean over pseudo-sessions)", fontsize=14)
title2 = "Pseudo-session decoding by RT group and ROI"
if CLIP_ABS_PSEUDO is not None:
    title2 += f" (|R2_pseudo_mean| ≤ {CLIP_ABS_PSEUDO})"
fig2.suptitle(title2, fontsize=16, y=0.995)
plt.tight_layout()
plt.show()


fig3, axes3 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.8 * ncols, 5.0 * nrows), sharey=True)
axes3 = np.array(axes3).reshape(-1)

for ax_i, region in enumerate(regions):
    ax = axes3[ax_i]
    sub = df[df["region"] == region].copy()
    draw_panel(ax, sub, "R2_corrected", region)

for k in range(n_regions, len(axes3)):
    axes3[k].axis("off")

fig3.supylabel("Corrected R² = (real) − (pseudo mean)", fontsize=14)
title3 = "Corrected prior decoding by RT group and ROI"
if CLIP_ABS_CORR is not None:
    title3 += f" (|R2_corrected| ≤ {CLIP_ABS_CORR})"
fig3.suptitle(title3, fontsize=16, y=0.995)
plt.tight_layout()
plt.show()


orbvl_sessions = (
    df.loc[df["region"] == "ORBvl", "eid"]
      .dropna()
      .unique()
)


