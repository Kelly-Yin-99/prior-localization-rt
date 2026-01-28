# import os
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
#
# PKL_PATH = os.path.expanduser("~/Documents/roi_summary.pkl")
# GROUP_ORDER = ["fast", "normal", "slow"]
# MIN_UNITS = 5
#
# # Group colors
# PALETTE = {"fast": "#d73027", "normal": "#4575b4", "slow": "#1a9850"}
#
# CLIP_ABS_UNCORR = None
# CLIP_ABS_PSEUDO = None
# CLIP_ABS_CORR = None
#
#
# with open(PKL_PATH, "rb") as f:
#     data = pickle.load(f)
#
# df = pd.DataFrame(data)
# print("\nColumns in roi_summary.pkl:")
# for c in df.columns:
#     print(" -", c)
#
# # Basic QC
# for col in ["R2_uncorrected", "R2_pseudo_mean", "N_units", "group", "region"]:
#     if col not in df.columns:
#         raise RuntimeError(f"Missing column '{col}' in roi_summary.pkl rows.")
#
# df = df[np.isfinite(df["R2_uncorrected"])]
# df = df[np.isfinite(df["R2_pseudo_mean"])]
# df = df[np.isfinite(df["N_units"])]
# df = df[df["N_units"] >= MIN_UNITS].copy()
#
#
# df["R2_corrected"] = df["R2_uncorrected"] - df["R2_pseudo_mean"]
#
# df = df[np.isfinite(df["R2_corrected"])].copy()
#
# df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
# regions = sorted(df["region"].unique().tolist())
#
# if len(regions) == 0:
#     raise RuntimeError("No regions found after filtering. Check MIN_UNITS or input file.")
#
# # Optional clipping to reduce extreme outliers for visualization only
# if CLIP_ABS_UNCORR is not None:
#     df = df[np.abs(df["R2_uncorrected"]) <= float(CLIP_ABS_UNCORR)].copy()
#
# if CLIP_ABS_PSEUDO is not None:
#     df = df[np.abs(df["R2_pseudo_mean"]) <= float(CLIP_ABS_PSEUDO)].copy()
#
# if CLIP_ABS_CORR is not None:
#     df = df[np.abs(df["R2_corrected"]) <= float(CLIP_ABS_CORR)].copy()
#
#
# def add_n_labels(ax, sub_df, y_col):
#     y_top = sub_df[y_col].max()
#     if not np.isfinite(y_top):
#         y_top = 0.0
#     y_n = y_top + 0.04
#
#     for j, g in enumerate(GROUP_ORDER):
#         n = int((sub_df["group"] == g).sum())
#         ax.text(j, y_n, f"n={n}", ha="center", va="bottom", fontsize=10)
#
# def draw_panel(ax, sub, y_col, title):
#     sns.boxplot(
#         data=sub,
#         x="group",
#         y=y_col,
#         hue="group",              # avoid seaborn palette deprecation warning
#         order=GROUP_ORDER,
#         hue_order=GROUP_ORDER,
#         palette=PALETTE,
#         width=0.6,
#         fliersize=0,
#         ax=ax,
#         legend=False,
#     )
#     sns.stripplot(
#         data=sub,
#         x="group",
#         y=y_col,
#         order=GROUP_ORDER,
#         color="black",
#         alpha=0.55,
#         jitter=True,
#         size=4,
#         ax=ax,
#     )
#     ax.axhline(0, color="k", linestyle="--", linewidth=1)
#     ax.set_title(title, fontsize=14)
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     add_n_labels(ax, sub, y_col)
#
#
# n_regions = len(regions)
# ncols = 2 if n_regions > 1 else 1
# nrows = int(np.ceil(n_regions / ncols))
#
# fig1, axes1 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.8 * ncols, 5.0 * nrows), sharey=True)
# axes1 = np.array(axes1).reshape(-1)
#
# for ax_i, region in enumerate(regions):
#     ax = axes1[ax_i]
#     sub = df[df["region"] == region].copy()
#     draw_panel(ax, sub, "R2_uncorrected", region)
#
# for k in range(n_regions, len(axes1)):
#     axes1[k].axis("off")
#
# fig1.supylabel("Uncorrected R² (real sessions)", fontsize=14)
# title1 = "Uncorrected prior decoding by RT group and ROI"
# if CLIP_ABS_UNCORR is not None:
#     title1 += f" (|R2_uncorrected| ≤ {CLIP_ABS_UNCORR})"
# fig1.suptitle(title1, fontsize=16, y=0.995)
# plt.tight_layout()
# plt.show()
#
#
# fig2, axes2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.8 * ncols, 5.0 * nrows), sharey=True)
# axes2 = np.array(axes2).reshape(-1)
#
# for ax_i, region in enumerate(regions):
#     ax = axes2[ax_i]
#     sub = df[df["region"] == region].copy()
#     draw_panel(ax, sub, "R2_pseudo_mean", region)
#
# for k in range(n_regions, len(axes2)):
#     axes2[k].axis("off")
#
# fig2.supylabel("Pseudo R² (mean over pseudo-sessions)", fontsize=14)
# title2 = "Pseudo-session decoding by RT group and ROI"
# if CLIP_ABS_PSEUDO is not None:
#     title2 += f" (|R2_pseudo_mean| ≤ {CLIP_ABS_PSEUDO})"
# fig2.suptitle(title2, fontsize=16, y=0.995)
# plt.tight_layout()
# plt.show()
#
#
# fig3, axes3 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.8 * ncols, 5.0 * nrows), sharey=True)
# axes3 = np.array(axes3).reshape(-1)
#
# for ax_i, region in enumerate(regions):
#     ax = axes3[ax_i]
#     sub = df[df["region"] == region].copy()
#     draw_panel(ax, sub, "R2_corrected", region)
#
# for k in range(n_regions, len(axes3)):
#     axes3[k].axis("off")
#
# fig3.supylabel("Corrected R² = (real) − (pseudo mean)", fontsize=14)
# title3 = "Corrected prior decoding by RT group and ROI"
# if CLIP_ABS_CORR is not None:
#     title3 += f" (|R2_corrected| ≤ {CLIP_ABS_CORR})"
# fig3.suptitle(title3, fontsize=16, y=0.995)
# plt.tight_layout()
# plt.show()
#
#
# orbvl_sessions = (
#     df.loc[df["region"] == "ORBvl", "eid"]
#       .dropna()
#       .unique()
# )
#


import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import kruskal, mannwhitneyu  # NEW


PKL_PATH = os.path.expanduser("~/Documents/roi_summary.pkl")
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


# -----------------------------
# Significance helpers (NEW)
# -----------------------------
def bh_fdr(pvals):
    """
    Benjamini–Hochberg FDR correction.
    Returns q-values in the same order as pvals.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return pvals

    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * m / (np.arange(1, m + 1))

    # enforce monotonicity
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    out = np.empty_like(q)
    out[order] = q
    return out


def p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def add_sig_bracket(ax, x1, x2, y, text, h=0.02, lw=1.2):
    """
    Draw a bracket from x1 to x2 at height y, with label text above.
    x1, x2 are category indices (0,1,2).
    """
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, c="k", clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=12, color="k")


def add_group_sig_annotations(ax, sub_df, y_col="R2_corrected", alpha=0.05):
    """
    For ONE ROI panel:
      1) Kruskal–Wallis across groups
      2) If significant, pairwise Mann–Whitney U with BH-FDR across the 3 pairs
      3) Plot only significant brackets (q < alpha)
    """
    # Extract values per group
    vals = {}
    for g in GROUP_ORDER:
        v = sub_df.loc[sub_df["group"] == g, y_col].to_numpy()
        v = v[np.isfinite(v)]
        vals[g] = v

    # Need at least 2 groups with >= 2 samples for KW to be meaningful
    n_nonempty = sum(len(vals[g]) >= 2 for g in GROUP_ORDER)
    if n_nonempty < 2:
        return

    # Kruskal–Wallis
    groups_for_test = [vals[g] for g in GROUP_ORDER if len(vals[g]) > 0]
    if len(groups_for_test) < 2:
        return

    try:
        kw_stat, kw_p = kruskal(*groups_for_test)
    except Exception:
        return

    if not (np.isfinite(kw_p) and kw_p < alpha):
        return  # only annotate if overall is significant

    # Pairwise MWU
    pairs = [("fast", "normal"), ("fast", "slow"), ("normal", "slow")]
    raw_ps = []
    ok_pairs = []

    for a, b in pairs:
        va, vb = vals[a], vals[b]
        if len(va) < 2 or len(vb) < 2:
            continue
        try:
            # two-sided
            _, p = mannwhitneyu(va, vb, alternative="two-sided")
            raw_ps.append(float(p))
            ok_pairs.append((a, b))
        except Exception:
            continue

    if not raw_ps:
        return

    qvals = bh_fdr(raw_ps)

    # Determine where to place brackets
    y_max = np.nanmax(sub_df[y_col].to_numpy())
    y_min = np.nanmin(sub_df[y_col].to_numpy())
    if not np.isfinite(y_max):
        y_max = 0.0
    if not np.isfinite(y_min):
        y_min = 0.0

    y_range = max(1e-6, y_max - y_min)
    base_y = y_max + 0.08 * y_range
    step = 0.08 * y_range
    h = 0.02 * y_range

    # Map group -> x position
    x = {g: i for i, g in enumerate(GROUP_ORDER)}

    # Plot significant ones only
    layer = 0
    for (a, b), q in sorted(zip(ok_pairs, qvals), key=lambda t: t[1]):  # smallest q first
        if q < alpha:
            stars = p_to_stars(q)
            label = f"{stars}" if stars else f"q={q:.3f}"
            y = base_y + layer * step
            add_sig_bracket(ax, x[a], x[b], y, label, h=h)
            layer += 1

    # Optional: add overall KW p-value in corner
    ax.text(
        0.02, 0.98,
        f"KW p={kw_p:.3g}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10
    )


# -----------------------------
# Plot helpers (unchanged + one hook for sig)
# -----------------------------
def add_n_labels(ax, sub_df, y_col):
    y_top = sub_df[y_col].max()
    if not np.isfinite(y_top):
        y_top = 0.0
    y_n = y_top + 0.04

    for j, g in enumerate(GROUP_ORDER):
        n = int((sub_df["group"] == g).sum())
        ax.text(j, y_n, f"n={n}", ha="center", va="bottom", fontsize=10)


def draw_panel(ax, sub, y_col, title, add_sig=False):
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

    if add_sig:
        add_group_sig_annotations(ax, sub, y_col=y_col, alpha=0.05)


# -----------------------------
# Figure 1: Uncorrected
# -----------------------------
n_regions = len(regions)
ncols = 2 if n_regions > 1 else 1
nrows = int(np.ceil(n_regions / ncols))

fig1, axes1 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.8 * ncols, 5.0 * nrows), sharey=True)
axes1 = np.array(axes1).reshape(-1)

for ax_i, region in enumerate(regions):
    ax = axes1[ax_i]
    sub = df[df["region"] == region].copy()
    draw_panel(ax, sub, "R2_uncorrected", region, add_sig=False)

for k in range(n_regions, len(axes1)):
    axes1[k].axis("off")

fig1.supylabel("Uncorrected R² (real sessions)", fontsize=14)
title1 = "Uncorrected prior decoding by RT group and ROI"
if CLIP_ABS_UNCORR is not None:
    title1 += f" (|R2_uncorrected| ≤ {CLIP_ABS_UNCORR})"
fig1.suptitle(title1, fontsize=16, y=0.995)
plt.tight_layout()
plt.show()


# -----------------------------
# Figure 2: Pseudo
# -----------------------------
fig2, axes2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.8 * ncols, 5.0 * nrows), sharey=True)
axes2 = np.array(axes2).reshape(-1)

for ax_i, region in enumerate(regions):
    ax = axes2[ax_i]
    sub = df[df["region"] == region].copy()
    draw_panel(ax, sub, "R2_pseudo_mean", region, add_sig=False)

for k in range(n_regions, len(axes2)):
    axes2[k].axis("off")

fig2.supylabel("Pseudo R² (mean over pseudo-sessions)", fontsize=14)
title2 = "Pseudo-session decoding by RT group and ROI"
if CLIP_ABS_PSEUDO is not None:
    title2 += f" (|R2_pseudo_mean| ≤ {CLIP_ABS_PSEUDO})"
fig2.suptitle(title2, fontsize=16, y=0.995)
plt.tight_layout()
plt.show()


# -----------------------------
# Figure 3: Corrected (with significance annotations)
# -----------------------------
fig3, axes3 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.8 * ncols, 5.0 * nrows), sharey=True)
axes3 = np.array(axes3).reshape(-1)

for ax_i, region in enumerate(regions):
    ax = axes3[ax_i]
    sub = df[df["region"] == region].copy()
    draw_panel(ax, sub, "R2_corrected", region, add_sig=True)   # <-- ONLY here

for k in range(n_regions, len(axes3)):
    axes3[k].axis("off")

fig3.supylabel("Corrected R² = (real) − (pseudo mean)", fontsize=14)
title3 = "Corrected prior decoding by RT group and ROI"
if CLIP_ABS_CORR is not None:
    title3 += f" (|R2_corrected| ≤ {CLIP_ABS_CORR})"
fig3.suptitle(title3, fontsize=16, y=0.995)
plt.tight_layout()
plt.show()


# Example: list ORBvl session IDs (unchanged)
orbvl_sessions = (
    df.loc[df["region"] == "ORBvl", "eid"]
      .dropna()
      .unique()
)
print("ORBvl sessions:", orbvl_sessions)

# -----------------------------
# NEW: print + save unique session IDs for specific ROIs
# -----------------------------
# NEW: union of session IDs across MOs and ACAd (deduped)
# -----------------------------
ROI_TO_DUMP = ["MOs", "ACAd", "MOp", "ACAd"]

OUT_EID_DIR = Path(os.path.expanduser("~/Documents/roi_eids_all"))
OUT_EID_DIR.mkdir(parents=True, exist_ok=True)

# union set
eid_union = set()

for roi in ROI_TO_DUMP:
    eids = (
        df.loc[df["region"] == roi, "eid"]
          .dropna()
          .astype(str)
          .unique()
    )
    eid_union.update(eids.tolist())

# deterministic sorted list
eid_union = sorted(eid_union)

print(f"\n[UNION {ROI_TO_DUMP}] unique session IDs (n={len(eid_union)}):")
for eid in eid_union:
    print(" ", eid)

# Save TXT
txt_path = OUT_EID_DIR / "eids_union.txt"
txt_path.write_text("\n".join(eid_union) + ("\n" if len(eid_union) else ""))

# Save PKL
pkl_path = OUT_EID_DIR / "eids_union.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(eid_union, f)

print(f"\nSaved union list to:\n - {txt_path}\n - {pkl_path}")
