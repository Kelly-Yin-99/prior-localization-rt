

from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import kruskal, mannwhitneyu


PEARSON_SUMMARY_PKL = Path(
    "./prior_localization_sessionfit_output/pearson_summary_MOs_ACAd_MOp_ORBvl.pkl"
)

GROUP_ORDER = ["fast", "normal", "slow"]


METRICS = [
    ("r_real", "Pearson r (real)", "all_rois_pearson_r_real_sig_n.pdf"),
    ("z_corr", r"z = (r_real − mean(r_pseudo)) / std(r_pseudo)", "all_rois_z_corr_sig_n_eq.pdf"),
    ("r2_corr", "Corrected R²", "all_rois_r2_corr_sig_n.pdf"),
]

OUT_DIR = Path("./prior_localization_sessionfit_output/plots_all_rois")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUT_DIR / "summary_all_rois_selected_metrics.csv"


MIN_N_PER_GROUP = 2
ALPHA = 0.05
DO_KW_GATE = True

TITLE_FONTSIZE = 12
TITLE_PAD = 18
N_LABEL_PAD_FRAC = 0.14
BRACKET_BASE_FRAC = 0.22
BRACKET_STEP_FRAC = 0.12
BRACKET_H_FRAC = 0.04


with open(PEARSON_SUMMARY_PKL, "rb") as f:
    rows = pickle.load(f)

df = pd.DataFrame(rows)

print("Loaded:", PEARSON_SUMMARY_PKL)
print("Columns:", df.columns.tolist())

required = ["roi", "group", "r_real", "z_corr", "r2_corr"]
for c in required:
    if c not in df.columns:
        raise RuntimeError(f"Missing column '{c}'")

df["roi"] = df["roi"].astype(str)
df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)

# coerce numerics
for col in ["r_real", "z_corr", "r2_corr"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

rois = sorted(df["roi"].dropna().unique())
print("ROIs:", rois)


summary_cols = ["r_real", "z_corr", "r2_corr"]
summary = (
    df.groupby(["roi", "group"], observed=True)[summary_cols]
      .agg(["count", "mean", "median", "std"])
)
summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.reset_index()
summary.to_csv(SUMMARY_CSV, index=False)
print("Saved summary CSV:", SUMMARY_CSV)


def bh_fdr(pvals):
    pvals = np.asarray(pvals, float)
    m = len(pvals)
    if m == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * m / (np.arange(1, m + 1))
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


def add_sig_bracket(ax, x1, x2, y, text, h):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="k", clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=9)


def add_pairwise_sig(ax, vals_by_group):

    if DO_KW_GATE:
        groups_for_kw = [v for v in vals_by_group.values() if len(v) >= MIN_N_PER_GROUP]
        if len(groups_for_kw) < 2:
            return
        try:
            _, p_kw = kruskal(*groups_for_kw)
        except Exception:
            return
        if not np.isfinite(p_kw) or p_kw >= ALPHA:
            return
        ax.text(
            0.02, 0.98, f"KW p={p_kw:.3g}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9
        )

    pairs = [("fast", "normal"), ("fast", "slow"), ("normal", "slow")]
    raw_ps, valid_pairs = [], []

    for a, b in pairs:
        va, vb = vals_by_group[a], vals_by_group[b]
        if len(va) >= MIN_N_PER_GROUP and len(vb) >= MIN_N_PER_GROUP:
            try:
                _, p = mannwhitneyu(va, vb, alternative="two-sided")
            except Exception:
                continue
            raw_ps.append(float(p))
            valid_pairs.append((a, b))

    if not raw_ps:
        return

    qvals = bh_fdr(raw_ps)

    all_vals = np.concatenate([v for v in vals_by_group.values() if len(v)], axis=0) \
        if any(len(v) for v in vals_by_group.values()) else np.array([0.0])
    y_max = float(np.nanmax(all_vals)) if np.isfinite(all_vals).any() else 0.0
    y_min = float(np.nanmin(all_vals)) if np.isfinite(all_vals).any() else 0.0
    yr = max(1e-6, y_max - y_min)

    base_y = y_max + BRACKET_BASE_FRAC * yr
    step = BRACKET_STEP_FRAC * yr
    h = BRACKET_H_FRAC * yr

    x = {"fast": 1, "normal": 2, "slow": 3}
    layer = 0

    for (a, b), q in sorted(zip(valid_pairs, qvals), key=lambda t: t[1]):
        if q < ALPHA:
            label = p_to_stars(q) or f"q={q:.3f}"
            y = base_y + layer * step
            add_sig_bracket(ax, x[a], x[b], y, label, h)
            layer += 1


def add_n_labels(ax, vals_by_group):

    all_vals = np.concatenate([v for v in vals_by_group.values() if len(v)], axis=0) \
        if any(len(v) for v in vals_by_group.values()) else np.array([0.0])
    y_max = float(np.nanmax(all_vals)) if np.isfinite(all_vals).any() else 0.0
    y_min = float(np.nanmin(all_vals)) if np.isfinite(all_vals).any() else 0.0
    yr = max(1e-6, y_max - y_min)

    y_n = y_max + N_LABEL_PAD_FRAC * yr
    for j, g in enumerate(GROUP_ORDER, start=1):
        n = len(vals_by_group[g])
        ax.text(j, y_n, f"n={n}", ha="center", va="bottom", fontsize=9)


    top = y_max + max(N_LABEL_PAD_FRAC + 0.22, 0.35) * yr
    cur_lo, cur_hi = ax.get_ylim()
    ax.set_ylim(cur_lo, max(cur_hi, top))

def plot_metric(metric: str, ylabel: str, out_pdf: Path):
    if metric not in df.columns:
        raise RuntimeError(f"Metric '{metric}' not found in df columns")

    ncols = 3
    nrows = int(np.ceil(len(rois) / ncols))

    with PdfPages(out_pdf) as pdf:
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(5.2 * ncols, 3.9 * nrows),
            sharey=False
        )
        axes = np.array(axes).reshape(-1)

        rng = np.random.default_rng(0)  #

        for i, roi in enumerate(rois):
            ax = axes[i]
            sub = df[df["roi"] == roi]

            vals_by_group = {}
            data = []
            for g in GROUP_ORDER:
                v = sub.loc[sub["group"] == g, metric].to_numpy()
                v = v[np.isfinite(v)]
                vals_by_group[g] = v
                data.append(v)

            ax.boxplot(
                data, positions=[1, 2, 3],
                widths=0.6, showfliers=False
            )

            # scatter with jitter
            for j, v in enumerate(data, start=1):
                if len(v):
                    ax.scatter(
                        j + rng.uniform(-0.15, 0.15, size=len(v)),
                        v, s=18, alpha=0.6
                    )

            ax.axhline(0, linestyle="--", linewidth=1)
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(GROUP_ORDER)
            ax.set_title(roi, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
            add_n_labels(ax, vals_by_group)
            add_pairwise_sig(ax, vals_by_group)

        for k in range(len(rois), len(axes)):
            axes[k].axis("off")

        fig.suptitle(f"{ylabel} by RT group", fontsize=14, y=0.995)
        fig.text(0.01, 0.5, ylabel, va="center", rotation="vertical", fontsize=12)

        # slightly more top margin
        fig.tight_layout(rect=[0.03, 0.02, 1, 0.965])

        pdf.savefig(fig)
        plt.close(fig)

    print("Saved:", out_pdf)



for col, ylabel, fname in METRICS:
    plot_metric(col, ylabel, OUT_DIR / fname)

print("Outputs in:", OUT_DIR)
print("Summary CSV:", SUMMARY_CSV)
