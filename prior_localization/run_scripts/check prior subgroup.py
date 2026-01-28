
import json
import pickle
from pathlib import Path
import numpy as np

pkl_path = Path("prior_localization_sessionfit_output/pearson_summary_MOp_2.pkl")
log_path = Path("prior_localization_sessionfit_output/run_log_MOp_2.json")

def _fmt(x, nd=4):
    """Format float nicely, handling None/NaN."""
    if x is None:
        return " nan"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(x):
        return " nan"
    return f"{x: .{nd}f}"

def _fmt_pm(mu, sd, nd=4):
    return f"{_fmt(mu, nd)}±{_fmt(sd, nd)}"


print("=== RUN LOG ===")
run_log = json.loads(log_path.read_text())
for r in run_log:
    print(r)


print("\n=== SUMMARY (rows) ===")
with open(pkl_path, "rb") as f:
    rows = pickle.load(f)

print("Type:", type(rows), "| N rows:", len(rows))
print("Keys:", sorted(rows[0].keys()))


rows_sorted = sorted(rows, key=lambda x: (x.get("eid", ""), x.get("group", "")))

print("\n=== PER-SESSION / PER-GROUP ===")
for row in rows_sorted:
    eid = row.get("eid", "NA")
    grp = row.get("group", "NA")
    n = int(row.get("n_trials_group_used", -1))

    # Pearson fields
    r_real = row.get("r_real", np.nan)
    z_r = row.get("z_corr", np.nan)
    p_r = row.get("p_emp", np.nan)
    rf_mu = row.get("r_fake_mean", np.nan)
    rf_sd = row.get("r_fake_std", np.nan)

    # R^2 fields (new)
    r2_real = row.get("r2_real", np.nan)
    r2_corr = row.get("r2_corr", np.nan)
    z_r2 = row.get("z_r2", np.nan)
    p_r2 = row.get("p_emp_r2", np.nan)
    r2f_mu = row.get("r2_fake_mean", np.nan)
    r2f_sd = row.get("r2_fake_std", np.nan)

    # One-line output with both blocks
    print(
        f"{eid} | {grp:6s} | n={n:4d} | "
        f"Pearson: r={_fmt(r_real, 4)} | null={_fmt_pm(rf_mu, rf_sd, 4)} | z={_fmt(z_r, 3)} | p={_fmt(p_r, 4)} || "
        f"R2: r2={_fmt(r2_real, 4)} | null={_fmt_pm(r2f_mu, r2f_sd, 4)} | r2_corr={_fmt(r2_corr, 4)} | "
        f"z={_fmt(z_r2, 3)} | p={_fmt(p_r2, 4)}"
    )
