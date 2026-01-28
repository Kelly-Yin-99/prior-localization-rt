# import pickle
# import numpy as np
# import pandas as pd
# from pathlib import Path
#
# pkl_path = Path("prior_localization_sessionfit_output/pearson_summary_MOs.pkl")
#
# with open(pkl_path, "rb") as f:
#     rows = pickle.load(f)
#
# print("Type:", type(rows))
# print("N rows:", len(rows))
# print("Keys example:", list(rows[0].keys()))
#
# df = pd.DataFrame(rows)
#
# # r_fake is an array in each row; make it easier to view
# df["r_fake_mean_check"] = df["r_fake"].apply(lambda x: float(np.mean(x)) if isinstance(x, (list, np.ndarray)) and len(x) else np.nan)
# df["r_fake_std_check"]  = df["r_fake"].apply(lambda x: float(np.std(x, ddof=1)) if isinstance(x, (list, np.ndarray)) and len(x) > 1 else np.nan)
#
# # show the main columns
# cols = ["eid", "subject", "roi", "group", "n_trials_group_used", "r_real", "r_fake_mean", "r_fake_std", "z_corr", "p_emp"]
# print(df[cols].sort_values(["eid", "group"]).to_string(index=False))
#
# # Print nice subgroup summaries pooled across sessions
# print("\n=== Subgroup summary across sessions ===")
# print(df.groupby("group")[["n_trials_group_used","r_real","z_corr","p_emp"]].agg(["count","mean","median"]).to_string())

# # inspect_pearson_outputs.py
# import json
# import pickle
# from pathlib import Path
# import numpy as np
#
# pkl_path = Path("prior_localization_sessionfit_output/pearson_summary_MOs.pkl")
# log_path = Path("prior_localization_sessionfit_output/run_log_MOs.json")
#
# # ---- run log ----
# print("=== RUN LOG ===")
# run_log = json.loads(log_path.read_text())
# for r in run_log:
#     print(r)
#
# # ---- pearson summary ----
# print("\n=== PEARSON SUMMARY (rows) ===")
# with open(pkl_path, "rb") as f:
#     rows = pickle.load(f)
#
# print("Type:", type(rows), "| N rows:", len(rows))
# print("Keys:", sorted(rows[0].keys()))
#
# # print nicely per session
# rows_sorted = sorted(rows, key=lambda x: (x["eid"], x["group"]))
# for row in rows_sorted:
#     eid = row["eid"]
#     grp = row["group"]
#     n = row["n_trials_group_used"]
#     r_real = row["r_real"]
#     z = row["z_corr"]
#     p = row["p_emp"]
#     rf_mu = row["r_fake_mean"]
#     rf_sd = row["r_fake_std"]
#
#     print(
#         f"{eid} | {grp:6s} | n={n:4d} | r_real={r_real: .4f} | "
#         f"r_fake={rf_mu: .4f}±{rf_sd: .4f} | z={z: .3f} | p_emp={p: .4f}"
#     )

# inspect_pearson_outputs.py
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

# ---- run log ----
print("=== RUN LOG ===")
run_log = json.loads(log_path.read_text())
for r in run_log:
    print(r)

# ---- summary rows ----
print("\n=== SUMMARY (rows) ===")
with open(pkl_path, "rb") as f:
    rows = pickle.load(f)

print("Type:", type(rows), "| N rows:", len(rows))
print("Keys:", sorted(rows[0].keys()))

# print nicely per session
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
