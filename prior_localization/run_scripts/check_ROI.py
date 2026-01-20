# import os
# import pickle
# import pandas as pd
#
# # ------------------------------------------------------------
# # Paths
# # ------------------------------------------------------------
# PKL_A = os.path.expanduser("~/Documents/roi_summary.pkl")    # file to CLEAN (A)
# PKL_B = os.path.expanduser("~/Downloads/roi_summary.pkl")    # file to SUBTRACT (B)
# OUT_A_MINUS_B = os.path.expanduser("~/Documents/roi_summary_A_minus_B.pkl")
#
# # ------------------------------------------------------------
# # Load
# # ------------------------------------------------------------
# with open(PKL_A, "rb") as f:
#     data_a = pickle.load(f)
#
# with open(PKL_B, "rb") as f:
#     data_b = pickle.load(f)
#
# df_a = pd.DataFrame(data_a)
# df_b = pd.DataFrame(data_b)
#
# print("Rows in A:", len(df_a))
# print("Rows in B:", len(df_b))
#
# # ------------------------------------------------------------
# # Define "same measurement" key
# # ------------------------------------------------------------
# KEY = ["eid", "region", "group"]
# if "probe" in df_a.columns and "probe" in df_b.columns:
#     KEY.append("probe")
#
# missing_a = [c for c in KEY if c not in df_a.columns]
# missing_b = [c for c in KEY if c not in df_b.columns]
# if missing_a or missing_b:
#     raise RuntimeError(f"Missing columns. A missing={missing_a}, B missing={missing_b}")
#
# # Normalize types for stable matching
# for c in KEY:
#     df_a[c] = df_a[c].astype(str)
#     df_b[c] = df_b[c].astype(str)
#
# # Drop rows missing keys
# df_a = df_a.dropna(subset=KEY).copy()
# df_b = df_b.dropna(subset=KEY).copy()
#
# # ------------------------------------------------------------
# # MULTISET subtraction:
# #   remove exactly as many rows per key as appear in B,
# #   not "remove all rows with that key".
# # ------------------------------------------------------------
#
# # Count how many of each key to remove (from B)
# b_counts = df_b.groupby(KEY).size().reset_index(name="remove_n")
#
# # Attach those counts to each row in A
# df_a2 = df_a.merge(b_counts, on=KEY, how="left")
# df_a2["remove_n"] = df_a2["remove_n"].fillna(0).astype(int)
#
# # Within each key group in A, mark first `remove_n` rows to drop
# df_a2["_rank_in_key"] = df_a2.groupby(KEY).cumcount()
#
# # Drop condition: rank < remove_n
# to_drop = df_a2["_rank_in_key"] < df_a2["remove_n"]
#
# print("Rows to drop from A (count-aware):", int(to_drop.sum()))
#
# df_clean = df_a2.loc[~to_drop].drop(columns=["remove_n", "_rank_in_key"]).copy()
#
# print("Rows remaining in A after count-aware removal:", len(df_clean))
#
# # ------------------------------------------------------------
# # Print how many left in each region (and region x group)
# # ------------------------------------------------------------
# print("\n=== Remaining rows per region ===")
# if len(df_clean) == 0:
#     print("(none)")
# else:
#     print(df_clean["region"].value_counts().sort_index().to_string())
#
# print("\n=== Remaining rows per region x group ===")
# if len(df_clean) == 0:
#     print("(none)")
# else:
#     rg = (
#         df_clean.groupby(["region", "group"])
#                 .size()
#                 .reset_index(name="n_rows")
#                 .sort_values(["region", "group"])
#     )
#     print(rg.to_string(index=False))
#
# print("\n=== Remaining UNIQUE sessions (n_eids) per region x group ===")
# if len(df_clean) == 0:
#     print("(none)")
# else:
#     rg_u = (
#         df_clean.groupby(["region", "group"])["eid"]
#                 .nunique()
#                 .reset_index(name="n_unique_eids")
#                 .sort_values(["region", "group"])
#     )
#     print(rg_u.to_string(index=False))
#
# # ------------------------------------------------------------
# # Save cleaned output
# # ------------------------------------------------------------
# out_rows = df_clean.to_dict(orient="records")
# with open(OUT_A_MINUS_B, "wb") as f:
#     pickle.dump(out_rows, f)
#
# print(f"\nSaved A-minus-B (count-aware) to: {OUT_A_MINUS_B}")
import os
import pickle
import numpy as np
import pandas as pd

PKL_PATH = os.path.expanduser("~/Documents/roi_summary_A_minus_B.pkl")

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

df = pd.DataFrame(data)

# --- filters you asked for ---
region = "ACAd"
group = "fast"
thr = 2.0

# Ensure columns exist
need = ["eid", "region", "group", "R2_uncorrected"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns in pkl: {missing}")

# Clean numeric
df["R2_uncorrected"] = pd.to_numeric(df["R2_corrected"], errors="coerce")

sub = df[(df["region"] == region) & (df["group"] == group) & (df["R2_corrected"] > thr)].copy()

# Print unique session IDs
eids = sorted(sub["eid"].dropna().astype(str).unique().tolist())

print(f"{region} / {group}: sessions with R2_uncorrected > {thr}  (n_sessions={len(eids)}, n_rows={len(sub)})")
for eid in eids:
    print(eid)

# Optional: show the rows so you can see duplicates / paths / units
cols_show = [c for c in ["eid","probe","N_units","R2_uncorrected","R2_pseudo_mean","R2_corrected","pkl_path"] if c in sub.columns]
if cols_show:
    print("\n--- matching rows (sorted by R2_uncorrected desc) ---")
    print(sub[cols_show].sort_values("R2_uncorrected", ascending=False).to_string(index=False))
