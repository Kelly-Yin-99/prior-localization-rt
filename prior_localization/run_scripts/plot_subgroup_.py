

import pickle, pandas as pd, os, numpy as np

pkl = os.path.expanduser("~/Downloads/pearson_r2_summary_20260123_232334.pkl")
rows = pickle.load(open(pkl, "rb"))
df = pd.DataFrame(rows)
rr = pd.to_numeric(df["r_fake_mean"], errors="coerce")

print("r_real non-null:", rr.notna().sum(), "/", len(rr))
print("r_real finite:", np.isfinite(rr).sum(), "/", len(rr))
print("r_real NaN:", rr.isna().sum())
print("r_real +inf:", np.isposinf(rr).sum(), " -inf:", np.isneginf(rr).sum())

print("\nSummary:")
print(rr.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]))


print("n rows:", len(df))
print("columns:", df.columns.tolist())
print("head:\n", df.head(5))
