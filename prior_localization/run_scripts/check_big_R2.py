import os
import pickle
from collections import defaultdict

PKL_PATH = os.path.expanduser("~/Documents/roi_summary.pkl")

# -----------------------
# Load data
# -----------------------
with open(PKL_PATH, "rb") as f:
    rows = pickle.load(f)

print(f"Loaded {len(rows)} rows")

# -----------------------
# Choose which R² to use
# -----------------------
R2_FIELD = "R2_corrected"   # or "R2_uncorrected"

# -----------------------
# Collect by region
# -----------------------
by_region = defaultdict(list)

for r in rows:
    r2 = r.get(R2_FIELD, None)
    if r2 is None:
        continue
    if r2 >= 1:
        by_region[r["region"]].append(
            (r["eid"], r2, r.get("group"), r.get("N_units"))
        )

# -----------------------
# Print results
# -----------------------
for region in sorted(by_region.keys()):
    entries = by_region[region]
    if len(entries) == 0:
        continue

    # sort descending by R²
    entries = sorted(entries, key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 80)
    print(f"Region: {region}")
    print(f"Sessions with {R2_FIELD} >= 1: {len(entries)}")
    print("-" * 80)

    for eid, r2, group, n_units in entries:
        print(
            f"eid={eid} | "
            f"{R2_FIELD}={r2:.3f} | "
            f"group={group} | "
            f"N_units={n_units}"
        )
