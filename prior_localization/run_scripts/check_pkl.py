import pickle
from pathlib import Path

p = Path("prior_localization_groups_output/roi_summary.pkl")

with open(p, "rb") as f:
    rows = pickle.load(f)

print(type(rows))
print("n_rows =", len(rows))
print("first item keys =", rows[0].keys())
print("first item =", rows[0])
print(rows)
