import os
import pickle

PKL_PATH = os.path.expanduser("~/Downloads/roi_summary.pkl")

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

# Filter rows
rows = [
    r for r in data
    if r.get("region") == "MOp"
    and str(r.get("group")).lower() == "slow"
    and (r.get("R2_corrected") is not None)
    and (r["R2_corrected"] > 2)
]

# Unique eids
eids = sorted({r["eid"] for r in rows})

print(f"Found {len(eids)} unique eids (MOp, slow, R2_corrected > 2):")
for eid in eids:
    print(eid)


print("\nDetails:")
for r in sorted(rows, key=lambda x: (x["eid"], x.get("probe", ""))):
    print(
        f"eid={r['eid']}  probe={r.get('probe')}  "
        f"R2_corr={r.get('R2_corrected'):.4f}  "
        f"R2_uncorr={r.get('R2_uncorrected'):.4f}  "
        f"R2_pseudo_mean={r.get('R2_pseudo_mean'):.4f}  "
        f"N_units={r.get('N_units')}"
    )
