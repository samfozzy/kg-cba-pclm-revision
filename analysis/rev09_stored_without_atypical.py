"""Country-clustered tests of every model against ridge regression on the 105 countries without the seven atypical
economies, from the stored out-of-fold estimates of the original run (out/oof_experiment_A.csv, written by
oof_from_original.py). Complements rev_tests_from_oof.py, which tests on all 112 countries and re-scores every
model on subsets of countries without testing.

Output: out/oof_experiment_A_tests_vs_reference_without_atypical.csv (same columns as the file for all countries).
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from kgcba_common import cluster_compare, ATYPICAL

OUT = os.environ.get("KGCBA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")) + os.sep
d = pd.read_csv(OUT + "oof_experiment_A.csv")
d = d[~d.iso.isin(ATYPICAL)]
assert d.iso.nunique() == 105
models = [c for c in d.columns if c not in ("iso", "year", "observed")]
ape = {m: 100 * np.abs(d.observed - d[m]).values / d.observed.values for m in models}
rows = []
for m in models:
    if m == "Ridge":
        continue
    r = cluster_compare(ape[m], ape["Ridge"], d.iso.values)
    rows.append({"first": m, "second": "Ridge", "MdAPE first": r["mdape_a"], "MdAPE second": r["mdape_b"], "delta (points)": r["delta"],
                 "CI lower": r["ci_lo"], "CI upper": r["ci_hi"], "p cluster bootstrap": r["p_boot"], "p per-country": r["p_country"],
                 "p observation level": r["p_obs"]})
t = pd.DataFrame(rows)
t.to_csv(OUT + "oof_experiment_A_tests_vs_reference_without_atypical.csv", index=False)
print(t.round(4).to_string(index=False))
