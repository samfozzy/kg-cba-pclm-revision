"""Tests and re-scoring that need the stored out-of-fold (OOF) estimates of the neural models.

Input: one CSV with one row per held-out country-year of Experiment A and the columns
    iso, year, observed, <model 1>, <model 2>, ...
where `observed` is per-capita CO2 (t per person) and each model column holds that model's
out-of-fold estimate on the same scale (use --log if the columns hold ln estimates).
Adapt load_oof() if your files are organised differently (e.g. one file per model).

Usage:
    python rev_tests_from_oof.py oof_experiment_A.csv --reference Ridge

Output (written next to the input file):
    *_tests_vs_reference.csv   every model against the reference: MdAPE, median paired difference in APE,
                               95% cluster-bootstrap interval, p (cluster bootstrap, per-country, observation level)
    *_rescored.csv             every model scored on (a) all countries, (b) without the seven atypical economies,
                               (c) the 87 countries without a filled run > 3 years, (d) the 25 countries with one,
                               (e) without the model's own six worst-fitted countries; and the share of squared
                               error contributed by the seven atypical economies and by the six worst-fitted countries
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from kgcba_common import metrics, ape, cluster_compare, ATYPICAL, build_panel


def load_oof(path, log=False):
    d = pd.read_csv(path)
    need = {"iso", "year", "observed"}
    assert need <= set(d.columns), f"the file must contain the columns {sorted(need)}"
    models = [c for c in d.columns if c not in need]
    if log:
        d[models] = np.exp(d[models])
    return d, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oof"); ap.add_argument("--reference", default="Ridge"); ap.add_argument("--log", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    d, models = load_oof(a.oof, a.log)
    assert a.reference in models, f"reference model {a.reference!r} not among {models}"
    c, iso = d.observed.values, d.iso.values
    tests = []
    for m in models:
        if m == a.reference: continue
        r = cluster_compare(ape(c, d[m].values), ape(c, d[a.reference].values), iso, n_boot=a.n_boot)
        tests.append({"first": m, "second": a.reference, "MdAPE first": r["mdape_a"], "MdAPE second": r["mdape_b"], "delta (points)": r["delta"],
                      "CI lower": r["ci_lo"], "CI upper": r["ci_hi"], "p cluster bootstrap": r["p_boot"], "p per-country": r["p_country"], "p observation level": r["p_obs"]})
    T = pd.DataFrame(tests); stem = os.path.splitext(a.oof)[0]
    T.to_csv(stem + "_tests_vs_reference.csv", index=False); print(T.round(4).to_string(index=False))
    try:
        _, audit = build_panel(return_audit=True); run = audit.set_index("iso").longest_run
        long_run = set(run[run > 3].index)
    except Exception as e:                        # source data not available: skip the missing-data subsets
        print("panel not available, skipping filled-run subsets:", e); long_run = None
    rows = []
    for m in models:
        ch = d[m].values; se = pd.Series((c - ch) ** 2).groupby(iso).sum().sort_values(ascending=False)
        top6 = se.head(6).index.tolist()
        subsets = {"All countries": np.ones(len(d), bool), "Without the seven atypical economies": ~np.isin(iso, ATYPICAL),
                   "Without the model's own six worst-fitted countries": ~np.isin(iso, top6)}
        if long_run is not None:
            subsets["Countries without a filled run > 3 years"] = ~np.isin(iso, list(long_run)); subsets["Countries with a filled run > 3 years"] = np.isin(iso, list(long_run))
        for lab, k in subsets.items():
            rows.append({"model": m, "subset": lab, "countries": int(len(np.unique(iso[k]))), **metrics(c[k], ch[k]),
                         "share of squared error, seven atypical (%)": 100 * se[[i for i in ATYPICAL if i in se.index]].sum() / se.sum(),
                         "share of squared error, six worst-fitted (%)": 100 * se.head(6).sum() / se.sum(), "six worst-fitted": " ".join(top6)})
    R = pd.DataFrame(rows); R.to_csv(stem + "_rescored.csv", index=False); print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
