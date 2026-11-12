"""Conformal coverage for ridge regression in Experiment C, like for like with the intervals of the hybrid.

The 90% intervals reported for the hybrid (E-PSO-KG-CBA) are those of Experiment C (76 countries, seven covariates),
calibrated on the inner validation countries of each fold: one-seventh of the training countries, rounded down,
that is 8 countries (original_pipeline/code/conformal.py). rev06_conformal.py analyses ridge regression in
Experiment A (112 countries) with 13 calibration countries. This script repeats that analysis on the countries,
covariates and folds of Experiment C, with calibration sets of the same size as for the hybrid:
  (a) 8 calibration countries drawn at random from the training countries of each fold (200 draws);
  (b) all 60 or 61 training countries, by grouped cross-conformal prediction;
  (c) an exchangeable reference: calibration on a random half of the held-out countries' own observations.
It also scores the stored intervals of the hybrid (original_pipeline/results/conformal_regen.npz) by fold and by
country. Output: out/rev06c.json.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from kgcba_common import *
from sklearn.model_selection import GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("KGCBA_OUT", os.path.join(HERE, "out")) + os.sep
PIPE = os.path.join(HERE, "..", "original_pipeline")

p = add_low_carbon(build_panel()); X, y, c, iso, yr = windows(p, FEATS7)
assert len(np.unique(iso)) == 76 and len(y) == 2052


def fitpred(tr, te):
    Z = standardise(X, tr).reshape(len(X), -1); m = Ridge(alpha=1.0).fit(Z[tr], y[tr]); return m.predict(Z[te])


def cq(scores, level):
    n = len(scores); k = int(np.ceil(level * (n + 1))); return np.sort(scores)[min(k, n) - 1]


outer = list(GroupKFold(5).split(X, y, iso))
res = {"n_countries": 76, "n_samples": int(len(y))}
# the folds are those of Experiment C in the original pipeline
meta = pd.read_csv(os.path.join(PIPE, "data", "meta_enriched.csv"))
assert (meta.iso.values == iso).all() and (meta.year.values == yr).all()
import pickle
with open(os.path.join(PIPE, "data", "folds_enriched.pkl"), "rb") as fh:
    folds_pipe = pickle.load(fh)
assert all(np.array_equal(a[1], b[1]) for a, b in zip(outer, folds_pipe)), "folds differ from the original pipeline"

# (a) 8 calibration countries per fold (one-seventh of the training countries, rounded down, as for the hybrid)
rng = np.random.default_rng(42); cov, fold_cov, width = [], [], []
for rep in range(200):
    hit = np.zeros(len(y), bool); w = np.zeros(len(y)); fc = []
    for k, (tr, te) in enumerate(outer):
        cs = np.unique(iso[tr]); cal_c = rng.choice(cs, size=len(cs) // 7, replace=False)
        cal = tr[np.isin(iso[tr], cal_c)]; fit = tr[~np.isin(iso[tr], cal_c)]
        q = cq(np.abs(y[cal] - fitpred(fit, cal)), 0.9); yh = fitpred(fit, te)
        hit[te] = np.abs(y[te] - yh) <= q; w[te] = np.exp(yh + q) - np.exp(yh - q); fc.append(hit[te].mean())
    cov.append(hit.mean()); fold_cov.append(fc); width.append(w.mean())
fc = np.array(fold_cov)
res["n_cal_countries_single"] = sorted({len(np.unique(iso[tr])) // 7 for tr, _ in outer})
res["single_split90"] = dict(mean=float(np.mean(cov)), p5=float(np.percentile(cov, 5)), p95=float(np.percentile(cov, 95)),
                             min=float(np.min(cov)), max=float(np.max(cov)), width=float(np.mean(width)))
res["single_split_fold90"] = dict(min=float(fc.min()), p5=float(np.percentile(fc, 5)), median=float(np.median(fc)),
                                  p95=float(np.percentile(fc, 95)), max=float(fc.max()))

# (b) all training countries, grouped cross-conformal
hit = np.zeros(len(y), bool); w = np.zeros(len(y)); fcb = []
for tr, te in outer:
    s = np.zeros(len(tr))
    for itr, ite in GroupKFold(5).split(X[tr], y[tr], iso[tr]):
        s[ite] = np.abs(y[tr][ite] - fitpred(tr[itr], tr[ite]))
    q = cq(s, 0.9); yh = fitpred(tr, te)
    hit[te] = np.abs(y[te] - yh) <= q; w[te] = np.exp(yh + q) - np.exp(yh - q); fcb.append(float(hit[te].mean()))
pcc = pd.Series(hit).groupby(iso).mean()
res["n_cal_countries_cross"] = sorted({len(np.unique(iso[tr])) for tr, _ in outer})
res["cross_conformal90"] = dict(coverage=float(hit.mean()), fold=fcb, width=float(w.mean()),
                                share_full=float((pcc == 1).mean()), share_le50=float((pcc <= 0.5).mean()))

# (c) exchangeable reference
oof = np.zeros(len(y))
for tr, te in outer: oof[te] = fitpred(tr, te)
rng = np.random.default_rng(7); cc = []
for rep in range(200):
    h = []
    for tr, te in outer:
        perm = rng.permutation(te); cal, ev = perm[: len(te) // 2], perm[len(te) // 2:]
        h.append(np.abs(y[ev] - oof[ev]) <= cq(np.abs(y[cal] - oof[cal]), 0.9))
    cc.append(np.concatenate(h).mean())
res["exchangeable_reference90"] = float(np.mean(cc))

# stored intervals of the hybrid, scored by fold and by country
cf = np.load(os.path.join(PIPE, "results", "conformal_regen.npz"))
inside = (c >= cf["lo"]) & (c <= cf["hi"])
ph = pd.Series(inside).groupby(iso).mean()
res["hybrid_stored"] = dict(coverage=float(inside.mean()), fold=[float(inside[te].mean()) for _, te in outer],
                            width=float((cf["hi"] - cf["lo"]).mean()), share_full=float((ph == 1).mean()),
                            share_le50=float((ph <= 0.5).mean()))
json.dump(res, open(OUT + "rev06c.json", "w"), indent=1)
print(json.dumps(res, indent=1))
