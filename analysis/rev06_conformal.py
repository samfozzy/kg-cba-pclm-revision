import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from kgcba_common import *
from sklearn.model_selection import GroupKFold
OUT = os.environ.get("KGCBA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")) + os.sep
p = build_panel(); X, y, c, iso, yr = windows(p)
def fitpred(tr, te):
    Z = standardise(X, tr).reshape(len(X), -1); m = Ridge(alpha=1.0).fit(Z[tr], y[tr]); return m.predict(Z[te])
def cq(scores, level):
    n = len(scores); k = int(np.ceil(level * (n + 1))); return np.sort(scores)[min(k, n) - 1]
levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
outer = list(GroupKFold(5).split(X, y, iso))
res = {}
# (a) single country-disjoint calibration split: 1/7 of the training countries, 200 random draws
rng = np.random.default_rng(42); cov_a = {l: [] for l in levels}; fold_cov_a = []; width_a = []; qs_a = []
for rep in range(200):
    hit = {l: np.zeros(len(y), bool) for l in levels}; w90 = np.zeros(len(y))
    fc = []
    for k, (tr, te) in enumerate(outer):
        cs = np.unique(iso[tr]); cal_c = rng.choice(cs, size=len(cs) // 7, replace=False)
        cal = tr[np.isin(iso[tr], cal_c)]; fit = tr[~np.isin(iso[tr], cal_c)]
        s = np.abs(y[cal] - fitpred(fit, cal)); yh = fitpred(fit, te)
        for l in levels:
            q = cq(s, l); hit[l][te] = np.abs(y[te] - yh) <= q
            if l == 0.9: w90[te] = np.exp(yh + q) - np.exp(yh - q); qs_a.append(q); fc.append(hit[l][te].mean())
    for l in levels: cov_a[l].append(hit[l].mean())
    fold_cov_a.append(fc); width_a.append(w90.mean())
res["single_split"] = {str(l): dict(mean=float(np.mean(v)), p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)), min=float(np.min(v)), max=float(np.max(v))) for l, v in cov_a.items()}
fc = np.array(fold_cov_a)
res["single_split_fold90"] = dict(min=float(fc.min()), p5=float(np.percentile(fc, 5)), median=float(np.median(fc)), p95=float(np.percentile(fc, 95)), max=float(fc.max()))
res["single_split_width90"] = float(np.mean(width_a)); res["single_split_q90_range"] = [float(np.min(qs_a)), float(np.max(qs_a))]
res["n_cal_countries_single"] = sorted({len(np.unique(iso[tr])) // 7 for tr, _ in outer})
# (b) grouped cross-conformal: out-of-fold residuals for all training countries (inner grouped 5-fold)
hit = {l: np.zeros(len(y), bool) for l in levels}; w90 = np.zeros(len(y)); fcb = []; qb = []; lo = np.zeros(len(y)); hi = np.zeros(len(y))
for k, (tr, te) in enumerate(outer):
    s = np.zeros(len(tr))
    for itr, ite in GroupKFold(5).split(X[tr], y[tr], iso[tr]):
        s[ite] = np.abs(y[tr][ite] - fitpred(tr[itr], tr[ite]))
    yh = fitpred(tr, te)
    for l in levels:
        q = cq(s, l); hit[l][te] = np.abs(y[te] - yh) <= q
        if l == 0.9: w90[te] = np.exp(yh + q) - np.exp(yh - q); fcb.append(float(hit[l][te].mean())); qb.append(float(q)); lo[te] = np.exp(yh - q); hi[te] = np.exp(yh + q)
res["cross_conformal"] = {str(l): float(hit[l].mean()) for l in levels}
res["cross_conformal_fold90"] = fcb; res["cross_conformal_q90"] = qb; res["cross_conformal_width90"] = float(w90.mean())
res["cross_conformal_rel_halfwidth90"] = [float(np.exp(min(qb)) - 1), float(np.exp(max(qb)) - 1)]
pcc = pd.Series(hit[0.9]).groupby(iso).mean()
res["cross_conformal_country"] = dict(share_ge90=float((pcc >= 0.9).mean()), share_full=float((pcc == 1).mean()), share_le50=float((pcc <= 0.5).mean()), share_zero=float((pcc == 0).mean()), median=float(pcc.median()))
res["n_cal_countries_cross"] = int(len(np.unique(iso[outer[0][0]])))
pd.DataFrame({"iso": iso, "year": yr, "observed": c, "lower90": lo, "upper90": hi, "covered": hit[0.9]}).to_csv(OUT + "ridge_conformal_intervals.csv", index=False)
# (c) exchangeable reference: calibrate on a random half of the SAME held-out countries' observations
rng = np.random.default_rng(7); cov_c = {l: [] for l in levels}
oof = np.zeros(len(y))
for tr, te in outer: oof[te] = fitpred(tr, te)
for rep in range(200):
    h = {l: [] for l in levels}
    for tr, te in outer:
        perm = rng.permutation(te); cal, ev = perm[: len(te) // 2], perm[len(te) // 2:]
        s = np.abs(y[cal] - oof[cal])
        for l in levels: h[l].append((np.abs(y[ev] - oof[ev]) <= cq(s, l)))
    for l in levels: cov_c[l].append(np.concatenate(h[l]).mean())
res["exchangeable_reference"] = {str(l): float(np.mean(v)) for l, v in cov_c.items()}
json.dump(res, open(OUT + "rev06.json", "w"), indent=1)
print(json.dumps(res, indent=0))
