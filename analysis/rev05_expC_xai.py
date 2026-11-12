import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from kgcba_common import *
from sklearn.model_selection import GroupKFold
import lightgbm as lgb, shap
OUT = os.environ.get("KGCBA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")) + os.sep
p, audit = build_panel(return_audit=True); q = add_low_carbon(p)
res = {}
res["subpanel_by_continent"] = q.groupby("continent").iso.nunique().to_dict()
res["panel_by_continent"] = p.groupby("continent").iso.nunique().to_dict()
res["subpanel_africa"] = sorted(q[q.continent == "Africa"].iso.unique().tolist())
res["atypical_in_subpanel"] = [i for i in ATYPICAL if i in set(q.iso)]
res["subpanel_long_fill"] = audit[audit.iso.isin(q.iso.unique()) & (audit.longest_run > 3)][["iso", "longest_run"]].values.tolist()
X7, y, c, iso, yr = windows(q, FEATS7); X6 = X7[:, :, :6]
mk_l = lambda k: lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5, random_state=42, verbose=-1, n_jobs=4)
o7, fold = loco_predict(X7, y, iso); o6, _ = loco_predict(X6, y, iso)
m7, m6 = metrics(c, np.exp(o7)), metrics(c, np.exp(o6))
# original manuscript's comparison: six-covariate model fitted on all 112 countries, scored on the 76
X112, y112, c112, iso112, _ = windows(p); o112, _ = loco_predict(X112, y112, iso112); m76 = np.isin(iso112, q.iso.unique())
res["expC_ridge"] = dict(seven=m7, six_same_folds=m6, six_from_112_fit=metrics(c112[m76], np.exp(o112[m76])),
                         test_7_vs_6_same_folds=cluster_compare(ape(c, np.exp(o7)), ape(c, np.exp(o6)), iso))
print("ridge 7:", m7); print("ridge 6 same folds:", m6); print("test:", res["expC_ridge"]["test_7_vs_6_same_folds"])
# per-country change for the atypical economies
pc = pd.DataFrame({"iso": iso, "six": ape(c, np.exp(o6)), "seven": ape(c, np.exp(o7))}).groupby("iso").median()
pc["lc_mean"] = q.groupby("iso").lc_share.mean()
pc.to_csv(OUT + "expC_per_country.csv")
print(pc.loc[[i for i in ATYPICAL if i in pc.index]].round(1))
res["expC_atypical"] = pc.loc[[i for i in ATYPICAL if i in pc.index]].round(2).reset_index().values.tolist()
se6 = pd.Series((c - np.exp(o6)) ** 2).groupby(iso).sum().sort_values(ascending=False); se7 = pd.Series((c - np.exp(o7)) ** 2).groupby(iso).sum().sort_values(ascending=False)
res["expC_top6_share_six"] = float(100 * se6.head(6).sum() / se6.sum()); res["expC_top6_share_seven"] = float(100 * se7.head(6).sum() / se7.sum())
res["expC_sse_ratio"] = float(se7.sum() / se6.sum())
res["expC_top6_seven"] = [(k, round(float(100 * v / se7.sum()), 1)) for k, v in se7.head(6).items()]
# also outlier-free comparison in the sub-panel
k2 = ~np.isin(iso, ATYPICAL)
res["expC_no_atypical"] = dict(six=metrics(c[k2], np.exp(o6[k2])), seven=metrics(c[k2], np.exp(o7[k2])), n_countries=int(len(np.unique(iso[k2]))))
# ---------------- drop-one-covariate (refit) and grouped permutation importance, held-out countries
rows = []
base = {"Ridge": (m7, o7)}
ol, _ = loco_predict(X7, y, iso, mk_l); base["LightGBM"] = (metrics(c, np.exp(ol)), ol)
print("LightGBM enriched baseline:", base["LightGBM"][0])
res["lgbm_enriched_baseline"] = base["LightGBM"][0]
rng = np.random.default_rng(42)
for mname, mk in [("Ridge", lambda k: Ridge(alpha=1.0)), ("LightGBM", mk_l)]:
    b_m, b_o = base[mname]
    b_logmse = float(np.mean((y - b_o) ** 2))
    # permutation: per fold, fit once, permute one covariate's whole 8-year window across test samples (30 repeats)
    perm = {f: [] for f in FEATS7}; perm_mse = {f: [] for f in FEATS7}
    preds = {f: np.zeros((30, len(y))) for f in FEATS7}
    for k, (tr, te) in enumerate(GroupKFold(5).split(X7, y, iso)):
        Z = standardise(X7, tr); m = mk(k).fit(Z[tr].reshape(len(tr), -1), y[tr])
        for j, f in enumerate(FEATS7):
            for r in range(30):
                Zp = Z[te].copy(); Zp[:, :, j] = Zp[rng.permutation(len(te)), :, j]
                preds[f][r, te] = m.predict(Zp.reshape(len(te), -1))
    for j, f in enumerate(FEATS7):
        md = np.array([np.median(ape(c, np.exp(preds[f][r]))) for r in range(30)])
        ms = np.array([np.mean((y - preds[f][r]) ** 2) for r in range(30)])
        keepj = [i for i in range(7) if i != j]
        od, _ = loco_predict(X7[:, :, keepj], y, iso, mk); md_drop = metrics(c, np.exp(od))
        rows.append(dict(model=mname, covariate=LABEL[f], base_MdAPE=b_m["MdAPE"], perm_MdAPE=md.mean(), perm_dMdAPE=md.mean() - b_m["MdAPE"], perm_sd=md.std(),
                         perm_logMSE_ratio=ms.mean() / b_logmse, drop_MdAPE=md_drop["MdAPE"], drop_dMdAPE=md_drop["MdAPE"] - b_m["MdAPE"],
                         drop_MAPE=md_drop["MAPE"], drop_R2=md_drop["R2"], drop_RMSE=md_drop["RMSE"]))
    print(mname, "importance done", flush=True)
I = pd.DataFrame(rows); I.to_csv(OUT + "xai_alternative.csv", index=False)
pd.set_option("display.width", 250); print(I.round(3).to_string())
# ---------------- SHAP (TreeExplainer) on LightGBM, enriched panel, out-of-fold
sv = np.zeros((len(y), 56))
for k, (tr, te) in enumerate(GroupKFold(5).split(X7, y, iso)):
    Z = standardise(X7, tr).reshape(len(y), -1); m = mk_l(k).fit(Z[tr], y[tr])
    sv[te] = shap.TreeExplainer(m).shap_values(Z[te])
S = np.abs(sv).reshape(len(y), W, 7)
by_cov = S.sum(axis=1).mean(axis=0); by_lag = S.sum(axis=2).mean(axis=0)
res["shap_by_cov_oof"] = {LABEL[f]: float(v) for f, v in zip(FEATS7, by_cov)}; res["shap_by_lag_oof"] = [float(v) for v in by_lag]
# full-sample fit variant (as a single model on all 76 countries)
Zall = ((X7 - X7.mean(axis=(0, 1))) / X7.std(axis=(0, 1))).reshape(len(y), -1); mall = mk_l(0).fit(Zall, y)
S2 = np.abs(shap.TreeExplainer(mall).shap_values(Zall)).reshape(len(y), W, 7)
res["shap_by_cov_full"] = {LABEL[f]: float(v) for f, v in zip(FEATS7, S2.sum(axis=1).mean(axis=0))}; res["shap_by_lag_full"] = [float(v) for v in S2.sum(axis=2).mean(axis=0)]
print("SHAP oof by cov:", {k: round(v, 4) for k, v in res["shap_by_cov_oof"].items()}); print("SHAP full by cov:", {k: round(v, 4) for k, v in res["shap_by_cov_full"].items()})
print("SHAP full by lag:", np.round(res["shap_by_lag_full"], 4))
# ridge standardised coefficients (summed |beta| over the window; last-step)
Zs = (X7 - X7.mean(axis=(0, 1))) / X7.std(axis=(0, 1)); rr = Ridge(alpha=1.0).fit(Zs.reshape(len(y), -1), y)
B = rr.coef_.reshape(W, 7)
res["ridge_coef_sum"] = {LABEL[f]: float(v) for f, v in zip(FEATS7, B.sum(axis=0))}
res["ridge_coef_abs_by_lag"] = [float(v) for v in np.abs(B).sum(axis=1)]
print("ridge summed standardised coefficients:", {k: round(v, 3) for k, v in res["ridge_coef_sum"].items()})
json.dump(res, open(OUT + "rev05.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k.startswith("sub") or k.startswith("aty") or k.startswith("expC_t") or k.startswith("expC_s") or k == "expC_no_atypical"}, indent=0, default=float))
