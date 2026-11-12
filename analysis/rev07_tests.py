import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from kgcba_common import *
OUT = os.environ.get("KGCBA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")) + os.sep
p = build_panel(); pz = build_panel_zero_fill(); q = add_low_carbon(p)
X, y, c, iso, yr = windows(p); Xz, _, _, _, _ = windows(pz)
M = classical_models(); res = {}
o_r, _ = loco_predict(X, y, iso); A = lambda o: ape(c, np.exp(o))
o_l, _ = loco_predict(X, y, iso, M["LightGBM"]); o_s, _ = loco_predict(X, y, iso, M["SVR"]); o_k, _ = loco_predict(X, y, iso, M["kNN"])
o_x, _ = loco_predict(X, y, iso, M["XGBoost"])
res["ridge_vs_lightgbm_refit"] = cluster_compare(A(o_r), A(o_l), iso)
res["ridge_vs_xgboost_refit"] = cluster_compare(A(o_r), A(o_x), iso)
res["ridge_vs_svr"] = cluster_compare(A(o_r), A(o_s), iso); res["ridge_vs_knn"] = cluster_compare(A(o_r), A(o_k), iso)
o_z, _ = loco_predict(Xz, y, iso); res["zero_fill_vs_original"] = cluster_compare(A(o_z), A(o_r), iso)
X3, _, _, _, _ = windows(p, ("e", "g", "p")); o_3, _ = loco_predict(X3, y, iso); res["no_fuel_shares_vs_six"] = cluster_compare(A(o_3), A(o_r), iso)
X1, _, _, _, _ = windows(p, ("e",)); o_1, _ = loco_predict(X1, y, iso); res["energy_only_vs_six"] = cluster_compare(A(o_1), A(o_r), iso)
o_last, _ = loco_predict(X, y, iso, last_step=True); res["last_year_vs_window"] = cluster_compare(A(o_last), A(o_r), iso); res["last_year_metrics"] = metrics(c, np.exp(o_last))
X7, y7, c7, iso7, _ = windows(q, FEATS7); o7, _ = loco_predict(X7, y7, iso7); o6, _ = loco_predict(X7[:, :, :6], y7, iso7)
res["expC_seven_vs_six"] = cluster_compare(ape(c7, np.exp(o7)), ape(c7, np.exp(o6)), iso7)
k = ~np.isin(iso7, ATYPICAL); res["expC_seven_vs_six_no_atypical"] = cluster_compare(ape(c7[k], np.exp(o7[k])), ape(c7[k], np.exp(o6[k])), iso7[k])
for lam in (0.01, 0.1, 1, 10, 100, 1000):
    o, _ = loco_predict(X, y, iso, lambda k_, lam=lam: Ridge(alpha=lam)); res[f"ridge_lambda_{lam}"] = metrics(c, np.exp(o))
# per-fold table for ridge & number of countries per fold
from sklearn.model_selection import GroupKFold
res["fold_sizes"] = [int(len(np.unique(iso[te]))) for tr, te in GroupKFold(5).split(X, y, iso)]
res["fold_countries"] = [sorted(np.unique(iso[te]).tolist()) for tr, te in GroupKFold(5).split(X, y, iso)]
# continental breakdown, ridge
df = pd.DataFrame({"iso": iso, "ape": A(o_r)}); df["continent"] = df.iso.map(CONTINENT)
cont = df.groupby("continent").ape.agg(median="median", mean="mean", p75=lambda s: s.quantile(.75), n="size").sort_values("median")
cont["countries"] = df.groupby("continent").iso.nunique(); cont.to_csv(OUT + "ridge_by_continent.csv"); print(cont.round(2))
json.dump(res, open(OUT + "rev07.json", "w"), indent=1)
for k_, v in res.items():
    if isinstance(v, dict) and "p_boot" in v: print(f"{k_:32s} MdAPE {v['mdape_a']:.2f} vs {v['mdape_b']:.2f}  delta {v['delta']:+.2f} [{v['ci_lo']:+.2f}, {v['ci_hi']:+.2f}]  p_boot {v['p_boot']:.4f}  p_country {v['p_country']:.2e}  p_obs {v['p_obs']:.1e} | diff of medians {v['diff_of_medians']:+.2f} [{v['dom_ci_lo']:+.2f},{v['dom_ci_hi']:+.2f}] p {v['dom_p_boot']:.4f}")
    elif isinstance(v, dict): print(k_, {a: round(b, 3) for a, b in v.items()})
print(res["fold_sizes"])
print("PUBLISHED: Ridge vs LightGBM  delta -3.10 [-4.91, -1.03]  p_boot 0.0058  p_country 0.026  p_obs 8.3e-25")
