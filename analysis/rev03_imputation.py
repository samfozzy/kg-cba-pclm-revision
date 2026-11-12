import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from kgcba_common import *
OUT = os.environ.get("KGCBA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")) + os.sep
p, audit = build_panel(return_audit=True); pz = build_panel_zero_fill()
X, y, c, iso, yr = windows(p); Xz, _, _, _, _ = windows(pz)
run = audit.set_index("iso").longest_run
le3 = np.isin(iso, run[run <= 3].index); le10 = np.isin(iso, run[run <= 10].index)
print("countries <=3:", int((run <= 3).sum()), " <=10:", int((run <= 10).sum()))
models = classical_models(); rows = []; store = {}
for name, mk in models.items():
    o, _ = loco_predict(X, y, iso, mk); store[(name, "orig")] = o
    oz, _ = loco_predict(Xz, y, iso, mk); store[(name, "zero")] = oz
    o87, _ = loco_predict(X[le3], y[le3], iso[le3], mk)
    o93, _ = loco_predict(X[le10], y[le10], iso[le10], mk)
    for lab, m in [("A. Original panel, all 112 countries", metrics(c, np.exp(o))),
                   ("B. Blank coal/gas shares set to zero, all 112 countries", metrics(c, np.exp(oz))),
                   ("C1. Original panel, scored on the 87 countries with no filled run > 3 yr", metrics(c[le3], np.exp(o[le3]))),
                   ("C2. Re-estimated on the 87 countries with no filled run > 3 yr", metrics(c[le3], np.exp(o87))),
                   ("D1. Original panel, scored on the 93 countries with no filled run > 10 yr", metrics(c[le10], np.exp(o[le10]))),
                   ("D2. Re-estimated on the 93 countries with no filled run > 10 yr", metrics(c[le10], np.exp(o93))),
                   ("E. Original panel, scored on the 25 countries with a filled run > 3 yr", metrics(c[~le3], np.exp(o[~le3])))]:
        rows.append(dict(model=name, treatment=lab, **m))
    print(name, "done")
# no-fuel-share ridge
X3, _, _, _, _ = windows(p, ("e", "g", "p")); o3, _ = loco_predict(X3, y, iso)
rows.append(dict(model="Ridge", treatment="F. No fuel shares (energy, GDP, population only), all 112 countries", **metrics(c, np.exp(o3))))
rows.append(dict(model="Ridge", treatment="F. No fuel shares, scored on the 25 countries with a filled run > 3 yr", **metrics(c[~le3], np.exp(o3[~le3]))))
T = pd.DataFrame(rows); T.to_csv(OUT + "imputation_sensitivity.csv", index=False)
pd.set_option("display.width", 250)
print(T[T.model.isin(["Ridge", "LightGBM", "SVR", "XGBoost", "Random forest", "kNN"])].round(3).to_string())
# paired cluster tests, ridge: original vs zero fill
a, b = ape(c, np.exp(store[("Ridge", "orig")])), ape(c, np.exp(store[("Ridge", "zero")]))
t = cluster_compare(b, a, iso); print("ridge zero-fill vs original:", {k: round(v, 4) for k, v in t.items()})
# how different are the predictions and inputs?
print("max |difference| in imputed share values (orig - zero):", float(np.abs(p.coal_share - pz.coal_share).max().round(4)), float(np.abs(p.gas_share - pz.gas_share).max().round(4)),
      " mean over filled cells:", float(p.coal_share[p.filled_coal_share].mean().round(4)), float(p.gas_share[p.filled_gas_share].mean().round(4)),
      " median:", float(p.coal_share[p.filled_coal_share].median().round(4)), float(p.gas_share[p.filled_gas_share].median().round(4)),
      " share of filled cells with value < 0.01:", float(((p.coal_share[p.filled_coal_share] < 0.01).mean()).round(3)), float(((p.gas_share[p.filled_gas_share] < 0.01).mean()).round(3)),
      " <0.05:", float(((p.coal_share[p.filled_coal_share] < 0.05).mean()).round(3)), float(((p.gas_share[p.filled_gas_share] < 0.05).mean()).round(3)))
# per-country MdAPE for African cases, ridge
pc = pd.DataFrame({"iso": iso, "orig": a, "zero": b, "nofuel": ape(c, np.exp(o3))}).groupby("iso").median()
pc["longest_run"] = run; pc["continent"] = pd.Series(CONTINENT)
pc.to_csv(OUT + "ridge_per_country_mdape.csv")
print(pc.loc[["ZAF", "EGY", "NGA", "GHA", "CMR", "RWA", "BEN", "NER", "SEN", "TUN", "LBY", "TZA", "MAR", "DZA", "COD", "MOZ", "BWA"]].round(1).to_string())
json.dump({"ridge_zero_vs_orig": t}, open(OUT + "rev03.json", "w"), indent=1)
