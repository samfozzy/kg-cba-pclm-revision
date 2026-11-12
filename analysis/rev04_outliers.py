import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from kgcba_common import *
OUT = os.environ.get("KGCBA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")) + os.sep
p = build_panel(); X, y, c, iso, yr = windows(p)
keep = ~np.isin(iso, ATYPICAL)
rows = []; res = {}
for name, mk in classical_models().items():
    o, _ = loco_predict(X, y, iso, mk)
    o105, _ = loco_predict(X[keep], y[keep], iso[keep], mk)
    rows.append(dict(model=name, scope="All 112 countries", **metrics(c, np.exp(o))))
    rows.append(dict(model=name, scope="Scored without the 7 atypical economies (105 countries; model unchanged)", **metrics(c[keep], np.exp(o[keep]))))
    rows.append(dict(model=name, scope="Re-estimated without the 7 atypical economies (105 countries)", **metrics(c[keep], np.exp(o105))))
    rows.append(dict(model=name, scope="The 7 atypical economies only (model unchanged)", **metrics(c[~keep], np.exp(o[~keep]))))
    if name == "Ridge":
        se = pd.Series((c - np.exp(o)) ** 2).groupby(iso).sum().sort_values(ascending=False)
        share = 100 * se / se.sum()
        res["ridge_sse_top"] = [(k, round(float(v), 2)) for k, v in share.head(12).items()]
        res["ridge_top6_share"] = float(share.head(6).sum()); res["ridge_atypical7_share"] = float(share[ATYPICAL].sum())
        res["ridge_top3_share"] = float(share.head(3).sum())
        pc = pd.DataFrame({"iso": iso, "ape": ape(c, np.exp(o)), "err": np.exp(o) - c, "c": c}).groupby("iso").agg(mdape=("ape", "median"), bias=("err", "mean"), mean_c=("c", "mean"))
        res["ridge_atypical_detail"] = pc.loc[ATYPICAL].round(2).reset_index().values.tolist()
        print(pc.loc[ATYPICAL].round(2))
    print(name, "done", flush=True)
T = pd.DataFrame(rows); T.to_csv(OUT + "outlier_sensitivity.csv", index=False)
pd.set_option("display.width", 250); print(T.round(3).to_string())
print(json.dumps(res, indent=0))
json.dump(res, open(OUT + "rev04.json", "w"), indent=1)
