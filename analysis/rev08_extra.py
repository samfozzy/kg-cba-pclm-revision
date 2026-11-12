import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from kgcba_common import *
OUT = os.environ.get("KGCBA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")) + os.sep
p = build_panel(); X, y, c, iso, yr = windows(p); M = classical_models(); res = {}
o_r, _ = loco_predict(X, y, iso); o_f, _ = loco_predict(X, y, iso, M["Random forest"])
res["ridge_vs_rf_refit"] = cluster_compare(ape(c, np.exp(o_r)), ape(c, np.exp(o_f)), iso)
# per-fold ridge MdAPE and counts
from sklearn.model_selection import GroupKFold
_, fold = loco_predict(X, y, iso)
res["ridge_fold_mdape"] = [float(np.median(ape(c, np.exp(o_r))[fold == k])) for k in range(5)]
# African summary
df = pd.DataFrame({"iso": iso, "ape": ape(c, np.exp(o_r))}); pc = df.groupby("iso").ape.median()
af = [i for i in pc.index if CONTINENT[i] == "Africa"]
res["africa_country_mdape"] = {i: float(pc[i]) for i in af}; res["africa_n_below_15"] = int((pc[af] < 15).sum()); res["africa_n_above_40"] = int((pc[af] > 40).sum())
res["all_country_mdape_quantiles"] = {str(q): float(pc.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
res["n_countries_mdape_below_20"] = int((pc < 20).sum()); res["n_countries_mdape_above_50"] = int((pc > 50).sum())
json.dump(res, open(OUT + "rev08.json", "w"), indent=1)
v = res["ridge_vs_rf_refit"]; print(f"ridge vs RF: {v['mdape_a']:.2f} vs {v['mdape_b']:.2f} delta {v['delta']:+.2f} [{v['ci_lo']:+.2f},{v['ci_hi']:+.2f}] p_boot {v['p_boot']:.4f} p_country {v['p_country']:.3g}")
print({k: (v if not isinstance(v, dict) else {a: round(b, 1) for a, b in v.items()}) for k, v in res.items() if k != "ridge_vs_rf_refit"})
