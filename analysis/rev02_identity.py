import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json
from kgcba_common import *
from kgcba_common import _raw
d = _raw(); p = build_panel()
d = d[d.iso_code.isin(p.iso.unique())].copy()
comp = ["coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2", "other_industry_co2"]
d["sum_reported"] = d[comp].sum(axis=1, skipna=True)
d["resid"] = d.co2 - d.sum_reported
d["resid_pct"] = 100 * d.resid / d.co2
m = d.coal_co2.isna() | d.gas_co2.isna()
print("country-years with coal or gas CO2 missing:", int(m.sum()), "coal:", int(d.coal_co2.isna().sum()), "gas:", int(d.gas_co2.isna().sum()))
print("residual (% of total) where a fuel component is missing:\n", d.loc[m, "resid_pct"].describe().round(4).to_string())
print("residual (% of total) where nothing missing:\n", d.loc[~m, "resid_pct"].describe().round(4).to_string())
print("share of missing-fuel country-years with |residual| < 0.5% of total:", round(float((d.loc[m, "resid_pct"].abs() < 0.5).mean()), 4), " <0.01%:", round(float((d.loc[m, "resid_pct"].abs() < 0.01).mean()), 4))
print(d.loc[m & (d.resid_pct.abs() >= 0.5), ["iso_code", "year", "co2", "coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2", "other_industry_co2", "resid_pct"]].head(20).to_string())
# first observed values after/before missing stretches: are they small?
for i in ["GHA", "SLV", "NER", "PAN", "JAM", "GTM", "CMR", "RWA", "BEN", "BOL", "YEM", "JOR", "LBN"]:
    g = d[d.iso_code == i].sort_values("year")
    out = []
    for f in ["coal", "gas"]:
        s = (g[f + "_co2"] / g.co2); obs = s.dropna()
        if s.isna().any():
            out.append(f"{f}: missing {int(s.isna().sum())} yrs; observed n={len(obs)} share min={obs.min():.4f} med={obs.median():.4f} max={obs.max():.4f}; filled value used ~ first/last obs {obs.iloc[0]:.4f}/{obs.iloc[-1]:.4f}")
    print(i, " | ".join(out))
