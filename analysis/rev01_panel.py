import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from kgcba_common import *
OUT = os.environ.get("KGCBA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")) + os.sep
p, audit = build_panel(return_audit=True)
q = add_low_carbon(p)
audit["in_enriched_subpanel"] = audit.iso.isin(q.iso.unique())
res = {}
# ---- size
X, y, c, iso, yr = windows(p)
res["size"] = dict(countries=int(p.iso.nunique()), years=int(p.year.nunique()), country_years=int(len(p)), windows=int(len(y)),
                   target_years=[int(yr.min()), int(yr.max())], targets_per_country=int(len(y) / p.iso.nunique()),
                   enriched_countries=int(q.iso.nunique()), enriched_country_years=int(len(q)), enriched_windows=int(q.iso.nunique() * 27))
# ---- descriptive statistics on the raw scale
desc = []
names = {"co2_per_capita": "Per-capita CO2 (t CO2 per person per year)", "energy_per_capita": "Primary energy per capita (kWh per person)",
         "gdp_pc": "GDP per capita (international-$ per person, 2011 prices)", "population": "Population (millions)",
         "coal_share": "Coal share of CO2", "oil_share": "Oil share of CO2", "gas_share": "Gas share of CO2"}
for v, lab in names.items():
    s = p[v] / (1e6 if v == "population" else 1)
    desc.append(dict(variable=lab, n=len(s), mean=s.mean(), sd=s.std(), min=s.min(), p25=s.quantile(.25), median=s.median(), p75=s.quantile(.75), max=s.max(),
                     filled=int(p["filled_" + v].sum()), filled_pct=100 * p["filled_" + v].mean()))
s = q.lc_share
desc.append(dict(variable="Low-carbon share of primary energy (76 countries)", n=len(s), mean=s.mean(), sd=s.std(), min=s.min(), p25=s.quantile(.25), median=s.median(), p75=s.quantile(.75), max=s.max(), filled=0, filled_pct=0.0))
desc = pd.DataFrame(desc); desc.to_csv(OUT + "descriptives.csv", index=False)
print(desc.round(3).to_string())
# ---- correlations (log-transformed, all 3,808 country-years)
cc = pd.DataFrame({"ln_co2pc": np.log(p.co2_per_capita), "log_epc": p.e, "log_gdppc": p.g, "log_pop": p.p, "coal": p.coal_share, "oil": p.oil_share, "gas": p.gas_share}).corr()
cc.to_csv(OUT + "correlations.csv"); print(cc.round(3).to_string())
res["corr"] = {k: float(v) for k, v in cc["ln_co2pc"].items()}
# ---- audit
audit = audit.sort_values("longest_run", ascending=False)
audit.to_csv(OUT + "imputation_audit_by_country.csv", index=False)
gt3 = audit[audit.longest_run > 3]; gt10 = audit[audit.longest_run > 10]
res["audit"] = dict(n_gt3=int(len(gt3)), n_gt10=int(len(gt10)), gt3=gt3.iso.tolist(), gt3_runs=gt3.longest_run.tolist(),
                    gt3_by_continent=gt3.continent.value_counts().to_dict(), gt10_by_continent=gt10.continent.value_counts().to_dict(),
                    filled={c_: int(audit["n_filled_" + c_].sum()) for c_ in RAW}, longest={c_: int(audit["run_" + c_].max()) for c_ in RAW},
                    countries_any_fill_excl_gdp=int(((audit[["n_filled_" + c_ for c_ in RAW if c_ != "gdp_pc"]].sum(axis=1)) > 0).sum()),
                    gdp_2023_filled=int(p[(p.year == 2023)].filled_gdp_pc.sum()), gdp_other=audit[audit.n_filled_gdp_pc > 1][["iso", "n_filled_gdp_pc", "run_gdp_pc"]].values.tolist())
print(json.dumps(res["audit"], indent=0)[:1500])
print(gt3[["iso", "country", "continent", "run_energy_per_capita", "run_gdp_pc", "run_coal_share", "run_gas_share", "longest_run", "in_enriched_subpanel"]].to_string())
# which variable/years are filled for a few countries
for i in ["GHA", "SLV", "YEM", "JOR", "LBN", "SRB", "AZE", "PAN", "TUN"]:
    g = p[p.iso == i]
    print(i, {c_: g.year[g["filled_" + c_]].tolist() for c_ in RAW if g["filled_" + c_].any() and c_ != "gdp_pc"})
res["range"] = dict(min=float(p.co2_per_capita.min()), max=float(p.co2_per_capita.max()), max_target=float(c.max()), min_target=float(c.min()))
json.dump(res, open(OUT + "rev01.json", "w"), indent=1)
print(res["size"], res["range"])
