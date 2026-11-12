"""Shared utilities for the revision analyses (PCLM-D-26-00445).

Rebuilds the harmonised panel from the public Our World in Data files and
provides the leave-country-out (grouped five-fold) evaluation helpers used by
every revision script. Data files are read from DATA_DIR:
  owid-co2-data.csv     (owid/co2-data,   commit 382ee6c, 2 June 2026)
  owid-energy-data.csv  (owid/energy-data, commit 7e387a1, 27 April 2026)
"""
import os, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

DATA_DIR = os.environ.get("KGCBA_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
CO2_FILE = os.path.join(DATA_DIR, "owid-co2-data.csv")
ENERGY_FILE = os.path.join(DATA_DIR, "owid-energy-data.csv")
Y0, Y1, W = 1990, 2023, 8
RAW = ["energy_per_capita", "gdp_pc", "population", "coal_share", "oil_share", "gas_share", "co2_per_capita"]
FEATS6 = ("e", "g", "p", "coal_share", "oil_share", "gas_share")
FEATS7 = FEATS6 + ("lc_share",)
LABEL = {"e": "Energy per capita", "g": "GDP per capita", "p": "Population", "coal_share": "Coal share",
         "oil_share": "Oil share", "gas_share": "Gas share", "lc_share": "Low-carbon share"}
CONTINENT = {}
for cont, isos in {
    "Africa": "BEN BWA CMR COD DZA EGY GHA LBY MAR MOZ NER NGA RWA SEN TUN TZA ZAF",
    "Asia": "AFG ARE ARM AZE BGD CHN GEO HKG IDN IND IRN IRQ ISR JOR JPN KAZ KGZ KOR KWT LBN MMR MYS OMN PAK PHL QAT SAU SGP SYR THA TJK TKM TUR TWN UZB VNM YEM",
    "Europe": "ALB AUT BEL BGR BIH BLR CHE CYP CZE DEU DNK ESP EST FIN FRA GBR GRC HRV HUN IRL ISL ITA LTU LUX LVA MDA MKD MLT NLD NOR POL PRT ROU RUS SRB SVK SVN SWE UKR",
    "North America": "CAN CUB DOM GTM JAM MEX PAN SLV USA",
    "South America": "ARG BOL BRA CHL COL PER URY VEN",
    "Oceania": "AUS NZL"}.items():
    for i in isos.split(): CONTINENT[i] = cont
ATYPICAL = ["QAT", "KWT", "SGP", "ISL", "NOR", "LUX", "SAU"]   # structurally atypical energy systems named in the manuscript


def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def _raw():
    d = pd.read_csv(CO2_FILE)
    d = d[d.iso_code.notna() & ~d.iso_code.astype(str).str.startswith("OWID")]
    d = d[(d.year >= Y0) & (d.year <= Y1)].copy()
    d["gdp_pc"] = d.gdp / d.population
    for f in ("coal", "oil", "gas"):
        d[f + "_share"] = d[f + "_co2"] / d.co2
    return d


def build_panel(fill="original", return_audit=False):
    """fill='original': linear interpolation of up to three consecutive missing years inside a
    series, then nearest-observation carry-forward/backward for everything still missing."""
    d = _raw()
    rows, audit = [], []
    for iso, g in d.groupby("iso_code"):
        name = g.country.iloc[0]
        g = g.set_index("year").reindex(range(Y0, Y1 + 1))
        G = g[RAW].astype(float).replace([np.inf, -np.inf], np.nan)
        if (G.notna().sum() == 0).any():
            continue
        F = G.interpolate(limit=3, limit_area="inside").ffill().bfill()
        if F.isna().any().any():
            continue
        miss = G.isna()
        a = {"iso": iso, "country": name, "continent": CONTINENT.get(iso, "?")}
        for c in RAW:
            a["n_filled_" + c] = int(miss[c].sum())
            a["run_" + c] = longest_run(miss[c].values)
        a["longest_run"] = max(a["run_" + c] for c in RAW)
        audit.append(a)
        F["iso"], F["country"], F["year"] = iso, name, F.index
        for c in RAW:
            F["filled_" + c] = miss[c].values
        rows.append(F.reset_index(drop=True))
    p = pd.concat(rows, ignore_index=True)
    p["continent"] = p.iso.map(CONTINENT)
    p["e"], p["g"], p["p"] = np.log10(p.energy_per_capita), np.log10(p.gdp_pc), np.log10(p.population)
    return (p, pd.DataFrame(audit)) if return_audit else p


def add_low_carbon(p, min_complete=0.90):
    e = pd.read_csv(ENERGY_FILE, usecols=["iso_code", "year", "low_carbon_share_energy"])
    e = e[(e.year >= Y0) & (e.year <= Y1) & e.iso_code.notna()]
    q = p.merge(e.rename(columns={"iso_code": "iso", "low_carbon_share_energy": "lc_raw"}), on=["iso", "year"], how="left")
    comp = q.groupby("iso").lc_raw.apply(lambda s: s.notna().mean())
    keep = comp[comp >= min_complete].index
    q = q[q.iso.isin(keep)].copy()
    q["lc_share"] = q.groupby("iso").lc_raw.transform(lambda s: s.interpolate(limit=3, limit_area="inside").ffill().bfill()) / 100.0
    q["filled_lc"] = q.lc_raw.isna()
    return q.reset_index(drop=True)


def windows(p, feats=FEATS6):
    X, y, c, iso, yr = [], [], [], [], []
    for i, g in p.groupby("iso", sort=True):
        g = g.sort_values("year")
        A, t = g[list(feats)].values, g.co2_per_capita.values
        for k in range(W - 1, len(g)):
            X.append(A[k - W + 1:k + 1]); y.append(np.log(t[k])); c.append(t[k]); iso.append(i); yr.append(g.year.values[k])
    return np.array(X), np.array(y), np.array(c), np.array(iso), np.array(yr)


def metrics(c, ch):
    ape = np.abs((c - ch) / c) * 100
    return dict(RMSE=float(np.sqrt(np.mean((c - ch) ** 2))), MAE=float(np.mean(np.abs(c - ch))), MAPE=float(ape.mean()),
                MdAPE=float(np.median(ape)), R2=float(1 - np.sum((c - ch) ** 2) / np.sum((c - c.mean()) ** 2)))


def standardise(X, tr):
    mu, sd = X[tr].mean(axis=(0, 1)), X[tr].std(axis=(0, 1))
    return (X - mu) / sd


def loco_predict(X, y, groups, make_model=lambda k: Ridge(alpha=1.0), last_step=False, n_splits=5):
    """Country-disjoint grouped k-fold; returns out-of-fold log predictions and fold ids."""
    oof, fold = np.zeros(len(y)), np.zeros(len(y), int)
    for k, (tr, te) in enumerate(GroupKFold(n_splits).split(X, y, groups)):
        Z = standardise(X, tr)
        Zf = Z[:, -1, :] if last_step else Z.reshape(len(Z), -1)
        m = make_model(k).fit(Zf[tr], y[tr])
        oof[te], fold[te] = m.predict(Zf[te]), k
    return oof, fold


def ape(c, ch):
    return np.abs((c - ch) / c) * 100


def cluster_compare(a, b, groups, n_boot=10000, seed=42):
    """Algorithm A3 of the manuscript: paired comparison of two APE vectors (a minus b) with countries as clusters.
    delta is the median of the paired differences; the bootstrap resamples countries with replacement."""
    from scipy.stats import wilcoxon
    a, b, groups = np.asarray(a), np.asarray(b), np.asarray(groups)
    dvec = a - b
    p_obs = wilcoxon(a, b).pvalue
    df = pd.DataFrame({"a": a, "b": b, "g": groups})
    med = df.groupby("g")[["a", "b"]].median()
    p_cty = wilcoxon(med.a, med.b).pvalue
    idx = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    keys = list(idx); rng = np.random.default_rng(seed); ds = np.empty(n_boot); dm = np.empty(n_boot)
    for r in range(n_boot):
        samp = np.concatenate([idx[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        ds[r] = np.median(dvec[samp]); dm[r] = np.median(a[samp]) - np.median(b[samp])
    lo, hi = np.percentile(ds, [2.5, 97.5])
    p_boot = min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()))
    lo2, hi2 = np.percentile(dm, [2.5, 97.5]); p2 = min(1.0, 2 * min((dm <= 0).mean(), (dm >= 0).mean()))
    return dict(mdape_a=float(np.median(a)), mdape_b=float(np.median(b)), delta=float(np.median(dvec)), ci_lo=float(lo), ci_hi=float(hi),
                p_obs=float(p_obs), p_country=float(p_cty), p_boot=float(p_boot), n_boot=n_boot,
                diff_of_medians=float(np.median(a) - np.median(b)), dom_ci_lo=float(lo2), dom_ci_hi=float(hi2), dom_p_boot=float(p2))


def build_panel_zero_fill():
    """Accounting-consistent alternative: a coal or gas share that is blank in the source is set to zero
    (the reported national total equals the sum of the reported components in every such country-year);
    energy and GDP per capita are treated as in the original panel."""
    p = build_panel()
    for f in ("coal_share", "gas_share"):
        p.loc[p["filled_" + f], f] = 0.0
    return p


def classical_models(seed=42):
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb, lightgbm as lgb
    return {
        "Ridge": lambda k: Ridge(alpha=1.0),
        "kNN": lambda k: KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "SVR": lambda k: SVR(kernel="rbf", C=10, epsilon=0.01),
        "Random forest": lambda k: RandomForestRegressor(n_estimators=600, max_depth=14, random_state=seed, n_jobs=-1),
        "XGBoost": lambda k: xgb.XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=6, subsample=0.85, random_state=seed, n_jobs=4),
        "LightGBM": lambda k: lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.85, colsample_bytree=0.85,
                                                reg_lambda=1.5, random_state=seed, verbose=-1, n_jobs=4),
    }
