"""
Build the analysis panel from the raw Our World in Data downloads.

Reproduces data/panel_clean.csv exactly (112 countries x 34 years, 3,808 country-years)
and the 76-country enriched sub-panel used in Experiment C.

Source note: energy_per_capita is taken from the OWID **CO2** file, not the energy file.
The two lineages are different vintages and disagree for 175 recent country-years; the
CO2-file column is the one the published panel uses.

Outputs
-------
data/panel_clean.csv          harmonised panel
data/imputation_flags.csv     per country-year, which values were filled
data/enriched.npz             (X, y) arrays for Experiment C
data/meta_enriched.csv        iso/year/y_raw for Experiment C
data/folds_enriched.pkl       grouped 5-fold split for Experiment C
"""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

RAW = 'data/raw'
OUT = 'data'
Y0, Y1 = 1990, 2023
W = 8
CORE = ['co2_per_capita', 'energy_per_capita', 'gdp_per_capita', 'population',
        'coal_share', 'oil_share', 'gas_share']


def fill_series(s):
    """Interior gaps of <=3 consecutive years by linear interpolation; edges by
    nearest-value extension (which carries no length cap -- see the imputation audit)."""
    return s.interpolate(limit=3, limit_area='inside').ffill().bfill()


def build():
    co2 = pd.read_csv(f'{RAW}/owid-co2-data.csv', low_memory=False)
    en = pd.read_csv(f'{RAW}/owid-energy-data.csv', low_memory=False)

    iso_ok = (co2.iso_code.notna()
              & ~co2.iso_code.astype(str).str.startswith('OWID')
              & (co2.iso_code.astype(str).str.len() == 3))
    c = co2[(co2.year.between(Y0, Y1)) & iso_ok][
        ['country', 'iso_code', 'year', 'population', 'gdp', 'co2', 'co2_per_capita',
         'energy_per_capita', 'coal_co2', 'oil_co2', 'gas_co2']].copy()

    lc = en[en.year.between(Y0, Y1)][['iso_code', 'year', 'low_carbon_share_energy']]
    d = c.merge(lc, on=['iso_code', 'year'], how='left')

    d['gdp_per_capita'] = d.gdp / d.population
    for f in ['coal', 'oil', 'gas']:
        d[f'{f}_share'] = d[f'{f}_co2'] / d.co2

    d = d.sort_values(['iso_code', 'year'])
    flags = d[['iso_code', 'year']].copy()
    for col in CORE:
        flags[col + '_imputed'] = d[col].isna().values
        d[col] = d.groupby('iso_code')[col].transform(fill_series)

    complete = d.groupby('iso_code').apply(
        lambda g: bool(g[CORE].notna().all().all() and len(g) == (Y1 - Y0 + 1)),
        include_groups=False)
    keep = set(complete[complete].index)
    d = d[d.iso_code.isin(keep)].copy()
    d['log_population'] = np.log10(d.population)

    os.makedirs(OUT, exist_ok=True)
    d.to_csv(f'{OUT}/panel_clean.csv', index=False)
    flags[flags.iso_code.isin(keep)].to_csv(f'{OUT}/imputation_flags.csv', index=False)
    print(f'panel: {d.iso_code.nunique()} countries x {Y1-Y0+1} years = {len(d)} country-years')
    return d


def build_enriched(d):
    """76-country sub-panel adding the low-carbon share of primary energy."""
    cov = d.groupby('iso_code').low_carbon_share_energy.apply(lambda s: s.notna().mean())
    sub = set(cov[cov >= 0.90].index)
    e = d[d.iso_code.isin(sub)].copy()
    e['low_carbon_share_energy'] = e.groupby('iso_code').low_carbon_share_energy.transform(fill_series) / 100.0  # OWID reports percent; panel uses a fraction
    e['log_epc'] = np.log10(e.energy_per_capita.clip(lower=1))
    e['log_gpc'] = np.log10(e.gdp_per_capita.clip(lower=100))
    e['y'] = np.log(e.co2_per_capita)
    FE = ['log_epc', 'log_gpc', 'log_population', 'coal_share', 'oil_share',
          'gas_share', 'low_carbon_share_energy']
    X, y, meta = [], [], []
    for iso, g in e.groupby('iso_code'):
        g = g.sort_values('year').reset_index(drop=True)
        for i in range(W - 1, len(g)):
            X.append(g[FE].iloc[i - W + 1:i + 1].values)
            y.append(g.y.iloc[i])
            meta.append((iso, int(g.year.iloc[i]), float(g.co2_per_capita.iloc[i])))
    X = np.asarray(X, np.float32); y = np.asarray(y, np.float32)
    meta = pd.DataFrame(meta, columns=['iso', 'year', 'y_raw'])
    np.savez(f'{OUT}/enriched.npz', X=X, y=y)
    meta.to_csv(f'{OUT}/meta_enriched.csv', index=False)
    folds = list(GroupKFold(n_splits=5).split(X, y, meta.iso.values))
    pickle.dump(folds, open(f'{OUT}/folds_enriched.pkl', 'wb'))
    print(f'enriched: {meta.iso.nunique()} countries, {len(y)} samples, X{X.shape}')


if __name__ == '__main__':
    build_enriched(build())
