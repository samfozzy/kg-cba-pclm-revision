"""
Country-clustered inference for the LOCO benchmark and Experiment B.

The out-of-fold sample contains 27 repeated years for each of 112 countries. Treating
those 3,024 rows as independent (observation-level Wilcoxon) understates standard errors
by roughly the square root of the within-country cluster size. This script reports, for
every model pair:

  p_obs      observation-level paired Wilcoxon on |APE| (the inflated test)
  p_country  paired Wilcoxon on per-country median APE  (n = 112 clusters)
  boot CI    95% cluster bootstrap CI for the difference in pooled median APE,
             resampling COUNTRIES with replacement (10,000 draws)
  p_boot     two-sided bootstrap p-value for that difference

Outputs: results/table_clustered_stats.csv, results/table_expB_clustered.csv,
         results/analysis_regen.json
"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import paths as P

RNG = np.random.default_rng(42)
B = 10000
OOF = P.OOF
OUT = P.RESULTS


def cluster_boot(diff_by_country, stat=np.median):
    """95% CI and two-sided p for a paired difference, resampling countries."""
    keys = list(diff_by_country)
    pooled = np.concatenate([diff_by_country[k] for k in keys])
    obs = stat(pooled)
    draws = np.empty(B)
    idx = RNG.integers(0, len(keys), size=(B, len(keys)))
    for b in range(B):
        draws[b] = stat(np.concatenate([diff_by_country[keys[i]] for i in idx[b]]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return obs, lo, hi, max(p, 1.0 / B)


def compare(name_a, name_b, ape_a, ape_b, iso, label):
    d = ape_a - ape_b
    _, p_obs = wilcoxon(ape_a, ape_b)
    df = pd.DataFrame(dict(iso=iso, a=ape_a, b=ape_b, d=d))
    per = df.groupby('iso')[['a', 'b']].median()
    _, p_ctry = wilcoxon(per.a.values, per.b.values)
    by = {k: g.d.values for k, g in df.groupby('iso')}
    obs, lo, hi, p_boot = cluster_boot(by)
    return dict(experiment=label, comparison=f'{name_a} vs {name_b}',
                med_a=round(float(np.median(ape_a)), 3), med_b=round(float(np.median(ape_b)), 3),
                n_obs=len(ape_a), n_countries=per.shape[0],
                p_obs=float(f'{p_obs:.3g}'), p_country=round(float(p_ctry), 4),
                delta_med=round(float(obs), 3), ci_lo=round(float(lo), 3),
                ci_hi=round(float(hi), 3), p_boot=round(float(p_boot), 4),
                verdict='significant' if p_boot < 0.05 else 'not significant')


def metrics(t, p):
    e = t - p
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                MAPE=float(np.mean(np.abs(e / t)) * 100),
                MdAPE=float(np.median(np.abs(e / t)) * 100),
                R2=float(1 - np.sum(e ** 2) / np.sum((t - t.mean()) ** 2)))


def main():
    out = {}
    meta = pd.read_csv(f'{OUT}/loco_meta.csv')
    t = meta.y_raw.values
    iso = meta.iso.values
    cz = np.load(f'{OUT}/oof_classical.npz')
    oA = {k: cz[k] for k in cz.files}
    for k in ['LSTM', 'GRU', 'CNN-LSTM', 'CNN-BiLSTM-Attn', 'KG-CBA', 'PSO-KG-CBA']:
        oA[k] = np.load(f'{OOF}/oof_{k}.npz')['oof']
    oA['Ensemble'] = (oA['Ridge'] + oA['GRU'] + oA['PSO-KG-CBA']) / 3.0
    out['panelA'] = {k: metrics(t, np.exp(v)) for k, v in oA.items()}
    apeA = {k: np.abs((t - np.exp(v)) / t) * 100 for k, v in oA.items()}

    metaE = pd.read_csv(P.data('meta_enriched.csv'))
    tE, isoE = metaE.y_raw.values, metaE.iso.values
    oC = {k: np.load(f'{OOF}/oofE_{k}.npz')['oof'] for k in ['Ridge', 'LightGBM', 'PSO-KG-CBA']}
    oC['Ensemble'] = (oC['Ridge'] + oC['PSO-KG-CBA']) / 2.0
    out['panelC'] = {k: metrics(tE, np.exp(v)) for k, v in oC.items()}
    apeC = {k: np.abs((tE - np.exp(v)) / tE) * 100 for k, v in oC.items()}

    pairs = [('Ensemble', 'Ridge', 'A'), ('Ensemble', 'GRU', 'A'),
             ('Ensemble', 'PSO-KG-CBA', 'A'), ('PSO-KG-CBA', 'CNN-BiLSTM-Attn', 'A'),
             ('KG-CBA', 'CNN-BiLSTM-Attn', 'A'), ('Ridge', 'LightGBM', 'A'),
             ('Ridge', 'CNN-BiLSTM-Attn', 'A'),
             ('Ensemble', 'Ridge', 'C'), ('PSO-KG-CBA', 'LightGBM', 'C'),
             ('Ensemble', 'PSO-KG-CBA', 'C')]
    rows = []
    for a, b, exp in pairs:
        if exp == 'A':
            rows.append(compare(a, b, apeA[a], apeA[b], iso, 'A'))
        else:
            rows.append(compare(a, b, apeC[a], apeC[b], isoE, 'C'))
    tab = pd.DataFrame(rows)
    tab.to_csv(f'{OUT}/table_clustered_stats.csv', index=False)
    print(tab[['experiment', 'comparison', 'med_a', 'med_b', 'p_obs',
               'p_country', 'delta_med', 'ci_lo', 'ci_hi', 'p_boot', 'verdict']].to_string(index=False))

    # ---- error concentration (Pareto) ----
    for m in ['Ensemble', 'PSO-KG-CBA']:
        se = pd.DataFrame(dict(iso=iso, se=(t - np.exp(oA[m])) ** 2)).groupby('iso').se.sum()
        top = se.sort_values(ascending=False).head(6)
        out.setdefault('pareto', {})[m] = dict(
            top6=list(top.index), share=float(top.sum() / se.sum()))
        print(m, 'top-6 share %.3f' % (top.sum() / se.sum()), list(top.index))

    # ---- continent + Ghana ----
    panel = pd.read_csv(P.data('panel_clean.csv'))[['iso_code', 'country']].drop_duplicates()
    cont = pd.read_csv(P.CONTINENTS) if os.path.exists(P.CONTINENTS) else None
    d = pd.DataFrame(dict(iso=iso, ape=apeA['Ensemble']))
    out['ghana'] = dict(n=int((d.iso == 'GHA').sum()),
                        median_ape=float(d[d.iso == 'GHA'].ape.median()),
                        mean_ape=float(d[d.iso == 'GHA'].ape.mean()))
    if cont is not None:
        d = d.merge(cont, on='iso', how='left')
        g = d.groupby('continent').ape.agg(['median', 'mean', 'count', lambda s: s.quantile(.75)])
        g.columns = ['median', 'mean', 'n_obs', 'q75']
        out['continent_ape_ensemble'] = g.round(2).to_dict()
        print(g.round(2).to_string())
    print('Ghana', {k: round(v, 3) if isinstance(v, float) else v for k, v in out['ghana'].items()})

    json.dump(out, open(f'{OUT}/analysis_regen.json', 'w'), indent=1)

    # ---- Experiment B, clustered ----
    rows = []
    for win in ['main', 'repl']:
        f = f'{OUT}/expB_errors_{win}.csv'
        if not os.path.exists(f):
            continue
        e = pd.read_csv(f)
        for h in sorted(e.h.unique()):
            sub = e[e.h == h]
            base = sub[sub.model == 'Persistence'].set_index(['iso', 'target_year']).abs_err
            for m in sorted(sub.model.unique()):
                if m == 'Persistence':
                    continue
                cur = sub[sub.model == m].set_index(['iso', 'target_year']).abs_err
                j = pd.concat([base.rename('p'), cur.rename('m')], axis=1).dropna().reset_index()
                dd = {k: (g.m - g.p).values for k, g in j.groupby('iso')}
                obs, lo, hi, pb = cluster_boot(dd, stat=np.mean)
                rows.append(dict(window=win, h=h, model=m,
                                 mae_model=round(j.m.mean(), 4), mae_persist=round(j.p.mean(), 4),
                                 delta_mae=round(float(obs), 4), ci_lo=round(float(lo), 4),
                                 ci_hi=round(float(hi), 4), p_boot=round(float(pb), 4),
                                 beats_persistence=bool(obs < 0),
                                 significant=bool(pb < 0.05)))
    if rows:
        tb = pd.DataFrame(rows)
        tb.to_csv(f'{OUT}/table_expB_clustered.csv', index=False)
        print('\n=== Experiment B vs persistence (country-clustered) ===')
        print(tb[tb.significant].to_string(index=False))


if __name__ == '__main__':
    main()
