"""
Regenerate every figure that depends on out-of-fold predictions, from the corrected
and fully reproducible results. Figures F01-F06 are unchanged and not rebuilt here.

Produces: F07 (supp), F08, F09, F10, F11, F12, F13, F14 (supp), F15.
All output is written as flattened RGB at 300 dpi for production.
"""
import os, json, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import paths as P

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
C = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161', '#949494', '#56B4E9']
FIG = P.FIGURES


def save(name):
    plt.savefig(f'{FIG}/{name}', facecolor='white')
    plt.close()


# ---------------- data ----------------
metaA = pd.read_csv(P.results('loco_meta.csv'))
tA = metaA.y_raw.values
cz = np.load(P.results('oof_classical.npz'))
oA = {k: cz[k] for k in cz.files}
for k in ['LSTM', 'GRU', 'CNN-LSTM', 'CNN-BiLSTM-Attn', 'KG-CBA', 'PSO-KG-CBA']:
    oA[k] = np.load(P.oof(f'oof_{k}.npz'))['oof']
oA['Ensemble'] = (oA['Ridge'] + oA['GRU'] + oA['PSO-KG-CBA']) / 3

metaE = pd.read_csv(P.data('meta_enriched.csv'))
tE = metaE.y_raw.values
oE = {k: np.load(P.oof(f'oofE_{k}.npz'))['oof'] for k in ['Ridge', 'LightGBM', 'PSO-KG-CBA']}
oE['Ensemble'] = (oE['Ridge'] + oE['PSO-KG-CBA']) / 2

an = json.load(open(P.results('analysis_regen.json')))


def summ(t, lp):
    p = np.exp(lp); e = t - p
    return dict(MAPE=np.mean(np.abs(e / t)) * 100, MdAPE=np.median(np.abs(e / t)) * 100,
                R2=1 - np.sum(e ** 2) / np.sum((t - t.mean()) ** 2))


# ---------------- F07 training dynamics (supplementary) ----------------
fig, axs = plt.subplots(1, 4, figsize=(11, 2.7), sharey=True)
for ax, k in zip(axs, ['LSTM', 'GRU', 'CNN-LSTM', 'CNN-BiLSTM-Attn']):
    d = np.load(P.oof(f'{k}_fold0.npz'))
    ax.plot(d['loss'], color=C[0], lw=1.4, label='train')
    ax.plot(d['val_loss'], color=C[3], lw=1.4, label='inner val')
    ax.set(title=k, xlabel='Epoch', yscale='log')
axs[0].set_ylabel('MSE (log target)')
axs[0].legend(frameon=False, fontsize=7.5)
plt.tight_layout(); save('F07_training.png')

# ---------------- F08 model comparison ----------------
order = ['Mean', 'kNN', 'SVR', 'RF', 'XGBoost', 'LightGBM', 'LSTM', 'CNN-LSTM',
         'CNN-BiLSTM-Attn', 'GRU', 'Ridge', 'KG-CBA', 'PSO-KG-CBA', 'Ensemble']
pa = an['panelA']
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4))
xs = np.arange(len(order))
for i, met in enumerate(['MAPE', 'MdAPE']):
    v = [min(pa[m][met], 60) for m in order]
    cols = [C[3] if m in ('KG-CBA', 'PSO-KG-CBA') else C[2] if m == 'Ensemble'
            else C[1] if m == 'Ridge' else C[0] for m in order]
    ax[i].bar(xs, v, color=cols)
    ax[i].set_xticks(xs); ax[i].set_xticklabels(order, rotation=55, ha='right', fontsize=7.5)
    ax[i].set(ylabel=f'{met} (%)', title=f'({"ab"[i]}) {met}')
    if met == 'MAPE':
        ax[i].text(0, 58, f"{pa['Mean']['MAPE']:.0f}% (clipped)", fontsize=7, ha='left')
plt.tight_layout(); save('F08_model_comparison.png')

# ---------------- F09 predicted vs observed ----------------
fig, axs = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
for ax, name in zip(axs, ['Ridge', 'GRU', 'PSO-KG-CBA', 'Ensemble']):
    p = np.exp(oA[name])
    ax.scatter(tA, p, s=5, alpha=.35, color=C[0], edgecolors='none')
    lim = [.03, 90]; ax.plot(lim, lim, 'k--', lw=.8)
    ax.set(xscale='log', yscale='log', xlim=lim, ylim=lim, xlabel='Observed (t/person)',
           title=f"{name}\nR$^2$={pa[name]['R2']:.3f}")
axs[0].set_ylabel('Predicted (t/person)')
plt.tight_layout(); save('F09_pred_vs_actual.png')

# ---------------- F10 Pareto ----------------
STABLE = ['SGP', 'QAT', 'KWT']
e2 = (tA - np.exp(oA['Ensemble'])) ** 2
sse = pd.DataFrame({'iso': metaA.iso, 'e2': e2}).groupby('iso').e2.sum().sort_values(ascending=False)
top = sse.head(15); cum = 100 * sse.cumsum() / sse.sum()
top6 = list(sse.index[:6])
fig, ax = plt.subplots(figsize=(7.4, 3.3))
cols = [C[3] if i in STABLE else C[1] if i in top6 else C[0] for i in top.index]
ax.bar(range(15), top.values, color=cols)
ax.set_xticks(range(15)); ax.set_xticklabels(top.index, rotation=45)
ax.set_ylabel('Sum of squared errors')
ax2 = ax.twinx(); ax2.plot(range(15), cum.iloc[:15].values, color=C[2], marker='o', ms=3.5, lw=1.4)
ax2.set_ylabel('Cumulative share of total SSE (%)', color=C[2]); ax2.set_ylim(0, 100)
ax2.spines['right'].set_visible(True)
ax.set_title('Error concentration (dark red = outlier in all six models; orange = this model\'s top six)')
save('F10_pareto.png')

# ---------------- F11 enrichment ----------------
iso76 = set(metaE.iso); mask = metaA.iso.isin(iso76).values
ms = ['Ridge', 'PSO-KG-CBA', 'Ensemble']
A76 = {m: summ(tA[mask], oA[m][mask]) for m in ms}
C76 = {m: summ(tE, oE[m]) for m in ms}
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.1))
xp = np.arange(3)
for i, (met, ttl) in enumerate([('MAPE', '(a) MAPE (%)'), ('MdAPE', '(b) MdAPE (%)')]):
    ax[i].bar(xp - .2, [A76[m][met] for m in ms], .4, color=C[6], label='6 covariates')
    ax[i].bar(xp + .2, [C76[m][met] for m in ms], .4, color=C[2], label='+ low-carbon share')
    ax[i].set_xticks(xp); ax[i].set_xticklabels(ms, fontsize=8)
    ax[i].set_title(ttl)
ax[0].legend(frameon=False, fontsize=8)
plt.suptitle('Identical 76 countries, with and without the low-carbon energy share', fontsize=9.5)
plt.tight_layout(); save('F11_enrichment.png')
json.dump(dict(A76=A76, C76=C76), open(P.results('enrichment_effect_regen.json'), 'w'), indent=1)

# ---------------- F12 African case studies ----------------
cf = np.load(P.results('conformal_regen.npz'))
cases = [c for c in ['ZAF', 'EGY', 'NGA', 'GHA'] if c in iso76]
runs = pd.read_csv(P.results('table_imputation_runs.csv')).set_index('iso').longest_run
fig, axs = plt.subplots(1, len(cases), figsize=(2.9 * len(cases), 2.9), sharex=True)
for ax, iso in zip(np.atleast_1d(axs), cases):
    m = (metaE.iso == iso).values
    yr = metaE.year.values[m]
    ax.fill_between(yr, cf['lo'][m], cf['hi'][m], color=C[0], alpha=.18, label='90% conformal')
    ax.plot(yr, tE[m], 'k-', lw=1.5, label='observed')
    ax.plot(yr, np.exp(oE['Ensemble'][m]), color=C[3], lw=1.4, ls='--', label='predicted')
    ape = np.median(np.abs((tE[m] - np.exp(oE['Ensemble'][m])) / tE[m])) * 100
    flag = '  [imputed run %d yr]' % runs.get(iso, 0) if runs.get(iso, 0) > 3 else ''
    ax.set(title=f'{iso}: MdAPE {ape:.1f}%{flag}', xlabel='Year')
np.atleast_1d(axs)[0].set_ylabel('t CO$_2$ person$^{-1}$')
np.atleast_1d(axs)[0].legend(frameon=False, fontsize=7)
plt.tight_layout(); save('F12_africa_cases.png')

# ---------------- F13 SHAP / F14 attention ----------------
FE = ['log_epc', 'log_gpc', 'log_population', 'coal_share', 'oil_share', 'gas_share',
      'low_carbon_share']
try:
    import shap
    from lightgbm import LGBMRegressor
    d = np.load(P.data('enriched.npz')); XE = d['X']; yE = d['y']
    foldsE = pickle.load(open(P.data('folds_enriched.pkl'), 'rb'))
    trn, tst = foldsE[0]
    mu, sd = XE[trn].mean((0, 1)), XE[trn].std((0, 1)) + 1e-8
    F = ((XE - mu) / sd).reshape(len(XE), -1)
    lg = LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.85,
                       colsample_bytree=0.85, reg_lambda=1.5, random_state=42,
                       verbose=-1).fit(F[trn], yE[trn])
    sv = shap.TreeExplainer(lg).shap_values(F[tst]).reshape(len(tst), 8, 7)
    by_f = {FE[i]: float(np.abs(sv[:, :, i]).mean()) for i in range(7)}
    by_l = {f't-{7-j}': float(np.abs(sv[:, j, :]).mean()) for j in range(8)}
    json.dump(dict(shap_by_feature=by_f, shap_by_lag=by_l),
              open(P.results('shap_regen.json'), 'w'), indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.0))
    k = sorted(by_f, key=by_f.get)
    ax[0].barh(k, [by_f[i] for i in k], color=C[0])
    ax[0].set(xlabel='mean |SHAP|', title='(a) By covariate')
    lg_ = list(by_l)
    ax[1].bar(lg_, [by_l[i] for i in lg_], color=C[4])
    ax[1].set(ylabel='mean |SHAP|', title='(b) By window position')
    plt.tight_layout(); save('F13_shap.png')
    print('SHAP by feature:', {a: round(b, 4) for a, b in by_f.items()})
    print('SHAP by lag    :', {a: round(b, 4) for a, b in by_l.items()})
except Exception as ex:
    print('SHAP skipped:', ex)

try:
    import tensorflow as tf
    from conformal import TemporalAttention, load_fold_model
    d = np.load(P.data('enriched.npz')); XE = d['X']
    foldsE = pickle.load(open(P.data('folds_enriched.pkl'), 'rb'))
    atts = []
    for fi, (trn, tst) in enumerate(foldsE):
        m = load_fold_model(P.oof(f'E-PSO-KG-CBA_fold{fi}.keras'), XE.shape[2])
        s = np.load(P.oof(f'E-PSO-KG-CBA_scaler_fold{fi}.npz'))
        sub = Model = None
        lay = [l for l in m.layers if isinstance(l, TemporalAttention)][0]
        ex = tf.keras.Model(m.input, lay.output[1])
        atts.append(ex.predict((XE[tst] - s['mu']) / s['sd'], verbose=0).reshape(len(tst), 8))
    att = np.concatenate(atts).mean(0)
    json.dump(att.tolist(), open(P.results('attention_regen.json'), 'w'))
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.bar(range(8), att, color=C[4]); ax.axhline(1 / 8, ls=':', color='gray', lw=1)
    ax.set_xticks(range(8)); ax.set_xticklabels([f't-{7-j}' for j in range(8)])
    ax.set(ylabel='Mean attention weight', title='Temporal attention (E-PSO-KG-CBA)')
    ax.text(7.1, 1 / 8 + .002, 'uniform', fontsize=7, color='gray')
    save('F14_attention.png')
    print('attention:', np.round(att, 4).tolist())
except Exception as ex:
    print('attention skipped:', ex)

# ---------------- F15 Experiment B (rebuilt from real results) ----------------
tb = pd.read_csv(P.results('table_experiment_b.csv'))
sel = ['Persistence', 'Drift', 'Ridge (growth)', 'RF (growth)', 'XGBoost (level)',
       'LightGBM (growth)']
fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.4), sharey=True)
for ax, (win, ttl) in zip(axs, [('main_2021_2023', '(a) Test 2021-2023'),
                                ('replication_2015_2019', '(b) Replication, test 2015-2019')]):
    w = .13
    for i, m in enumerate(sel):
        v = [tb[(tb.window == win) & (tb.h == h) & (tb.model == m)].MASE.values[0] for h in [1, 3, 5]]
        ax.bar(np.arange(3) + (i - 2.5) * w, v, w, color=C[i % len(C)], label=m)
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(['h = 1', 'h = 3', 'h = 5'])
    ax.set_title(ttl)
axs[0].set_ylabel('MASE (vs in-sample naive)')
axs[0].legend(frameon=False, fontsize=7, ncol=2)
plt.suptitle('Experiment B: learners beat persistence only marginally, and not consistently across windows',
             fontsize=9.5)
plt.tight_layout(); save('F15_temporal.png')

# ---------------- flatten every figure to RGB ----------------
from PIL import Image
for f in sorted(os.listdir(FIG)):
    if not f.endswith('.png'):
        continue
    im = Image.open(f'{FIG}/{f}')
    if im.mode != 'RGB':
        bg = Image.new('RGB', im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1] if im.mode in ('RGBA', 'LA') else None)
        bg.save(f'{FIG}/{f}', dpi=(300, 300))
print('figures rebuilt and flattened to RGB')
