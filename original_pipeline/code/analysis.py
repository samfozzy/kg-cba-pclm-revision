"""
LEGACY - not part of the run order in REPRODUCE.md. Kept as the record of the first submission.
Its inputs are read from the standard folders (paths.py) and its outputs are written to results/legacy/,
so running it never overwrites the outputs of the run order.
Superseded by clustered_stats.py (step 7), conformal.py (step 8) and rebuild_figures.py (step 10).
Two known problems are kept as they were run: the Wilcoxon tests are observation-level (see clustered_stats.py),
and the SHAP values are reshaped to (N, F, W) instead of (N, W, F) (REPRODUCE.md, note 2).
"""
import os, json, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import paths as P
import numpy as np, pandas as pd, tensorflow as tf
from scipy.stats import wilcoxon

out = {}

# ---------- Panel A (112 countries, 6 covariates) ----------
from deep_loco import meta as metaA, X as XA, y as yA, folds as foldsA
oA = {}
cz = np.load(P.results('oof_classical.npz'))
for k in cz.files: oA[k] = cz[k]
for k in ['LSTM','GRU','CNN-LSTM','CNN-BiLSTM-Attn']: oA[k] = np.load(P.oof(f'oof_{k}.npz'))['oof']
oA['KG-CBA'] = np.load(P.oof('oof_KG-CBA.npz'))['oof']
oA['PSO-KG-CBA'] = np.load(P.oof('oof_PSO-KG-CBA.npz'))['oof']
oA['Ensemble'] = (oA['Ridge']+oA['GRU']+oA['PSO-KG-CBA'])/3.0
tA = metaA.y_raw.values

def summarize(t, p_log):
    p = np.exp(p_log); e = t-p
    return dict(RMSE=float(np.sqrt(np.mean(e**2))), MAE=float(np.mean(np.abs(e))),
        MAPE=float(np.mean(np.abs(e/t))*100), MdAPE=float(np.median(np.abs(e/t))*100),
        R2=float(1-np.sum(e**2)/np.sum((t-t.mean())**2)))

resA = {k: summarize(tA, v) for k,v in oA.items()}
out['panelA'] = resA

# ---------- Panel C (76 countries, 7 covariates) ----------
metaE = pd.read_csv(P.data('meta_enriched.csv')); dE=np.load(P.data('enriched.npz')); XE,yE=dE['X'],dE['y']
oE = {'Ridge': np.load(P.oof('oofE_Ridge.npz'))['oof'], 'LightGBM': np.load(P.oof('oofE_LightGBM.npz'))['oof'],
      'PSO-KG-CBA': np.load(P.oof('oofE_PSO-KG-CBA.npz'))['oof']}
oE['Ensemble'] = (oE['Ridge']+oE['PSO-KG-CBA'])/2.0
oE['Ensemble3'] = (oE['Ridge']+oE['LightGBM']+oE['PSO-KG-CBA'])/3.0
tE = metaE.y_raw.values
resE = {k: summarize(tE, v) for k,v in oE.items()}
out['panelC'] = resE
for k,v in resE.items(): print('C:',k, {a:round(b,3) for a,b in v.items()})

# ---------- Significance: Wilcoxon signed-rank on |APE| ----------
def ape(t,p_log): return np.abs((t-np.exp(p_log))/t)
sig = {}
pairs = [('Ensemble','Ridge','A'),('Ensemble','GRU','A'),('Ensemble','PSO-KG-CBA','A'),
         ('PSO-KG-CBA','CNN-BiLSTM-Attn','A'),('Ridge','LightGBM','A'),
         ('Ensemble','Ridge','C'),('PSO-KG-CBA','LightGBM','C'),('Ensemble','PSO-KG-CBA','C')]
for a,b,panel in pairs:
    t,o = (tA,oA) if panel=='A' else (tE,oE)
    s,pv = wilcoxon(ape(t,o[a]), ape(t,o[b]))
    sig[f'{panel}:{a}_vs_{b}'] = dict(W=float(s), p=float(pv),
        medA=float(np.median(ape(t,o[a]))*100), medB=float(np.median(ape(t,o[b]))*100))
out['wilcoxon'] = sig

# ---------- Per-continent / Africa & Ghana ----------
cont = pd.read_csv(os.path.join(P.RAW, 'owid-co2-data.csv'))[['iso_code','country']].drop_duplicates()
# continent of each country (the file used in the first submission is not in the archive; code/continents.csv
# holds the same assignment)
syn = pd.read_csv(P.CONTINENTS).rename(columns={'continent':'Continent'})
mA = metaA.merge(syn, on='iso', how='left')
best='Ensemble'
mA['ape'] = ape(tA,oA[best])*100
by_cont = mA.groupby('Continent')['ape'].agg(['median','mean','count']).round(2)
out['continent_ape_ensemble'] = by_cont.to_dict()
print('\nBy continent (Ensemble APE):\n', by_cont)
gha = mA[mA.iso=='GHA']
out['ghana'] = dict(n=int(len(gha)), median_ape=float(gha.ape.median()), mean_ape=float(gha.ape.mean()))
print('Ghana median APE %.2f%%' % gha.ape.median())

# ---------- MC-dropout uncertainty on E-PSO-KG-CBA ----------
from conformal import TemporalAttention, load_fold_model   # importing train_enriched would train the networks
folds_E = pickle.load(open(P.data('folds_enriched.pkl'),'rb'))
mc_mean=np.full(len(yE),np.nan); mc_std=np.full(len(yE),np.nan)
T=60
for fi,(trn,tst) in enumerate(folds_E):
    m=load_fold_model(P.oof(f'E-PSO-KG-CBA_fold{fi}.keras'), XE.shape[2])
    s=np.load(P.oof(f'E-PSO-KG-CBA_scaler_fold{fi}.npz'))
    Z=(XE[tst]-s['mu'])/s['sd']
    sims=np.stack([m(Z,training=True).numpy().ravel() for _ in range(T)])
    mc_mean[tst]=sims.mean(0); mc_std[tst]=sims.std(0)
z90=1.645
lo,hi=np.exp(mc_mean-z90*mc_std), np.exp(mc_mean+z90*mc_std)
picp=float(np.mean((tE>=lo)&(tE<=hi))); mpiw=float(np.mean(hi-lo))
# calibrated interval via inflation factor on log-residual std
resid=np.abs(np.log(tE)-mc_mean); infl=float(np.quantile(resid/np.clip(mc_std,1e-6,None),0.90))
lo2,hi2=np.exp(mc_mean-infl*mc_std), np.exp(mc_mean+infl*mc_std)
out['uncertainty']=dict(T=T, picp_raw=picp, mpiw_raw=mpiw, inflation=infl,
    picp_cal=float(np.mean((tE>=lo2)&(tE<=hi2))), mpiw_cal=float(np.mean(hi2-lo2)))
np.savez(P.legacy('mc_dropout.npz'), mean=mc_mean, std=mc_std, lo=lo2, hi=hi2)
print('\nMC-dropout: raw PICP=%.3f | calibrated PICP=%.3f (infl=%.2f)'%(picp,out['uncertainty']['picp_cal'],infl))

# ---------- Attention weights (manual recompute, fold 0) ----------
m0=load_fold_model(P.oof('E-PSO-KG-CBA_fold0.keras'), XE.shape[2])
s0=np.load(P.oof('E-PSO-KG-CBA_scaler_fold0.npz'))
att_layer=[l for l in m0.layers if isinstance(l,TemporalAttention)][0]
# submodel up to dropout before attention
drop_out=None
for l in m0.layers:
    if 'bidirectional' in l.name: bi=l
sub=tf.keras.Model(m0.input, m0.get_layer([l.name for l in m0.layers if 'dropout' in l.name][0]).output)
H=sub.predict((XE-s0['mu'])/s0['sd'],verbose=0)   # (N,T,D)
W1,b,u=att_layer.get_weights()
sc=np.tanh(H@W1+b)@u                              # (N,T,1)
a=np.exp(sc)/np.exp(sc).sum(1,keepdims=True)
att_mean=a.mean(0).ravel()
out['attention_mean_by_lag']=[float(x) for x in att_mean]
print('Attention by lag (t-7..t):', np.round(att_mean,3))

# ---------- SHAP on enriched LightGBM (fold 0) ----------
import lightgbm as lgb, shap
from sklearn.preprocessing import StandardScaler
FE=['log_epc','log_gpc','log_population','coal_share','oil_share','gas_share','low_carbon_share']
trn,tst=folds_E[0]
Xf=XE.reshape(len(XE),-1)
scal=StandardScaler().fit(Xf[trn])
lg=lgb.LGBMRegressor(n_estimators=800,learning_rate=0.03,num_leaves=63,subsample=0.85,colsample_bytree=0.85,reg_lambda=1.5,random_state=42,verbose=-1).fit(scal.transform(Xf[trn]),yE[trn])
ex=shap.TreeExplainer(lg)
sv=ex.shap_values(scal.transform(Xf[tst]))
names=[f'{f}_t-{7-j}' if j<7 else f'{f}_t' for f in FE for j in range(8)]
# aggregate |SHAP| by covariate and by lag
sv_abs=np.abs(sv).reshape(len(sv),len(FE),8)
out['shap_by_feature']={FE[i]: float(sv_abs[:,i,:].mean()) for i in range(len(FE))}
out['shap_by_lag']={f't-{7-j}': float(sv_abs[:,:,j].mean()) for j in range(8)}
np.savez(P.legacy('shap_fold0.npz'), sv=sv, Xtst=scal.transform(Xf[tst]), tst=tst)
print('SHAP by feature:', {k:round(v,4) for k,v in out['shap_by_feature'].items()})

json.dump(out, open(P.legacy('analysis.json'),'w'), indent=1)
print('\nSaved results/legacy/analysis.json')
