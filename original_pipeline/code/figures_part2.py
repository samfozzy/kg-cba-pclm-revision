"""
LEGACY - not part of the run order in REPRODUCE.md. Kept as the record of the first submission.
Its inputs are read from the standard folders (paths.py) and its outputs are written to results/legacy/,
so running it never overwrites the outputs of the run order.
Superseded by rebuild_figures.py (step 10) and conformal.py (step 8). Its figures go to results/legacy/.
It needs results/legacy/analysis.json from analysis.py.
"""
import os, json, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import paths as P
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':110,'savefig.dpi':300,'savefig.bbox':'tight'})
C=['#0173B2','#DE8F05','#029E73','#D55E00','#CC78BC','#CA9161','#949494','#56B4E9']
FIGL=P.legacy('figures'); os.makedirs(FIGL,exist_ok=True)

from deep_loco import meta as metaA, X as XA, y as yA
oA={k:np.load(P.results('oof_classical.npz'))[k] for k in np.load(P.results('oof_classical.npz')).files}
for k in ['LSTM','GRU','CNN-LSTM','CNN-BiLSTM-Attn']: oA[k]=np.load(P.oof(f'oof_{k}.npz'))['oof']
oA['KG-CBA']=np.load(P.oof('oof_KG-CBA.npz'))['oof']; oA['PSO-KG-CBA']=np.load(P.oof('oof_PSO-KG-CBA.npz'))['oof']
oA['Ensemble']=(oA['Ridge']+oA['GRU']+oA['PSO-KG-CBA'])/3
tA=metaA.y_raw.values

# F9 - pred vs actual (panel A)
fig,axs=plt.subplots(1,4,figsize=(11,2.9),sharey=True)
for ax,name in zip(axs,['Ridge','GRU','PSO-KG-CBA','Ensemble']):
    p=np.exp(oA[name])
    ax.scatter(tA,p,s=5,alpha=.35,color=C[0],edgecolors='none')
    lim=[.03,80]; ax.plot(lim,lim,'k--',lw=.8)
    ax.set(xscale='log',yscale='log',xlim=lim,ylim=lim,xlabel='Observed (t/person)',
           title=f"{name}\nR$^2$={1-np.sum((tA-p)**2)/np.sum((tA-tA.mean())**2):.3f}")
axs[0].set_ylabel('Predicted (t/person)')
plt.tight_layout(); plt.savefig(FIGL+'/F09_pred_vs_actual.png'); plt.close()

# F10 - Pareto of SSE by country (Ensemble, panel A)
e2=(tA-np.exp(oA['Ensemble']))**2
sse=pd.DataFrame({'iso':metaA.iso,'e2':e2}).groupby('iso').e2.sum().sort_values(ascending=False)
top=sse.head(15); cum=100*sse.cumsum()/sse.sum()
fig,ax=plt.subplots(figsize=(7,3.2))
ax.bar(range(15),top.values,color=[C[3] if i in ['SGP','ISL','QAT','NOR','KWT','LUX'] else C[0] for i in top.index])
ax.set_xticks(range(15),top.index,rotation=45); ax.set_ylabel('Sum of squared errors')
ax2=ax.twinx(); ax2.plot(range(15),cum.iloc[:15].values,color=C[1],marker='o',ms=3.5,lw=1.4)
ax2.set_ylabel('Cumulative share of total SSE (%)',color=C[1]); ax2.set_ylim(0,100); ax2.spines['right'].set_visible(True)
ax.set_title('Error concentration: structurally atypical energy systems dominate (red = identified outliers)')
plt.savefig(FIGL+'/F10_pareto.png'); plt.close()

# F11 - covariate enrichment effect (A vs C)
resA=json.load(open(P.legacy('analysis.json')))['panelA']; resC=json.load(open(P.legacy('analysis.json')))['panelC']
# recompute panel-A metrics restricted to the 76 enriched countries for a fair comparison
metaE=pd.read_csv(P.data('meta_enriched.csv')); iso76=set(metaE.iso)
maskA76=metaA.iso.isin(iso76).values
def summ(t,p_log):
    p=np.exp(p_log); e=t-p
    return dict(MAPE=np.mean(np.abs(e/t))*100, MdAPE=np.median(np.abs(e/t))*100,
                R2=1-np.sum(e**2)/np.sum((t-t.mean())**2))
A76={m:summ(tA[maskA76],oA[m][maskA76]) for m in ['Ridge','PSO-KG-CBA','Ensemble']}
oE={'Ridge':np.load(P.oof('oofE_Ridge.npz'))['oof'],'PSO-KG-CBA':np.load(P.oof('oofE_PSO-KG-CBA.npz'))['oof']}
oE['Ensemble']=(oE['Ridge']+oE['PSO-KG-CBA'])/2
tE=metaE.y_raw.values
C76={m:summ(tE,oE[m]) for m in oE}
fig,ax=plt.subplots(1,2,figsize=(8.4,3.1))
ms=['Ridge','PSO-KG-CBA','Ensemble']; xp=np.arange(3)
for i,(met,ttl) in enumerate([('MAPE','(a) MAPE (%)'),('MdAPE','(b) MdAPE (%)')]):
    ax[i].bar(xp-.2,[A76[m][met] for m in ms],.4,color=C[6],label='6 covariates')
    ax[i].bar(xp+.2,[C76[m][met] for m in ms],.4,color=C[2],label='+ low-carbon share')
    ax[i].set_xticks(xp,ms); ax[i].set_title(ttl)
ax[0].legend(frameon=False,fontsize=8)
fig.suptitle('Effect of energy-mix covariate enrichment (identical 76 countries)',y=1.03,fontsize=10)
plt.tight_layout(); plt.savefig(FIGL+'/F11_enrichment.png'); plt.close()
json.dump(dict(A76=A76,C76=C76),open(P.legacy('enrichment_effect.json'),'w'),indent=1,default=float)

# ---- Per-fold split-conformal (proper) for E-PSO-KG-CBA ----
from conformal import TemporalAttention, load_fold_model   # importing train_enriched would train the networks
dE=np.load(P.data('enriched.npz')); XE,yE=dE['X'],dE['y']
folds_E=pickle.load(open(P.data('folds_enriched.pkl'),'rb'))
mc_mean=np.full(len(yE),np.nan); mc_std=np.full(len(yE),np.nan); q_by_fold={}
T=60
lo=np.full(len(yE),np.nan); hi=np.full(len(yE),np.nan)
for fi,(trn,tst) in enumerate(folds_E):
    m=load_fold_model(P.oof(f'E-PSO-KG-CBA_fold{fi}.keras'), XE.shape[2])
    s=np.load(P.oof(f'E-PSO-KG-CBA_scaler_fold{fi}.npz'))
    # reconstruct this fold's inner validation countries (same RNG as training)
    tr_iso=metaE.iloc[trn].iso.unique(); rng=np.random.RandomState(fi)
    val_iso=set(rng.choice(tr_iso,size=max(1,len(tr_iso)//7),replace=False))
    iv=metaE.iloc[trn].iso.isin(val_iso).values; VA=trn[iv]
    Zva=(XE[VA]-s['mu'])/s['sd']; Zte=(XE[tst]-s['mu'])/s['sd']
    sims_va=np.stack([m(Zva,training=True).numpy().ravel() for _ in range(T)])
    sims_te=np.stack([m(Zte,training=True).numpy().ravel() for _ in range(T)])
    mu_va,sd_va=sims_va.mean(0),sims_va.std(0)+1e-6
    mu_te,sd_te=sims_te.mean(0),sims_te.std(0)+1e-6
    scores=np.abs(yE[VA]-mu_va)/sd_va
    n=len(scores); q=float(np.quantile(scores, min(1,np.ceil(0.9*(n+1))/n)))
    q_by_fold[fi]=q
    mc_mean[tst],mc_std[tst]=mu_te,sd_te
    lo[tst],hi[tst]=np.exp(mu_te-q*sd_te),np.exp(mu_te+q*sd_te)
picp=float(np.mean((tE>=lo)&(tE<=hi))); mpiw=float(np.mean(hi-lo))
unc=dict(T=T,q_by_fold=q_by_fold,picp=picp,mpiw=mpiw)
json.dump(unc,open(P.legacy('conformal.json'),'w'),indent=1)
np.savez(P.legacy('conformal.npz'),mean=mc_mean,std=mc_std,lo=lo,hi=hi)
print('Split-conformal: PICP=%.3f MPIW=%.2f q=%s'%(picp,mpiw,{k:round(v,2) for k,v in q_by_fold.items()}))

# F12 - case studies with conformal bands
cases=['GHA','ZAF','DZA','EGY']
fig,axs=plt.subplots(1,4,figsize=(11,2.8))
for ax,iso in zip(axs,cases):
    idx=metaE.index[metaE.iso==iso]
    yr=metaE.loc[idx,'year']; obs=metaE.loc[idx,'y_raw']
    ax.plot(yr,obs,'k-',lw=1.5,label='Observed')
    ax.plot(yr,np.exp(mc_mean[idx]),color=C[0],lw=1.4,label='E-PSO-KG-CBA')
    ax.fill_between(yr,lo[idx],hi[idx],color=C[0],alpha=.2,label='90% conformal PI')
    ax.set(title=iso,xlabel='Year')
axs[0].set_ylabel('t CO$_2$ person$^{-1}$')
axs[0].legend(fontsize=6.5,frameon=False)
fig.suptitle('African case studies: leave-country-out estimation with conformal intervals',y=1.04,fontsize=10)
plt.tight_layout(); plt.savefig(FIGL+'/F12_africa_cases.png'); plt.close()

# F13 - SHAP aggregation
an=json.load(open(P.legacy('analysis.json')))
sf=an['shap_by_feature']; sl=an['shap_by_lag']
nice={'log_epc':'log energy p.c.','log_gpc':'log GDP p.c.','log_population':'log population',
      'coal_share':'coal share','oil_share':'oil share','gas_share':'gas share','low_carbon_share':'low-carbon share'}
fig,ax=plt.subplots(1,2,figsize=(8.6,3))
ks=sorted(sf,key=sf.get)
ax[0].barh([nice[k] for k in ks],[sf[k] for k in ks],color=C[2])
ax[0].set_title('(a) Mean |SHAP| by covariate')
lags=[f't-{i}' for i in range(7,-1,-1)]; lags[-1]='t'
vals=[sl[k] for k in ['t-7','t-6','t-5','t-4','t-3','t-2','t-1','t-0']]
ax[1].plot(range(8),vals,marker='o',color=C[3])
ax[1].set_xticks(range(8),lags); ax[1].set_title('(b) Mean |SHAP| by lag'); ax[1].set_xlabel('Window position')
plt.tight_layout(); plt.savefig(FIGL+'/F13_shap.png'); plt.close()

# F14 - attention weights
att=an['attention_mean_by_lag']
fig,ax=plt.subplots(figsize=(4.6,2.8))
ax.bar(range(8),att,color=C[4]); ax.axhline(1/8,ls=':',color='gray',lw=1)
ax.set_xticks(range(8),lags); ax.set(ylabel='Mean attention weight',title='Temporal attention distribution (E-PSO-KG-CBA)')
ax.text(7.1,1/8+.002,'uniform',fontsize=7,color='gray')
plt.savefig(FIGL+'/F14_attention.png'); plt.close()

# F15 - temporal benchmark (Experiment B)
tb={1:{'Persist':(0.317,6.43),'Drift':(0.38,7.78),'SVR':(0.324,6.82),'LGBM':(0.459,9.44)},
    3:{'Persist':(0.531,10.24),'Drift':(0.68,13.84),'SVR':(0.681,12.83),'LGBM':(1.694,31.38)},
    5:{'Persist':(0.738,14.35),'Drift':(0.933,20.39),'SVR':(1.22,21.52),'LGBM':(3.206,56.15)}}
fig,ax=plt.subplots(figsize=(6.4,3.2))
w=.2; ms=['Persist','Drift','SVR','LGBM']
for i,m in enumerate(ms):
    ax.bar(np.arange(3)+(i-1.5)*w,[tb[h][m][0] for h in [1,3,5]],w,color=C[i],label=m)
ax.set_xticks(range(3),['h = 1','h = 3','h = 5']); ax.set_ylabel('Test MAE (t CO$_2$ person$^{-1}$)')
ax.set_title('Experiment B: temporal forecasting - naive persistence dominates all horizons (2021-2023)')
ax.legend(frameon=False)
plt.savefig(FIGL+'/F15_temporal.png'); plt.close()

# ---- master tables ----
rows=[]
for m,v in an['panelA'].items(): rows.append(dict(Experiment='A (112, 6 cov.)',Model=m,**v))
for m,v in an['panelC'].items(): rows.append(dict(Experiment='C (76, 7 cov.)',Model=m,**v))
pd.DataFrame(rows).round(3).to_csv(P.legacy('table_master_results.csv'),index=False)
wil=pd.DataFrame(an['wilcoxon']).T.round(4); wil.to_csv(P.legacy('table_wilcoxon.csv'))
print(wil)
print('figures:',len(os.listdir('figures')))
