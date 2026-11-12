import os, json, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import paths as P

plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':110,'savefig.dpi':300,'savefig.bbox':'tight'})
C=['#0173B2','#DE8F05','#029E73','#D55E00','#CC78BC','#CA9161','#949494','#ECE133','#56B4E9','#7A4FA3','#117733','#882255']
df=pd.read_csv(P.data('panel_clean.csv'))

# F1 - global trends
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
gm=df.groupby('year').co2_per_capita.mean()
ax[0].plot(gm.index,gm.values,color=C[0],lw=2)
ax[0].fill_between(gm.index, df.groupby('year').co2_per_capita.quantile(.25),
                   df.groupby('year').co2_per_capita.quantile(.75), color=C[0], alpha=.18)
ax[0].set(title='(a) Panel mean per-capita CO$_2$ (IQR shaded)',xlabel='Year',ylabel='t CO$_2$ person$^{-1}$ yr$^{-1}$')
for i,iso in enumerate(['USA','CHN','IND','DEU','GHA','QAT']):
    g=df[df.iso_code==iso]
    ax[1].plot(g.year,g.co2_per_capita,label=iso,color=C[i],lw=1.6)
ax[1].set(title='(b) Selected national trajectories',xlabel='Year',yscale='log')
ax[1].legend(ncol=3,fontsize=7.5,frameon=False)
plt.tight_layout(); plt.savefig(P.FIGURES+'/F01_trends.png'); plt.close()

# F2 - correlation heatmap (log-space covariates, panel A)
d2=df.copy()
d2['log_epc']=np.log10(d2.energy_per_capita.clip(lower=1)); d2['log_gpc']=np.log10(d2.gdp_per_capita.clip(lower=100))
d2['log_co2pc']=np.log(d2.co2_per_capita)
cols=['log_co2pc','log_epc','log_gpc','log_population','coal_share','oil_share','gas_share']
labs=['ln CO$_2$pc','log EPC','log GDPpc','log Pop','Coal sh.','Oil sh.','Gas sh.']
R=d2[cols].corr().values
fig,ax=plt.subplots(figsize=(4.6,4))
im=ax.imshow(R,vmin=-1,vmax=1,cmap='RdBu_r')
ax.set_xticks(range(len(cols)),labs,rotation=45,ha='right'); ax.set_yticks(range(len(cols)),labs)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j,i,f'{R[i,j]:.2f}',ha='center',va='center',fontsize=7,
                color='white' if abs(R[i,j])>.6 else 'black')
plt.colorbar(im,shrink=.8); ax.set_title('Pearson correlations (n = 3,808 country-years)')
plt.savefig(P.FIGURES+'/F02_corr.png'); plt.close()

# F3 - distribution by continent
syn=pd.read_csv(P.CONTINENTS).rename(columns={'continent':'Continent'})
d3=df.merge(syn,left_on='iso_code',right_on='iso')
order=d3.groupby('Continent').co2_per_capita.median().sort_values().index
fig,ax=plt.subplots(figsize=(6,3.2))
bp=ax.boxplot([d3[d3.Continent==c].co2_per_capita for c in order],showfliers=False,patch_artist=True)
ax.set_xticks(range(1,len(order)+1)); ax.set_xticklabels(order)   # 'labels=' was removed in recent Matplotlib
for p,c in zip(bp['boxes'],C): p.set_facecolor(c); p.set_alpha(.6)
ax.set(yscale='log',ylabel='t CO$_2$ person$^{-1}$ yr$^{-1}$',title='Per-capita CO$_2$ by continent, 1990-2023')
plt.savefig(P.FIGURES+'/F03_continent_dist.png'); plt.close()

# F4 - methodology flowchart
fig,ax=plt.subplots(figsize=(9.2,4.6)); ax.axis('off')
def box(x,y,w,h,text,fc='#EAF2F8'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',fc=fc,ec='#34495E',lw=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=7.6)
def arr(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,color='#34495E',lw=1.1))
box(.01,.72,.17,.2,'OWID CO$_2$ &\nenergy datasets\n(real, 1990-2023)')
box(.01,.42,.17,.2,'Cleaning & QC\n112 countries\ninterp. ≤3 yr gaps')
box(.01,.12,.17,.2,'Log transforms,\nfuel shares,\n8-yr windows')
box(.24,.60,.17,.24,'Experiment A\nLOCO 5-fold\n(country-disjoint)\n6 covariates')
box(.24,.28,.17,.24,'Experiment C\nenriched (+low-\ncarbon share)\n76 countries')
box(.24,.01,.17,.2,'Experiment B\ntemporal naive\nbenchmark h=1-5')
box(.455,.60,.19,.3,'Models (12)\nMean/Ridge/kNN/SVR/\nRF/XGB/LGBM/LSTM/\nGRU/CNN-LSTM/\nKG-CBA/PSO-KG-CBA')
box(.455,.24,.19,.22,'PSO tuning\n6 particles x 5 iters\ncountry-disjoint val.')
box(.685,.60,.14,.24,'Evaluation\nMAPE/MdAPE/\nRMSE/R$^2$ +\nWilcoxon')
box(.685,.26,.14,.24,'Uncertainty\nMC-dropout +\nsplit-conformal')
box(.865,.60,.125,.24,'XAI\nSHAP +\nattention')
box(.865,.26,.125,.24,'Case studies\nAfrica / Ghana')
arr(.095,.72,.095,.64); arr(.095,.42,.095,.34); arr(.18,.22,.24,.4)
arr(.18,.22,.24,.72); arr(.18,.22,.24,.11); arr(.41,.72,.455,.75)
arr(.41,.4,.455,.72); arr(.545,.60,.545,.48); arr(.645,.75,.685,.73)
arr(.645,.75,.685,.4); arr(.825,.72,.865,.72); arr(.825,.38,.865,.38)
ax.set_title('Experimental framework overview',fontsize=10)
plt.savefig(P.FIGURES+'/F04_flowchart.png'); plt.close()

# F5 - KG-CBA architecture schematic
fig,ax=plt.subplots(figsize=(9.2,3.6)); ax.axis('off')
box(.01,.38,.13,.26,'Input window\n8 yr x F covariates')
box(.18,.62,.15,.24,'Kaya linear head\nDense(1)\n(Ridge warm start)',fc='#FDEBD0')
box(.18,.08,.15,.24,'Conv1D x2\n(37 filters, k=4)')
box(.37,.08,.15,.24,'BiLSTM (40)\n+ dropout 0.23')
box(.56,.08,.16,.24,'Additive temporal\nattention pooling')
box(.76,.08,.13,.24,'Dense(32)+drop\n-> residual head')
box(.76,.62,.13,.24,'$\\Sigma$  ln CO$_2$pc',fc='#D5F5E3')
arr(.14,.55,.18,.72); arr(.14,.45,.18,.22); arr(.33,.2,.37,.2); arr(.52,.2,.56,.2)
arr(.72,.2,.76,.2); arr(.825,.32,.825,.62); arr(.33,.74,.76,.74)
ax.text(.5,.95,'Kaya-guided CNN-BiLSTM-Attention (KG-CBA): physics-informed linear backbone + deep residual branch',
        ha='center',fontsize=9.5)
plt.savefig(P.FIGURES+'/F05_architecture.png'); plt.close()

# F6 - PSO convergence
st=json.load(open(P.PSO_STATE))
fs=[l['f'] for l in st['log']]
best=np.minimum.accumulate(fs)
fig,ax=plt.subplots(figsize=(5,3))
ax.plot(range(1,len(fs)+1),fs,'o',ms=4,color=C[1],alpha=.6,label='particle evaluation')
ax.step(range(1,len(fs)+1),best,where='post',color=C[0],lw=2,label='global best')
ax.set(xlabel='Fitness evaluation',ylabel='Validation MSE (log-space)',title='PSO convergence (6 particles x 5 iterations)')
ax.legend(frameon=False)
plt.savefig(P.FIGURES+'/F06_pso.png'); plt.close()

# F7 and F8 of the first submission need the outputs of the legacy scripts run_arch.py (hist_<model>.npz) and
# analysis.py (analysis.json). rebuild_figures.py (step 10) rebuilds both figures from the outputs of the run
# order, so here they are drawn only when those legacy outputs exist.
if all(os.path.exists(P.legacy(f'hist_{k}.npz')) for k in ['LSTM','GRU','CNN-LSTM','CNN-BiLSTM-Attn']) \
        and os.path.exists(P.legacy('analysis.json')):
    # F7 - training curves
    fig,ax=plt.subplots(1,2,figsize=(8.6,3))
    for i,k in enumerate(['LSTM','GRU','CNN-LSTM','CNN-BiLSTM-Attn']):
        h=np.load(P.legacy(f'hist_{k}.npz'))
        ax[0].plot(h['f0_loss'],color=C[i],lw=1.4,label=k)
        ax[1].plot(h['f0_val_loss'],color=C[i],lw=1.4,label=k)
    ax[0].set(title='(a) Training loss (fold 1)',xlabel='Epoch',ylabel='MSE',yscale='log')
    ax[1].set(title='(b) Validation loss (fold 1)',xlabel='Epoch',yscale='log')
    ax[1].legend(frameon=False,fontsize=8)
    plt.tight_layout(); plt.savefig(P.FIGURES+'/F07_training.png'); plt.close()

    # F8 - model comparison bars (panel A)
    res=json.load(open(P.legacy('analysis.json')))['panelA']
    order=['Mean','kNN','SVR','RF','XGBoost','LightGBM','LSTM','CNN-LSTM','CNN-BiLSTM-Attn','GRU','KG-CBA','PSO-KG-CBA','Ridge','Ensemble']
    mape=[res[m]['MAPE'] for m in order]; mdape=[res[m]['MdAPE'] for m in order]
    fig,ax=plt.subplots(figsize=(9,3.4))
    xp=np.arange(len(order))
    ax.bar(xp-.2,mape,.4,color=C[0],label='MAPE')
    ax.bar(xp+.2,mdape,.4,color=C[1],label='MdAPE')
    ax.set_xticks(xp,order,rotation=38,ha='right')
    ax.set(ylabel='%',title='Experiment A: leave-country-out error by model (112 countries)')
    ax.set_ylim(0,60); ax.axhline(res['Ensemble']['MdAPE'],ls=':',color='gray',lw=1)
    for i,v in enumerate(mape):
        if v>60: ax.text(xp[i]-.2,57,f'{v:.0f}',ha='center',fontsize=7,rotation=90,color=C[0])
    ax.legend(frameon=False)
    plt.savefig(P.FIGURES+'/F08_model_comparison.png'); plt.close()
else:
    print('F07 and F08 skipped here; rebuild_figures.py rebuilds them')
print('figures part 1 done:', sorted(os.listdir(P.FIGURES)))
