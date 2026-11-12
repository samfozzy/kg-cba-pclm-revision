import os, json, pickle, random
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as L, Model, callbacks, optimizers

SEED=42
def set_seed(s=SEED):
    np.random.seed(s); random.seed(s); tf.random.set_seed(s)
set_seed()

# ---------------- Data ----------------
df = pd.read_csv('panel_clean.csv').sort_values(['iso_code','year'])
covs = ['energy_per_capita','gdp_per_capita','log_population','coal_share','oil_share','gas_share']
df = df.dropna(subset=covs+['co2_per_capita']).copy()
df['log_epc'] = np.log10(df.energy_per_capita.clip(lower=1))
df['log_gpc'] = np.log10(df.gdp_per_capita.clip(lower=100))
FEATS = ['log_epc','log_gpc','log_population','coal_share','oil_share','gas_share']
df['target'] = np.log(df.co2_per_capita)

W=8
X,y,meta=[],[],[]
for iso,g in df.groupby('iso_code'):
    g=g.sort_values('year').reset_index(drop=True)
    for i in range(W-1,len(g)):
        X.append(g[FEATS].iloc[i-W+1:i+1].values)
        y.append(g['target'].iloc[i])
        meta.append((iso,int(g.year.iloc[i]),float(g.co2_per_capita.iloc[i]),str(g.country.iloc[i])))
X=np.array(X,np.float32); y=np.array(y,np.float32)
meta=pd.DataFrame(meta,columns=['iso','year','y_raw','country'])
print('LOCO dataset:',X.shape,'| countries:',meta.iso.nunique())

from sklearn.model_selection import GroupKFold
gkf=GroupKFold(n_splits=5)
folds=list(gkf.split(X,y,meta.iso.values))

# ---------------- Models ----------------
class TemporalAttention(L.Layer):
    """Additive (Bahdanau-style) attention pooling over time steps."""
    def build(self, shp):
        d=shp[-1]
        self.W1=self.add_weight(name='W1',shape=(d,d),initializer='glorot_uniform',trainable=True)
        self.b =self.add_weight(name='b',shape=(d,),initializer='zeros',trainable=True)
        self.u =self.add_weight(name='u',shape=(d,1),initializer='glorot_uniform',trainable=True)
    def call(self,h):                      # h: (B,T,D)
        s=tf.tanh(tf.matmul(h,self.W1)+self.b)      # (B,T,D)
        a=tf.nn.softmax(tf.matmul(s,self.u),axis=1) # (B,T,1)
        return tf.reduce_sum(a*h,axis=1), a

def build_model(kind, hp=None):
    hp = hp or {}
    f  = int(hp.get('filters',48)); k=int(hp.get('kernel',3))
    u  = int(hp.get('units',48));   dr=float(hp.get('dropout',0.25))
    lr = float(hp.get('lr',1e-3))
    inp=L.Input((W,len(FEATS)))
    if kind=='LSTM':
        h=L.LSTM(u,return_sequences=False)(inp); h=L.Dropout(dr)(h)
    elif kind=='GRU':
        h=L.GRU(u,return_sequences=False)(inp); h=L.Dropout(dr)(h)
    elif kind=='CNN-LSTM':
        c=L.Conv1D(f,k,padding='same',activation='relu')(inp)
        c=L.Conv1D(f,k,padding='same',activation='relu')(c)
        h=L.LSTM(u,return_sequences=False)(c); h=L.Dropout(dr)(h)
    elif kind=='CNN-BiLSTM-Attn':
        c=L.Conv1D(f,k,padding='same',activation='relu')(inp)
        c=L.Conv1D(f,k,padding='same',activation='relu')(c)
        s=L.Bidirectional(L.LSTM(u,return_sequences=True))(c)
        s=L.Dropout(dr)(s)
        h,att=TemporalAttention()(s)
    else: raise ValueError(kind)
    h=L.Dense(32,activation='relu')(h); h=L.Dropout(dr)(h)
    out=L.Dense(1)(h)
    m=Model(inp,out)
    m.compile(optimizers.Adam(lr),loss='mse')
    return m

def train_eval(kind, hp=None, tag=None, save_dir=None):
    tag = tag or kind
    oof=np.full(len(y),np.nan); hist={}
    for fi,(trn,tst) in enumerate(folds):
        set_seed(SEED+fi)
        # inner split by country for early stopping
        tr_iso=meta.iloc[trn].iso.unique(); rng=np.random.RandomState(fi)
        val_iso=set(rng.choice(tr_iso,size=max(1,len(tr_iso)//7),replace=False))
        inner_val=meta.iloc[trn].iso.isin(val_iso).values
        tr_idx=trn[~inner_val]; va_idx=trn[inner_val]
        mu=X[tr_idx].mean((0,1)); sd=X[tr_idx].std((0,1))+1e-8
        Z=lambda A:(A-mu)/sd
        m=build_model(kind,hp)
        cb=[callbacks.EarlyStopping(patience=9,restore_best_weights=True,monitor='val_loss')]
        h=m.fit(Z(X[tr_idx]),y[tr_idx],validation_data=(Z(X[va_idx]),y[va_idx]),
                epochs=90,batch_size=96,verbose=0,callbacks=cb)
        oof[tst]=m.predict(Z(X[tst]),verbose=0).ravel()
        hist[fi]=h.history
        if save_dir:
            os.makedirs(save_dir,exist_ok=True)
            m.save(f'{save_dir}/{tag}_fold{fi}.keras')
            np.savez(f'{save_dir}/{tag}_scaler_fold{fi}.npz',mu=mu,sd=sd)
    raw_p=np.exp(oof); raw_t=meta.y_raw.values
    e=raw_t-raw_p
    res=dict(RMSE=float(np.sqrt(np.mean(e**2))),MAE=float(np.mean(np.abs(e))),
             MAPE=float(np.mean(np.abs(e/raw_t))*100),MdAPE=float(np.median(np.abs(e/raw_t))*100),
             R2=float(1-np.sum(e**2)/np.sum((raw_t-raw_t.mean())**2)))
    return res,oof,hist

if __name__=='__main__':
    all_res={}; all_oof={}; all_hist={}
    for kind in ['LSTM','GRU','CNN-LSTM','CNN-BiLSTM-Attn']:
        res,oof,hist=train_eval(kind,save_dir='models' if kind=='CNN-BiLSTM-Attn' else None)
        all_res[kind]=res; all_oof[kind]=oof; all_hist[kind]=hist
        print(f'{kind:18s}',{k:round(v,3) for k,v in res.items()},flush=True)
    np.savez('deep_oof.npz',**all_oof)
    pickle.dump(dict(res=all_res,hist=all_hist,meta=meta,folds=folds,X=X,y=y,FEATS=FEATS),open('deep_loco.pkl','wb'))
    json.dump(all_res,open('results_deep.json','w'),indent=1)
