import os, sys, json, time, pickle, random
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers as L, Model, callbacks, optimizers
from sklearn.linear_model import Ridge

SEED=42
def set_seed(s=SEED):
    np.random.seed(s); random.seed(s); tf.random.set_seed(s)

class TemporalAttention(L.Layer):
    def build(self, shp):
        d=shp[-1]
        self.W1=self.add_weight(name='W1',shape=(d,d),initializer='glorot_uniform',trainable=True)
        self.b =self.add_weight(name='b',shape=(d,),initializer='zeros',trainable=True)
        self.u =self.add_weight(name='u',shape=(d,1),initializer='glorot_uniform',trainable=True)
    def call(self,h):
        s=tf.tanh(tf.matmul(h,self.W1)+self.b)
        a=tf.nn.softmax(tf.matmul(s,self.u),axis=1)
        return tf.reduce_sum(a*h,axis=1), a

d=np.load('enriched.npz'); X,y=d['X'],d['y']
meta=pd.read_csv('meta_enriched.csv')
folds=pickle.load(open('folds_enriched.pkl','rb'))
HP=json.load(open('pso_state.json'))
from_hp=np.array(HP['gbest'])
BATCHES=[48,64,96,128]
HP=dict(filters=int(round(from_hp[0])),kernel=int(round(from_hp[1])),units=int(round(from_hp[2])),
        dropout=float(from_hp[3]),lr=float(10**from_hp[4]),batch=BATCHES[int(round(from_hp[5]))])
print('hp:',HP)
NF=X.shape[2]

def build_kg(rw,rb):
    set_seed(123)
    inp=L.Input((8,NF))
    last=L.Lambda(lambda t: t[:,-1,:])(inp)
    lin=L.Dense(1,name='kaya_linear')(last)
    c=L.Conv1D(HP['filters'],HP['kernel'],padding='same',activation='relu')(inp)
    c=L.Conv1D(HP['filters'],HP['kernel'],padding='same',activation='relu')(c)
    s=L.Bidirectional(L.LSTM(HP['units'],return_sequences=True))(c)
    s=L.Dropout(HP['dropout'])(s)
    h,att=TemporalAttention()(s)
    h=L.Dense(32,activation='relu')(h); h=L.Dropout(HP['dropout'])(h)
    resid=L.Dense(1,name='deep_residual',kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3))(h)
    out=L.Add()([lin,resid])
    m=Model(inp,out)
    m.get_layer('kaya_linear').set_weights([rw,rb])
    m.compile(optimizers.Adam(HP['lr']),loss='mse')
    return m

os.makedirs('models',exist_ok=True); t0=time.time()
for fi,(trn,tst) in enumerate(folds):
    mp=f'models/E-PSO-KG-CBA_fold{fi}.keras'
    if os.path.exists(mp): continue
    if time.time()-t0>190: print('PAUSE at fold',fi); sys.exit(0)
    set_seed(42+fi)
    tr_iso=meta.iloc[trn].iso.unique(); rng=np.random.RandomState(fi)
    val_iso=set(rng.choice(tr_iso,size=max(1,len(tr_iso)//7),replace=False))
    iv=meta.iloc[trn].iso.isin(val_iso).values
    TR,VA=trn[~iv],trn[iv]
    mu=X[TR].mean((0,1)); sd=X[TR].std((0,1))+1e-8; Z=lambda A:(A-mu)/sd
    r=Ridge(1.0).fit(Z(X[TR])[:,-1,:],y[TR])
    m=build_kg(r.coef_.astype('float32').reshape(-1,1),np.array([r.intercept_],'float32'))
    m.fit(Z(X[TR]),y[TR],validation_data=(Z(X[VA]),y[VA]),epochs=90,batch_size=HP['batch'],verbose=0,
          callbacks=[callbacks.EarlyStopping(patience=9,restore_best_weights=True)])
    m.save(mp); np.savez(f'models/E-PSO-KG-CBA_scaler_fold{fi}.npz',mu=mu,sd=sd)
    print('fold',fi,'saved',round(time.time()-t0,1),'s')

if all(os.path.exists(f'models/E-PSO-KG-CBA_fold{fi}.keras') for fi in range(5)):
    oof=np.full(len(y),np.nan)
    for fi,(trn,tst) in enumerate(folds):
        m=tf.keras.models.load_model(f'models/E-PSO-KG-CBA_fold{fi}.keras',
            custom_objects={'TemporalAttention':TemporalAttention},safe_mode=False)
        s=np.load(f'models/E-PSO-KG-CBA_scaler_fold{fi}.npz')
        oof[tst]=m.predict((X[tst]-s['mu'])/s['sd'],verbose=0).ravel()
    p=np.exp(oof); t=meta.y_raw.values; e=t-p
    res=dict(RMSE=float(np.sqrt(np.mean(e**2))),MAE=float(np.mean(np.abs(e))),
        MAPE=float(np.mean(np.abs(e/t))*100),MdAPE=float(np.median(np.abs(e/t))*100),
        R2=float(1-np.sum(e**2)/np.sum((t-t.mean())**2)))
    np.savez('oofE_PSO-KG-CBA.npz',oof=oof)
    json.dump({'E-PSO-KG-CBA':res},open('res_E-PSO-KG-CBA.json','w'),indent=1)
    print('E-PSO-KG-CBA',{k:round(v,3) for k,v in res.items()})
