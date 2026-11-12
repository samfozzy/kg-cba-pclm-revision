import os, sys, json, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
from tensorflow.keras import callbacks
from deep_loco import meta, X, y, folds, FEATS, set_seed
from pso import build_kg, decode
import tensorflow as tf

TAG = sys.argv[1]          # 'PSO-KG-CBA' or 'KG-CBA'
if TAG=='PSO-KG-CBA':
    st=json.load(open('pso_state.json')); HP=decode(np.array(st['gbest']))
else:
    HP=dict(filters=48,kernel=3,units=48,dropout=0.25,lr=1e-3,batch=96)
print(TAG,'hp:',HP)

os.makedirs('models',exist_ok=True)
t0=time.time()
for fi,(trn,tst) in enumerate(folds):
    mpath=f'models/{TAG}_fold{fi}.keras'
    if os.path.exists(mpath): continue
    if time.time()-t0>190: print('PAUSE at fold',fi); sys.exit(0)
    set_seed(42+fi)
    tr_iso=meta.iloc[trn].iso.unique(); rng=np.random.RandomState(fi)
    val_iso=set(rng.choice(tr_iso,size=max(1,len(tr_iso)//7),replace=False))
    iv=meta.iloc[trn].iso.isin(val_iso).values
    TR,VA=trn[~iv],trn[iv]
    mu=X[TR].mean((0,1)); sd=X[TR].std((0,1))+1e-8; Z=lambda A:(A-mu)/sd
    # refit ridge warm start on this fold's training data
    from sklearn.linear_model import Ridge
    import pso as _p
    _p.RW[:], _p.RB[:] = 0,0
    r=Ridge(1.0).fit(Z(X[TR])[:,-1,:], y[TR])
    _p.RW[:,0]=r.coef_.astype('float32'); _p.RB[0]=np.float32(r.intercept_)
    m=build_kg(HP)
    m.fit(Z(X[TR]),y[TR],validation_data=(Z(X[VA]),y[VA]),epochs=90,batch_size=HP['batch'],
          verbose=0,callbacks=[callbacks.EarlyStopping(patience=9,restore_best_weights=True)])
    m.save(mpath); np.savez(f'models/{TAG}_scaler_fold{fi}.npz',mu=mu,sd=sd)
    print('fold',fi,'saved', round(time.time()-t0,1),'s')

# if all folds exist -> compute OOF
if all(os.path.exists(f'models/{TAG}_fold{fi}.keras') for fi in range(5)):
    from pso import TemporalAttention
    oof=np.full(len(y),np.nan)
    for fi,(trn,tst) in enumerate(folds):
        m=tf.keras.models.load_model(f'models/{TAG}_fold{fi}.keras',
             custom_objects={'TemporalAttention':TemporalAttention}, safe_mode=False)
        s=np.load(f'models/{TAG}_scaler_fold{fi}.npz')
        oof[tst]=m.predict((X[tst]-s['mu'])/s['sd'],verbose=0).ravel()
    raw_p=np.exp(oof); raw_t=meta.y_raw.values; e=raw_t-raw_p
    res=dict(RMSE=float(np.sqrt(np.mean(e**2))),MAE=float(np.mean(np.abs(e))),
        MAPE=float(np.mean(np.abs(e/raw_t))*100),MdAPE=float(np.median(np.abs(e/raw_t))*100),
        R2=float(1-np.sum(e**2)/np.sum((raw_t-raw_t.mean())**2)))
    np.savez(f'oof_{TAG}.npz',oof=oof)
    json.dump({TAG:res},open(f'res_{TAG}.json','w'),indent=1)
    print(TAG,{k:round(v,3) for k,v in res.items()})
