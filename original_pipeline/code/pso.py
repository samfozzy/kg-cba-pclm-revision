"""
Particle swarm search for the hyperparameters of the Kaya-guided hybrid (PSO-KG-CBA). Step 4 of REPRODUCE.md.

Six particles x five iterations over six dimensions (filters, kernel, LSTM units, dropout, log10 learning rate,
batch-size index into BATCHES). Fitness is the lowest inner-validation MSE (log target) of one training run on
the first leave-country-out fold of Experiment A, with one-seventh of its training countries held out
(numpy RandomState(0)). The swarm is initialised with RandomState(42), velocities are updated with
RandomState(1000 + iteration), and every fitness evaluation builds its network after set_seed(123).

The state, including the log of every evaluation, is saved to results/pso_state.json after each evaluation,
so the search can be interrupted and resumed: each invocation stops after TIME_BUDGET seconds (default 190,
or the environment variable PSO_TIME_BUDGET), and the script is run again until it prints 'PSO COMPLETE'.
run_deep.py and regen_enriched.py use the configuration the search returned in the original run, written
out as PSO_BEST.
"""
import os, json, time, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
import numpy as np, tensorflow as tf
from tensorflow.keras import layers as L, Model, callbacks, optimizers, regularizers
from sklearn.linear_model import Ridge
from deep_loco import meta, X, y, folds, FEATS, TemporalAttention, set_seed
import paths as P

STATE=P.PSO_STATE
TIME_BUDGET=float(os.environ.get('PSO_TIME_BUDGET', 190))  # seconds per invocation
N_PART, N_ITER = 6, 5
# dims: filters, kernel, units, dropout, log10(lr), batch
LO = np.array([16, 2, 16, 0.05, np.log10(2e-4), 0])
HI = np.array([64, 4, 96, 0.45, np.log10(3e-3), 3])
BATCHES=[48,64,96,128]

trn,tst = folds[0]
tr_iso = meta.iloc[trn].iso.unique(); rng=np.random.RandomState(0)
val_iso=set(rng.choice(tr_iso,size=len(tr_iso)//7,replace=False))
iv=meta.iloc[trn].iso.isin(val_iso).values
TR,VA = trn[~iv], trn[iv]
MU=X[TR].mean((0,1)); SD=X[TR].std((0,1))+1e-8
Z=lambda A:(A-MU)/SD
# Ridge warm start for the Kaya linear layer (last-step covariates -> log co2pc)
ridge=Ridge(1.0).fit(Z(X[TR])[:,-1,:], y[TR])
RW, RB = ridge.coef_.astype('float32').reshape(-1,1), np.array([ridge.intercept_],'float32')

def decode(p):
    f=int(round(p[0])); k=int(round(p[1])); u=int(round(p[2]))
    return dict(filters=f, kernel=k, units=u, dropout=float(p[3]),
                lr=float(10**p[4]), batch=BATCHES[int(round(p[5]))])

def build_kg(hp):
    set_seed(123)
    inp=L.Input((8,len(FEATS)))
    last=L.Lambda(lambda t: t[:,-1,:])(inp)
    lin=L.Dense(1,name='kaya_linear')(last)
    c=L.Conv1D(hp['filters'],hp['kernel'],padding='same',activation='relu')(inp)
    c=L.Conv1D(hp['filters'],hp['kernel'],padding='same',activation='relu')(c)
    s=L.Bidirectional(L.LSTM(hp['units'],return_sequences=True))(c)
    s=L.Dropout(hp['dropout'])(s)
    h,att=TemporalAttention()(s)
    h=L.Dense(32,activation='relu')(h); h=L.Dropout(hp['dropout'])(h)
    resid=L.Dense(1,name='deep_residual',
                  kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3))(h)
    out=L.Add()([lin,resid])
    m=Model(inp,out)
    m.get_layer('kaya_linear').set_weights([RW,RB])
    m.compile(optimizers.Adam(hp['lr']),loss='mse')
    return m

def fitness(pos):
    hp=decode(pos)
    m=build_kg(hp)
    h=m.fit(Z(X[TR]),y[TR],validation_data=(Z(X[VA]),y[VA]),epochs=60,
            batch_size=hp['batch'],verbose=0,
            callbacks=[callbacks.EarlyStopping(patience=6,restore_best_weights=True)])
    return float(np.min(h.history['val_loss']))

def init_state():
    r=np.random.RandomState(42)
    P=r.uniform(LO,HI,(N_PART,len(LO)))
    V=r.uniform(-1,1,(N_PART,len(LO)))*(HI-LO)*0.15
    return dict(iter=0, part=0, P=P.tolist(), V=V.tolist(),
                pbest=P.tolist(), pbest_f=[1e9]*N_PART,
                gbest=P[0].tolist(), gbest_f=1e9, log=[])

def main():
    st=json.load(open(STATE)) if os.path.exists(STATE) else init_state()
    t0=time.time()
    while st['iter']<N_ITER:
        P=np.array(st['P']); V=np.array(st['V'])
        while st['part']<N_PART:
            if time.time()-t0>TIME_BUDGET:
                json.dump(st,open(STATE,'w')); print('PAUSE',st['iter'],st['part'],'best=%.4f'%st['gbest_f']); return
            i=st['part']; f=fitness(P[i])
            st['log'].append(dict(iter=st['iter'],part=i,f=f,hp=decode(P[i])))
            if f<st['pbest_f'][i]:
                st['pbest_f'][i]=f; st['pbest'][i]=P[i].tolist()
            if f<st['gbest_f']:
                st['gbest_f']=f; st['gbest']=P[i].tolist()
            st['part']+=1; json.dump(st,open(STATE,'w'))
        # velocity/position update
        r=np.random.RandomState(1000+st['iter'])
        r1,r2=r.rand(N_PART,len(LO)),r.rand(N_PART,len(LO))
        pb=np.array(st['pbest']); gb=np.array(st['gbest'])
        V=0.7*V+1.5*r1*(pb-P)+1.5*r2*(gb-P)
        V=np.clip(V,-(HI-LO)*0.3,(HI-LO)*0.3)
        P=np.clip(P+V,LO,HI)
        st['P'],st['V']=P.tolist(),V.tolist()
        st['iter']+=1; st['part']=0; json.dump(st,open(STATE,'w'))
        print('iter',st['iter'],'done. gbest=%.4f'%st['gbest_f'],decode(np.array(st['gbest'])))
    print('PSO COMPLETE. gbest=%.4f'%st['gbest_f'], decode(np.array(st['gbest'])))

if __name__=='__main__':
    main()
