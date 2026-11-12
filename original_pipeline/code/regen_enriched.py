"""
Experiment C — enriched 76-country panel (adds low-carbon share of primary energy).
Regenerates out-of-fold predictions for Ridge, LightGBM and the enriched hybrid
E-PSO-KG-CBA, resumable per fold. Writes results/oof_parts/oofE_<MODEL>.npz.
"""
import os, sys, json, time, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as L, Model, callbacks, optimizers
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from deep_loco import TemporalAttention, set_seed
import paths as P

PSO_BEST = dict(filters=37, kernel=4, units=40, dropout=0.231, lr=9.96e-4, batch=48)
OUT = P.OOF

d = np.load(P.data('enriched.npz')); XE, yE = d['X'], d['y']
metaE = pd.read_csv(P.data('meta_enriched.csv'))
foldsE = pickle.load(open(P.data('folds_enriched.pkl'), 'rb'))
NF = XE.shape[2]


def build_kg(hp, RW, RB):
    set_seed(123)
    inp = L.Input((8, NF))
    last = L.Lambda(lambda t: t[:, -1, :])(inp)
    lin = L.Dense(1, name='kaya_linear')(last)
    c = L.Conv1D(hp['filters'], hp['kernel'], padding='same', activation='relu')(inp)
    c = L.Conv1D(hp['filters'], hp['kernel'], padding='same', activation='relu')(c)
    s = L.Bidirectional(L.LSTM(hp['units'], return_sequences=True))(c)
    s = L.Dropout(hp['dropout'])(s)
    h, att = TemporalAttention()(s)
    h = L.Dense(32, activation='relu')(h)
    h = L.Dropout(hp['dropout'])(h)
    resid = L.Dense(1, name='deep_residual',
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3))(h)
    m = Model(inp, L.Add()([lin, resid]))
    m.get_layer('kaya_linear').set_weights([RW, RB])
    m.compile(optimizers.Adam(hp['lr']), loss='mse')
    return m


def score(t, p):
    e = t - p
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                MAPE=float(np.mean(np.abs(e / t)) * 100),
                MdAPE=float(np.median(np.abs(e / t)) * 100),
                R2=float(1 - np.sum(e ** 2) / np.sum((t - t.mean()) ** 2)))


def shallow():
    oR = np.full(len(yE), np.nan); oL = np.full(len(yE), np.nan)
    for trn, tst in foldsE:
        mu = XE[trn].mean((0, 1)); sd = XE[trn].std((0, 1)) + 1e-8
        Z, Zt = (XE[trn] - mu) / sd, (XE[tst] - mu) / sd
        oR[tst] = Ridge(1.0).fit(Z.reshape(len(trn), -1), yE[trn]).predict(Zt.reshape(len(tst), -1))
        oL[tst] = LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63,
                                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5,
                                random_state=42, n_jobs=-1, verbose=-1
                                ).fit(Z.reshape(len(trn), -1), yE[trn]).predict(Zt.reshape(len(tst), -1))
    np.savez(f'{OUT}/oofE_Ridge.npz', oof=oR)
    np.savez(f'{OUT}/oofE_LightGBM.npz', oof=oL)
    t = metaE.y_raw.values
    print('Ridge   ', {k: round(v, 3) for k, v in score(t, np.exp(oR)).items()})
    print('LightGBM', {k: round(v, 3) for k, v in score(t, np.exp(oL)).items()})


def hybrid(budget=1e9):
    t0 = time.time()
    for fi, (trn, tst) in enumerate(foldsE):
        p = f'{OUT}/E-PSO-KG-CBA_fold{fi}.npz'
        if os.path.exists(p):
            continue
        if time.time() - t0 > budget:
            print('PAUSE before fold', fi, flush=True); return
        set_seed(42 + fi)
        tr_iso = metaE.iloc[trn].iso.unique(); rng = np.random.RandomState(fi)
        val_iso = set(rng.choice(tr_iso, size=max(1, len(tr_iso) // 7), replace=False))
        iv = metaE.iloc[trn].iso.isin(val_iso).values
        tr_idx, va_idx = trn[~iv], trn[iv]
        mu = XE[tr_idx].mean((0, 1)); sd = XE[tr_idx].std((0, 1)) + 1e-8
        Z = lambda A: (A - mu) / sd
        r = Ridge(1.0).fit(Z(XE[tr_idx])[:, -1, :], yE[tr_idx])
        m = build_kg(PSO_BEST, r.coef_.astype('float32').reshape(-1, 1),
                     np.array([r.intercept_], 'float32'))
        m.fit(Z(XE[tr_idx]), yE[tr_idx], validation_data=(Z(XE[va_idx]), yE[va_idx]),
              epochs=90, batch_size=PSO_BEST['batch'], verbose=0,
              callbacks=[callbacks.EarlyStopping(patience=9, restore_best_weights=True)])
        np.savez(p, idx=tst, pred=m.predict(Z(XE[tst]), verbose=0).ravel())
        m.save(f'{OUT}/E-PSO-KG-CBA_fold{fi}.keras')
        np.savez(f'{OUT}/E-PSO-KG-CBA_scaler_fold{fi}.npz', mu=mu, sd=sd)
        print('fold', fi, f'{time.time()-t0:.0f}s', flush=True)

    if all(os.path.exists(f'{OUT}/E-PSO-KG-CBA_fold{fi}.npz') for fi in range(5)):
        oof = np.full(len(yE), np.nan)
        for fi in range(5):
            z = np.load(f'{OUT}/E-PSO-KG-CBA_fold{fi}.npz'); oof[z['idx']] = z['pred']
        np.savez(f'{OUT}/oofE_PSO-KG-CBA.npz', oof=oof)
        print('E-PSO-KG-CBA', {k: round(v, 3) for k, v in
                               score(metaE.y_raw.values, np.exp(oof)).items()})


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('all', 'shallow'):
        shallow()
    if what in ('all', 'hybrid'):
        hybrid(float(sys.argv[2]) if len(sys.argv) > 2 else 1e9)
