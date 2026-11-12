"""
Regenerate out-of-fold predictions for every deep model in Experiment A (and the
enriched Experiment C hybrid), one fold at a time, resumable across invocations.

Usage:  python code/run_deep.py <MODEL> [max_seconds]
        MODEL in {LSTM, GRU, CNN-LSTM, CNN-BiLSTM-Attn, KG-CBA, PSO-KG-CBA}

Per-fold predictions land in results/oof_parts/<MODEL>_fold<k>.npz; once all five folds exist
the assembled vector is written to results/oof_parts/oof_<MODEL>.npz. Re-running skips folds
already on disk, so long jobs can be resumed.

The PSO-converged configuration is the one reported in Table B1.
"""
import os, sys, json, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L, Model, callbacks, optimizers
from sklearn.linear_model import Ridge

from deep_loco import meta, X, y, folds, FEATS, TemporalAttention, set_seed, build_model
import paths as P

PSO_BEST = dict(filters=37, kernel=4, units=40, dropout=0.231, lr=9.96e-4, batch=48)
FIXED = dict(filters=48, kernel=3, units=48, dropout=0.25, lr=1e-3, batch=96)
OUT = P.OOF


def build_kg(hp, RW, RB):
    """Kaya-guided CNN-BiLSTM-Attention: linear head warm-started at the ridge solution."""
    set_seed(123)
    inp = L.Input((8, len(FEATS)))
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


def run_fold(kind, fi, trn, tst):
    set_seed(42 + fi)
    tr_iso = meta.iloc[trn].iso.unique()
    rng = np.random.RandomState(fi)
    val_iso = set(rng.choice(tr_iso, size=max(1, len(tr_iso) // 7), replace=False))
    iv = meta.iloc[trn].iso.isin(val_iso).values
    tr_idx, va_idx = trn[~iv], trn[iv]
    mu = X[tr_idx].mean((0, 1)); sd = X[tr_idx].std((0, 1)) + 1e-8
    Z = lambda A: (A - mu) / sd

    if kind in ('KG-CBA', 'PSO-KG-CBA'):
        hp = PSO_BEST if kind == 'PSO-KG-CBA' else FIXED
        r = Ridge(1.0).fit(Z(X[tr_idx])[:, -1, :], y[tr_idx])
        RW = r.coef_.astype('float32').reshape(-1, 1)
        RB = np.array([r.intercept_], 'float32')
        m = build_kg(hp, RW, RB)
        bs = hp['batch']
    else:
        m = build_model(kind)
        bs = 96

    cb = [callbacks.EarlyStopping(patience=9, restore_best_weights=True, monitor='val_loss')]
    h = m.fit(Z(X[tr_idx]), y[tr_idx], validation_data=(Z(X[va_idx]), y[va_idx]),
              epochs=90, batch_size=bs, verbose=0, callbacks=cb)
    pred = m.predict(Z(X[tst]), verbose=0).ravel()
    np.savez(f'{OUT}/{kind}_fold{fi}.npz', idx=tst, pred=pred,
             val_loss=np.array(h.history['val_loss']), loss=np.array(h.history['loss']))
    return len(h.history['loss'])


def main():
    kind = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
    t0 = time.time()
    for fi, (trn, tst) in enumerate(folds):
        p = f'{OUT}/{kind}_fold{fi}.npz'
        if os.path.exists(p):
            continue
        if time.time() - t0 > budget:
            print('PAUSE before fold', fi, flush=True); return
        ep = run_fold(kind, fi, trn, tst)
        print(f'{kind} fold {fi}: {ep} epochs, {time.time()-t0:.0f}s', flush=True)

    if all(os.path.exists(f'{OUT}/{kind}_fold{fi}.npz') for fi in range(5)):
        oof = np.full(len(y), np.nan)
        for fi in range(5):
            d = np.load(f'{OUT}/{kind}_fold{fi}.npz')
            oof[d['idx']] = d['pred']
        np.savez(f'{OUT}/oof_{kind}.npz', oof=oof)
        t = meta.y_raw.values; p = np.exp(oof); e = t - p
        res = dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                   MAPE=float(np.mean(np.abs(e / t)) * 100),
                   MdAPE=float(np.median(np.abs(e / t)) * 100),
                   R2=float(1 - np.sum(e ** 2) / np.sum((t - t.mean()) ** 2)))
        json.dump({kind: res}, open(f'{OUT}/res_{kind}.json', 'w'), indent=1)
        print(kind, {k: round(v, 3) for k, v in res.items()}, flush=True)


if __name__ == '__main__':
    main()
