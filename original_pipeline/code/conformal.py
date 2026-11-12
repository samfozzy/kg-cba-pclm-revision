"""
Monte-Carlo dropout and per-fold split-conformal prediction intervals for the Kaya-guided hybrid of
Experiment C (E-PSO-KG-CBA, 76 countries, seven covariates). Step 8 of REPRODUCE.md.

RECONSTRUCTED IN REVISION (November 2026). The original conformal.py was not included in the archive;
its outputs were (results/conformal_regen.json and results/conformal_regen.npz). This script was
rebuilt from the method stated in the manuscript (Methods, "Prediction intervals"; S1 Text, section F)
and from the stored fold models, and was checked against those stored outputs. See VERIFICATION below.

Method, for each of the five leave-country-out folds of Experiment C:
  1. Load the trained fold network (results/oof_parts/E-PSO-KG-CBA_fold<k>.keras) and its
     standardization constants (..._scaler_fold<k>.npz). The network is rebuilt in code and only its
     weights are loaded, so that the Lambda layer stored in the .keras file is never deserialized; this
     works on any Python version and reproduces the stored deterministic estimates exactly.
  2. Recreate the fold's inner validation countries exactly as in regen_enriched.py
     (one-seventh of the training countries, numpy RandomState(fold)).
  3. Monte-Carlo dropout: T = 60 forward passes with dropout active give the predictive mean ybar and
     standard deviation sigma (population standard deviation) of y = ln(CO2 per person).
  4. Split-conformal calibration on the inner validation countries: scores s = |y - ybar| / sigma;
     q = the ceil(0.9 (n + 1))-th smallest score.
  5. 90% interval for every held-out sample: [exp(ybar - q sigma), exp(ybar + q sigma)].
     Uncalibrated intervals use q = 1.6449 (the 95th percentile of the standard normal).
Coverage (PICP) is the share of held-out observations inside their interval; MPIW is the mean width in
t CO2 per person.

Usage (from the original_pipeline folder or anywhere else):
    python code/conformal.py              # writes results/conformal_regen.json and .npz (read by rebuild_figures.py)
                                          # or, if the stored ones exist, results/conformal_rerun.* next to them
    python code/conformal.py --seed 7     # another Monte-Carlo seed (fold k uses seed + k)
    python code/conformal.py --overwrite  # always writes results/conformal_regen.json and .npz

VERIFICATION (November 2026; Python 3.13, TensorFlow 2.21, Keras 3.15):
  - The rebuilt fold networks reproduce the stored deterministic estimates of E-PSO-KG-CBA exactly
    (maximum difference 0.0 in every fold).
  - Monte-Carlo dropout is random, and the seeding sequence of the original run is not known, so the
    stored intervals cannot be reproduced bit for bit. Across 27 runs with different seeds and orders of
    evaluation, coverage ranged from 81.0% to 83.0% (stored 81.9%), mean width from 5.40 to 6.04 t CO2
    per person (stored 5.68), uncalibrated coverage from 12.5% to 13.2% (stored 13.2%) and uncalibrated
    width from 0.460 to 0.468 (stored 0.469); the calibration factor of every fold was close to the
    stored value (fold 1, the largest: 66.6 to 75.6, stored 69.2). The stored results are therefore
    consistent with this method, and differences of about one percentage point in coverage are Monte
    Carlo variation.
  - An earlier implementation of the same method survives in figures_part2.py (legacy). It differs in two
    details: the calibration factor is np.quantile(scores, ceil(0.9 (n + 1)) / n) with linear interpolation
    instead of the order statistic of step 4, and 1e-6 is added to sigma. The difference is a fraction of
    one order statistic. In 13 runs of that variant with different seeds and orders of evaluation, coverage
    ranged from 81.3% to 82.7% and mean width from 5.41 to 5.68; it does not reproduce the stored calibration
    factors exactly either. This script keeps the order statistic, which is the definition given in S1 Text.
  - Every run prints a comparison with the stored results/conformal_regen.json and .npz.
"""
import os, sys, json, zipfile, pickle, random, argparse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as L, Model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P

DATA, RES, OOF = P.DATA, P.RESULTS, P.OOF
TAG = "E-PSO-KG-CBA"
T = 60                    # Monte-Carlo dropout passes
LEVEL = 0.90              # nominal coverage
Z_RAW = 1.6449            # two-sided 90% normal quantile for the uncalibrated intervals
SEED = 42


def set_seed(s):
    np.random.seed(s); random.seed(s); tf.random.set_seed(s)


class TemporalAttention(L.Layer):
    """Additive attention pooling over time steps (identical to deep_loco.py)."""
    def build(self, shp):
        d = shp[-1]
        self.W1 = self.add_weight(name="W1", shape=(d, d), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="b", shape=(d,), initializer="zeros", trainable=True)
        self.u = self.add_weight(name="u", shape=(d, 1), initializer="glorot_uniform", trainable=True)

    def call(self, h):
        s = tf.tanh(tf.matmul(h, self.W1) + self.b)
        a = tf.nn.softmax(tf.matmul(s, self.u), axis=1)
        return tf.reduce_sum(a * h, axis=1), a


def load_fold_model(path, n_feat):
    """Rebuild the hybrid with the layer names and settings stored in the .keras file, then load its weights."""
    layers = json.loads(zipfile.ZipFile(path).read("config.json"))["config"]["layers"]
    names, cfg = {}, {}
    for l in layers:
        names.setdefault(l["class_name"], []).append(l["config"]["name"]); cfg[l["config"]["name"]] = l["config"]
    conv = [cfg[n] for n in names["Conv1D"]]
    units = cfg[names["Bidirectional"][0]]["layer"]["config"]["units"]
    rate = cfg[names["Dropout"][0]]["rate"]
    dense32 = [n for n in names["Dense"] if n not in ("kaya_linear", "deep_residual")][0]
    inp = L.Input((8, n_feat), name=names["InputLayer"][0])
    last = L.Lambda(lambda t: t[:, -1, :], name=names["Lambda"][0])(inp)
    lin = L.Dense(1, name="kaya_linear")(last)
    x = L.Conv1D(conv[0]["filters"], conv[0]["kernel_size"][0], padding="same", activation="relu", name=names["Conv1D"][0])(inp)
    x = L.Conv1D(conv[1]["filters"], conv[1]["kernel_size"][0], padding="same", activation="relu", name=names["Conv1D"][1])(x)
    s = L.Bidirectional(L.LSTM(units, return_sequences=True), name=names["Bidirectional"][0])(x)
    s = L.Dropout(rate, name=names["Dropout"][0])(s)
    h, _ = TemporalAttention(name=names["TemporalAttention"][0])(s)
    h = L.Dense(32, activation="relu", name=dense32)(h)
    h = L.Dropout(rate, name=names["Dropout"][1])(h)
    resid = L.Dense(1, name="deep_residual")(h)
    m = Model(inp, L.Add(name=names["Add"][0])([lin, resid]))
    m.load_weights(path)
    return m


def mc_dropout(m, Z):
    P = np.stack([m(Z, training=True).numpy().ravel() for _ in range(T)])
    return P.mean(axis=0), P.std(axis=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--overwrite", action="store_true"); ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    d = np.load(os.path.join(DATA, "enriched.npz"), allow_pickle=False); X, y = d["X"], d["y"]
    meta = pd.read_csv(os.path.join(DATA, "meta_enriched.csv")); t = meta.y_raw.values
    with open(os.path.join(DATA, "folds_enriched.pkl"), "rb") as fh:   # contains numpy index arrays only
        folds = pickle.load(fh)
    lo, hi, lo_raw, hi_raw = (np.full(len(y), np.nan) for _ in range(4))
    q_by_fold, check = {}, []
    for fi, (trn, tst) in enumerate(folds):
        set_seed(a.seed + fi)
        m = load_fold_model(os.path.join(OOF, f"{TAG}_fold{fi}.keras"), X.shape[2])
        sc = np.load(os.path.join(OOF, f"{TAG}_scaler_fold{fi}.npz"), allow_pickle=False)
        Z = lambda A: ((A - sc["mu"]) / sc["sd"]).astype("float32")
        stored = np.load(os.path.join(OOF, f"{TAG}_fold{fi}.npz"), allow_pickle=False)
        check.append(float(np.abs(m.predict(Z(X[tst]), verbose=0).ravel() - stored["pred"]).max()))
        tr_iso = meta.iloc[trn].iso.unique(); rng = np.random.RandomState(fi)
        val_iso = set(rng.choice(tr_iso, size=max(1, len(tr_iso) // 7), replace=False))
        va = trn[meta.iloc[trn].iso.isin(val_iso).values]
        mv, sv = mc_dropout(m, Z(X[va]))
        mt, st = mc_dropout(m, Z(X[tst]))
        scores = np.abs(y[va] - mv) / sv; n = len(scores)
        q = float(np.sort(scores)[min(int(np.ceil(LEVEL * (n + 1))), n) - 1]); q_by_fold[str(fi)] = round(q, 2)
        lo[tst], hi[tst] = np.exp(mt - q * st), np.exp(mt + q * st)
        lo_raw[tst], hi_raw[tst] = np.exp(mt - Z_RAW * st), np.exp(mt + Z_RAW * st)
        print(f"fold {fi}: {len(set(val_iso))} calibration countries ({n} samples), q = {q:.2f}", flush=True)
    res = dict(T=T, seed=a.seed, q_by_fold=q_by_fold, picp=float(((t >= lo) & (t <= hi)).mean()), mpiw=float((hi - lo).mean()),
               picp_raw=float(((t >= lo_raw) & (t <= hi_raw)).mean()), mpiw_raw=float((hi_raw - lo_raw).mean()))
    ref_json, ref_npz = os.path.join(RES, "conformal_regen.json"), os.path.join(RES, "conformal_regen.npz")
    compare = os.path.exists(ref_json) and not a.overwrite      # keep the stored intervals and compare with them
    name = "conformal_rerun" if compare else "conformal_regen"
    json.dump(res, open(os.path.join(RES, f"{name}.json"), "w"), indent=1)
    np.savez(os.path.join(RES, f"{name}.npz"), lo=lo, hi=hi)
    print(json.dumps(res, indent=1))
    print("\nVERIFICATION")
    print(f"  rebuilt fold networks vs stored deterministic estimates: max |difference| = {max(check):.2e}")
    if compare:
        ref = json.load(open(ref_json)); rn = np.load(ref_npz, allow_pickle=False)
        for k in ("picp", "mpiw", "picp_raw", "mpiw_raw"):
            print(f"  {k:9s} this run {res[k]:.4f}   stored {ref[k]:.4f}")
        print(f"  q by fold this run {list(q_by_fold.values())}   stored {list(ref['q_by_fold'].values())}")
        w, w0 = hi - lo, rn["hi"] - rn["lo"]
        print(f"  interval width vs stored: median relative difference {100 * np.median(np.abs(w - w0) / w0):.1f}%, "
              f"largest absolute difference {np.abs(w - w0).max():.3g} t CO2 per person (very wide intervals of high emitters)")


if __name__ == "__main__":
    main()
