"""
Experiment A — classical / statistical LOCO baselines, with out-of-fold predictions saved.

Regenerates oof_classical.npz (one out-of-fold log-prediction vector per model), which
analysis.py, figures_part2.py and clustered_stats.py all consume. Deterministic.

Ridge, like the other classical learners, is fitted on the flattened W x F window (eq. 6); the
last-step ridge is used only to warm-start the Kaya head of the hybrids (REPRODUCE.md, note 1). Standardisation uses training-fold
moments only, pooled over samples and time steps, matching deep_loco.py.
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import paths as P

SEED = 42
W = 8
DATA = os.environ.get("PANEL", P.data("panel_clean.csv"))
OUT = P.RESULTS

FEATS = ["log_epc", "log_gpc", "log_population", "coal_share", "oil_share", "gas_share"]


def build():
    df = pd.read_csv(DATA).sort_values(["iso_code", "year"])
    covs = ["energy_per_capita", "gdp_per_capita", "log_population",
            "coal_share", "oil_share", "gas_share"]
    df = df.dropna(subset=covs + ["co2_per_capita"]).copy()
    df["log_epc"] = np.log10(df.energy_per_capita.clip(lower=1))
    df["log_gpc"] = np.log10(df.gdp_per_capita.clip(lower=100))
    df["target"] = np.log(df.co2_per_capita)
    X, y, meta = [], [], []
    for iso, g in df.groupby("iso_code"):
        g = g.sort_values("year").reset_index(drop=True)
        for i in range(W - 1, len(g)):
            X.append(g[FEATS].iloc[i - W + 1:i + 1].values)
            y.append(g["target"].iloc[i])
            meta.append((iso, int(g.year.iloc[i]), float(g.co2_per_capita.iloc[i]),
                         str(g.country.iloc[i])))
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float32)
    meta = pd.DataFrame(meta, columns=["iso", "year", "y_raw", "country"])
    return X, y, meta


def models(seed=SEED):
    return {
        "kNN": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "SVR": SVR(C=10.0, epsilon=0.01),
        "RF": RandomForestRegressor(n_estimators=600, max_depth=14,
                                    random_state=seed, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=6,
                                subsample=0.85, random_state=seed, n_jobs=-1, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63,
                                  random_state=seed, n_jobs=-1, verbose=-1),
    }


def score(t, p):
    e = t - p
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                MAPE=float(np.mean(np.abs(e / t)) * 100),
                MdAPE=float(np.median(np.abs(e / t)) * 100),
                R2=float(1 - np.sum(e ** 2) / np.sum((t - t.mean()) ** 2)))


def main():
    X, y, meta = build()
    folds = list(GroupKFold(n_splits=5).split(X, y, meta.iso.values))
    names = ["Mean", "Ridge"] + list(models().keys())
    oof = {n: np.full(len(y), np.nan) for n in names}

    for fi, (trn, tst) in enumerate(folds):
        mu = X[trn].mean((0, 1)); sd = X[trn].std((0, 1)) + 1e-8
        Ztr, Zte = (X[trn] - mu) / sd, (X[tst] - mu) / sd
        oof["Mean"][tst] = y[trn].mean()
        oof["Ridge"][tst] = Ridge(1.0).fit(Ztr.reshape(len(trn), -1), y[trn]).predict(Zte.reshape(len(tst), -1))
        Ftr, Fte = Ztr.reshape(len(trn), -1), Zte.reshape(len(tst), -1)
        for name, mdl in models().items():
            oof[name][tst] = mdl.fit(Ftr, y[trn]).predict(Fte)
        print("fold", fi, "done")

    np.savez(f"{OUT}/oof_classical.npz", **oof)
    meta.to_csv(f"{OUT}/loco_meta.csv", index=False)
    t = meta.y_raw.values
    res = {n: score(t, np.exp(oof[n])) for n in names}
    json.dump(res, open(f"{OUT}/results_classical_loco_regen.json", "w"), indent=1)
    print(pd.DataFrame(res).T.round(3).to_string())


if __name__ == "__main__":
    main()
