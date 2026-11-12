"""
Experiment B — temporal forecasting benchmark against naive baselines.

Recasts the same panel as a conventional h-step-ahead forecasting task in which the
target country's OWN emission history IS available (unlike the LOCO task), so that
persistence and drift are well defined.

Design
------
Features : 8-year window of [log_epc, log_gpc, log_population, coal_share, oil_share,
           gas_share, y=ln c] -> 56 flattened predictors. The target's own lagged
           emissions are deliberately included; otherwise the comparison against
           persistence would be unfair to the learners.
Targets  : 'level'  -> y_{t+h} = ln c_{t+h}            (prediction: exp(yhat))
           'growth' -> d_{t+h} = y_{t+h} - y_t         (prediction: exp(y_t + dhat))
Splits   : main        train on pairs with t+h <= 2017, test on t+h in {2021,2022,2023}
           replication train on pairs with t+h <= 2012, test on t+h in 2015..2019
Baselines: persistence  chat_{t+h} = c_t
           drift        yhat_{t+h} = y_t + h * mean(last 3 log-changes)
Metrics  : MAE, MAPE, MdAPE on the raw scale, plus MASE_h scaled per country by that
           country's in-sample naive h-step MAE (eq. 34).

Outputs  : results/experiment_b.json, results/table_experiment_b.csv,
           results/expB_errors_<window>_<form>.csv (per-observation errors for the
           country-clustered significance tests in clustered_stats.py)

Deterministic: no TensorFlow, seed 42 throughout.
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

SEED = 42
W = 8
HORIZONS = [1, 3, 5]
DATA = os.environ.get("PANEL", "data/panel_clean.csv")
OUT = "results"
os.makedirs(OUT, exist_ok=True)

FEATS = ["log_epc", "log_gpc", "log_population", "coal_share", "oil_share", "gas_share", "y"]


def load_panel():
    df = pd.read_csv(DATA).sort_values(["iso_code", "year"])
    covs = ["energy_per_capita", "gdp_per_capita", "log_population",
            "coal_share", "oil_share", "gas_share"]
    df = df.dropna(subset=covs + ["co2_per_capita"]).copy()
    df["log_epc"] = np.log10(df.energy_per_capita.clip(lower=1))
    df["log_gpc"] = np.log10(df.gdp_per_capita.clip(lower=100))
    df["y"] = np.log(df.co2_per_capita)
    return df


def build_supervised(df, h):
    """One row per (country, origin year t) with a full W-window and an observed t+h."""
    X, rows = [], []
    for iso, g in df.groupby("iso_code"):
        g = g.sort_values("year").reset_index(drop=True)
        yv = g["y"].values
        cv = g["co2_per_capita"].values
        yrs = g["year"].values
        for i in range(W - 1, len(g) - h):
            X.append(g[FEATS].iloc[i - W + 1:i + 1].values.ravel())
            d3 = np.mean(np.diff(yv[i - 3:i + 1])) if i >= 3 else 0.0
            rows.append(dict(iso=iso, country=g.country.iloc[0],
                             t=int(yrs[i]), target_year=int(yrs[i + h]),
                             y_t=yv[i], c_t=cv[i],
                             y_target=yv[i + h], c_target=cv[i + h],
                             drift3=d3))
    return np.asarray(X, np.float64), pd.DataFrame(rows)


def naive_scale(df, h, train_end):
    """Per-country in-sample naive h-step MAE on the training period (eq. 34)."""
    sc = {}
    for iso, g in df.groupby("iso_code"):
        g = g.sort_values("year")
        c = g.loc[g.year <= train_end, "co2_per_capita"].values
        sc[iso] = np.mean(np.abs(c[h:] - c[:-h])) if len(c) > h else np.nan
    return sc


def learners(seed=SEED):
    return {
        "Ridge": Ridge(alpha=1.0),
        "kNN": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "SVR": SVR(C=10.0, epsilon=0.01),
        "RF": RandomForestRegressor(n_estimators=600, max_depth=14,
                                    random_state=seed, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=6,
                                subsample=0.85, random_state=seed, n_jobs=-1,
                                verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63,
                                  random_state=seed, n_jobs=-1, verbose=-1),
    }


def score(c_true, c_pred, scale):
    e = c_true - c_pred
    ape = np.abs(e / c_true) * 100
    return dict(MAE=float(np.mean(np.abs(e))),
                MAPE=float(np.mean(ape)),
                MdAPE=float(np.median(ape)),
                MASE=float(np.mean(np.abs(e) / scale)))


def run_window(df, label, train_end, test_years):
    out, err_rows = {}, []
    for h in HORIZONS:
        X, M = build_supervised(df, h)
        scale_map = naive_scale(df, h, train_end)
        M["scale"] = M.iso.map(scale_map)
        tr = (M.target_year <= train_end).values
        te = M.target_year.isin(test_years).values
        ok = te & M.scale.notna().values & (M.scale.values > 0)
        te = ok
        Xtr, Xte = X[tr], X[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        Mte = M[te].reset_index(drop=True)
        scale = Mte.scale.values
        c_true = Mte.c_target.values

        preds = {}
        preds["Persistence"] = Mte.c_t.values
        preds["Drift"] = np.exp(Mte.y_t.values + h * Mte.drift3.values)

        for form in ("level", "growth"):
            if form == "level":
                ttr = M.y_target.values[tr]
            else:
                ttr = (M.y_target.values - M.y_t.values)[tr]
            for name, mdl in learners().items():
                mdl.fit(Ztr, ttr)
                p = mdl.predict(Zte)
                yhat = p if form == "level" else Mte.y_t.values + p
                preds[f"{name} ({form})"] = np.exp(np.clip(yhat, -12, 8))

        for name, p in preds.items():
            out[f"h={h}|{name}"] = score(c_true, p, scale)
            err_rows.append(pd.DataFrame(dict(
                h=h, model=name, iso=Mte.iso.values, target_year=Mte.target_year.values,
                abs_err=np.abs(c_true - p), ape=np.abs((c_true - p) / c_true) * 100,
                scaled_err=np.abs(c_true - p) / scale)))
        print(f"[{label}] h={h}: n_train={tr.sum()} n_test={te.sum()}  "
              f"persistence MAE={out[f'h={h}|Persistence']['MAE']:.3f}")

    pd.concat(err_rows).to_csv(f"{OUT}/expB_errors_{label}.csv", index=False)
    return out


WINDOWS = {
    "main_2021_2023": ("main", 2017, [2021, 2022, 2023]),
    "replication_2015_2019": ("repl", 2012, [2015, 2016, 2017, 2018, 2019]),
}


def main(only=None):
    df = load_panel()
    res = {}
    for key, (lab, tend, tys) in WINDOWS.items():
        part = f"{OUT}/_expB_{lab}.json"
        if only and key != only and os.path.exists(part):
            res[key] = json.load(open(part)); continue
        if only and key != only:
            continue
        res[key] = run_window(df, lab, tend, tys)
        json.dump(res[key], open(part, "w"), indent=1)
    if len(res) < len(WINDOWS):
        return
    json.dump(res, open(f"{OUT}/experiment_b.json", "w"), indent=1)

    rows = []
    for win, d in res.items():
        for key, m in d.items():
            h, model = key.split("|")
            rows.append(dict(window=win, h=int(h.split("=")[1]), model=model, **m))
    tab = pd.DataFrame(rows).round(3)
    tab.to_csv(f"{OUT}/table_experiment_b.csv", index=False)
    print("\n=== MAIN WINDOW (test 2021-2023) ===")
    print(tab[tab.window == "main_2021_2023"].pivot_table(
        index="model", columns="h", values=["MAE", "MASE"]).round(3).to_string())
    print("\n=== REPLICATION (test 2015-2019) ===")
    print(tab[tab.window == "replication_2015_2019"].pivot_table(
        index="model", columns="h", values=["MAE", "MASE"]).round(3).to_string())


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
