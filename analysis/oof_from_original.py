"""Convert the stored out-of-fold (OOF) estimates of the original pipeline into the CSV format read by
rev_tests_from_oof.py.

Reads   ../original_pipeline/results/oof_classical.npz, oof_parts/oof_<model>.npz, oof_parts/oofE_<model>.npz,
        loco_meta.csv and ../original_pipeline/data/meta_enriched.csv
Writes  out/oof_experiment_A.csv (3,024 rows) and out/oof_experiment_C.csv (2,052 rows), with columns
        iso, year, observed (t CO2 per person) and one column per model (estimate, t CO2 per person).
The ensembles are the exploratory ensembles of the article (mean of the log estimates).
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "original_pipeline", "results")
D = os.path.join(HERE, "..", "original_pipeline", "data")
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)

meta = pd.read_csv(os.path.join(R, "loco_meta.csv"))
cz = np.load(os.path.join(R, "oof_classical.npz"), allow_pickle=False)
logA = {k: cz[k] for k in cz.files}
for k in ["LSTM", "GRU", "CNN-LSTM", "CNN-BiLSTM-Attn", "KG-CBA", "PSO-KG-CBA"]:
    logA[k] = np.load(os.path.join(R, "oof_parts", f"oof_{k}.npz"), allow_pickle=False)["oof"]
logA["Ensemble"] = (logA["Ridge"] + logA["GRU"] + logA["PSO-KG-CBA"]) / 3
A = pd.DataFrame({"iso": meta.iso, "year": meta.year, "observed": meta.y_raw, **{k: np.exp(v) for k, v in logA.items()}})
A.to_csv(os.path.join(OUT, "oof_experiment_A.csv"), index=False)

metaE = pd.read_csv(os.path.join(D, "meta_enriched.csv"))
logC = {k: np.load(os.path.join(R, "oof_parts", f"oofE_{k}.npz"), allow_pickle=False)["oof"] for k in ["Ridge", "LightGBM", "PSO-KG-CBA"]}
logC["Ensemble"] = (logC["Ridge"] + logC["PSO-KG-CBA"]) / 2
C = pd.DataFrame({"iso": metaE.iso, "year": metaE.year, "observed": metaE.y_raw, **{("E-PSO-KG-CBA" if k == "PSO-KG-CBA" else k): np.exp(v) for k, v in logC.items()}})
C.to_csv(os.path.join(OUT, "oof_experiment_C.csv"), index=False)
print("Experiment A:", A.shape, "Experiment C:", C.shape)
