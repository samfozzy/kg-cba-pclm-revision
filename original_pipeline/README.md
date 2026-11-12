# Original analysis pipeline

This folder is the complete analysis pipeline of the study as archived in
`samfozzy/kg-cba-loco-co2-reproducibility` (commit `67b355c`, 12 August 2026), copied from the archive
and revised in November 2026 (see "Revisions to the archive" below); the methods and the stored estimates are unchanged. It contains the code of every model, including the
deep networks and the Kaya-guided hybrids, and their stored out-of-fold estimates. The run order, the
software versions and the expected values are in [`REPRODUCE.md`](REPRODUCE.md).

## Contents

| Path | What it contains |
|---|---|
| `code/build_panel.py` | Builds the harmonized panel from the two raw files |
| `code/loco_classical.py` | Mean predictor, ridge regression, kNN, SVR, random forest, XGBoost, LightGBM |
| `code/run_deep.py`, `code/deep_loco.py`, `code/run_arch.py` | LSTM, GRU, CNN-LSTM, CNN-BiLSTM-Attn, KG-CBA, PSO-KG-CBA (leave-country-out training) |
| `code/pso.py`, `code/train_final.py` | Particle swarm search for the hybrid and the final per-fold training |
| `code/train_enriched.py`, `code/regen_enriched.py` | Experiment C (76 countries, seven covariates) |
| `code/experiment_b.py` | Experiment B (temporal forecasting against persistence) |
| `code/clustered_stats.py`, `code/analysis.py` | Country-clustered tests and summary results |
| `code/conformal.py` | Monte-Carlo dropout and split-conformal intervals for the hybrid of Experiment C (reconstructed in revision) |
| `code/figures_part1.py`, `code/figures_part2.py`, `code/rebuild_figures.py` | Figures, SHAP and attention results |
| `data/raw/` | The two Our World in Data source files (identical to `../data/owid-*.csv`) |
| `data/panel_clean.csv`, `data/enriched.npz`, `data/meta_enriched.csv`, `data/folds_enriched.pkl` | Harmonized panel, Experiment C arrays and fold assignment |
| `results/oof_classical.npz` | Out-of-fold estimates (natural log of t CO₂ per person) of the seven classical models, Experiment A |
| `results/oof_parts/oof_<model>.npz` | Out-of-fold estimates of the six deep models, Experiment A |
| `results/oof_parts/<model>_fold<k>.npz` | Per-fold estimates, row indices and training curves |
| `results/oof_parts/oofE_<model>.npz` | Out-of-fold estimates of ridge, LightGBM and E-PSO-KG-CBA, Experiment C |
| `results/oof_parts/E-PSO-KG-CBA_fold<k>.keras`, `..._scaler_fold<k>.npz` | Trained hybrid networks of Experiment C and their standardization constants |
| `results/loco_meta.csv` | Country, year and observed value of each row of the Experiment A estimates |
| `results/conformal_regen.*` | Monte-Carlo dropout and split-conformal intervals of the original run (E-PSO-KG-CBA, Experiment C) |
| `results/pso_state.json`, `results/pso_rerun_check.csv` | Log of the particle swarm search (regenerated) and its comparison with Figure S4 |
| `results/table_imputation_audit.csv`, `results/table_imputation_runs.csv` | Every filled country-year and the length of each filled run |
| `results/legacy/` | Outputs of the legacy scripts (not part of the run order) |
| `results/conformal_rerun.*` | The same intervals from the reconstructed `code/conformal.py` (seed 42) |
| `results/*.json`, `results/*.csv` | Values behind the tables and figures of the original submission |

## Checked against the revised manuscript

The stored estimates reproduce Tables 3, 4 and 6 of the revised manuscript exactly. The ridge regression
of the revision analyses (`../analysis/`) reproduces the stored ridge estimates to within 0.0003 on the log
scale. `../analysis/oof_from_original.py` converts the stored estimates to CSV
(`../analysis/out/oof_experiment_A.csv`, `oof_experiment_C.csv`), and `../analysis/rev_tests_from_oof.py`
uses them for the country-clustered tests of every model against ridge regression
(`../analysis/out/oof_experiment_*_tests_vs_reference.csv`) and for re-scoring every model on subsets of
countries (`../analysis/out/oof_experiment_*_rescored.csv`).

## Revisions to the archive

- `code/paths.py` defines the folder layout once. Every script now reads and writes `data/`, `results/`,
  `results/oof_parts/` and `figures/` through it, so the run order in `REPRODUCE.md` works from any working
  directory. Current-run scripts write to the folders listed above; scripts that produced earlier variants
  of the analysis are marked "LEGACY" in their docstrings and write to `results/legacy/`.
- `code/conformal.py` was not in the archive and was reconstructed (step 8 of `REPRODUCE.md`). Only its
  outputs were archived (`results/conformal_regen.*`: the 90% intervals of the hybrid E-PSO-KG-CBA in
  Experiment C, 76 countries, 2,052 rows). The rebuilt script loads only the weights of the stored fold
  networks, which reproduce the stored estimates exactly. Monte-Carlo dropout is random and the seeding of the
  original run is unknown, so the intervals cannot be reproduced bit for bit: in 27 runs with different
  seeds, coverage ranged from 81.0% to 83.0% (stored 81.9%) and mean width from 5.40 to 6.04 t CO2 per person
  (stored 5.68).
- `results/pso_state.json`, the log of the particle swarm search, was missing from the archive. It was
  regenerated by `code/pso.py`. The search reproduces the published configuration as its global best, and
  its fitness path matches the values digitised from the original Figure S4 (`results/pso_rerun_check.csv`).
- `results/table_imputation_audit.csv` and `results/table_imputation_runs.csv` had no writing script. They
  are now written by `build_panel.py` and match the archived files (same rows and values).
- `figures_part1.py` no longer uses the Matplotlib argument removed in version 3.11 for box plots.
- A clean-copy run of the whole pipeline, and what it reproduces and does not, is described in
  `REPRODUCE.md`.

## Analysis of the revision

`../analysis/` contains the analyses added in revision: country-clustered tests, re-scoring on subsets of
countries, the atypical-economy exclusion (`rev09_stored_without_atypical.py`), and like-for-like conformal
coverage of ridge regression in Experiment C (`rev06c_conformal_expC.py`, output `analysis/out/rev06c.json`).
