# REPRODUCE.md

Everything in the manuscript regenerates from this repository. Total runtime is roughly
45 minutes on a single CPU core, plus about 8 minutes of particle swarm search (step 4); no GPU is required.

Run every command from the folder that contains `code/`, `data/` and `results/` (the `original_pipeline`
folder of the repository), or from anywhere else: since the revision, all scripts locate their folders from
their own position (`code/paths.py`), so the working directory does not matter.

## Environment

```
Python 3.12.3
tensorflow-cpu==2.21.0   scikit-learn==1.8.0   xgboost==3.4.0
lightgbm==4.7.0          shap==0.52.0          pandas / numpy / scipy / matplotlib
```

Global seed 42 with per-fold offsets. `TF_ENABLE_ONEDNN_OPTS=0` is set so that runs are
reproducible across machines. The conformal script (step 8) was last run with Python 3.13 and Keras 3.15;
the `.keras` files store a Lambda layer, so the script rebuilds the network in code and loads only the weights.

## Run order

| Step | Command | Produces |
|---|---|---|
| 1 | `python code/build_panel.py` | `data/panel_clean.csv`, `data/enriched.npz`, `data/meta_enriched.csv`, `data/folds_enriched.pkl` |
| 2 | `python code/loco_classical.py` | `results/oof_classical.npz`, `results/loco_meta.csv`, classical rows of Table 1 |
| 3 | `python code/run_deep.py <MODEL>` for each of LSTM, GRU, CNN-LSTM, CNN-BiLSTM-Attn, KG-CBA, PSO-KG-CBA | `results/oof_parts/oof_<MODEL>.npz`, per-fold checkpoints, deep rows of Table 1 |
| 4 | `python code/pso.py` (repeat until it prints `PSO COMPLETE`) | `results/pso_state.json`, Table B1 |
| 5 | `python code/regen_enriched.py all` | `results/oof_parts/oofE_*.npz`, Table 4 |
| 6 | `python code/experiment_b.py main_2021_2023` then `python code/experiment_b.py replication_2015_2019` then `python code/experiment_b.py` | `results/experiment_b.json`, `results/table_experiment_b.csv`, `results/expB_errors_*.csv`, Table 6 |
| 7 | `python code/clustered_stats.py` | `results/table_clustered_stats.csv` (Table 2), `results/table_expB_clustered.csv`, `results/analysis_regen.json` |
| 8 | `python code/conformal.py` | `results/conformal_regen.json` and `.npz` (or `conformal_rerun.*` if the stored files are present), Section 4.5 |
| 9 | `python code/figures_part1.py` | F01-F06 (F07 and F08 are skipped unless the legacy files of `results/legacy/` exist; step 10 draws them) |
| 10 | `python code/rebuild_figures.py` | F07-F15, SHAP and attention results, all figures flattened to RGB |

Step 1 also writes `results/table_imputation_audit.csv` and `results/table_imputation_runs.csv`, the record of
every filled country-year (Table 3, Table B5 and Section 4.4).

Step 4 is time-limited: each invocation runs for `PSO_TIME_BUDGET` seconds (default 190; set the environment
variable to change it) and saves its state to `results/pso_state.json`, so the same command is repeated until
it prints `PSO COMPLETE`. The published configuration (37 filters, kernel 4, 40 units, dropout 0.231, learning
rate 9.96e-4, batch 48) is the global best of the stored log. `results/pso_rerun_check.csv` compares a
fresh search with the fitness values digitised from the original Figure S4.

Steps 8, 9 and 10 need the trained fold networks in `results/oof_parts/` (step 3) and do not depend on
the legacy scripts. Scripts in `code/` marked "LEGACY" in their docstrings (`deep_loco.py` as `__main__`,
`run_arch.py`, `train_final.py`, `train_enriched.py`, `analysis.py`, `figures_part2.py`) are not part of the run
order; they wrote the earlier variants of the analysis and their outputs are kept under `results/legacy/`.

Steps 3, 5 and 6 are resumable: each writes per-fold or per-window partial results and
skips work already on disk, so a long job can be interrupted and restarted.

## Expected values

The following reproduce to three decimals on any machine: Mean, Ridge, kNN, SVR, Random
Forest, XGBoost, LSTM, GRU, CNN-LSTM, CNN-BiLSTM-Attention (Table 1), and Ridge on the
enriched panel (Table 4). The two Kaya-guided hybrids contain a warm-started dense layer
whose training path is sensitive to floating-point accumulation order; across repeated
runs their RMSE varies by up to about 4% and MdAPE by about 0.5 percentage points. No
conclusion in the paper depends on that variation — the clustered comparisons in Table 2
that involve the hybrids have bootstrap intervals an order of magnitude wider.

## Clean-copy check (November 2026)

Steps 1 to 10 were run on a clean copy of `code/` and `data/raw/` from an unrelated working directory
(Python 3.13, TensorFlow 2.21, Keras 3.15, Matplotlib 3.11), with step 4 run separately. Compared with the
stored outputs in `results/`:

- Classical models (step 2) match the stored values to three decimals.
- Experiment C (step 5): the out-of-fold predictions of E-PSO-KG-CBA agree with the stored ones to within
  4 x 10^-6 on the log scale in every fold, so its metrics are unchanged. The Random Forest rows of
  Experiment B (step 6) differ slightly (up to 0.001 in MdAPE), because tree ensembles are not bit-reproducible
  across machines; the other rows match.
- Experiment A deep models (step 3) differ more, because the hybrids are retrained from scratch and the
  warm-started dense layer is sensitive to floating-point accumulation order. The MdAPE of KG-CBA on all 112
  countries is 15.39 (stored 15.66), of PSO-KG-CBA 16.50 (stored 16.72) and of the ensemble 15.02 (stored
  15.12). The country-clustered comparisons (step 7) keep their conclusions: the ensemble does not differ from
  ridge (p = 0.14; stored p = 0.22), the ensemble is better than PSO-KG-CBA (p = 0.046; stored p = 0.022), and
  the hybrids are better than CNN-BiLSTM-Attention (p = 0.0014 and 0.0012; stored 0.0016 and 0.0018).
- Experiment A on the 76-country subset is more variable for PSO-KG-CBA: R^2 0.50 (stored 0.39) and MAPE 25.2
  (stored 27.6), with MdAPE 16.4 (stored 16.3). The Experiment A tables in the manuscript come from the stored
  estimates.
- Conformal intervals (step 8) give coverage 82.3% and mean width 5.64 (stored 81.9% and 5.68), with
  calibration factors by fold of 10.8, 73.8, 30.8, 8.9 and 21.4 (stored 10.7, 69.2, 31.1, 8.8 and 21.9). The
  difference is Monte-Carlo variation (see the script docstring).

The clean run wrote to a scratch copy; the stored outputs in `results/` were not overwritten.

## Two specification notes

1. **Ridge is fitted on the flattened window**, not on the most recent window step
   (Section 3.3, equation 6). The last-step variant, which is used only to warm-start the
   hybrid's Kaya head, gives RMSE 3.343 and R² 0.761 rather than 3.056 and 0.800.
2. **SHAP aggregation is lag-major.** The flattened design matrix has column index
   `tau * F + j`, so the attribution array must be reshaped to `(N, W, F)` before
   aggregating. Reshaping to `(N, F, W)` silently interchanges covariates and lags and
   produces a spurious single-lag importance spike.

## Panel reproduction, verified

`build_panel.py` regenerates the panel from the two raw downloads and reproduces the
published `panel_clean.csv` exactly: the same 112 countries, all 3,808 country-years, and
a maximum absolute difference of 0.0 on every one of the seven modelled series. The
enriched arrays and the Experiment C fold assignment likewise match bit-for-bit.

Two source subtleties are load-bearing and are documented in the script:

- **`energy_per_capita` comes from the OWID CO2 file, not the energy file.** The two
  lineages are different vintages and disagree for 175 recent country-years (up to
  4,610 kWh person⁻¹). Using the energy-file column silently changes the panel.
- **`low_carbon_share_energy` is reported by OWID in percent** but is stored in the panel
  as a fraction, so it is divided by 100 during construction.

## Data provenance

`data/raw/` contains the two Our World in Data source files as downloaded on 26 July 2026.
`code/build_panel.py` documents every filtering, interpolation and edge-extension step.
`results/table_imputation_audit.csv` and `results/table_imputation_runs.csv` record exactly
which country-years were filled and how long each filled run is; these underpin Table 3,
Table B5 and the sensitivity analysis in Section 4.4.
