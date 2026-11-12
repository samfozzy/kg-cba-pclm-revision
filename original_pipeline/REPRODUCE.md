# REPRODUCE.md

Everything in the manuscript regenerates from this repository. Total runtime is roughly
45 minutes on a single CPU core; no GPU is required.

## Environment

```
Python 3.12.3
tensorflow-cpu==2.21.0   scikit-learn==1.8.0   xgboost==3.4.0
lightgbm==4.7.0          shap==0.52.0          pandas / numpy / scipy / matplotlib
```

Global seed 42 with per-fold offsets. `TF_ENABLE_ONEDNN_OPTS=0` is set so that runs are
reproducible across machines.

## Run order

| Step | Command | Produces |
|---|---|---|
| 1 |  `python code/build_panel.py` | `data/panel_clean.csv`, `data/enriched.npz`, `data/meta_enriched.csv`, `data/folds_enriched.pkl` |
| 2 | `python code/loco_classical.py` | `results/oof_classical.npz`, `results/loco_meta.csv`, classical rows of Table 1 |
| 3 | `python code/run_deep.py <MODEL>` for each of LSTM, GRU, CNN-LSTM, CNN-BiLSTM-Attn, KG-CBA, PSO-KG-CBA | `oof_parts/oof_<MODEL>.npz`, per-fold checkpoints, deep rows of Table 1 |
| 4 | `python code/pso.py` (repeat until it reports convergence) | `pso_state.json`, Table B1 |
| 5 | `python code/regen_enriched.py all` | `oof_parts/oofE_*.npz`, Table 4 |
| 6 | `python code/experiment_b.py main_2021_2023` then `python code/experiment_b.py replication_2015_2019` then `python code/experiment_b.py` | `results/experiment_b.json`, `results/table_experiment_b.csv`, `results/expB_errors_*.csv`, Table 6 |
| 7 | `python code/clustered_stats.py` | `results/table_clustered_stats.csv` (Table 2), `results/table_expB_clustered.csv`, `results/analysis_regen.json` |
| 8 | `python code/conformal.py` | `results/conformal_regen.json`, Section 4.5 |
| 9 | `python code/figures_part1.py` | F01-F06 |
| 10 | `python code/rebuild_figures.py` | F07-F15, SHAP and attention results, all figures flattened to RGB |

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
