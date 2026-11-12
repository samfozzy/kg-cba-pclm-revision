# Original analysis pipeline

This folder is the complete analysis pipeline of the study as archived in
`samfozzy/kg-cba-loco-co2-reproducibility` (commit `67b355c`, 12 August 2026), copied without changes
except that this README replaces the archive's own. It contains the code of every model, including the
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
| `code/figures_part1.py`, `code/figures_part2.py`, `code/rebuild_figures.py` | Figures, SHAP and attention results |
| `data/raw/` | The two Our World in Data source files (identical to `../data/owid-*.csv`) |
| `data/panel_clean.csv`, `data/enriched.npz`, `data/meta_enriched.csv`, `data/folds_enriched.pkl` | Harmonized panel, Experiment C arrays and fold assignment |
| `results/oof_classical.npz` | Out-of-fold estimates (natural log of t CO₂ per person) of the seven classical models, Experiment A |
| `results/oof_parts/oof_<model>.npz` | Out-of-fold estimates of the six deep models, Experiment A |
| `results/oof_parts/<model>_fold<k>.npz` | Per-fold estimates, row indices and training curves |
| `results/oof_parts/oofE_<model>.npz` | Out-of-fold estimates of ridge, LightGBM and E-PSO-KG-CBA, Experiment C |
| `results/oof_parts/E-PSO-KG-CBA_fold<k>.keras`, `..._scaler_fold<k>.npz` | Trained hybrid networks of Experiment C and their standardization constants |
| `results/loco_meta.csv` | Country, year and observed value of each row of the Experiment A estimates |
| `results/conformal_regen.*` | Monte-Carlo dropout and split-conformal intervals |
| `results/*.json`, `results/*.csv` | Values behind the tables and figures of the original submission |

## Checked against the revised manuscript

The stored estimates reproduce Tables 3, 4 and 6 of the revised manuscript exactly. The ridge regression
of the revision analyses (`../analysis/`) reproduces the stored ridge estimates to within 0.0003 on the log
scale. `../analysis/oof_from_original.py` converts the stored estimates to CSV
(`../analysis/out/oof_experiment_A.csv`, `oof_experiment_C.csv`), and `../analysis/rev_tests_from_oof.py`
uses them for the country-clustered tests of every model against ridge regression
(`../analysis/out/oof_experiment_*_tests_vs_reference.csv`) and for re-scoring every model on subsets of
countries (`../analysis/out/oof_experiment_*_rescored.csv`).

## Known gaps in this archive

- `code/conformal.py`, step 8 of `REPRODUCE.md`, is not in the archive. Its outputs are in
  `results/conformal_regen.json` and `results/conformal_regen.npz`. They are the intervals of the hybrid
  of Experiment C (E-PSO-KG-CBA, 76 countries, 2,052 rows).
- `pso_state.json`, the log of the particle swarm search read by `code/pso.py` and `code/figures_part1.py`,
  is not in the archive.
- The scripts do not share one working-directory convention. `build_panel.py`, `loco_classical.py` and
  `experiment_b.py` read `data/` and write `results/` relative to this folder. `run_deep.py`,
  `regen_enriched.py` and `clustered_stats.py` read and write `oof_parts/` without the `results/` prefix,
  and several scripts read `enriched.npz`, `meta_enriched.csv`, `folds_enriched.pkl` or `panel_clean.csv`
  from the current directory. The run order in `REPRODUCE.md` therefore needs these paths adjusted before
  it runs end to end.
