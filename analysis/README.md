# Revision analyses for PCLM-D-26-00445

Scripts for the analyses added in the revision of "Kaya-guided hybrid deep learning for
leave-country-out estimation of national CO2 emissions". They rebuild the panel from the two public
source files and need neither the neural-network code nor a GPU. Add this folder to the
reproducibility repository before archiving it.

## 1. Source data (exact versions)

The two files are already included in `../data/`. To fetch them yourself:

    git clone https://github.com/owid/co2-data    && git -C co2-data    checkout 382ee6c662b0ece26e111f263b44c029afad7787
    git clone https://github.com/owid/energy-data && git -C energy-data checkout 7e387a16f70a510e433f8aac7efeac6faa1e5059
    cp co2-data/owid-co2-data.csv energy-data/owid-energy-data.csv ../data/

(Commits of 2 June 2026 and 27 April 2026, the versions that were current on 26 July 2026.)
Set `KGCBA_DATA` to use another folder; `KGCBA_OUT` for the output folder (default `out/`).

## 2. Environment

Python 3.13, numpy, pandas, scipy, scikit-learn 1.9, xgboost 3.4, lightgbm 4.7, shap 0.54, matplotlib,
pillow, openpyxl. Ridge, kNN, SVR and the mean predictor are deterministic; the tree ensembles use
random state 42.

## 3. Run order and what each script produces

| Script | Produces | Used in |
|---|---|---|
| `rev01_panel.py` | panel size, descriptive statistics, correlations, fill audit by country | Table 1, S1 Table, S2 Fig |
| `rev02_identity.py` | check that blank coal/gas entries are zeros (national total = sum of components) | Methods, Data |
| `rev03_imputation.py` | missing-data sensitivity, six models (about 15 min) | Table 5, S2 Table A |
| `rev04_outliers.py` | analyses without the seven atypical economies, six models (about 20 min) | Table 5, Fig 5, S2 Table B |
| `rev05_expC_xai.py` | Experiment C with identical folds; permutation and drop-one importance; SHAP (about 20 min) | Tables 4, 6, 7, S2 Table E |
| `rev06_conformal.py` | conformal coverage for ridge under three calibration schemes | Results, S2 Table F |
| `rev07_tests.py` | country-clustered tests; ridge penalty; results by continent | Tables 4, 5, S2 Table C, D, H |
| `rev08_extra.py` | ridge against random forest; per-country summaries | Results, S2 Table C |
| `make_si_tables.py` | S1_Table.xlsx and S2_Table.xlsx from the outputs above | Supporting Information |
| `fig_diagrams.py`, `fig_data.py`, `fig_results.py` | Figs 1-5, S2 Fig (and conversion of the retained original figures) | Figures |
| `rev_tests_from_oof.py` | tests of every model against ridge, and re-scoring of every model, from stored out-of-fold estimates | Table 4 (to be completed) |

The folder `out/` contains the outputs of the run reported in the revised manuscript, for comparison.

## 4. Checks

`rev01_panel.py` must report 112 countries, 3,808 country-years, 3,024 samples and 28 / 114 / 244 / 291
filled values for energy, GDP, coal share and gas share. `rev07_tests.py` (or any script that fits
ridge) must reproduce RMSE 3.056, MAE 1.443, MAPE 25.2, MdAPE 16.3 and R2 0.800 for Experiment A, and
RMSE 3.037, MdAPE 7.7 for ridge with seven covariates in Experiment C.

## 5. Still to do with the stored predictions of the neural models

    python rev_tests_from_oof.py oof_experiment_A.csv --reference Ridge

The input needs the columns `iso, year, observed` and one column of out-of-fold estimates per model
(t CO2 per person; add `--log` if they are ln estimates). The script writes the cluster-bootstrap tests of
every model against ridge (the rows for KG-CBA and PSO-KG-CBA belong in Table 4) and the accuracy of
every model without the seven atypical economies and on the countries with and without long filled runs.
