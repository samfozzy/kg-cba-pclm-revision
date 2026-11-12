import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from kgcba_common import *
from sklearn.model_selection import GroupKFold
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
OUT = os.environ.get("KGCBA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")) + os.sep; DEST = os.environ.get("KGCBA_SI", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")) + os.sep
F = Font(name="Arial", size=10); FB = Font(name="Arial", size=10, bold=True); FT = Font(name="Arial", size=12, bold=True)
HFILL = PatternFill("solid", fgColor="E8EEF5"); THIN = Side(style="thin", color="999999")

def write_sheet(wb, name, df, title, notes=(), fmt=None, widths=None):
    ws = wb.create_sheet(name); ws["A1"] = title; ws["A1"].font = FT
    r = 2
    for n in notes: ws.cell(row=r, column=1, value=n).font = F; r += 1
    r += 1; hdr = r
    for j, c in enumerate(df.columns, 1):
        cell = ws.cell(row=hdr, column=j, value=c); cell.font = FB; cell.fill = HFILL; cell.alignment = Alignment(wrap_text=True, vertical="center"); cell.border = Border(bottom=THIN, top=THIN)
    for i, row in enumerate(df.itertuples(index=False), hdr + 1):
        for j, v in enumerate(row, 1):
            if isinstance(v, (np.floating, float)): v = None if (v != v) else float(v)
            elif isinstance(v, (np.integer,)): v = int(v)
            elif isinstance(v, (np.bool_,)): v = bool(v)
            cell = ws.cell(row=i, column=j, value=v); cell.font = F
            f_ = (fmt or {}).get(df.columns[j - 1])
            if f_ and isinstance(v, float): cell.number_format = f_
    for j, c in enumerate(df.columns, 1):
        w = (widths or {}).get(c) or max(10, min(46, max(len(str(c)) * 0.62 + 3, df[c].astype(str).str.len().max() + 2)))
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.row_dimensions[hdr].height = 44; ws.freeze_panes = ws.cell(row=hdr + 1, column=2)
    return ws

def readme(wb, title, lines):
    ws = wb.active; ws.title = "README"; ws["A1"] = title; ws["A1"].font = FT
    for i, l in enumerate(lines, 3):
        c = ws.cell(row=i, column=1, value=l); c.font = F; c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 120
    return ws

# ---------------------------------------------------------------- S1 Table
p, audit = build_panel(return_audit=True); q = add_low_carbon(p)
X, y, c, iso, yr = windows(p)
fold = {}
for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, iso)):
    for i_ in np.unique(iso[te]): fold[i_] = k + 1
pc = pd.read_csv(OUT + "ridge_per_country_mdape.csv").set_index("iso"); ec = pd.read_csv(OUT + "expC_per_country.csv").set_index("iso")
A = audit.set_index("iso")
T = pd.DataFrame({"ISO code": A.index, "Country": A.country.values, "Continent": A.continent.values,
                  "In 76-country panel (Experiment C)": ["Yes" if i in ec.index else "No" for i in A.index],
                  "Cross-validation fold": [fold[i] for i in A.index],
                  "Filled years, energy per capita": A.n_filled_energy_per_capita.values, "Filled years, GDP per capita": A.n_filled_gdp_pc.values,
                  "Filled years, coal share": A.n_filled_coal_share.values, "Filled years, gas share": A.n_filled_gas_share.values,
                  "Longest filled run, coal share (years)": A.run_coal_share.values, "Longest filled run, gas share (years)": A.run_gas_share.values,
                  "Longest filled run, any variable (years)": A.longest_run.values,
                  "Ridge MdAPE (%), Experiment A": [pc.orig[i] for i in A.index],
                  "Ridge MdAPE (%), blank shares set to zero": [pc.zero[i] for i in A.index],
                  "Ridge MdAPE (%), without fuel shares": [pc.nofuel[i] for i in A.index],
                  "Ridge MdAPE (%), 76-country panel, six covariates": [ec.six[i] if i in ec.index else np.nan for i in A.index],
                  "Ridge MdAPE (%), 76-country panel, seven covariates": [ec.seven[i] if i in ec.index else np.nan for i in A.index]}).sort_values("ISO code")
wb = Workbook()
readme(wb, "S1 Table. Countries and data audit", [
    "Supporting information for: Kaya-guided hybrid deep learning for leave-country-out estimation of national CO2 emissions: a 112-country benchmark with country-clustered inference, conformal uncertainty, and explainable AI.",
    "Sheet 'Countries' lists the 112 countries of the panel (1990-2023) with their continent, membership of the 76-country panel of Experiment C and cross-validation fold.",
    "Filled years: number of the 34 years in which the value was missing in the source and was completed by the fill rule described in the article (linear interpolation of up to three consecutive years, then last available value carried forward, then first observed value carried backward). The target (per-capita CO2), population and the oil share needed no filling. For GDP per capita, the year 2023 is filled for every country because the Maddison Project Database 2023 ends in 2022.",
    "Longest filled run: largest number of consecutive filled years. A blank coal or gas entry in the source occurs where that fuel does not appear in the national total (see the article, section Data).",
    "Ridge MdAPE: median absolute percentage error of ridge regression for the country when it is held out (leave-country-out, five folds). Experiment A: 112 countries, six covariates, original fill. The next two columns repeat this with blank coal and gas shares set to zero and without the three fuel shares. The last two columns are for the 76-country panel with identical folds, without and with the low-carbon share of primary energy.",
    "Source data: Our World in Data, owid-co2-data.csv (repository version of 2 June 2026) and owid-energy-data.csv (version of 27 April 2026). Values produced by the scripts rev01_panel.py, rev03_imputation.py and rev05_expC_xai.py of the reproducibility archive."])
write_sheet(wb, "Countries", T, "Countries of the panel, data audit and per-country error of ridge regression",
            fmt={c_: "0.0" for c_ in T.columns if "MdAPE" in c_}, widths={"Country": 30, "Continent": 15})
wb.save(DEST + "S1_Table.xlsx"); print("S1 Table rows", len(T), "enriched", (T.iloc[:, 3] == "Yes").sum())

# ---------------------------------------------------------------- S2 Table
wb = Workbook()
readme(wb, "S2 Table. Sensitivity analyses and additional results", [
    "Supporting information for: Kaya-guided hybrid deep learning for leave-country-out estimation of national CO2 emissions: a 112-country benchmark with country-clustered inference, conformal uncertainty, and explainable AI.",
    "All results are leave-country-out results (five folds, countries as groups). RMSE and MAE are in t CO2 per person; MAPE and MdAPE are in %.",
    "Sheet A: missing-data analyses for ridge regression and the five classical learners.",
    "Sheet B: analyses without the seven atypical economies (Qatar, Kuwait, Saudi Arabia, Singapore, Luxembourg, Norway, Iceland).",
    "Sheet C: paired comparisons with countries as clusters (Algorithm C of S1 Text), with the classical learners re-estimated in revision. Delta is the median of the paired differences in absolute percentage error (first minus second); the interval is the 95% percentile interval of a cluster bootstrap over countries (10,000 draws).",
    "Sheet D: absolute percentage error by continent for the exploratory ensemble (original run) and for ridge regression.",
    "Sheet E: covariate importance on held-out error in Experiment C (76 countries): permutation importance (30 repetitions) and drop-one-covariate re-estimation.",
    "Sheet F: coverage of conformal prediction intervals for ridge regression in Experiment A under three calibration schemes (S1 Text, section F).",
    "Sheet F2: the same for ridge regression on the countries, covariates and folds of Experiment C, with calibration sets of the size used for the hybrid (8 countries), and the coverage of the stored intervals of the hybrid E-PSO-KG-CBA.",
    "Sheet G: values of the original run and of the re-estimation in revision for the models that were re-estimated. The mean predictor, ridge regression, kNN and SVR are deterministic and reproduce exactly; the three tree ensembles depend on the random state and differ slightly, and LightGBM was re-estimated with the column subsample (0.85) and L2 penalty (1.5) of Experiment C, which the original run used only in Experiment C.",
    "Sheet H: sensitivity of ridge regression to the penalty lambda.",
    "Sheet I: paired comparisons of every model with ridge regression from the stored out-of-fold estimates of the original run (all thirteen models and the ensemble), in Experiment A on all 112 countries and on the 105 countries without the seven atypical economies, and in Experiment C.",
    "Sheet J: every model of the original run re-scored from its stored out-of-fold estimates on subsets of countries, with the share of the squared error contributed by the seven atypical economies and by the model's own six worst-fitted countries.",
    "Values produced by the scripts rev03_imputation.py, rev04_outliers.py, rev05_expC_xai.py, rev06_conformal.py, rev06c_conformal_expC.py, rev07_tests.py, rev08_extra.py, rev_tests_from_oof.py and rev09_stored_without_atypical.py of the reproducibility archive; the ensemble rows of sheet D are from the original run."])
mfmt = {"RMSE": "0.000", "MAE": "0.000", "MAPE": "0.0", "MdAPE": "0.0", "R2": "0.000"}
S = pd.read_csv(OUT + "imputation_sensitivity.csv"); S["treatment"] = S.treatment.str.replace(r"^[A-F]\d?\.\s*", "", regex=True)
S = S.rename(columns={"model": "Model", "treatment": "Analysis", "R2": "R2"})
write_sheet(wb, "A_missing_data", S, "A. Missing-data analyses", ["Original panel = fill rule of the article. A long filled run is a run of more than three (or ten) consecutive filled years."], fmt=mfmt, widths={"Analysis": 70, "Model": 16})
O = pd.read_csv(OUT + "outlier_sensitivity.csv").rename(columns={"model": "Model", "scope": "Analysis"})
write_sheet(wb, "B_atypical_economies", O, "B. Analyses without the seven atypical economies", fmt=mfmt, widths={"Analysis": 70, "Model": 16})
r7 = json.load(open(OUT + "rev07.json")); r8 = json.load(open(OUT + "rev08.json"))
lab = {"ridge_vs_lightgbm_refit": ("A", "Ridge", "LightGBM (re-estimated)"), "ridge_vs_xgboost_refit": ("A", "Ridge", "XGBoost (re-estimated)"), "ridge_vs_rf_refit": ("A", "Ridge", "Random forest (re-estimated)"),
       "ridge_vs_svr": ("A", "Ridge", "SVR"), "ridge_vs_knn": ("A", "Ridge", "kNN"), "zero_fill_vs_original": ("A", "Ridge, blank shares set to zero", "Ridge, original fill"),
       "no_fuel_shares_vs_six": ("A", "Ridge without fuel shares", "Ridge, six covariates"), "energy_only_vs_six": ("A", "Ridge, energy per capita only", "Ridge, six covariates"),
       "last_year_vs_window": ("A", "Ridge, most recent year only", "Ridge, eight-year window"), "expC_seven_vs_six": ("C", "Ridge, seven covariates", "Ridge, six covariates (same countries and folds)"),
       "expC_seven_vs_six_no_atypical": ("C, 69 countries without the atypical economies", "Ridge, seven covariates", "Ridge, six covariates")}
rows = []
allr = dict(r7); allr["ridge_vs_rf_refit"] = r8["ridge_vs_rf_refit"]
for k, (ex, a, b) in lab.items():
    v = allr[k]; rows.append({"Experiment": ex, "First model": a, "Second model": b, "MdAPE first (%)": v["mdape_a"], "MdAPE second (%)": v["mdape_b"], "Delta (points)": v["delta"],
                              "95% CI lower": v["ci_lo"], "95% CI upper": v["ci_hi"], "p, cluster bootstrap": v["p_boot"], "p, per-country test": v["p_country"], "p, observation-level test": v["p_obs"]})
Ct = pd.DataFrame(rows)
write_sheet(wb, "C_paired_tests", Ct, "C. Paired comparisons with countries as clusters", ["A cluster-bootstrap p of 0 means that none of the 10,000 draws had the opposite sign (p < 0.0001)."],
            fmt={"MdAPE first (%)": "0.00", "MdAPE second (%)": "0.00", "Delta (points)": "+0.00;-0.00;0.00", "95% CI lower": "+0.00;-0.00;0.00", "95% CI upper": "+0.00;-0.00;0.00", "p, cluster bootstrap": "0.0000", "p, per-country test": "0.0E+00", "p, observation-level test": "0.0E+00"},
            widths={"First model": 34, "Second model": 44, "Experiment": 22})
ens = pd.DataFrame([("Europe", 12.50, 23.60, 25.67, 1053), ("South America", 14.84, 20.45, 26.24, 216), ("Asia", 15.83, 23.43, 26.93, 999), ("Africa", 16.77, 26.63, 26.77, 459), ("Oceania", 18.76, 18.03, 25.22, 54), ("North America", 19.63, 21.46, 27.41, 243)],
                   columns=["Continent", "Median APE (%)", "Mean APE (%)", "75th percentile of APE (%)", "Country-years"]); ens.insert(0, "Model", "Ensemble (original run)")
rc = pd.read_csv(OUT + "ridge_by_continent.csv"); rc = rc.rename(columns={"continent": "Continent", "median": "Median APE (%)", "mean": "Mean APE (%)", "p75": "75th percentile of APE (%)", "n": "Country-years", "countries": "Countries"}); rc.insert(0, "Model", "Ridge regression")
ens["Countries"] = ens.Continent.map(rc.set_index("Continent").Countries)
D = pd.concat([rc[["Model", "Continent", "Countries", "Country-years", "Median APE (%)", "Mean APE (%)", "75th percentile of APE (%)"]], ens[["Model", "Continent", "Countries", "Country-years", "Median APE (%)", "Mean APE (%)", "75th percentile of APE (%)"]]])
write_sheet(wb, "D_continents", D, "D. Absolute percentage error by continent, Experiment A", fmt={"Median APE (%)": "0.0", "Mean APE (%)": "0.0", "75th percentile of APE (%)": "0.0"}, widths={"Model": 26})
I = pd.read_csv(OUT + "xai_alternative.csv")
I = I.rename(columns={"model": "Model", "covariate": "Covariate", "base_MdAPE": "Reference MdAPE (%)", "perm_MdAPE": "MdAPE after permutation (%)", "perm_dMdAPE": "Permutation: change in MdAPE (points)", "perm_sd": "Permutation: SD over 30 repetitions",
                      "perm_logMSE_ratio": "Permutation: ratio of mean squared error of y", "drop_MdAPE": "MdAPE without the covariate (%)", "drop_dMdAPE": "Drop-one: change in MdAPE (points)", "drop_MAPE": "MAPE without the covariate (%)", "drop_R2": "R2 without the covariate", "drop_RMSE": "RMSE without the covariate"})
write_sheet(wb, "E_importance", I, "E. Covariate importance on held-out error, Experiment C (76 countries)", ["LightGBM was re-estimated for this analysis (random state 42); its reference MdAPE (13.2%) differs slightly from the original run (13.1%)."],
            fmt={c_: "0.00" for c_ in I.columns if c_ not in ("Model", "Covariate")}, widths={"Covariate": 20})
r6 = json.load(open(OUT + "rev06.json"))
rows = []
for l in ["0.5", "0.6", "0.7", "0.8", "0.9", "0.95"]:
    ss = r6["single_split"][l]
    rows.append({"Nominal level (%)": float(l) * 100, "(a) 13 training countries: mean coverage (%)": 100 * ss["mean"], "(a) 5th percentile over 200 draws (%)": 100 * ss["p5"], "(a) 95th percentile over 200 draws (%)": 100 * ss["p95"],
                 "(b) all training countries: coverage (%)": 100 * r6["cross_conformal"][l], "(c) exchangeable reference: coverage (%)": 100 * r6["exchangeable_reference"][l]})
Fd = pd.DataFrame(rows)
ws = write_sheet(wb, "F_conformal", Fd, "F. Coverage of conformal prediction intervals for ridge regression, Experiment A (calibration on 13 training countries, one-seventh of 89 or 90 rounded to the nearest integer)",
                 [f"At the 90% level: coverage by fold under (b) = {', '.join(f'{100 * v:.1f}%' for v in r6['cross_conformal_fold90'])}; coverage of single folds under (a) ranges from {100 * r6['single_split_fold90']['min']:.0f}% to {100 * r6['single_split_fold90']['max']:.0f}%.",
                  f"Under (b), {100 * r6['cross_conformal_country']['share_full']:.0f}% of the 112 countries have every year inside the 90% interval and {100 * r6['cross_conformal_country']['share_le50']:.0f}% have half of their years or fewer inside it; the mean interval width is {r6['cross_conformal_width90']:.2f} t CO2 per person."],
                 fmt={c_: "0.0" for c_ in Fd.columns})
r6c = json.load(open(OUT + "rev06c.json")); hs = r6c["hybrid_stored"]; ss = r6c["single_split90"]; cc = r6c["cross_conformal90"]
F2 = pd.DataFrame([
    {"Model and calibration set": "Ridge regression, (a) 8 training countries (200 random draws)", "Calibration countries per fold": "8", "Coverage of 90% interval (%)": 100 * ss["mean"],
     "Lowest fold (%)": 100 * r6c["single_split_fold90"]["min"], "Highest fold (%)": 100 * r6c["single_split_fold90"]["max"], "Mean width (t CO2 per person)": ss["width"],
     "Countries with every year inside (%)": None, "Countries with half of their years or fewer inside (%)": None},
    {"Model and calibration set": "Ridge regression, (b) all training countries (grouped cross-conformal)", "Calibration countries per fold": "60 or 61", "Coverage of 90% interval (%)": 100 * cc["coverage"],
     "Lowest fold (%)": 100 * min(cc["fold"]), "Highest fold (%)": 100 * max(cc["fold"]), "Mean width (t CO2 per person)": cc["width"],
     "Countries with every year inside (%)": 100 * cc["share_full"], "Countries with half of their years or fewer inside (%)": 100 * cc["share_le50"]},
    {"Model and calibration set": "Ridge regression, (c) exchangeable reference (200 random halves)", "Calibration countries per fold": "held-out countries", "Coverage of 90% interval (%)": 100 * r6c["exchangeable_reference90"],
     "Lowest fold (%)": None, "Highest fold (%)": None, "Mean width (t CO2 per person)": None, "Countries with every year inside (%)": None, "Countries with half of their years or fewer inside (%)": None},
    {"Model and calibration set": "E-PSO-KG-CBA, Monte-Carlo dropout and split-conformal calibration (stored intervals of the original run)", "Calibration countries per fold": "8", "Coverage of 90% interval (%)": 100 * hs["coverage"],
     "Lowest fold (%)": 100 * min(hs["fold"]), "Highest fold (%)": 100 * max(hs["fold"]), "Mean width (t CO2 per person)": hs["width"],
     "Countries with every year inside (%)": 100 * hs["share_full"], "Countries with half of their years or fewer inside (%)": 100 * hs["share_le50"]}])
write_sheet(wb, "F2_conformal_expC", F2, "F2. Coverage of 90% conformal prediction intervals in Experiment C (76 countries, seven covariates, folds of Experiment C)",
            [f"Under (a), coverage over the 200 draws ranges from {100 * ss['min']:.1f}% to {100 * ss['max']:.1f}%. By fold under (b): {', '.join(f'{100 * v:.1f}%' for v in cc['fold'])}. By fold for E-PSO-KG-CBA: {', '.join(f'{100 * v:.1f}%' for v in hs['fold'])}.",
             "The calibration sets of (a) have the size of the inner validation sets on which the intervals of the hybrid were calibrated (one-seventh of the training countries, rounded down)."],
            fmt={c_: "0.0" for c_ in F2.columns if c_ not in ("Model and calibration set", "Calibration countries per fold", "Mean width (t CO2 per person)")} | {"Mean width (t CO2 per person)": "0.00"},
            widths={"Model and calibration set": 60})
orig = {"Mean predictor": (7.415, 4.408, 337.8, 63.3, -0.178), "Ridge": (3.056, 1.443, 25.2, 16.3, 0.800), "kNN": (5.119, 2.729, 49.7, 39.0, 0.439), "SVR": (4.881, 2.100, 32.3, 21.3, 0.490),
        "Random forest": (4.233, 1.824, 28.7, 19.1, 0.616), "XGBoost": (4.043, 1.775, 28.0, 18.4, 0.650), "LightGBM": (3.943, 1.813, 28.4, 18.6, 0.667)}
re_ = O[O.Analysis == "All 112 countries"].set_index("Model")
oofm = np.zeros(len(y))
for tr, te in GroupKFold(5).split(X, y, iso): oofm[te] = y[tr].mean()
mm = metrics(c, np.exp(oofm))
rows = []
for m, v in orig.items():
    rr = mm if m == "Mean predictor" else re_.loc[m].to_dict()
    rows.append({"Model": m, "RMSE, original run": v[0], "RMSE, re-estimated": rr["RMSE"], "MAE, original run": v[1], "MAE, re-estimated": rr["MAE"], "MAPE, original run": v[2], "MAPE, re-estimated": rr["MAPE"],
                 "MdAPE, original run": v[3], "MdAPE, re-estimated": rr["MdAPE"], "R2, original run": v[4], "R2, re-estimated": rr["R2"]})
G = pd.DataFrame(rows)
write_sheet(wb, "G_reproduction", G, "G. Original run and re-estimation in revision, Experiment A", ["Original-run values are those of Table 3 of the article."],
            fmt={c_: ("0.000" if c_.split(",")[0] in ("RMSE", "MAE", "R2") else "0.0") for c_ in G.columns if c_ != "Model"})
H = pd.DataFrame([{"lambda": float(k.split("_")[-1]), **v} for k, v in r7.items() if k.startswith("ridge_lambda_")])
write_sheet(wb, "H_ridge_penalty", H, "H. Ridge regression: sensitivity to the penalty lambda, Experiment A", ["The article uses lambda = 1."], fmt=mfmt)
tA = pd.read_csv(OUT + "oof_experiment_A_tests_vs_reference.csv"); tA.insert(0, "Experiment", "A, 112 countries")
tA0 = pd.read_csv(OUT + "oof_experiment_A_tests_vs_reference_without_atypical.csv"); tA0.insert(0, "Experiment", "A, 105 countries without the seven atypical economies")
tC = pd.read_csv(OUT + "oof_experiment_C_tests_vs_reference.csv"); tC.insert(0, "Experiment", "C, 76 countries")
It = pd.concat([tA, tA0, tC]).rename(columns={"first": "First model", "second": "Second model", "MdAPE first": "MdAPE first (%)", "MdAPE second": "MdAPE second (%)", "delta (points)": "Delta (points)",
                                              "CI lower": "95% CI lower", "CI upper": "95% CI upper", "p cluster bootstrap": "p, cluster bootstrap", "p per-country": "p, per-country test", "p observation level": "p, observation-level test"})
It["First model"] = It["First model"].replace({"RF": "Random forest", "Mean": "Mean predictor"})
write_sheet(wb, "I_tests_stored", It, "I. Every model against ridge regression, from the stored out-of-fold estimates of the original run",
            ["A positive delta means that the model has the larger error than ridge regression. A cluster-bootstrap p of 0 means that none of the 10,000 draws had the opposite sign (p < 0.0001).",
             "Computed by rev_tests_from_oof.py (all countries) and rev09_stored_without_atypical.py (105 countries) from out/oof_experiment_A.csv and out/oof_experiment_C.csv, which oof_from_original.py writes from original_pipeline/results/."],
            fmt={"MdAPE first (%)": "0.00", "MdAPE second (%)": "0.00", "Delta (points)": "+0.00;-0.00;0.00", "95% CI lower": "+0.00;-0.00;0.00", "95% CI upper": "+0.00;-0.00;0.00", "p, cluster bootstrap": "0.0000", "p, per-country test": "0.0E+00", "p, observation-level test": "0.0E+00"},
            widths={"Experiment": 44, "First model": 20, "Second model": 14})
jA = pd.read_csv(OUT + "oof_experiment_A_rescored.csv"); jA.insert(0, "Experiment", "A")
jC = pd.read_csv(OUT + "oof_experiment_C_rescored.csv"); jC.insert(0, "Experiment", "C")
J = pd.concat([jA, jC]).rename(columns={"model": "Model", "subset": "Countries scored", "countries": "N"})
J["Model"] = J["Model"].replace({"RF": "Random forest", "Mean": "Mean predictor"})
write_sheet(wb, "J_rescored_stored", J, "J. Every model re-scored from the stored out-of-fold estimates on subsets of countries",
            ["The estimates are those of the original run; only the set of countries that is scored changes. A long filled run is a run of more than three consecutive filled years."],
            fmt=mfmt | {c_: "0.0" for c_ in J.columns if c_.startswith("share")}, widths={"Countries scored": 50, "Model": 18})
wb.save(DEST + "S2_Table.xlsx"); print("S2 Table sheets", wb.sheetnames)
