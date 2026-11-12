import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import textwrap

def canvas(w, h):
    fig = plt.figure(figsize=(w, h)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); return fig, ax

def box(ax, x, y, w, h, head, lines=(), fc="#eef3fa", ec="#33506e", head_size=7.6, size=6.9, lw=0.9, head_color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=ec, lw=lw))
    n = len(lines); lh = 0.135
    total = (0.16 if head else 0) + n * lh
    top = y + h / 2 + total / 2
    if head:
        ax.text(x + w / 2, top - 0.075, head, ha="center", va="center", fontsize=head_size, fontweight="bold", color=head_color)
        top -= 0.17
    for i, l in enumerate(lines):
        ax.text(x + w / 2, top - (i + 0.5) * lh, l, ha="center", va="center", fontsize=size, color=INK)

def arrow(ax, p0, p1, color="#33506e"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=color, shrinkA=0, shrinkB=0))

def stage(ax, y, h, num, label):
    ax.add_patch(FancyBboxPatch((0.08, y), 1.02, h, boxstyle="round,pad=0,rounding_size=0.06", fc="#33506e", ec="#33506e", lw=0))
    ax.text(0.59, y + h / 2 + 0.16, str(num), ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    for i, l in enumerate(label.split("\n")):
        ax.text(0.59, y + h / 2 - 0.03 - i * 0.135, l, ha="center", va="center", fontsize=7.0, color="white")

# ----------------------------------------------------------------- Fig 1: workflow
Wd, Ht = 7.5, 8.34
fig, ax = canvas(Wd, Ht)
X0, X1 = 1.22, 7.42; FW = X1 - X0
LB, LO, LG, LN = "#eef3fa", "#fdf0e8", "#e9f6f0", "#f3f2ee"
rows = {}
# 1 problem
y, h = 7.46, 0.80; rows[1] = (y, h); stage(ax, y, h, 1, "Research\nquestion")
box(ax, X0, y, FW, h, "Estimating national CO$_2$ where the emission record cannot be used",
    ["Q1  How accurately can per-capita fossil CO$_2$ be estimated for a country that is absent from training?",
     "Q2  Do deep and hybrid networks outperform a linear model motivated by the Kaya identity?",
     "Q3  Is error reduced more by better models or by better covariate data?     Q4  Are the answers robust?"], fc=LN, ec="#6b6a65")
# 2 data
y, h = 6.36, 0.82; rows[2] = (y, h); stage(ax, y, h, 2, "Data")
w = (FW - 2 * 0.28) / 3
box(ax, X0, y, w, h, "Open sources", ["Our World in Data compilation of", "Global Carbon Budget, Energy", "Institute and Maddison data"], fc=LB)
box(ax, X0 + w + 0.28, y, w, h, "Balanced panel", ["112 countries × 34 years", "(1990–2023) = 3,808 country-years;", "target and six covariates"], fc=LB)
box(ax, X0 + 2 * (w + 0.28), y, w, h, "Modelling samples", ["8-year covariate window → next", "value of ln(CO$_2$ per person)", "27 per country = 3,024 samples"], fc=LB)
arrow(ax, (X0 + w, y + h / 2), (X0 + w + 0.28, y + h / 2)); arrow(ax, (X0 + 2 * w + 0.28, y + h / 2), (X0 + 2 * w + 0.56, y + h / 2))
# 3 experiments
y, h = 5.14, 0.94; rows[3] = (y, h); stage(ax, y, h, 3, "Experiments")
box(ax, X0, y, w, h, "Experiment A (main)", ["Benchmark of all models", "112 countries, six covariates", "answers Q1 and Q2"], fc=LO, ec="#a8501f")
box(ax, X0 + w + 0.28, y, w, h, "Experiment C", ["Adds low-carbon share of energy", "76 countries, seven covariates", "answers Q3"], fc=LO, ec="#a8501f")
box(ax, X0 + 2 * (w + 0.28), y, w, h, "Experiment B (supporting)", ["Conventional temporal forecasting", "against naive persistence", "reported in S2 Text"], fc=LN, ec="#6b6a65")
# 4 models
y, h = 3.78, 1.08; rows[4] = (y, h); stage(ax, y, h, 4, "Models\n(13 + ensemble)")
g = 0.12; w5 = (FW - 4 * g) / 5
fam = [("Baseline", ["Mean predictor", "(training-set mean", "of the target)"], LN, "#6b6a65"),
       ("Linear", ["Ridge regression", "(log-linear,", "Kaya-motivated)"], LO, "#a8501f"),
       ("Classical ML", ["kNN, SVR,", "random forest,", "XGBoost, LightGBM"], LB, "#33506e"),
       ("Deep networks", ["LSTM, GRU,", "CNN-LSTM,", "CNN-BiLSTM-Attn"], LG, "#1f7a58"),
       ("Kaya-guided", ["hybrid networks:", "KG-CBA, PSO-KG-CBA", "(+ exploratory", "ensemble)"], LO, "#a8501f")]
for i, (hd, ls, fc, ec) in enumerate(fam):
    box(ax, X0 + i * (w5 + g), y, w5, h, hd, ls, fc=fc, ec=ec, size=6.6, head_size=7.3)
# 5 validation
y, h = 2.54, 0.96; rows[5] = (y, h); stage(ax, y, h, 5, "Validation and\ncomparison")
g = 0.2; w4 = (FW - 3 * g) / 4
val = [("Leave-country-out", ["5-fold cross-validation", "with countries as groups;", "inner country split"]),
       ("Accuracy", ["Median APE (primary),", "MAPE, MAE, RMSE, R²", "on held-out countries"]),
       ("Clustered inference", ["Per-country Wilcoxon test;", "cluster bootstrap over", "countries (10,000 draws)"]),
       ("Selection rule", ["Simplest model that is not", "significantly less accurate", "than the best"])]
for i, (hd, ls) in enumerate(val):
    box(ax, X0 + i * (w4 + g), y, w4, h, hd, ls, fc=LB, size=6.6, head_size=7.3)
    if i < 3: arrow(ax, (X0 + i * (w4 + g) + w4, y + h / 2), (X0 + (i + 1) * (w4 + g), y + h / 2))
# 6 diagnostics
y, h = 1.30, 0.96; rows[6] = (y, h); stage(ax, y, h, 6, "Diagnostics and\nrobustness")
dia = [("Error anatomy", ["Squared-error share", "by country; atypical", "energy systems"]),
       ("Sensitivity", ["Missing-data treatment;", "atypical economies", "excluded; covariate ablation"]),
       ("Uncertainty", ["Monte-Carlo dropout and", "split-conformal intervals;", "held-out coverage"]),
       ("Explainability", ["SHAP, permutation and", "drop-one-covariate", "importance"])]
for i, (hd, ls) in enumerate(dia):
    box(ax, X0 + i * (w4 + g), y, w4, h, hd, ls, fc=LG, ec="#1f7a58", size=6.6, head_size=7.3)
# 7 outcome
y, h = 0.08, 0.96; rows[7] = (y, h); stage(ax, y, h, 7, "Outcome and\nimplications")
out = [("Model", ["Ridge regression retained as the", "reference estimator; no deep or", "hybrid model is significantly better"]),
       ("Data", ["One energy-mix covariate halves", "median error: data availability,", "not model capacity, limits accuracy"]),
       ("Practice", ["Roadmap for inventory gap-filling:", "energy-mix statistics, grouped", "validation, clustered inference"])]
for i, (hd, ls) in enumerate(out):
    box(ax, X0 + i * (w + 0.28), y, w, h, hd, ls, fc=LN, ec="#6b6a65")
# vertical connectors between rows
cx = X0 + FW / 2
for a, b in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]:
    arrow(ax, (cx, rows[a][0]), (cx, rows[b][0] + rows[b][1]))
print("fig1", save(fig, "Fig1"))

# ----------------------------------------------------------------- Fig 3: architecture
Wd, Ht = 7.5, 2.95
fig, ax = canvas(Wd, Ht)
bw, bh = 1.22, 0.74
ytop, ybot = 1.92, 0.36
box(ax, 0.12, 1.02, 1.12, 0.92, "Input window", ["8 years × F", "standardised", "covariates"], fc=LN, ec="#6b6a65")
box(ax, 1.72, ytop, 2.4, bh, "Kaya linear head", ["one dense unit on the most recent year;", "initialised at the ridge solution"], fc=LO, ec="#a8501f")
xs = [1.72, 3.14, 4.56, 5.98]; w_ = 1.3
labs = [("Conv1D × 2", ["local temporal", "features"]), ("BiLSTM", ["forward and", "backward states"]), ("Attention pooling", ["additive weights", "over the 8 years"]), ("Dense (32)", ["dropout; output", "initialised near 0"])]
for x, (hd, ls) in zip(xs, labs):
    box(ax, x, ybot, w_, bh, hd, ls, fc=LB, size=6.6, head_size=7.3)
for i in range(3): arrow(ax, (xs[i] + w_, ybot + bh / 2), (xs[i + 1], ybot + bh / 2))
box(ax, 5.98, ytop, w_, bh, "Sum", ["predicted", "ln(CO$_2$ per person)"], fc=LG, ec="#1f7a58", size=6.6)
arrow(ax, (1.24, 1.62), (1.72, ytop + bh / 2 - 0.1)); arrow(ax, (1.24, 1.34), (1.72, ybot + bh / 2 + 0.1))
arrow(ax, (1.72 + 2.4, ytop + bh / 2), (5.98, ytop + bh / 2)); arrow(ax, (5.98 + w_ / 2, ybot + bh), (5.98 + w_ / 2, ytop))
ax.text(5.05, ytop + bh / 2 + 0.11, "log-linear term", ha="center", va="bottom", fontsize=6.8, color=MUTED)
ax.text(5.98 + w_ / 2 + 0.07, (ybot + bh + ytop) / 2, "non-linear\nresidual", ha="left", va="center", fontsize=6.8, color=MUTED)
ax.text(4.5, 0.13, "Deep residual branch", ha="center", va="center", fontsize=7.2, color=MUTED, style="italic")
print("fig3", save(fig, "Fig3"))
