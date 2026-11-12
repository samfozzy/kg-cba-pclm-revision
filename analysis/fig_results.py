import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json
from figstyle import *
from PIL import Image
OUT = os.environ.get("KGCBA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")) + os.sep
# ------------------------------------------------ Fig 4: Experiment A, all models (values of Table 3 in the manuscript)
T = [("Mean predictor", "base", 7.415, 63.3), ("Ridge regression", "kaya", 3.056, 16.3), ("kNN", "ml", 5.119, 39.0), ("SVR", "ml", 4.881, 21.3), ("Random forest", "ml", 4.233, 19.1),
     ("XGBoost", "ml", 4.043, 18.4), ("LightGBM", "ml", 3.943, 18.6), ("LSTM", "deep", 3.815, 18.2), ("GRU", "deep", 3.629, 18.5), ("CNN-LSTM", "deep", 3.773, 20.8),
     ("CNN-BiLSTM-Attn", "deep", 3.812, 20.8), ("KG-CBA", "kaya", 4.595, 15.7), ("PSO-KG-CBA", "kaya", 4.748, 16.7), ("Ensemble (exploratory)", "base", 3.557, 15.1)]
col = {"base": GREY, "kaya": ORANGE, "ml": BLUE, "deep": AQUA}
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.9), sharey=True)
ypos = np.arange(len(T))[::-1]
for ax, idx, lab, lim, ref in ((axes[0], 3, "Median absolute percentage error (%)", (14, 23), 16.3), (axes[1], 2, "Root-mean-square error (t CO$_2$ per person)", (2.8, 5.4), 3.056)):
    ax.axvline(ref, color=ORANGE, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for (name, fam, rmse, md), yy in zip(T, ypos):
        v = (name, fam, rmse, md)[idx]
        if v > lim[1]:
            ax.annotate("", xy=(lim[1] - 0.02 * (lim[1] - lim[0]), yy), xytext=(lim[1] - 0.12 * (lim[1] - lim[0]), yy), arrowprops=dict(arrowstyle="-|>", color=col[fam], lw=1.2))
            ax.text(lim[1] - 0.14 * (lim[1] - lim[0]), yy, f"{v:.1f}" if idx == 3 else f"{v:.2f}", ha="right", va="center", fontsize=7.3, color=INK)
        else:
            ax.plot([lim[0], v], [yy, yy], color=GRID, lw=1.0, zorder=1); ax.scatter([v], [yy], s=34, color=col[fam], zorder=3, edgecolor="white", linewidth=0.8)
            ax.text(v + 0.018 * (lim[1] - lim[0]), yy, f"{v:.1f}" if idx == 3 else f"{v:.2f}", ha="left", va="center", fontsize=7.3, color=INK, zorder=4, bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xlim(*lim); ax.set_xlabel(lab); ax.set_yticks(ypos); ax.set_yticklabels([t[0] for t in T]); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.7, len(T) - 0.3)
axes[0].set_title("(a)", loc="left", fontweight="bold"); axes[1].set_title("(b)", loc="left", fontweight="bold")
axes[0].text(16.3, len(T) - 0.42, "ridge", color=ORANGE, fontsize=7, ha="center", va="bottom"); axes[1].text(3.056, len(T) - 0.42, "ridge", color=ORANGE, fontsize=7, ha="center", va="bottom")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", color=col[k], markersize=6, label=l) for k, l in (("kaya", "Kaya-informed (ridge, hybrids)"), ("ml", "Classical machine learning"), ("deep", "Unguided deep networks"), ("base", "Naive baseline / exploratory ensemble"))]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, -0.01), handletextpad=0.2, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.055, 1, 1), w_pad=1.5); print("fig4", save(fig, "Fig4"))
# ------------------------------------------------ Fig 5: error concentration (ridge) and effect of excluding atypical economies
r4 = json.load(open(OUT + "rev04.json")); S = pd.read_csv(OUT + "outlier_sensitivity.csv")
ATY = ["QAT", "KWT", "SGP", "ISL", "NOR", "LUX", "SAU"]
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axes[0]; top = r4["ridge_sse_top"][:12]; xs = np.arange(len(top))
ax.bar(xs, [v for _, v in top], color=[ORANGE if k in ATY else BLUE for k, _ in top], width=0.72)
for x, (k, v) in zip(xs, top): ax.text(x, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=6.8, color=INK)
ax.set_xticks(xs); ax.set_xticklabels([k for k, _ in top], rotation=45, ha="right"); ax.set_ylabel("Share of total squared error (%)"); ax.set_ylim(0, 27); ax.set_yticks([0, 5, 10, 15, 20, 25])
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_title("(a)", loc="left", fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ORANGE, label="Seven atypical economies (77.9% in total)"), Patch(color=BLUE, label="Other countries")], frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.03), fontsize=7.2)
ax = axes[1]; order = ["Ridge", "LightGBM", "XGBoost", "Random forest", "SVR", "kNN"]; yy = np.arange(len(order))[::-1]
for name, y_ in zip(order, yy):
    a = S[(S.model == name) & (S.scope == "All 112 countries")].RMSE.iloc[0]; b = S[(S.model == name) & S.scope.str.startswith("Scored without")].RMSE.iloc[0]
    ax.plot([b, a], [y_, y_], color=GRID, lw=2, zorder=1); ax.scatter([a], [y_], s=34, color=GREY, zorder=3, edgecolor="white", linewidth=0.8); ax.scatter([b], [y_], s=34, color=BLUE, zorder=3, edgecolor="white", linewidth=0.8)
    ax.text(a + 0.1, y_, f"{a:.2f}", ha="left", va="center", fontsize=7.2); ax.text(b - 0.1, y_, f"{b:.2f}", ha="right", va="center", fontsize=7.2)
ax.set_yticks(yy); ax.set_yticklabels(order); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); ax.set_xlim(0.7, 5.9); ax.set_ylim(-0.6, len(order) - 0.2)
ax.set_xlabel("Root-mean-square error (t CO$_2$ per person)"); ax.text(-0.42, 1.045, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=8.5)
ax.scatter([], [], s=34, color=BLUE, label="Seven atypical economies excluded"); ax.scatter([], [], s=34, color=GREY, label="All 112 countries"); ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.2, handletextpad=0.2, columnspacing=1.0)
fig.tight_layout(w_pad=1.6); print("fig5", save(fig, "Fig5"))
# ------------------------------------------------ Figures retained from the original submission: strip in-image titles, resize, convert
SRC = os.environ.get("KGCBA_ORIGINAL_FIGS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "original_figures")) + os.sep
def ink_rows(im):
    a = np.asarray(im.convert("L")); return (a < 245).sum(axis=1)
def strip_title(im, max_frac=0.2, min_gap=10):
    """Remove the in-image title: find the first inked block, then the first band of >= min_gap blank rows
    below it, and crop just under the title (never inside it)."""
    ink = ink_rows(im); H = len(ink); y = int(np.where(ink > 0)[0][0]); run = 0; title_end = y
    while y < H:
        if ink[y] == 0:
            run += 1
            if run == 1: title_end = y
            if run >= min_gap: break
        else: run = 0
        y += 1
    while y < H and ink[y] == 0: y += 1
    assert y < H * max_frac, (y, H)
    return im.crop((0, max(title_end, y - 14), im.width, im.height))
jobs = [("image9.png", "S1_Fig", False), ("image10.png", "S3_Fig", 2), ("image11.png", "Fig6", True), ("image12.png", "Fig7", False), ("image13.png", "Fig8", False),
        ("image6.png", "S4_Fig", True), ("image7.png", "S5_Fig", False), ("image14.png", "S6_Fig", True), ("image15.png", "S2Text_FigA", True)]
for src, name, strip in jobs:
    im = Image.open(SRC + src).convert("RGB")
    if strip: im = strip_title(im, min_gap=10 if strip is True else strip)
    if im.width > 2250: im = im.resize((2250, round(im.height * 2250 / im.width)), Image.LANCZOS)
    assert 789 <= im.width <= 2250 and im.height <= 2625, (name, im.size)
    im.save(FIGDIR + name + ".tif", compression="tiff_lzw", dpi=(300, 300)); im.save(FIGDIR + name + ".png", dpi=(300, 300)); print(name, im.size)
