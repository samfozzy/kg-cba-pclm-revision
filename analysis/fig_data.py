import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from figstyle import *
from kgcba_common import *
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
p = build_panel(); q = add_low_carbon(p)
# ------------------------------------------------ Fig 2: time series of the target and all covariates
panels = [("co2_per_capita", "CO$_2$ per capita (target)", "t CO$_2$ per person", True, p, 1),
          ("energy_per_capita", "Energy per capita", "MWh per person", True, p, 1e-3),
          ("gdp_pc", "GDP per capita", "thousand int-\\$ per person", True, p, 1e-3),
          ("population", "Population", "million", True, p, 1e-6),
          ("coal_share", "Coal share", "fraction of CO$_2$", False, p, 1),
          ("oil_share", "Oil share", "fraction of CO$_2$", False, p, 1),
          ("gas_share", "Gas share", "fraction of CO$_2$", False, p, 1),
          ("lc_share", "Low-carbon share", "fraction of primary energy", False, q, 1)]
fig, axes = plt.subplots(2, 4, figsize=(7.5, 4.3)); letters = "abcdefgh"
for ax, (v, title, unit, logy, d, sc), L in zip(axes.ravel(), panels, letters):
    g = d.groupby("year")[v]; yrs = np.array(sorted(d.year.unique()))
    q10, q25, q50, q75, q90 = [g.quantile(x).values * sc for x in (.1, .25, .5, .75, .9)]
    ax.fill_between(yrs, q10, q90, color=BLUE, alpha=0.13, lw=0); ax.fill_between(yrs, q25, q75, color=BLUE, alpha=0.30, lw=0)
    ax.plot(yrs, q50, color=BLUE, lw=1.6)
    if logy:
        ax.set_yscale("log"); ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5))); ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}")); ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_ylim(0, 1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticks([1990, 2000, 2010, 2020]); ax.set_xlim(1990, 2023)
    ax.set_title(f"({L}) {title}", loc="left", fontsize=7.8, fontweight="bold", pad=4)
    ax.set_ylabel(unit, fontsize=7.2, labelpad=2); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(length=2.5)
ax = axes[0, 0]
ax.plot([], [], color=BLUE, lw=1.6, label="Median"); ax.fill_between([], [], [], color=BLUE, alpha=0.30, lw=0, label="25th–75th percentile"); ax.fill_between([], [], [], color=BLUE, alpha=0.13, lw=0, label="10th–90th percentile")
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.045, 1, 1), w_pad=0.9, h_pad=1.2)
print("fig2", save(fig, "Fig2"))
# ------------------------------------------------ S Fig: correlation matrix (diverging, neutral midpoint)
from matplotlib.colors import LinearSegmentedColormap
cc = pd.DataFrame({"ln CO$_2$ pc": np.log(p.co2_per_capita), "log energy pc": p.e, "log GDP pc": p.g, "log population": p.p, "Coal share": p.coal_share, "Oil share": p.oil_share, "Gas share": p.gas_share}).corr()
cm = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f2f1ec", "#2a78d6"])
fig, ax = plt.subplots(figsize=(4.6, 3.9))
im = ax.imshow(cc.values, vmin=-1, vmax=1, cmap=cm)
ax.set_xticks(range(7)); ax.set_yticks(range(7)); ax.set_xticklabels(cc.columns, rotation=40, ha="right"); ax.set_yticklabels(cc.index)
for i in range(7):
    for j in range(7):
        ax.text(j, i, f"{cc.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(cc.values[i, j]) > 0.75 else INK)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0); cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.outline.set_visible(False); cb.ax.tick_params(length=2, labelsize=7); cb.set_label("Pearson r", fontsize=7.5)
fig.tight_layout(); print("S corr", save(fig, "S2_Fig"))
