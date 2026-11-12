import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURF = "#1a1a1a", "#5a5a57", "#dcdbd6", "#ffffff"
GREY = "#8c8b86"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "figure.dpi": 100, "savefig.dpi": 300, "figure.facecolor": "white", "axes.facecolor": "white", "mathtext.default": "regular"})
FIGDIR = os.environ.get("KGCBA_FIGS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")) + os.sep
os.makedirs(FIGDIR, exist_ok=True)
def save(fig, name):
    """Write a 300-dpi PNG for review and a PLOS-compliant TIFF (RGB, LZW, no alpha, <=2250 px wide)."""
    png = FIGDIR + name + ".png"; fig.savefig(png, dpi=300, facecolor="white"); plt.close(fig)
    im = Image.open(png).convert("RGB")
    if im.width > 2250: im = im.resize((2250, round(im.height * 2250 / im.width)), Image.LANCZOS)
    assert im.height <= 2625, (name, im.size)
    im.save(FIGDIR + name + ".tif", compression="tiff_lzw", dpi=(300, 300)); im.save(png, dpi=(300, 300))
    return im.size
