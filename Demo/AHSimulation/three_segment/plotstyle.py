"""Matplotlib style for the report figures (reference data-viz palette, light surface)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
# categorical slots in fixed order: blue, orange, aqua, yellow, magenta
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
STOCK, NEW, THIRD, FOURTH = C[0], C[1], C[2], C[3]
GOOD_BAND = "#1baf7a"


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "semibold", "axes.labelsize": 9.5, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "grid.linestyle": "-", "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2, "font.size": 9.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    })


def note(ax, text, xy=(0.01, 0.99), **kw):
    ax.text(*xy, text, transform=ax.transAxes, va="top", ha="left", fontsize=8, color=INK2, **kw)
