"""Shared publication-grade matplotlib style for every paper figure.

Conventions:
  * vector PDF is the manuscript format (a 300-dpi PNG is also written for
    quick previews); figure widths target the print layout: SINGLE = 90 mm,
    ONEHALF = 140 mm, DOUBLE = 190 mm;
  * Okabe-Ito colourblind-safe palette;
  * no conclusion-style titles inside the figure — interpretation lives in the
    LaTeX caption; multi-panel figures are lettered (a), (b), ...
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(os.environ.get("PAPER_FIG_DIR",
                          Path(__file__).resolve().parent.parent / "figures"))
FIG.mkdir(parents=True, exist_ok=True)

# Okabe-Ito
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}

C_CRYPTO, C_TRAD, C_STOCK = OI["vermillion"], OI["blue"], OI["blue"]
CLASS_COLOR = {"crypto": OI["vermillion"], "index": OI["blue"], "stock": OI["sky"],
               "commodity": OI["orange"], "bond": OI["green"], "fx": OI["purple"]}
MODEL_COLOR = {"ising": OI["vermillion"], "free": OI["blue"], "l2": OI["sky"],
               "l1": OI["purple"], "markov1": OI["green"], "markov2": OI["orange"],
               "base": "0.55", "gbm": OI["green"], "mlp": OI["purple"]}

SINGLE, ONEHALF, DOUBLE = 3.54, 5.51, 7.48      # inches = 90 / 140 / 190 mm
TEXT = 6.5      # \linewidth of the manuscript (article, 10pt, 1in margins).
                # savefig crops to a tight bbox, so a figure is *delivered*
                # narrower than its canvas by the label margin: size the canvas
                # so the crop lands on TEXT and \includegraphics[width=\linewidth]
                # places it 1:1, leaving label point sizes untouched.

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name: str):
    """Write docs/figures/{name}.pdf (manuscript) + .png (preview)."""
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


def letter(ax, s: str):
    """Panel letter '(a)', '(b)', ... in the top-left corner of an axes."""
    ax.text(0.02, 0.98, s, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontweight="bold")
