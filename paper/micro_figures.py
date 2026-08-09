"""Figure for the microstructure-artifact defense (paper.gap_test, paper.liquidity).

    uv run --active python -m paper.micro_figures

Output (docs/figures/): microstructure.pdf/png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .style import OI, ONEHALF, letter, plt, save

OUT = Path(__file__).resolve().parent / "out"
C_CR, C_ST = OI["vermillion"], OI["blue"]


def main():
    gap = json.loads((OUT / "gap_test.json").read_text())
    liq = json.loads((OUT / "liquidity.json").read_text())

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(ONEHALF, 2.3))

    # (a) per-asset AUC with vs without a one-bar gap
    for cls, col, z in (("stock", C_ST, 1), ("crypto", C_CR, 2)):
        sub = [r for r in gap if r["class"] == cls]
        ax.scatter([r["auc_ising"] for r in sub], [r["auc_gap_ising"] for r in sub],
                   s=5, color=col, alpha=0.55, lw=0, label=cls, zorder=z)
    lims = (0.475, 0.575)
    ax.plot(lims, lims, color="0.4", lw=0.6, ls="--")
    ax.axhline(0.5, color="0.6", lw=0.5, ls=":")
    ax.axvline(0.5, color="0.6", lw=0.5, ls=":")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("OOS AUC (no gap)")
    ax.set_ylabel("OOS AUC (one-bar gap)")
    ax.legend(fontsize=6, loc="lower right")
    letter(ax, "(a)")

    # (b) crypto AUC by liquidity quintile, with flat-bar-robust AUC overlay
    cr = [r for r in liq["assets"] if r["class"] == "crypto" and r.get("volume")]
    cr.sort(key=lambda r: r["volume"])
    qs = np.array_split(np.arange(len(cr)), 5)
    x = np.arange(5)
    auc_q = [np.mean([cr[i]["auc_ising"] for i in idx]) for idx in qs]
    nz_q = [np.mean([cr[i]["auc_nz_ising"] for i in idx
                     if cr[i]["auc_nz_ising"] is not None]) for idx in qs]
    bx.bar(x - 0.18, auc_q, 0.36, color=C_CR, label="all bars")
    bx.bar(x + 0.18, nz_q, 0.36, color=C_CR, alpha=0.45, label="flat bars excluded")
    st = [r for r in liq["assets"] if r["class"] == "stock"]
    bx.axhline(np.mean([r["auc_ising"] for r in st]), color=C_ST, lw=1.0, ls="--")
    bx.text(4.45, np.mean([r["auc_ising"] for r in st]) + 0.0012, "stocks (mean)",
            ha="right", fontsize=6, color=C_ST)
    bx.axhline(0.5, color="0.4", lw=0.7, ls=":")
    bx.set_ylim(0.49, 0.545)
    bx.set_xticks(x, ["Q1\nthinnest", "Q2", "Q3", "Q4", "Q5\nmost liquid"], fontsize=6)
    bx.set_xlabel("crypto volume quintile")
    bx.set_ylabel("mean OOS AUC")
    bx.legend(fontsize=6, loc="upper right")
    letter(bx, "(b)")

    fig.tight_layout()
    save(fig, "microstructure")


if __name__ == "__main__":
    main()
