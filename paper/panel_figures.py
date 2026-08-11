"""The wrapper-panel money plot: does an instrument's 15m signal track its
underlying's, or its venue's?

One point per RTH-matched leg. NAV-linked wrappers (filled) are plotted
against their underlying's RTH AUC; instruments merely *correlated* with an
underlying (open) are plotted at the same x, so vertical distance from the
identity line reads as failure to inherit. The shaded band is the RTH-only
187-stock universe (mean +/- 2 SD): where a listed instrument with no live
underlying signal should sit.

    uv run python -m paper.natural_figures
"""

from __future__ import annotations

import json
from pathlib import Path

from .style import OI, SINGLE, letter, plt, save

OUT = Path(__file__).resolve().parent / "out"


def main():
    d = json.loads((OUT / "natural.json").read_text())
    panel = d["panel"]
    ref = panel["reference_rth_stocks"]
    rth = {r["asset"]: r for r in d["rows"] if r["rth"] and not r["gap"]}
    und = {r["experiment"]: r for r in d["rows"]
           if r["rth"] and not r["gap"] and r["role"] == "underlying"}

    fig, ax = plt.subplots(figsize=(SINGLE * 1.25, SINGLE * 1.1))

    # stock-universe band
    m, s = ref["mean"], ref["sd"]
    ax.axhspan(m - 2 * s, m + 2 * s, color="0.85", alpha=0.55, zorder=0)
    ax.axhline(m, color="0.55", lw=0.7, ls="--", zorder=1)
    lo, hi = 0.492, 0.542
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=0.8, zorder=1)

    # NAV wrappers
    for p in panel["pairs"]:
        thin = p["wrapper"] in ("pplt", "pall")
        ax.scatter(p["auc_underlying"], p["auc_wrapper"], s=42,
                   color=OI["vermillion"], edgecolor="k", linewidths=0.6,
                   marker="s" if thin else "o", zorder=3)
        OFF = {"ibit": (0.0012, 0.0022), "etha": (-0.0062, 0.0012),
               "fbtc": (0.0015, -0.0012), "feth": (-0.0068, -0.0028),
               "gld": (0.0012, 0.0015), "slv": (-0.0060, 0.0012),
               "pplt": (0.0012, 0.0015), "pall": (-0.0062, 0.0015)}
        dx, dy = OFF.get(p["wrapper"], (0.0012, 0.0015))
        ax.annotate(p["wrapper"].upper(), (p["auc_underlying"], p["auc_wrapper"]),
                    xytext=(p["auc_underlying"] + dx, p["auc_wrapper"] + dy), fontsize=6)

    # correlated legs at their underlying's x
    for r in d["rows"]:
        if not (r["rth"] and not r["gap"] and r["role"] == "correlated"):
            continue
        u = und[r["experiment"]]
        ax.scatter(u["auc_ising"], r["auc_ising"], s=38, facecolor="none",
                   edgecolor=OI["blue"], linewidths=1.1, zorder=3)
        OFF2 = {"coin": (0.0015, -0.0005), "mstr": (0.0015, -0.0005),
                "gdx": (0.0015, -0.0022), "nem": (0.0015, 0.0008)}
        dx, dy = OFF2.get(r["asset"], (0.0012, 0.0012))
        ax.annotate(r["asset"].upper(), (u["auc_ising"], r["auc_ising"]),
                    xytext=(u["auc_ising"] + dx, r["auc_ising"] + dy),
                    fontsize=6, color=OI["blue"])

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("underlying RTH out-of-sample AUC")
    ax.set_ylabel("listed instrument RTH AUC")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=OI["vermillion"],
               markeredgecolor="k", label="NAV-linked wrapper"),
        Line2D([], [], marker="s", ls="", color=OI["vermillion"],
               markeredgecolor="k", label="thin wrapper (PPLT, PALL)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor=OI["blue"], label="correlated, not NAV-linked"),
        Patch(facecolor="0.85", label="187 RTH stocks (mean $\\pm$ 2 SD)"),
    ], loc="upper left", fontsize=6)
    save(fig, "wrapper_panel")


if __name__ == "__main__":
    main()
