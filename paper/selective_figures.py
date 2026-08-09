"""Figures for the selective-prediction / break-even analysis (paper.selective).

    uv run --active python -m paper.selective_figures

Outputs (docs/figures/): selective.pdf/png, selective_deciles.pdf/png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .style import OI, ONEHALF, letter, plt, save

OUT = Path(__file__).resolve().parent / "out"

C_IS, C_FR = OI["vermillion"], OI["blue"]
COST_LO = 5.0           # cheapest realistic spot round-trip (bp)


def fig_selective(sel):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(ONEHALF, 2.3))

    for cell, ls in (("btc-15m", "-"), ("eth-15m", "--")):
        cdat = sel["focal"][cell]["models"]
        for m, col in (("ising", C_IS), ("free", C_FR)):
            cur = [c for c in cdat[m]["curve"] if c["coverage"] > 0.02]
            cov = np.array([c["coverage"] for c in cur]) * 100
            acc = np.array([c["accuracy"] for c in cur]) * 100
            ax.plot(cov, acc, ls, color=col, lw=1.1,
                    label=f"{cell.split('-')[0].upper()} {('Ising' if m=='ising' else 'free logit')}")
            if m == "ising":
                lo = np.array([c["acc_ci"][0] for c in cur]) * 100
                hi = np.array([c["acc_ci"][1] for c in cur]) * 100
                ax.fill_between(cov, lo, hi, color=col, alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.set_xticks([100, 30, 10, 3], ["100", "30", "10", "3"])
    ax.invert_xaxis()
    ax.axhline(50, color="0.4", lw=0.7, ls=":")
    ax.set_xlabel("coverage: % of candles traded")
    ax.set_ylabel("directional accuracy (%)")
    ax.legend(loc="upper left", fontsize=6, bbox_to_anchor=(0.0, 0.93))
    letter(ax, "(a)")

    for cell, ls in (("btc-15m", "-"), ("eth-15m", "--")):
        cur = [c for c in sel["focal"][cell]["models"]["ising"]["curve"]
               if c["coverage"] > 0.02]
        cov = np.array([c["coverage"] for c in cur]) * 100
        e = np.array([c["edge_bp"] for c in cur])
        lo = np.array([c["edge_ci"][0] for c in cur])
        hi = np.array([c["edge_ci"][1] for c in cur])
        bx.plot(cov, e, ls, color=C_IS, lw=1.1, label=cell.split("-")[0].upper())
        bx.fill_between(cov, lo, hi, color=C_IS, alpha=0.15, lw=0)
    ylim = 6.4
    bx.set_xscale("log")
    bx.set_xticks([100, 30, 10, 3], ["100", "30", "10", "3"])
    bx.invert_xaxis()
    bx.set_ylim(-1.2, ylim)
    bx.axhline(0, color="0.4", lw=0.7, ls=":")
    bx.axhspan(COST_LO, ylim, color="0.55", alpha=0.25, lw=0)
    bx.text(0.97, 0.965, "spot round-trip costs:\nmaker $\\geq$5 bp, taker 10-20 bp",
            transform=bx.transAxes, ha="right", va="top", fontsize=6, color="0.25")
    bx.set_xlabel("coverage: % of candles traded")
    bx.set_ylabel("gross edge per trade (bp of notional)")
    bx.legend(loc="center left", fontsize=6)
    letter(bx, "(b)")

    fig.tight_layout()
    save(fig, "selective")


def fig_deciles(sel):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(ONEHALF, 2.3))

    for m, col, lab in (("ising", C_IS, "Ising"), ("free", C_FR, "free logit")):
        for cell, ls, mk in (("btc-15m", "-", "o"), ("eth-15m", "--", "s")):
            d = sel["focal"][cell]["models"][m]["deciles"]
            ax.plot([x["decile"] for x in d], [x["accuracy"] * 100 for x in d],
                    ls, marker=mk, ms=2.5, color=col, lw=1.0,
                    label=f"{cell.split('-')[0].upper()} {lab}")
    ax.axhline(50, color="0.4", lw=0.7, ls=":")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("model-confidence decile (10 = most confident)")
    ax.set_ylabel("directional accuracy (%)")
    ax.legend(fontsize=6, loc="upper left")
    letter(ax, "(a)")

    ws = sel["wide_summary"]
    taus = ("tau0", "tau0.02")
    xs = np.arange(len(taus))
    w = 0.32
    for off, cls, col in ((-w / 2, "crypto", C_IS), (w / 2, "stock", C_FR)):
        vals = [ws[cls][f"ising_{t}"]["median_edge_bp"] for t in taus]
        bx.bar(xs + off, vals, w, color=col, label=f"{cls} (median)")
    bx.axhline(0, color="0.3", lw=0.7)
    bx.set_xticks(xs, ["$\\tau=0$\n(all candles)", "$\\tau=0.02$"])
    bx.set_ylabel("gross edge per trade (bp of notional)")
    bx.legend(fontsize=6, loc="upper center")
    bx.text(0.5, 0.72, "370 assets; spot round-trip\ncosts $\\geq$5-20 bp",
            transform=bx.transAxes, ha="center", va="top", fontsize=6, color="0.25")
    letter(bx, "(b)")

    fig.tight_layout()
    save(fig, "selective_deciles")


def main():
    sel = json.loads((OUT / "selective.json").read_text())
    fig_selective(sel)
    fig_deciles(sel)


if __name__ == "__main__":
    main()
