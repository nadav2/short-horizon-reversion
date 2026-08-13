"""Figure for the forced-flow identification (docs/paper.tex sec:forced).

  m_forced.pdf  (a) next-bar sign-flip rate by the prior bar's forced-order
                class (none / below-median / above-median move-side forced
                notional), USDT-margined tape, per coin, binomial 95% CIs;
                (b) the matched contrast per coin-tape cell: above-median
                forced minus no-forced flip rate within size x imbalance
                cells, moving-block bootstrap CIs (filled), the below-median
                class (small gray), and the opposed-side placebo (open).

Reads out/liq_test.json only; no recomputation.

    uv run --active python -m paper.liq_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .style import OI, TEXT, letter, plt, save

OUT = Path(__file__).resolve().parent / "out"
COINS = ["btc", "eth", "sol", "xrp"]
COIN_COLOR = {"btc": OI["vermillion"], "eth": OI["blue"],
              "sol": OI["green"], "xrp": OI["purple"]}
CLASS_LABELS = ["none", "forced $\\leq$ med", "forced $>$ med"]


def main():
    d = json.loads((OUT / "liq_test.json").read_text())
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(TEXT, 2.8), gridspec_kw={"width_ratios": [1.0, 1.15]})

    # (a) dose-response on the USDT-margined tape
    x = np.arange(3)
    for k, coin in enumerate(COINS):
        cells = d[coin]["um"]["flip_by_forced_class_raw"]
        r = np.array([c["rate"] for c in cells])
        n = np.array([c["n"] for c in cells])
        se = np.sqrt(r * (1 - r) / n)
        ax_a.errorbar(x + (k - 1.5) * 0.06, r, yerr=1.96 * se,
                      color=COIN_COLOR[coin], marker="o", ms=3.5,
                      capsize=2, lw=1.2, label=coin.upper())
    ax_a.axhline(0.5, color="0.6", lw=0.6, ls=":")
    ax_a.set_xticks(x, CLASS_LABELS)
    ax_a.set_xlim(-0.3, 2.3)
    ax_a.set_ylabel("next-bar sign-flip rate")
    ax_a.set_xlabel("forced orders in the prior flow-driven bar")
    ax_a.legend(loc="lower right", ncol=2, columnspacing=1.0,
                handletextpad=0.5)
    letter(ax_a, "(a)")

    # (b) forest of matched contrasts, all eight coin-tape cells
    XLIM = (-0.085, 0.14)
    rows = [(coin, tape) for coin in COINS for tape in ("um", "cm")]
    ys = np.arange(len(rows))[::-1]
    for y, (coin, tape) in zip(ys, rows):
        r = d[coin][tape]
        hi = r["matched_gradient"]["delta_high"]
        lo = r["matched_gradient"]["delta_low"]
        pl = r["placebo_opposed_liq"]
        ci = hi["ci"]
        col = COIN_COLOR[coin]
        ax_b.plot(ci, [y, y], color=col, lw=1.2, solid_capstyle="butt")
        ax_b.plot(hi["value"], y, "o", color=col, ms=4.5, zorder=3)
        ax_b.plot(lo["value"], y, "o", color="0.55", ms=2.5, zorder=2)
        if pl < XLIM[0]:  # off-scale noisy placebo, annotated at the edge
            ax_b.plot(XLIM[0] + 0.004, y, "o", mfc="none", mec=col,
                      ms=4.5, mew=0.9, zorder=3, clip_on=False)
            ax_b.annotate(f"{pl:+.2f}", (XLIM[0] + 0.009, y),
                          fontsize=6, va="center", color="0.35")
        else:
            ax_b.plot(pl, y, "o", mfc="none", mec=col, ms=4.5, mew=0.9,
                      zorder=3)
    ax_b.axvline(0, color="0.3", lw=0.7)
    ax_b.set_xlim(*XLIM)
    ax_b.set_yticks(ys, [f"{c.upper()}·{t.upper()}" for c, t in rows])
    ax_b.set_xlabel("matched flip-rate difference vs. no-forced bars")
    ax_b.set_ylim(-0.6, len(rows) - 0.4)
    # legend from proxy artists (colors vary per row), above the panel
    proxies = [
        plt.Line2D([], [], marker="o", color="0.25", ms=4.5, lw=1.2,
                   label="forced $>$ med (CI)"),
        plt.Line2D([], [], marker="o", color="0.55", ms=2.5, lw=0,
                   label="forced $\\leq$ med"),
        plt.Line2D([], [], marker="o", mfc="none", mec="0.25", ms=4.5,
                   lw=0, label="placebo (opposed)"),
    ]
    ax_b.legend(handles=proxies, loc="lower left", ncol=3,
                bbox_to_anchor=(-0.02, 1.0), columnspacing=0.9,
                handletextpad=0.4, fontsize=6.5)
    letter(ax_b, "(b)")

    fig.tight_layout(w_pad=1.6)
    save(fig, "m_forced")


if __name__ == "__main__":
    main()
