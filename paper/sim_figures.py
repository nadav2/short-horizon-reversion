"""Figures for the trading simulation.

    uv run --active python -m paper.sim_figures  ->  docs/figures/sim_*.pdf

1. sim_equity — non-compounding equity curves (BTC/ETH 15m, rho=1) per model.
2. sim_why    — accuracy does not explain Sharpe across models; calibration
                (log-loss) does.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import CANDLES_PER_DAY, load_merged
from .models import ARLogit
from .simulate import LABEL, MODEL_ORDER, simulate
from .style import DOUBLE, MODEL_COLOR, letter, save
from .walkforward import walk_forward, window_candles

OUT = Path(__file__).resolve().parent / "out"


def equity_curves(cells=(("btc", "15m"), ("eth", "15m")), rho=1.0):
    fig, axes = plt.subplots(1, len(cells), figsize=(DOUBLE, 2.8))
    if len(cells) == 1:
        axes = [axes]
    for j, (ax, (coin, interval)) in enumerate(zip(axes, cells)):
        dts, ch, ups = load_merged(coin, interval)
        tr, te = window_candles(interval)
        res, _ = walk_forward(ch, ups, tr, te, n_lags=12)
        ref_res, _ = walk_forward(ch, ups, tr, te, n_lags=1,
                                  models=lambda: [ARLogit("free", n_lags=1)])
        ref = ref_res["free"]["probs"]
        actual = res["ising"]["actuals"].astype(int)
        ipy = CANDLES_PER_DAY[interval] * 365
        m = 0.5 + rho * (ref - 0.5)
        for s in MODEL_ORDER:
            _, eq = simulate(res[s]["probs"], actual, m, ipy)
            ax.plot(eq, color=MODEL_COLOR[s], lw=1.4 if s == "ising" else 0.9,
                    alpha=1.0 if s == "ising" else 0.85, label=LABEL[s])
        ax.axhline(1.0, color="k", lw=0.6)
        ax.set_xlabel("out-of-sample candle")
        letter(ax, f"({chr(97 + j)}) {coin.upper()} {interval}")
        if j == 0:
            ax.set_ylabel("equity (base capital = 1)")
    # one shared legend below the panels: an in-axes legend collides with the
    # panel letter at the top-left and with the equity curves everywhere else
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.4, handlelength=1.6,
               borderaxespad=0.2)
    save(fig, "sim_equity")


def why_ising(rho="1.0"):
    S = json.loads((OUT / "simulation.json").read_text())
    pts = []
    for cell, d in S["cells"].items():
        for s, m in d["by_rho"][rho].items():
            pts.append((s, m["accuracy"], m["log_loss"], m["sharpe"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.9))
    for j, (ax, xi, xlab) in enumerate([(a1, 1, "out-of-sample accuracy"),
                                        (a2, 2, "out-of-sample log-loss")]):
        for s in MODEL_ORDER:
            xs = [p[xi] for p in pts if p[0] == s]
            ys = [p[3] for p in pts if p[0] == s]
            ax.scatter(xs, ys, c=MODEL_COLOR[s], s=30, label=LABEL[s],
                       edgecolor="k" if s == "ising" else "none", linewidths=0.6,
                       zorder=3 if s == "ising" else 2)
        ax.set_xlabel(xlab)
        letter(ax, f"({chr(97 + j)})")
        if xi == 2:
            ax.invert_xaxis()   # lower log-loss (better calibration) to the right
    a1.set_ylabel(f"annualized Sharpe ($\\rho$={rho})")
    # shared legend below both panels: "upper left" collided with the (b) letter
    handles, labels = a1.get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.4, handlelength=1.0,
               borderaxespad=0.2)
    save(fig, "sim_why")


def main():
    equity_curves()
    why_ising()
    print("done")


if __name__ == "__main__":
    main()
