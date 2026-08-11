"""AUC vs kernel length N, from out/nbaseline.json.

The visual companion to the one-lag baseline (Table nbaseline): betting
against the previous candle captures most of the crypto detection, the
kernel's increment is complete by N ~ 3, and stocks sit at no-skill at
every N.

    uv run --active python -m paper.nbaseline_figure  ->  docs/figures/nbaseline.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import C_CRYPTO, C_STOCK, SINGLE, save

OUT = Path(__file__).resolve().parent / "out"


def main():
    d = json.loads((OUT / "nbaseline.json").read_text())
    ns = [str(n) for n in d["config"]["lag_ns"]]
    x = np.arange(len(ns))

    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    for cls, color, label in ((d["summary"]["crypto"], C_CRYPTO, "crypto (183)"),
                              (d["summary"]["stock"], C_STOCK, "stocks/ETFs (187)")):
        y = [cls[n]["mean_auc"] for n in ns]
        ax.plot(x, y, "o-", color=color, label=label, markersize=4)

    ax.axhline(0.5, color="0.4", lw=0.6, ls=":")
    ax.set_xticks(x, ns)
    ax.set_xlabel("kernel length $N$ (lags)")
    ax.set_ylabel("class-mean OOS AUC")
    ax.set_ylim(0.493, 0.536)
    ax.legend(loc="center right")
    save(fig, "nbaseline")


if __name__ == "__main__":
    main()
