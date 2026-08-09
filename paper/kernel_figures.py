"""Kernel-shape identifiability figure from out/kernel_shape.json.

    uv run --active python -m paper.kernel_figures  ->  docs/figures/kernel_shape.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import MODEL_COLOR, OI, ONEHALF, letter, save

OUT = Path(__file__).resolve().parent / "out"
FAM_COLOR = {"powerlaw": OI["vermillion"], "exponential": OI["blue"], "flat": "0.45"}
FAM_LABEL = {"powerlaw": "power-law", "exponential": "exponential", "flat": "flat"}


def main():
    d = json.loads((OUT / "kernel_shape.json").read_text())
    cells = d["cells"]
    rep = next(c for c in cells if c["cell"] == "btc-15m")
    within = [c for c in cells if c.get("study") == "within"]

    fig, axes = plt.subplots(1, 2, figsize=(ONEHALF + 1.4, 2.8),
                             gridspec_kw={"width_ratios": [1, 1.35]})

    # (a) mean fitted kernels on the representative cell
    k = np.arange(1, len(rep["kernels"]["powerlaw"]) + 1)
    for fam in ("powerlaw", "exponential", "flat"):
        axes[0].plot(k, rep["kernels"][fam], "o-", markersize=3,
                     color=FAM_COLOR[fam], label=FAM_LABEL[fam])
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xlabel("lag $k$ (candles)")
    axes[0].set_ylabel("mean fitted weight $A\\,w_k$")
    axes[0].legend()
    letter(axes[0], "(a)")

    # (b) per-cell paired OOS AUC deltas vs the power-law kernel
    labels = [c["cell"] for c in within]
    y = np.arange(len(within))[::-1]
    for off, tag, color, lab in [(-0.17, "pl_minus_exp", FAM_COLOR["exponential"],
                                  "power $-$ exponential"),
                                 (0.17, "pl_minus_flat", FAM_COLOR["flat"],
                                  "power $-$ flat")]:
        mean = np.array([c[tag]["d_auc_mean"] for c in within])
        lo = np.array([c[tag]["d_auc_ci"][0] for c in within])
        hi = np.array([c[tag]["d_auc_ci"][1] for c in within])
        axes[1].errorbar(mean, y + off, xerr=[mean - lo, hi - mean], fmt="o",
                         markersize=3, color=color, elinewidth=0.8, capsize=1.5,
                         label=lab)
    axes[1].axvline(0, color="k", lw=0.8, ls="--")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=6.5)
    axes[1].set_xlabel("paired OOS AUC difference")
    axes[1].legend()
    letter(axes[1], "(b)")

    save(fig, "kernel_shape")
    print("done")


if __name__ == "__main__":
    main()
