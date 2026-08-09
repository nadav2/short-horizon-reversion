"""DFA / Hurst figures from out/dfa.json.

    uv run --active python -m paper.dfa_figures  ->  docs/figures/dfa_wide.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import C_CRYPTO, C_STOCK, DOUBLE, letter, save

OUT = Path(__file__).resolve().parent / "out"


def main():
    d = json.loads((OUT / "dfa.json").read_text())
    rows = [r for r in d["assets"] if r["class"] in ("crypto", "stock")
            and abs(r["A"]) <= 1]          # drop degenerate pegged-stablecoin fits
    cr = [r for r in rows if r["class"] == "crypto"]
    st = [r for r in rows if r["class"] == "stock"]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.9))

    # (a) H distributions
    H_c = np.array([r["H"] for r in cr])
    H_s = np.array([r["H"] for r in st])
    bins = np.linspace(0.40, 0.60, 36)
    axes[0].hist(H_s, bins=bins, color=C_STOCK, alpha=0.65,
                 label=f"stocks/ETFs (n={len(st)})")
    axes[0].hist(H_c, bins=bins, color=C_CRYPTO, alpha=0.65,
                 label=f"crypto (n={len(cr)})")
    axes[0].axvline(0.5, color="k", lw=0.8, ls="--")
    axes[0].set_xlabel("DFA-1 Hurst exponent $H$ (15 m returns)")
    axes[0].set_ylabel("number of assets")
    axes[0].legend()
    letter(axes[0], "(a)")

    # (b) H vs fitted coupling A
    A_c = np.array([r["A"] for r in cr])
    A_s = np.array([r["A"] for r in st])
    axes[1].scatter(H_s, A_s, s=10, c=C_STOCK, alpha=0.6, label="stocks/ETFs")
    axes[1].scatter(H_c, A_c, s=10, c=C_CRYPTO, alpha=0.6, label="crypto")
    axes[1].axvline(0.5, color="k", lw=0.6, ls="--")
    axes[1].axhline(0, color="k", lw=0.6, ls="--")
    H_all = np.concatenate([H_c, H_s])
    A_all = np.concatenate([A_c, A_s])
    r_all = np.corrcoef(H_all, A_all)[0, 1]
    b, a = np.polyfit(H_all, A_all, 1)
    xs = np.linspace(H_all.min(), H_all.max(), 20)
    axes[1].plot(xs, a + b * xs, "k-", lw=0.9)
    axes[1].text(0.03, 0.04, f"$r = {r_all:+.2f}$", transform=axes[1].transAxes,
                 fontsize=7, va="bottom")
    axes[1].set_xlabel("DFA-1 Hurst exponent $H$")
    axes[1].set_ylabel("fitted coupling $A$ (15 m)")
    letter(axes[1], "(b)")

    save(fig, "dfa_wide")
    print("done")


if __name__ == "__main__":
    main()
