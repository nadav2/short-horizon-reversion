"""Figure for the model-free multi-lag sign result (Sec. "Model-free confirmation").

(a) mean sign autocorrelation rho_k^sign by lag, crypto vs stocks -- the decaying
    reversal kernel, measured with nothing fitted;
(b) distribution of the kernel-weighted sign correlation R across the 370-asset
    universe -- the model-free counterpart of the fitted-coupling panel;
(c) sign-flip rate by decile of |r_{t-1}| on the focal coins, the mechanism that
    reconciles a strongly negative sign autocorrelation with rho_1(returns) ~ 0.

    uv run python -m paper.signlag_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .style import C_CRYPTO, C_STOCK, DOUBLE, letter, plt, save

OUT = Path(__file__).resolve().parent / "out"


def main():
    d = json.loads((OUT / "signlag.json").read_text())
    s, assets = d["summary"], d["assets"]

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.35))

    # (a) mean sign autocorrelation by lag
    ax = axes[0]
    lags = np.arange(1, len(s["crypto"]["mean_rho_by_lag"]) + 1)
    for cls, col, lab in ((("crypto"), C_CRYPTO, "crypto (183)"),
                          (("stock"), C_STOCK, "stocks/ETFs (187)")):
        ax.plot(lags, s[cls]["mean_rho_by_lag"], "o-", color=col, ms=3, label=lab)
    ax.axhline(0, color="0.4", lw=0.6, ls="--")
    ax.set_xlabel(r"lag $k$ (15-minute candles)")
    ax.set_ylabel(r"mean $\rho_k$ of $\mathrm{sign}(r_t)$")
    ax.set_xticks([1, 3, 6, 9, 12])
    ax.legend(loc="lower right")
    letter(ax, "(a)")

    # (b) distribution of the kernel-weighted sign correlation R
    ax = axes[1]
    for cls, col, lab in (("crypto", C_CRYPTO, "crypto"), ("stock", C_STOCK, "stocks/ETFs")):
        vals = [r["R_kernel"] for r in assets if r["class"] == cls]
        ax.hist(vals, bins=32, color=col, alpha=0.65, label=lab)
    ax.axvline(0, color="0.4", lw=0.6, ls="--")
    ax.set_xlabel(r"kernel-weighted sign correlation $R$")
    ax.set_ylabel("assets")
    ax.legend(loc="upper left")
    letter(ax, "(b)")

    # (c) sign-flip rate by decile of |r_{t-1}|
    ax = axes[2]
    md = s["magnitude_decomposition"]
    first = True
    for coin, style in (("btc", "o-"), ("eth", "s-"), ("xrp", "^-"), ("sol", "d-")):
        if coin not in md:
            continue
        bins = md[coin]["bins"]
        ax.plot([b["decile"] for b in bins], [b["flip_rate"] * 100 for b in bins],
                style, color=C_CRYPTO, ms=3, alpha=0.75,
                label="BTC / ETH / XRP / SOL" if first else None)
        first = False
    if "aapl" in md:
        bins = md["aapl"]["bins"]
        ax.plot([b["decile"] for b in bins], [b["flip_rate"] * 100 for b in bins],
                "o-", color=C_STOCK, ms=3, label="AAPL")
    ax.axhline(50, color="0.4", lw=0.6, ls="--")
    ax.set_xlabel(r"decile of $|r_{t-1}|$")
    ax.set_ylabel("sign-flip rate (%)")
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.legend(loc="lower right", frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.9)
    letter(ax, "(c)")

    fig.tight_layout()
    save(fig, "signlag")


if __name__ == "__main__":
    main()
