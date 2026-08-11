"""Figure for the model-free multi-lag sign result (Sec. "Model-free confirmation").

(a) mean sign autocorrelation rho_k^sign by lag, crypto vs stocks -- the decaying
    reversal kernel, measured with nothing fitted;
(b) distribution of the kernel-weighted sign correlation R across the 370-asset
    universe -- the model-free counterpart of the fitted-coupling panel;
(c) class-pooled sign-flip rate by decile of |r_{t-1}| (flat bars excluded,
    mean +- 2 s.e. across assets), the mechanism that reconciles a strongly
    negative sign autocorrelation with rho_1(returns) ~ 0;
(d) R vs the fitted model's out-of-sample AUC across the 370 assets -- the
    two-line statistic predicts the model's skill (r = -0.72).

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

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.6))
    axes = axes.ravel()

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

    # (c) class-pooled sign-flip rate by decile of |r_{t-1}| (flat bars excluded)
    ax = axes[2]
    fc = s["flip_by_class"]
    dec = np.arange(1, len(fc["crypto"]["mean"]) + 1)
    for cls, col, lab in (("crypto", C_CRYPTO, f"crypto ({fc['crypto']['n_assets']})"),
                          ("stock", C_STOCK, f"stocks/ETFs ({fc['stock']['n_assets']})")):
        m = np.array(fc[cls]["mean"]) * 100
        se = np.array(fc[cls]["se"]) * 100
        ax.plot(dec, m, "o-", color=col, ms=3, label=lab)
        ax.fill_between(dec, m - 2 * se, m + 2 * se, color=col, alpha=0.25, lw=0)
    ax.axhline(50, color="0.4", lw=0.6, ls="--")
    ax.set_xlabel(r"decile of $|r_{t-1}|$")
    ax.set_ylabel("sign-flip rate (%)")
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.86))
    letter(ax, "(c)")

    # (d) model-free R vs the fitted model's out-of-sample AUC
    ax = axes[3]
    for cls, col, lab in (("crypto", C_CRYPTO, "crypto"), ("stock", C_STOCK, "stocks/ETFs")):
        sel = [r for r in assets if r["class"] == cls]
        ax.scatter([r["R_kernel"] for r in sel], [r["auc_ising"] for r in sel],
                   s=5, color=col, alpha=0.55, lw=0, label=lab)
    ax.axhline(0.5, color="0.4", lw=0.6, ls="--")
    ax.axvline(0, color="0.4", lw=0.6, ls="--")
    ax.text(0.03, 0.03, rf"$r={d['summary']['corr_Rkernel_auc']:+.2f}$",
            transform=ax.transAxes, ha="left", va="bottom")
    ax.set_xlabel(r"sign correlation $R$")
    ax.set_ylabel("out-of-sample AUC")
    letter(ax, "(d)")

    fig.tight_layout()
    save(fig, "signlag")


if __name__ == "__main__":
    main()
