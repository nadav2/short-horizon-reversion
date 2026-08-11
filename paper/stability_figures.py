"""Five-year stability figure (Sec. "Robustness battery" / multi-year stability).

(a) per-year out-of-sample AUC of the fitted model, four focal coins,
    2021--2026 (out/multiyear.json), with SPY scored identically as the
    no-effect reference (out/multiyear_stocks.json) -- the model-based
    reading;
(b) per-quarter kernel-weighted sign correlation R (alpha = 1, nothing
    fitted), class mean with min--max envelope across assets
    (out/stability.json) -- the model-free reading of the same stability.

    uv run python -m paper.stability_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .style import C_CRYPTO, C_STOCK, DOUBLE, OI, letter, plt, save

OUT = Path(__file__).resolve().parent / "out"


def quarter_x(q: str) -> float:
    """'2023Q2' -> 2023.375 (mid-quarter in fractional years)."""
    y, n = q.split("Q")
    return int(y) + (int(n) - 1) / 4 + 0.125


def main():
    my = json.loads((OUT / "multiyear.json").read_text())
    spy = [r for r in json.loads((OUT / "multiyear_stocks.json").read_text())
           if r["coin"] == "spx"]
    st = json.loads((OUT / "stability.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))

    # (a) per-year OOS AUC, four focal coins + SPY as the no-effect reference
    ax = axes[0]
    colors = {"btc": OI["vermillion"], "eth": OI["blue"],
              "sol": OI["green"], "xrp": OI["orange"]}
    coins = sorted({r["coin"] for r in my})
    for ci, (coin, rows_) in enumerate(
            [(c, [r for r in my if r["coin"] == c]) for c in coins]
            + [("SPY", spy)]):
        sel = sorted(rows_, key=lambda r: r["year"])
        x = np.array([r["year"] for r in sel], float) + (ci - 2) * 0.12
        a = np.array([r["auc_ising"] for r in sel])
        lo = np.array([r["auc_ci"][0] for r in sel])
        hi = np.array([r["auc_ci"][1] for r in sel])
        col = colors.get(coin, "0.45")
        ax.errorbar(x, a, yerr=[a - lo, hi - a], fmt="o", color=col,
                    capsize=1.5, ms=2.8, lw=0.9, label=coin.upper())
    ax.axhline(0.5, color="0.4", lw=0.6, ls="--")
    ax.set_xlabel("year")
    ax.set_ylabel("out-of-sample AUC (15 m)")
    ax.legend(ncol=3, loc="lower left", columnspacing=1.0)
    letter(ax, "(a)")

    # (b) quarterly model-free R, class mean with min-max envelope
    ax = axes[1]
    qs = st["quarters"]
    x = np.array([quarter_x(q) for q in qs])
    for cls, col, lab in (("crypto", C_CRYPTO, "crypto (4 coins)"),
                          ("traditional", C_STOCK, "US-listed (8)")):
        s = st["summary"][cls]
        mean = np.array([s["mean"][q] for q in qs])
        lo = np.array([s["lo"][q] for q in qs])
        hi = np.array([s["hi"][q] for q in qs])
        ax.plot(x, mean, "o-", color=col, ms=2.8, label=lab)
        ax.fill_between(x, lo, hi, color=col, alpha=0.20, lw=0)
    ax.axhline(0, color="0.4", lw=0.6, ls="--")
    ax.set_xlabel("quarter")
    ax.set_ylabel(r"sign correlation $R$ ($\alpha\equiv 1$, per quarter)")
    ax.set_xticks(range(2021, 2027))
    ax.legend(loc="lower right")
    letter(ax, "(b)")

    fig.tight_layout()
    save(fig, "stability")


if __name__ == "__main__":
    main()
