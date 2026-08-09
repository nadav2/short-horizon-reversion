"""Generate the within-crypto figures from out/results.json and the OOS .npz dumps.

    uv run --active python -m paper.figures  ->  docs/figures/*.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import load_merged
from .models import ARLogit, IsingLogit
from .style import DOUBLE, MODEL_COLOR, ONEHALF, SINGLE, letter, save
from .walkforward import window_candles

OUT = Path(__file__).resolve().parent / "out"
C_ISING, C_FREE, C_L2, C_MK, C_BASE = (MODEL_COLOR["ising"], MODEL_COLOR["free"],
                                       MODEL_COLOR["l2"], MODEL_COLOR["markov1"],
                                       MODEL_COLOR["base"])


def _results():
    return json.loads((OUT / "results.json").read_text())


# ── Fig: power-law memory kernel across intervals (BTC) ──────────────────────

def fig_kernel(R):
    fig, ax = plt.subplots(figsize=(SINGLE, 2.7))
    for iv in ["5m", "15m", "1h", "4h"]:
        cell = R["cells"].get(f"btc-{iv}")
        if not cell:
            continue
        a = float(np.mean(cell["ising_params"]["alpha"]))
        A = float(np.mean(cell["ising_params"]["A"]))
        k = np.arange(1, 13)
        ax.plot(k, A / (k ** a), "o-", markersize=3,
                label=fr"{iv}  ($\alpha$={a:.2f}, $A$={A:+.2f})")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("lag $k$ (candles)")
    ax.set_ylabel(r"effective coupling $A\,k^{-\alpha}$")
    ax.legend()
    save(fig, "kernel")


# ── Fig: free AR coefficient instability vs tied Ising kernel ────────────────

def _sign_stability(W):
    """Mean over lags of the fraction of folds whose weight agrees with the modal sign.
    1.0 = the weight keeps the same sign in every fold; 0.5 = sign is a coin flip."""
    pos = (W > 0).mean(0)
    return float(np.mean(np.maximum(pos, 1 - pos)))


def fig_coef_instability(coin="btc", interval="15m", N=24):
    """Per-fold weights as a 'spaghetti' plot. Free weights and Ising weights live on
    different feature scales, so each panel is z-scored by its own across-fold-and-lag
    RMS to compare *shape stability*, not magnitude."""
    dts, ch, ups = load_merged(coin, interval)
    tr, te = window_candles(interval)
    n = len(ch)
    free_coefs, ising_kernels = [], []
    start = 0
    while start + tr < n:
        tlo, thi = start, start + tr
        if thi + te > n and free_coefs:
            break
        fm = ARLogit("free", n_lags=N); fm.fit(ch, ups, tlo, thi)
        free_coefs.append(fm.coef())
        im = IsingLogit(n_lags=N); im.fit(ch, ups, tlo, thi)
        ising_kernels.append(im.implied_weights())
        start += te
    free_coefs = np.array(free_coefs)
    ising_kernels = np.array(ising_kernels)
    k = np.arange(1, N + 1)
    free_sign = _sign_stability(free_coefs)
    ising_sign = _sign_stability(ising_kernels)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), sharex=True)
    for ax, W, color, pl, sgn in [
            (axes[0], free_coefs / np.sqrt((free_coefs ** 2).mean()), C_FREE, "(a)", free_sign),
            (axes[1], ising_kernels / np.sqrt((ising_kernels ** 2).mean()), C_ISING, "(b)", ising_sign)]:
        for row in W:
            ax.plot(k, row, color=color, lw=0.5, alpha=0.3)
        ax.plot(k, W.mean(0), color="k", lw=1.5)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("lag $k$ (candles)")
        letter(ax, pl)
        ax.text(0.98, 0.02, f"sign-stable in {sgn*100:.0f}% of folds",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    axes[0].set_ylabel("weight (z-scored per panel)")
    save(fig, "coef_instability")
    return {"free_sign_stability": free_sign, "ising_sign_stability": ising_sign, "N": N}


# ── Fig: train vs OOS log-loss as lag order grows (overfitting curve) ────────

def fig_overfit(R):
    cells = list(R["lag_sweep"].keys())
    fig, axes = plt.subplots(1, len(cells), figsize=(DOUBLE, 2.6), sharey=False)
    if len(cells) == 1:
        axes = [axes]
    for j, (ax, key) in enumerate(zip(axes, cells)):
        sweep = R["lag_sweep"][key]
        Ns = sorted(int(n) for n in sweep)
        for short, color, lab in [("free", C_FREE, "free AR-logit"),
                                  ("l2", C_L2, "ridge (L2)"),
                                  ("ising", C_ISING, "Ising (3 params)")]:
            tr = [sweep[str(N)][short]["train_ll"] for N in Ns]
            te = [sweep[str(N)][short]["test_ll"] for N in Ns]
            ax.plot(Ns, tr, "--", color=color, lw=0.8, alpha=0.7)
            ax.plot(Ns, te, "o-", color=color, lw=1.2, markersize=2.5, label=lab)
        ax.axhline(np.log(2), color=C_BASE, lw=0.7, ls=":", label="coin flip ($\\ln 2$)")
        ax.set_xlabel("AR order $N$ (lags)")
        letter(ax, f"({chr(97 + j)}) {key.upper()}")
    axes[0].set_ylabel("log-loss (solid OOS, dashed train)")
    axes[0].legend()
    save(fig, "overfit")


# ── Fig: equity curves (trading, stylized even-money) ────────────────────────

def fig_equity(coin="btc", interval="15m", fee=0.02):
    f = OUT / f"oos_{coin}_{interval}.npz"
    if not f.exists():
        print(f"skip equity ({f.name} missing)")
        return
    d = np.load(f)
    actual = d["actual"]
    fig, ax = plt.subplots(figsize=(ONEHALF, 2.8))
    for short, color, lab in [("ising", C_ISING, "Ising"), ("free", C_FREE, "free AR-logit"),
                              ("l2", C_L2, "ridge (L2)"), ("markov1", C_MK, "Markov-1"),
                              ("base", C_BASE, "base rate")]:
        key = f"p_{short}"
        if key not in d:
            continue
        p = d[key]
        correct = (p > 0.5) == (actual > 0.5)
        pnl = np.where(correct, 1.0 - fee, -1.0 - fee)
        ax.plot(np.cumsum(pnl), color=color, lw=1.2 if short == "ising" else 0.9, label=lab)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("out-of-sample bet number")
    ax.set_ylabel("cumulative P\\&L (stake units)")
    ax.legend()
    save(fig, "equity")


# ── Fig: fitted alpha across intervals and coins ─────────────────────────────

def fig_alpha(R):
    coins = ["btc", "eth", "sol", "xrp"]
    intervals = ["5m", "15m", "1h", "4h"]
    fig, ax = plt.subplots(figsize=(ONEHALF, 2.6))
    width = 0.2
    x = np.arange(len(intervals))
    from .style import OI
    cc = [OI["vermillion"], OI["blue"], OI["green"], OI["orange"]]
    for i, coin in enumerate(coins):
        means, sds = [], []
        for iv in intervals:
            cell = R["cells"].get(f"{coin}-{iv}")
            if cell:
                a = cell["ising_params"]["alpha"]
                means.append(np.mean(a)); sds.append(np.std(a))
            else:
                means.append(np.nan); sds.append(0)
        ax.bar(x + (i - 1.5) * width, means, width, yerr=sds, capsize=1.5,
               color=cc[i], label=coin.upper())
    ax.set_xticks(x); ax.set_xticklabels(intervals)
    ax.set_xlabel("sampling interval")
    ax.set_ylabel(r"fitted decay exponent $\alpha$")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    save(fig, "alpha")


# ── Fig: reliability / calibration diagram ───────────────────────────────────

def fig_calibration(coin="btc", interval="15m"):
    f = OUT / f"oos_{coin}_{interval}.npz"
    if not f.exists():
        print(f"skip calibration ({f.name} missing)")
        return
    d = np.load(f)
    actual = d["actual"]
    fig, ax = plt.subplots(figsize=(SINGLE, 3.0))
    ax.plot([0.4, 0.6], [0.4, 0.6], "k:", lw=0.8, label="perfect")
    for short, color, lab in [("ising", C_ISING, "Ising"), ("free", C_FREE, "free AR-logit")]:
        p = d[f"p_{short}"]
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
        mids, obs, err = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p < hi)
            if m.sum() > 30:
                mids.append(p[m].mean())
                q = actual[m].mean()
                obs.append(q)
                err.append(np.sqrt(q * (1 - q) / m.sum()))   # binomial SE per bin
        ax.errorbar(mids, obs, yerr=err, fmt="o-", color=color, markersize=3,
                    lw=1.0, capsize=1.5, label=lab)
    ax.set_xlabel("predicted $P(\\mathrm{up})$")
    ax.set_ylabel("observed frequency")
    ax.legend()
    save(fig, "calibration")


def main():
    R = _results()
    fig_kernel(R)
    extra = fig_coef_instability()
    fig_overfit(R)
    fig_equity()
    fig_alpha(R)
    fig_calibration()
    (OUT / "figure_stats.json").write_text(json.dumps(extra, indent=2))
    print("done")


if __name__ == "__main__":
    main()
