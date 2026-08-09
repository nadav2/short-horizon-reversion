"""Cross-market figures from out/markets.json (+ out/multiyear.json).

    uv run --active python -m paper.market_figures  ->  docs/figures/mkt_*.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import (CLASS_COLOR, C_CRYPTO, C_TRAD, OI, ONEHALF, SINGLE,
                    letter, save)

OUT = Path(__file__).resolve().parent / "out"
C_FX = CLASS_COLOR["fx"]


def rows():
    return json.loads((OUT / "markets.json").read_text())


def _color(r):
    if r["class"] == "crypto":
        return C_CRYPTO
    return C_FX if r["asset_class"] == "fx" else C_TRAD


# Fig A: per-asset 15m Ising AUC with bootstrap CIs.
def fig_auc_15m(R):
    sel = [r for r in R if r["interval"] == "15m"]
    order = {"crypto": 0, "fx": 1}
    sel.sort(key=lambda r: (order.get(r["asset_class"] if r["asset_class"] == "fx"
                                      else r["class"], 2), -r["ising_auc"]))
    labels = [f'{r["asset"].upper()} ({r["asset_class"]})' for r in sel]
    auc = np.array([r["ising_auc"] for r in sel])
    lo = np.array([r["ising_auc_ci"][0] for r in sel])
    hi = np.array([r["ising_auc_ci"][1] for r in sel])
    colors = [_color(r) for r in sel]
    y = np.arange(len(sel))[::-1]

    fig, ax = plt.subplots(figsize=(ONEHALF, 0.32 * len(sel) + 0.7))
    ax.errorbar(auc, y, xerr=[auc - lo, hi - auc], fmt="none", ecolor="0.55",
                elinewidth=0.8, capsize=2)
    ax.scatter(auc, y, c=colors, s=26, zorder=3)
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("out-of-sample AUC (Ising), 95% block-bootstrap CI")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
               [(C_CRYPTO, "crypto"), (C_FX, "FX"), (C_TRAD, "equity/commodity/bond")]]
    ax.legend(handles=handles, loc="lower right")
    save(fig, "mkt_auc_15m")


# Fig B: horizon decay of mean AUC by group.
def fig_horizon(R):
    intervals = ["15m", "1h", "4h"]
    groups = [("crypto", lambda r: r["class"] == "crypto", C_CRYPTO),
              ("FX", lambda r: r["asset_class"] == "fx", C_FX),
              ("traditional (ex FX)", lambda r: r["class"] == "traditional"
                                                and r["asset_class"] != "fx", C_TRAD)]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    for label, fsel, color in groups:
        means, ses = [], []
        for iv in intervals:
            a = [r["ising_auc"] for r in R if fsel(r) and r["interval"] == iv]
            means.append(np.mean(a))
            ses.append(np.std(a) / max(1, np.sqrt(len(a))))
        ax.errorbar(range(len(intervals)), means, yerr=ses, fmt="o-", color=color,
                    capsize=2.5, lw=1.2, markersize=3.5, label=label)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(intervals)))
    ax.set_xticklabels(intervals)
    ax.set_xlabel("sampling interval")
    ax.set_ylabel("mean out-of-sample AUC")
    ax.legend()
    save(fig, "mkt_horizon")


# Fig C: coupling A by asset class at 15m.
def fig_coupling(R):
    classes = ["crypto", "fx", "index", "stock", "commodity", "bond"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    means, ses, cols = [], [], []
    for c in classes:
        a = [r["A"] for r in R if r["asset_class"] == c and r["interval"] == "15m"]
        means.append(np.mean(a))
        ses.append(np.std(a) / max(1, np.sqrt(len(a))))
        cols.append(CLASS_COLOR[c])
    ax.bar(range(len(classes)), means, yerr=ses, capsize=2.5, color=cols, width=0.7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=7)
    ax.set_ylabel("mean coupling $A$ (15 m)")
    save(fig, "mkt_coupling")


# Fig D: multi-year stability of the crypto 15m edge (out/multiyear.json).
def fig_multiyear():
    f = OUT / "multiyear.json"
    if not f.exists():
        print("skip mkt_multiyear (multiyear.json missing)")
        return
    rows = json.loads(f.read_text())
    coins = sorted({r["coin"] for r in rows})
    colors = {"btc": OI["vermillion"], "eth": OI["blue"],
              "sol": OI["green"], "xrp": OI["orange"]}
    fig, ax = plt.subplots(figsize=(ONEHALF, 2.8))
    for ci, coin in enumerate(coins):
        sel = sorted([r for r in rows if r["coin"] == coin], key=lambda r: r["year"])
        x = np.array([r["year"] for r in sel], float) + (ci - 1.5) * 0.09
        a = np.array([r["auc_ising"] for r in sel])
        lo = np.array([r["auc_ci"][0] for r in sel])
        hi = np.array([r["auc_ci"][1] for r in sel])
        ax.errorbar(x, a, yerr=[a - lo, hi - a], fmt="o", color=colors.get(coin, "k"),
                    capsize=2, markersize=3.5, lw=1, label=coin.upper())
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xlabel("year")
    ax.set_ylabel("out-of-sample AUC (15 m)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save(fig, "mkt_multiyear")


def main():
    R = rows()
    fig_auc_15m(R)
    fig_horizon(R)
    fig_coupling(R)
    fig_multiyear()
    print("done")


if __name__ == "__main__":
    main()
