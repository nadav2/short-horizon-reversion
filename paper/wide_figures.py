"""Figures for the wide (hundreds-of-assets) study, from out/wide.json and
out/dependence.json.

    uv run --active python -m paper.wide_figures  ->  docs/figures/wide_*.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import C_CRYPTO, C_STOCK, DOUBLE, SINGLE, letter, save

OUT = Path(__file__).resolve().parent / "out"


def fig_dist(cr, st):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.8))
    auc_c = np.array([r["auc_ising"] for r in cr])
    auc_s = np.array([r["auc_ising"] for r in st])
    bins = np.linspace(0.46, 0.57, 40)
    axes[0].hist(auc_s, bins=bins, color=C_STOCK, alpha=0.65, label=f"stocks/ETFs (n={len(st)})")
    axes[0].hist(auc_c, bins=bins, color=C_CRYPTO, alpha=0.65, label=f"crypto (n={len(cr)})")
    axes[0].axvline(0.5, color="k", lw=0.8, ls="--")
    axes[0].set_xlabel("out-of-sample AUC, constrained logit (15 m)")
    axes[0].set_ylabel("number of assets")
    axes[0].legend()
    letter(axes[0], "(a)")

    A_c = np.array([r["A"] for r in cr])
    A_s = np.array([r["A"] for r in st])
    bins2 = np.linspace(-0.5, 0.3, 40)
    axes[1].hist(A_s, bins=bins2, color=C_STOCK, alpha=0.65, label="stocks/ETFs")
    axes[1].hist(A_c, bins=bins2, color=C_CRYPTO, alpha=0.65, label="crypto")
    axes[1].axvline(0, color="k", lw=0.8, ls="--")
    axes[1].set_xlabel("fitted coupling $A$ (15 m)")
    letter(axes[1], "(b)")
    save(fig, "wide_dist")


def fig_sharpe(cr, st):
    fig, ax = plt.subplots(figsize=(SINGLE, 2.7))
    sh_c = np.array([r["sharpe_ising"] for r in cr if np.isfinite(r["sharpe_ising"])])
    sh_s = np.array([r["sharpe_ising"] for r in st if np.isfinite(r["sharpe_ising"])])
    bins = np.linspace(-3, 7, 45)
    ax.hist(sh_s, bins=bins, color=C_STOCK, alpha=0.65,
            label=f"stocks/ETFs (median {np.median(sh_s):.2f})")
    ax.hist(sh_c, bins=bins, color=C_CRYPTO, alpha=0.65,
            label=f"crypto (median {np.median(sh_c):.2f})")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"annualized Sharpe, constrained-logit strategy ($\rho$=1)")
    ax.set_ylabel("number of assets")
    ax.legend()
    save(fig, "wide_sharpe")


def fig_gapboot():
    f = OUT / "dependence.json"
    if not f.exists():
        print("skip wide_gapboot (dependence.json missing)")
        return
    dep = json.loads(f.read_text())
    g = dep["joint_gap"]["ising"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    if "gap_samples" in g:
        samples = np.array(g["gap_samples"])
        lo = min(0.0, samples.min()) - 0.002
        ax.hist(samples, bins=30, range=(lo, samples.max() + 0.002),
                color=C_CRYPTO, alpha=0.65, label="joint bootstrap")
    ax.axvline(g["obs_gap"], color="k", lw=1.2, label="observed gap")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("class-mean AUC gap (crypto $-$ stocks, 15 m)")
    ax.set_ylabel("resamples")
    ax.legend(loc="upper left")
    save(fig, "wide_gapboot")


def main():
    rows = json.loads((OUT / "wide.json").read_text())
    cr = [r for r in rows if r["class"] == "crypto"]
    st = [r for r in rows if r["class"] == "stock"]
    fig_dist(cr, st)
    fig_sharpe(cr, st)
    fig_gapboot()
    print("done")


if __name__ == "__main__":
    main()
