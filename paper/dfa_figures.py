"""DFA / Hurst figures from out/dfa.json.

    uv run --active python -m paper.dfa_figures  ->  docs/figures/dfa_wide.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import C_CRYPTO, C_STOCK, ONEHALF, save

OUT = Path(__file__).resolve().parent / "out"


def main():
    d = json.loads((OUT / "dfa.json").read_text())
    # H needs no logit fit, so every asset enters -- no degenerate-fit filter.
    cr = [r for r in d["assets"] if r["class"] == "crypto"]
    st = [r for r in d["assets"] if r["class"] == "stock"]

    fig, ax = plt.subplots(figsize=(ONEHALF, 2.9))

    H_c = np.array([r["H"] for r in cr])
    H_s = np.array([r["H"] for r in st])
    bins = np.linspace(0.40, 0.60, 36)
    ax.hist(H_s, bins=bins, color=C_STOCK, alpha=0.65,
            label=f"stocks/ETFs (n={len(st)})")
    ax.hist(H_c, bins=bins, color=C_CRYPTO, alpha=0.65,
            label=f"crypto (n={len(cr)})")
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xlabel("DFA-1 Hurst exponent $H$ (15 m returns)")
    ax.set_ylabel("number of assets")
    ax.legend()

    save(fig, "dfa_wide")
    print(f"crypto n={len(cr)} mean H={H_c.mean():.4f} "
          f"median={np.median(H_c):.4f} below 0.5={100*(H_c < 0.5).mean():.1f}%")
    print(f"stocks n={len(st)} mean H={H_s.mean():.4f} "
          f"median={np.median(H_s):.4f} below 0.5={100*(H_s < 0.5).mean():.1f}%")


if __name__ == "__main__":
    main()
