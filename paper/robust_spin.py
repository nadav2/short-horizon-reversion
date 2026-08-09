"""Volatility-standardized spin-scale robustness (cross-class fairness check).

The fixed spin scale lambda=150 puts asset classes at different points of the
tanh nonlinearity (mean |s| ~ 0.39 for crypto vs ~ 0.20 for stocks at 15m), so
the cross-class comparison of the fitted coupling A could in principle be an
artifact of saturation. Here the Ising model is refit with a per-asset,
per-fold standardized scale lambda = VOL_SPIN_C / std(train returns), which
equalizes the spin distribution across assets. If the crypto/traditional
contrast is real, it must survive this re-parameterization.

    uv run --active python -m paper.robust_spin
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .compare_markets import (BLOCK, CRYPTO, N_BOOT, SEED, TRADITIONAL,
                              WINDOWS, block_boot_idx, load_span)
from .models import IsingLogit
from .walkforward import walk_forward

OUT = Path(__file__).resolve().parent / "out"
INTERVAL = "15m"


def models():
    vol = IsingLogit(n_lags=12, spin_mode="vol")
    vol.short, vol.name = "ising_vol", "Ising (vol-standardized spins)"
    return [vol, IsingLogit(n_lags=12)]


def main():
    tr, te = WINDOWS[INTERVAL]
    rows = []
    for asset in CRYPTO + TRADITIONAL:
        try:
            _, ch, ups = load_span(asset, INTERVAL)
        except FileNotFoundError:
            continue
        if len(ch) < tr + te:
            print(f"  {asset}: too short, skipped")
            continue
        res, nf = walk_forward(ch, ups, tr, te, n_lags=12, models=models)
        y = res["ising"]["actuals"].astype(int)
        if len(np.unique(y)) < 2:
            continue
        rng = np.random.default_rng(SEED)
        row = {"asset": asset, "n_oos": int(len(y)), "n_folds": nf,
               "class": "crypto" if asset in CRYPTO else "traditional"}
        for key in ("ising_vol", "ising"):
            p = res[key]["probs"]
            ge = 0
            n_eff = 0
            for _ in range(N_BOOT):
                idx = block_boot_idx(len(y), BLOCK[INTERVAL], rng)
                yb = y[idx]
                if yb.min() == yb.max():
                    continue
                n_eff += 1
                ge += roc_auc_score(yb, p[idx]) <= 0.5
            row[key] = {
                "auc": float(roc_auc_score(y, p)),
                "p_auc_gt05": ge / max(n_eff, 1),
                "A": float(np.mean([fp["A"] for fp in res[key]["fold_params"]])),
                "mean_lambda": float(np.mean([fp["spin_scale"] for fp in res[key]["fold_params"]])),
            }
        rows.append(row)
        print(f"  {asset:5s} fixed AUC={row['ising']['auc']:.4f} (A={row['ising']['A']:+.3f})  "
              f"vol AUC={row['ising_vol']['auc']:.4f} (A={row['ising_vol']['A']:+.3f}, "
              f"mean λ={row['ising_vol']['mean_lambda']:.0f})  p={row['ising_vol']['p_auc_gt05']:.3f}")

    (OUT / "robust_spin.json").write_text(json.dumps(rows, indent=2))
    for cls in ("crypto", "traditional"):
        sel = [r for r in rows if r["class"] == cls]
        if sel:
            print(f"{cls}: mean vol-spin AUC={np.mean([r['ising_vol']['auc'] for r in sel]):.4f} "
                  f"(fixed {np.mean([r['ising']['auc'] for r in sel]):.4f}), "
                  f"mean vol-spin A={np.mean([r['ising_vol']['A'] for r in sel]):+.3f} "
                  f"(fixed {np.mean([r['ising']['A'] for r in sel]):+.3f})")
    print(f"Wrote {OUT/'robust_spin.json'}")


if __name__ == "__main__":
    main()
