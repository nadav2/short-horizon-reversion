"""Refit-inclusive JOINT bootstrap of the class-mean gap (referee response, Tier 2 #8).

`full_boot.py` establishes that conditioning on the fitted models biases per-asset AUC
upward, and that the bias is class-asymmetric: on its 50-asset subset the crypto mean
falls 0.5271 -> 0.5203 while the stock mean rises 0.4994 -> 0.5005, so the class gap
falls from +0.0278 to +0.0198 (29%). But `full_boot` draws each asset's blocks with an
independent seed, so its replicates cannot be assembled into a class-mean interval: the
cross-sectional dependence that the paper's joint bootstrap exists to preserve is
destroyed by the independent draws.

This module supplies the missing object. In each replicate ONE set of moving blocks is
drawn on the shared 15m grid and applied to every asset simultaneously (as in
`dependence.py`), and both models are then REFIT inside that replicate on every asset
(as in `full_boot.py`), so the resulting class-mean gap distribution carries both the
cross-sectional dependence and the estimation uncertainty. Cost is B x n_assets
walk-forward refits, so it runs on a stratified subset rather than the full 370.

    uv run --active python -m paper.refit_joint --per-class 12 --boot 150
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .models import ARLogit, IsingLogit
from .walkforward import walk_forward
from .wide import BULK, INTERVAL, OUT, TRAIN, TEST, load

SLOT = 900
BLOCK = 384
SEED = 20260811


def pick(per_class: int) -> list[tuple[str, str]]:
    """Evenly spaced AUC quantiles per class, matching full_boot.pick_subset.

    Selecting on series length instead (the obvious alternative) is length-biased:
    the longest crypto histories are the established coins, whose class mean sits
    well above the universe's, so the resulting subset is not a fair sample of the
    class mean the gap is built from. Quantile spacing keeps the subset's
    fit-conditional gap close to the full-universe value, which is what makes the
    refit-inclusive comparison interpretable.
    """
    rows = json.loads((OUT / "wide.json").read_text())
    out = []
    for cls in ("crypto", "stock"):
        sub = sorted((r for r in rows if r["class"] == cls), key=lambda r: r["auc_ising"])
        # require enough history to support the walk-forward at all
        sub = [r for r in sub if r["n_oos"] >= 2000]
        idx = np.unique(np.linspace(0, len(sub) - 1, per_class).round().astype(int))
        out += [(sub[i]["asset"], cls) for i in idx]
    return out


def _replicate(args):
    """One joint replicate: shared block draw on the grid, refit every asset."""
    assets, rep_seed = args
    rng = np.random.default_rng(rep_seed)
    per = {"crypto": [], "stock": []}
    # shared block starts, expressed as fractions of the grid so each asset maps the
    # same time blocks onto its own (differently long) series
    nb = int(np.ceil(1.0 / (BLOCK / 40000)))
    fracs = rng.random(nb)
    for asset, cls in assets:
        path = BULK / f"{asset}-{INTERVAL}.json"
        if not path.exists():
            continue
        try:
            ch, ups, _ = load(path)
        except Exception:
            continue
        n = len(ch)
        if n < TRAIN + TEST + 2000:
            continue
        k = int(np.ceil(n / BLOCK))
        starts = (fracs[:k] * max(1, n - BLOCK + 1)).astype(int)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        factory = lambda: [IsingLogit(n_lags=12), ARLogit("free", n_lags=12)]
        try:
            res, _ = walk_forward(ch[idx], ups[idx], TRAIN, TEST, n_lags=12, models=factory)
        except Exception:
            continue
        y = res["ising"]["actuals"].astype(int)
        if len(y) < 2000 or len(np.unique(y)) < 2:
            continue
        per[cls].append(float(roc_auc_score(y, res["ising"]["probs"])))
    if not per["crypto"] or not per["stock"]:
        return None
    return float(np.mean(per["crypto"])), float(np.mean(per["stock"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--boot", type=int, default=150)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()

    assets = pick(a.per_class)
    print(f"{len(assets)} assets, B={a.boot}, {a.workers} workers")

    # point estimate: fit-conditional, same subset
    pt = {"crypto": [], "stock": []}
    for asset, cls in assets:
        ch, ups, _ = load(BULK / f"{asset}-{INTERVAL}.json")
        factory = lambda: [IsingLogit(n_lags=12), ARLogit("free", n_lags=12)]
        res, _ = walk_forward(ch, ups, TRAIN, TEST, n_lags=12, models=factory)
        y = res["ising"]["actuals"].astype(int)
        pt[cls].append(float(roc_auc_score(y, res["ising"]["probs"])))
    cond_gap = float(np.mean(pt["crypto"]) - np.mean(pt["stock"]))
    print(f"fit-conditional gap on this subset: {cond_gap:+.5f}")

    seeds = np.random.SeedSequence(SEED).spawn(a.boot)
    with Pool(a.workers) as pool:
        reps = pool.map(_replicate, [(assets, s) for s in seeds])
    reps = [r for r in reps if r is not None]
    g = np.array([c - s for c, s in reps])
    res = {
        "n_assets": len(assets), "per_class": a.per_class,
        "n_boot_requested": a.boot, "n_boot_used": len(g),
        "fit_conditional_gap": cond_gap,
        "fit_conditional_crypto": float(np.mean(pt["crypto"])),
        "fit_conditional_stock": float(np.mean(pt["stock"])),
        "refit_joint_gap_mean": float(g.mean()),
        "refit_joint_gap_ci": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
        "refit_joint_gap_sd": float(g.std(ddof=1)),
        "refit_crypto_mean": float(np.mean([c for c, _ in reps])),
        "refit_stock_mean": float(np.mean([s for _, s in reps])),
        "bias_vs_conditional": float(g.mean() - cond_gap),
    }
    (OUT / "refit_joint.json").write_text(json.dumps(res, indent=2))
    print(f"refit-inclusive joint gap {res['refit_joint_gap_mean']:+.5f} "
          f"{res['refit_joint_gap_ci']}  sd {res['refit_joint_gap_sd']:.5f}")
    print(f"bias vs fit-conditional {res['bias_vs_conditional']:+.5f}")


if __name__ == "__main__":
    main()
