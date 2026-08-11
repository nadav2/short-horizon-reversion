"""Population-level pipeline null: shuffled labels over the FULL wide universe.

The strict shuffled-label control of negative_controls.py runs on the 14 focal
instruments only, while the paper's central claim rests on the 370-asset wide
study. This module closes that gap: for every asset in bulk_data/ the label
series is randomly permuted (features untouched, destroying all feature-label
dependence) and the IDENTICAL wide-study pipeline is run -- same walk-forward
geometry, same two models, same moving-block bootstrap, and the same joint
Benjamini-Hochberg FDR across the full universe. Any significant count above
the FDR's nominal expectation would indict the population-scale machinery
itself.

    uv run --active python -m paper.wide_null
"""

from __future__ import annotations

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .fdr import bh_mask
from .models import ARLogit, IsingLogit
from .walkforward import walk_forward
from .wide import BLOCK, BULK, INTERVAL, N_BOOT, OUT, STOCK_SET, TEST, TRAIN, block_idx, load

SEED = 20260809
N_WORKERS = 8


def evaluate_null(path: Path) -> dict | None:
    base = path.name[:-(len(INTERVAL) + 6)]
    ch, ups, _ = load(path)
    if len(ch) < TRAIN + TEST + 2000:
        return None
    # deterministic per-asset shuffle so the run is reproducible asset-by-asset
    rng = np.random.default_rng([SEED, hash(base) & 0x7FFFFFFF])
    ups_null = rng.permutation(ups)

    factory = lambda: [IsingLogit(n_lags=12), ARLogit("free", n_lags=12)]
    res, nf = walk_forward(ch, ups_null, TRAIN, TEST, n_lags=12, models=factory)
    y = res["ising"]["actuals"].astype(int)
    if len(y) < 2000 or len(np.unique(y)) < 2:
        return None
    p_is, p_fr = res["ising"]["probs"], res["free"]["probs"]
    auc_is, auc_fr = roc_auc_score(y, p_is), roc_auc_score(y, p_fr)

    brng = np.random.default_rng(5)          # same bootstrap seed as wide.py
    ge_is = ge_fr = 0
    for _ in range(N_BOOT):
        idx = block_idx(len(y), brng)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        ge_is += roc_auc_score(yb, p_is[idx]) <= 0.5
        ge_fr += roc_auc_score(yb, p_fr[idx]) <= 0.5
    return {"asset": base,
            "class": "stock" if base in STOCK_SET else "crypto",
            "auc_ising": float(auc_is), "auc_free": float(auc_fr),
            "p_ising": ge_is / N_BOOT, "p_free": ge_fr / N_BOOT,
            "n_oos": int(len(y)), "n_folds": nf}


def main():
    files = sorted(BULK.glob(f"*-{INTERVAL}.json"))
    print(f"{len(files)} assets; shuffled-label null, {N_WORKERS} workers")
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        rows = [r for r in pool.map(evaluate_null, files) if r is not None]
    print(f"kept {len(rows)} assets in {time.time()-t0:.0f}s")

    # identical FDR treatment to fdr.py: floor exact zeros, BH on the conjunction
    floor = 1.0 / (N_BOOT + 1)
    p_conj = np.array([max(max(r["p_ising"], floor), max(r["p_free"], floor))
                       for r in rows])
    sig = bh_mask(p_conj, 0.05)
    raw_sig = np.array([max(r["p_ising"], r["p_free"]) < 0.05 for r in rows])

    summary = {}
    for cls in ("crypto", "stock"):
        sub = [i for i, r in enumerate(rows) if r["class"] == cls]
        aucs_is = np.array([rows[i]["auc_ising"] for i in sub])
        aucs_fr = np.array([rows[i]["auc_free"] for i in sub])
        summary[cls] = {
            "n": len(sub),
            "mean_auc_ising": float(aucs_is.mean()),
            "mean_auc_free": float(aucs_fr.mean()),
            "n_two_model_sig_raw": int(raw_sig[sub].sum()),
            "n_two_model_sig_bh": int(sig[sub].sum()),
        }
    gap = summary["crypto"]["mean_auc_ising"] - summary["stock"]["mean_auc_ising"]
    out = {"config": {"interval": INTERVAL, "train": TRAIN, "test": TEST,
                      "block": BLOCK, "n_boot": N_BOOT, "seed": SEED,
                      "models": ["ising", "free"], "n_lags": 12},
           "summary": summary,
           "class_mean_gap_ising": float(gap),
           "n_total": len(rows),
           "n_sig_bh_total": int(sig.sum()),
           "n_sig_raw_total": int(raw_sig.sum()),
           "rows": rows}
    (OUT / "wide_null.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    print(f"Wrote {OUT/'wide_null.json'}")


if __name__ == "__main__":
    main()
