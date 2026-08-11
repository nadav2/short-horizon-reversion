"""Gold-standard uncertainty check: bootstrap the ENTIRE procedure.

The paper's per-asset inference (wide.py, compare_markets.py) resamples the
concatenated OOS prediction series, holding the fitted walk-forward models
fixed -- it quantifies sampling variability CONDITIONAL on the fits and is
acknowledged to understate total uncertainty. This module runs the expensive
complement on a representative subset: a moving-block bootstrap of the RAW
(feature, label) series, with the complete walk-forward -- including the
per-fold alpha profiling and MLE refit of every model -- re-run inside every
resample. If the refit-inclusive intervals barely widen relative to the
conditional ones, the conditioning is innocuous.

Subset: the focal coins (btc, eth, xrp) plus N_PER_CLASS crypto pairs and
N_PER_CLASS stocks chosen as evenly spaced quantiles of the wide-study Ising
AUC (deterministic, spans each class distribution).

    uv run --active python -m paper.full_boot [--b 200] [--per-class 25]
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .models import ARLogit, IsingLogit
from .walkforward import walk_forward
from .wide import BLOCK, BULK, INTERVAL, N_BOOT, OUT, TEST, TRAIN, block_idx, load

SEED = 20260810
N_WORKERS = 8
FOCAL = ["btc", "eth", "xrp"]


def pick_subset(per_class: int) -> list[str]:
    """Evenly spaced Ising-AUC quantiles per class from the frozen wide study,
    focal coins forced into the crypto slice."""
    rows = json.loads((OUT / "wide.json").read_text())
    chosen: list[str] = []
    for cls in ("crypto", "stock"):
        sub = sorted((r for r in rows if r["class"] == cls),
                     key=lambda r: r["auc_ising"])
        idx = np.unique(np.linspace(0, len(sub) - 1, per_class).round().astype(int))
        names = [sub[i]["asset"] for i in idx]
        if cls == "crypto":
            names = list(dict.fromkeys(FOCAL + names))[:max(per_class, len(FOCAL))]
        chosen += names
    return chosen


def series_block_boot(ch, ups, rng):
    """Jointly resample the raw (feature, label) series in moving blocks."""
    idx = block_idx(len(ch), rng)
    return ch[idx], ups[idx]


def one_replication(args):
    ch, ups, rep_seed = args
    rng = np.random.default_rng(rep_seed)
    ch_b, ups_b = series_block_boot(ch, ups, rng)
    factory = lambda: [IsingLogit(n_lags=12), ARLogit("free", n_lags=12)]
    try:
        res, _ = walk_forward(ch_b, ups_b, TRAIN, TEST, n_lags=12, models=factory)
    except Exception:
        return None
    y = res["ising"]["actuals"].astype(int)
    if len(y) < 2000 or len(np.unique(y)) < 2:
        return None
    return (roc_auc_score(y, res["ising"]["probs"]),
            roc_auc_score(y, res["free"]["probs"]))


def conditional_ci(y, p, n_boot=N_BOOT, seed=5):
    """The paper's standard conditional bootstrap, for the width comparison."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = block_idx(len(y), rng)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, p[idx]))
    aucs = np.array(aucs)
    return ([float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
            float(np.mean(aucs <= 0.5)))


def evaluate_asset(asset: str, B: int, pool: Pool) -> dict | None:
    path = BULK / f"{asset}-{INTERVAL}.json"
    if not path.exists():
        return None
    ch, ups, _ = load(path)
    if len(ch) < TRAIN + TEST + 2000:
        return None

    # point estimate + conditional CI on the original series
    factory = lambda: [IsingLogit(n_lags=12), ARLogit("free", n_lags=12)]
    res, _ = walk_forward(ch, ups, TRAIN, TEST, n_lags=12, models=factory)
    y = res["ising"]["actuals"].astype(int)
    p_is, p_fr = res["ising"]["probs"], res["free"]["probs"]
    auc_is = float(roc_auc_score(y, p_is))
    ci_cond, p_cond = conditional_ci(y, p_is)

    # full-procedure bootstrap: refit everything inside each resample
    seeds = np.random.SeedSequence([SEED, hash(asset) & 0x7FFFFFFF]).spawn(B)
    reps = pool.map(one_replication, [(ch, ups, s) for s in seeds])
    reps = [r for r in reps if r is not None]
    if len(reps) < B * 0.9:
        return None
    full_is = np.array([r[0] for r in reps])
    full_fr = np.array([r[1] for r in reps])
    ci_full = [float(np.percentile(full_is, 2.5)), float(np.percentile(full_is, 97.5))]
    return {"asset": asset,
            "auc_ising": auc_is,
            "auc_free": float(roc_auc_score(y, p_fr)),
            "cond_ci": ci_cond, "cond_p": p_cond,
            "full_ci": ci_full,
            "full_p": float(np.mean(full_is <= 0.5)),
            "full_p_free": float(np.mean(full_fr <= 0.5)),
            "full_mean": float(full_is.mean()),
            "cond_halfwidth": (ci_cond[1] - ci_cond[0]) / 2,
            "full_halfwidth": (ci_full[1] - ci_full[0]) / 2,
            "n_reps": len(reps), "n_oos": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=200)
    ap.add_argument("--per-class", type=int, default=25)
    args = ap.parse_args()

    assets = pick_subset(args.per_class)
    rows_by_class = json.loads((OUT / "wide.json").read_text())
    cls_of = {r["asset"]: r["class"] for r in rows_by_class}
    print(f"{len(assets)} assets, B={args.b}, {N_WORKERS} workers")

    results = []
    with Pool(N_WORKERS) as pool:
        for i, a in enumerate(assets):
            t0 = time.time()
            r = evaluate_asset(a, args.b, pool)
            if r is None:
                print(f"  {a}: skipped", flush=True)
                continue
            r["class"] = cls_of.get(a, "crypto")
            results.append(r)
            print(f"  [{i+1}/{len(assets)}] {a:8s} auc={r['auc_ising']:.4f}  "
                  f"cond=[{r['cond_ci'][0]:.4f},{r['cond_ci'][1]:.4f}] "
                  f"full=[{r['full_ci'][0]:.4f},{r['full_ci'][1]:.4f}]  "
                  f"p_cond={r['cond_p']:.3f} p_full={r['full_p']:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    hw_ratio = np.array([r["full_halfwidth"] / r["cond_halfwidth"]
                         for r in results if r["cond_halfwidth"] > 0])
    crypto = [r for r in results if r["class"] == "crypto"]
    stocks = [r for r in results if r["class"] == "stock"]
    summary = {
        "n_assets": len(results), "B": args.b,
        "median_halfwidth_ratio": float(np.median(hw_ratio)),
        "q25_q75_halfwidth_ratio": [float(np.percentile(hw_ratio, 25)),
                                    float(np.percentile(hw_ratio, 75))],
        "crypto_sig_cond": sum(r["cond_p"] < 0.05 for r in crypto),
        "crypto_sig_full": sum(r["full_p"] < 0.05 for r in crypto),
        "crypto_sig_full_two_model": sum(max(r["full_p"], r["full_p_free"]) < 0.05
                                         for r in crypto),
        "crypto_n": len(crypto),
        "stock_sig_cond": sum(r["cond_p"] < 0.05 for r in stocks),
        "stock_sig_full": sum(r["full_p"] < 0.05 for r in stocks),
        "stock_n": len(stocks),
        "mean_full_minus_point": float(np.mean([r["full_mean"] - r["auc_ising"]
                                                for r in results])),
    }
    out = {"config": {"interval": INTERVAL, "train": TRAIN, "test": TEST,
                      "block": BLOCK, "seed": SEED, "B": args.b},
           "summary": summary, "rows": results}
    (OUT / "full_boot.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT/'full_boot.json'}")


if __name__ == "__main__":
    main()
