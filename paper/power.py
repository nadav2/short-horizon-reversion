"""Power of the equity null: what effect sizes does "0.498, 3% significant"
actually rule out?

The cross-market contrast leans on a negative result for stocks, and absence
of evidence licenses evidence of absence only with a power calculation. This
module supplies it, entirely from the frozen wide-study outputs
(out/wide_oos/{asset}.npz):

  (a) per-stock: the moving-block-bootstrap SD of out-of-sample AUC (the same
      block geometry as the per-asset test) gives, under a normal
      approximation of the bootstrap law, the power of the one-sided 5% test
      against a true effect of the crypto class-mean size
      (AUC 0.5 + GAP with GAP = 0.031), and the minimum per-asset effect
      detectable with 99% power;

  (b) class level: a JOINT moving-block bootstrap of the stock class-mean AUC
      on the shared 15m grid (as in paper.dependence) gives a one-sided 95%
      upper bound on any common equity-class effect.

    uv run --active python -m paper.power

Writes out/power.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from .dependence import fast_auc
from .fetch_bulk_stocks import UNIVERSE

OUT = Path(__file__).resolve().parent / "out"
OOS = OUT / "wide_oos"
SLOT = 900
BLOCK, N_BOOT_ASSET, N_BOOT_JOINT, SEED = 384, 500, 1000, 17
GAP = 0.031                     # crypto class-mean AUC excess (primary test)
ALPHAS = (0.05, 0.01)
STOCK_SET = {s.lower().replace(".", "-") for s in UNIVERSE}


def asset_boot_sd(y, p, rng):
    """Moving-block-bootstrap SD of AUC, identical block geometry to wide.py."""
    n = len(y)
    nb = int(np.ceil(n / BLOCK))
    aucs = []
    for _ in range(N_BOOT_ASSET):
        starts = rng.integers(0, max(1, n - BLOCK + 1), size=nb)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(fast_auc(yb, p[idx]))
    return float(np.std(aucs, ddof=1)), len(aucs)


def main():
    rng = np.random.default_rng(SEED)
    per_stock, stock_series = [], []
    for f in sorted(OOS.glob("*.npz")):
        if f.stem not in STOCK_SET:
            continue
        d = np.load(f)
        y = d["actual"].astype(np.int8)
        p = d["p_ising"].astype(np.float64)
        sd, nb = asset_boot_sd(y, p, rng)
        row = {"asset": f.stem, "n_oos": int(len(y)),
               "auc": fast_auc(y, p), "boot_sd": sd}
        for a in ALPHAS:
            z_a = norm.ppf(1 - a)
            row[f"power_{a}"] = float(norm.cdf(GAP / sd - z_a))
        # minimum true AUC excess detectable with 99% power at the 5% level
        row["mde_99"] = float((norm.ppf(0.95) + norm.ppf(0.99)) * sd)
        per_stock.append(row)
        stock_series.append((d["ts"] // SLOT, y, p))
    print(f"{len(per_stock)} stocks")

    # ── joint bootstrap of the stock class-mean AUC ──────────────────────────
    lo = min(int(s.min()) for s, _, _ in stock_series)
    hi = max(int(s.max()) for s, _, _ in stock_series)
    T = hi - lo + 1
    dense = []
    for s, y, p in stock_series:
        ry = np.full(T, -1, np.int8)
        rp = np.full(T, np.nan, np.float32)
        ry[s - lo] = y
        rp[s - lo] = p
        dense.append((ry, rp))

    def class_mean(sel):
        vals = []
        for ry, rp in dense:
            yb = ry[sel]
            ok = (yb >= 0) & ~np.isnan(rp[sel])
            if ok.sum() < 500 or len(np.unique(yb[ok])) < 2:
                continue
            vals.append(fast_auc(yb[ok], rp[sel][ok]))
        return float(np.mean(vals)) if vals else float("nan")

    obs_mean = class_mean(np.arange(T))
    nb = int(np.ceil(T / BLOCK))
    boots = []
    for _ in range(N_BOOT_JOINT):
        starts = rng.integers(0, max(1, T - BLOCK + 1), size=nb)
        sel = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:T]
        m = class_mean(sel)
        if m == m:
            boots.append(m)
    boots = np.array(boots)

    powers = {str(a): np.array([r[f"power_{a}"] for r in per_stock]) for a in ALPHAS}
    sds = np.array([r["boot_sd"] for r in per_stock])
    mdes = np.array([r["mde_99"] for r in per_stock])
    res = {
        "config": {"gap": GAP, "block": BLOCK, "n_boot_asset": N_BOOT_ASSET,
                   "n_boot_joint": N_BOOT_JOINT, "seed": SEED},
        "n_stocks": len(per_stock),
        "n_oos_median": float(np.median([r["n_oos"] for r in per_stock])),
        "boot_sd": {"median": float(np.median(sds)), "p95": float(np.percentile(sds, 95)),
                    "max": float(sds.max())},
        "power_vs_crypto_gap": {
            a: {"min": float(powers[a].min()), "median": float(np.median(powers[a])),
                "frac_ge_0.99": float(np.mean(powers[a] >= 0.99))}
            for a in powers},
        "mde_99": {"median": float(np.median(mdes)), "p95": float(np.percentile(mdes, 95)),
                   "max": float(mdes.max())},
        "class_mean": {"obs": obs_mean,
                       "upper95_one_sided": float(np.percentile(boots, 95)),
                       "ci95": [float(np.percentile(boots, 2.5)),
                                float(np.percentile(boots, 97.5))]},
        "per_stock": per_stock,
    }
    (OUT / "power.json").write_text(json.dumps(res, indent=2))

    print(f"median n_OOS {res['n_oos_median']:.0f}; boot SD median "
          f"{res['boot_sd']['median']:.4f} (p95 {res['boot_sd']['p95']:.4f})")
    for a in powers:
        pw = res["power_vs_crypto_gap"][a]
        print(f"power vs +{GAP} at alpha={a}: min {pw['min']:.6f}  "
              f"median {pw['median']:.6f}  frac>=0.99 {pw['frac_ge_0.99']:.3f}")
    print(f"99%-power minimum detectable excess: median {res['mde_99']['median']:.4f}, "
          f"p95 {res['mde_99']['p95']:.4f}, max {res['mde_99']['max']:.4f}")
    cm = res["class_mean"]
    print(f"stock class mean {cm['obs']:.4f}; one-sided 95% upper bound "
          f"{cm['upper95_one_sided']:.4f}")
    print(f"Wrote {OUT/'power.json'}")


if __name__ == "__main__":
    main()
