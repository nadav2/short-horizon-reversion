"""Robustness / corroboration analyses for the cross-market paper.

1. Variance-ratio test (Lo-MacKinlay, heteroskedasticity-robust): a *model-free*
   test for mean-reversion. VR(q) < 1 => negative serial dependence (mean-reverting);
   VR(q) = 1 => random walk; VR(q) > 1 => momentum. Corroborates the Ising A<0 finding
   without using the Ising model at all.
2. Between-group permutation test on 15m AUC: is crypto's predictability significantly
   higher than traditional's, treating each asset's AUC as one observation?
3. Sub-period robustness: does the crypto-15m AUC hold in BOTH halves of the window?

    uv run --active python -m paper.robustness
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .compare_markets import CRYPTO, TRADITIONAL, ASSET_CLASS, WINDOWS, load_span, models
from .common import classification_metrics
from .walkforward import walk_forward

OUT = Path(__file__).resolve().parent / "out"


# ── 1. Variance-ratio test ───────────────────────────────────────────────────

def variance_ratio(x, q):
    """Lo-MacKinlay VR(q) with heteroskedasticity-robust z-stat.
    x = per-period returns (we use intra-candle returns; for 24h-continuous series
    these compound into the price path). Returns (VR, z, p_two_sided)."""
    x = np.asarray(x, float)
    n = len(x)
    mu = x.mean()
    dev = x - mu
    s2 = np.sum(dev ** 2)
    if s2 == 0 or n <= q:
        return float("nan"), float("nan"), float("nan")
    var1 = s2 / (n - 1)                                      # 1-period variance
    yq = np.convolve(x, np.ones(q), mode="valid")           # overlapping q-period sums
    m = q * (n - q + 1) * (1 - q / n)                        # Lo-MacKinlay scaling (incl. 1/q)
    varq = np.sum((yq - q * mu) ** 2) / m
    vr = varq / var1                                         # = 1 under a random walk
    # heteroskedasticity-consistent variance of VR(q): theta = sum_k [2(q-k)/q]^2 * delta_k
    theta = 0.0
    for k in range(1, q):
        num = np.sum((dev[k:] ** 2) * (dev[:-k] ** 2))
        delta = num / (s2 ** 2)
        theta += ((2 * (q - k) / q) ** 2) * delta
    z = (vr - 1) / np.sqrt(theta) if theta > 0 else float("nan")
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))) if np.isfinite(z) else float("nan")
    return float(vr), float(z), float(p)


def vr_table(interval="15m", qs=(2, 4, 8, 16)):
    rows = []
    for asset in CRYPTO + TRADITIONAL:
        try:
            _, ch, _ = load_span(asset, interval)
        except Exception:
            continue
        if len(ch) < 1000:
            continue
        cls = "crypto" if asset in CRYPTO else "traditional"
        rec = {"asset": asset, "class": cls, "asset_class": ASSET_CLASS[asset], "n": len(ch)}
        for q in qs:
            vr, z, p = variance_ratio(ch, q)
            rec[f"vr{q}"] = vr; rec[f"z{q}"] = z; rec[f"p{q}"] = p
        rows.append(rec)
    return rows


# ── 2. Between-group permutation test on 15m AUC ─────────────────────────────

def permutation_auc_diff(markets, interval="15m", n_perm=100000, seed=11):
    sel = [r for r in markets if r["interval"] == interval]
    aucs = np.array([r["ising_auc"] for r in sel])
    is_crypto = np.array([r["class"] == "crypto" for r in sel])
    nc = is_crypto.sum()
    obs = aucs[is_crypto].mean() - aucs[~is_crypto].mean()
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(aucs))
        grp = perm[:nc]
        diff = aucs[grp].mean() - np.delete(aucs, grp).mean()
        if diff >= obs:
            ge += 1
    return {"obs_diff": float(obs), "p_one_sided": (ge + 1) / (n_perm + 1),
            "n_crypto": int(nc), "n_trad": int((~is_crypto).sum()),
            "crypto_mean": float(aucs[is_crypto].mean()),
            "trad_mean": float(aucs[~is_crypto].mean())}


# ── 3. Sub-period robustness for crypto 15m ──────────────────────────────────

def subperiod(interval="15m", assets=None):
    assets = assets or [a for a in CRYPTO if a != "sol"]   # sol lacks 15m
    tr, te = WINDOWS[interval]
    from sklearn.metrics import roc_auc_score
    out = []
    for asset in assets:
        _, ch, ups = load_span(asset, interval)
        res, _ = walk_forward(ch, ups, tr, te, n_lags=12, models=models)
        p = res["ising"]["probs"]; y = res["ising"]["actuals"].astype(int)
        h = len(p) // 2
        out.append({"asset": asset, "n": int(len(p)),
                    "auc_full": float(roc_auc_score(y, p)),
                    "auc_h1": float(roc_auc_score(y[:h], p[:h])),
                    "auc_h2": float(roc_auc_score(y[h:], p[h:]))})
    return out


def main():
    markets = json.loads((OUT / "markets.json").read_text())

    print("=== 1. Variance-ratio test (15m) — VR<1 = mean-reverting (model-free) ===")
    vr = vr_table("15m")
    print(f"  {'asset':5s} {'class':11s}  VR2     VR4     VR8     VR16    (z16, p16)")
    for r in vr:
        star = "*" if (np.isfinite(r["p16"]) and r["p16"] < 0.05) else " "
        print(f"  {r['asset']:5s} {r['asset_class']:11s} {r['vr2']:.3f}  {r['vr4']:.3f}  "
              f"{r['vr8']:.3f}  {r['vr16']:.3f}  (z={r['z16']:+.2f} p={r['p16']:.3f}){star}")
    for cls in ("crypto", "traditional"):
        sub = [r for r in vr if r["class"] == cls]
        for q in (2, 4, 8, 16):
            pass
        mvr = {q: np.mean([r[f"vr{q}"] for r in sub]) for q in (2, 4, 8, 16)}
        nsig = sum(1 for r in sub if np.isfinite(r["p16"]) and r["p16"] < 0.05 and r["vr16"] < 1)
        print(f"  MEAN {cls:11s}: VR2={mvr[2]:.3f} VR4={mvr[4]:.3f} VR8={mvr[8]:.3f} "
              f"VR16={mvr[16]:.3f}  ({nsig}/{len(sub)} sig VR16<1)")

    print("\n=== 2. Permutation test: crypto vs traditional 15m AUC ===")
    pt = permutation_auc_diff(markets, "15m")
    print(f"  crypto mean AUC={pt['crypto_mean']:.4f} ({pt['n_crypto']}), "
          f"traditional={pt['trad_mean']:.4f} ({pt['n_trad']})  "
          f"diff={pt['obs_diff']:+.4f}  permutation p={pt['p_one_sided']:.5f}")

    print("\n=== 3. Sub-period robustness, crypto 15m (AUC in each half) ===")
    sp = subperiod("15m")
    for r in sp:
        print(f"  {r['asset']:5s} full={r['auc_full']:.4f}  H1={r['auc_h1']:.4f}  H2={r['auc_h2']:.4f}")

    (OUT / "robustness.json").write_text(json.dumps(
        {"variance_ratio": vr, "permutation": pt, "subperiod": sp}, indent=2))
    print(f"\nWrote {OUT/'robustness.json'}")


if __name__ == "__main__":
    main()
