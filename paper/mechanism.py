"""Mechanism tests: WHEN does the crypto directional signal concentrate?

If the effect is microstructural overreaction-correction, predictability should
(a) vary with the trading session / time-of-day, and (b) be stronger in
high-volatility periods (more overreaction to correct). We test both by tagging
every out-of-sample prediction with its UTC hour and its trailing realized
volatility, then computing AUC within each bucket, pooled by asset class.

    uv run --active python -m paper.mechanism
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .compare_markets import CRYPTO, TRADITIONAL, WINDOWS, load_span, models
from .walkforward import walk_forward

OUT = Path(__file__).resolve().parent / "out"
VOL_WIN = 96            # trailing window for realized vol (1 day at 15m)
CRYPTO_15M = [a for a in CRYPTO if a != "sol"]   # sol lacks 15m matched data


def collect(asset, interval="15m"):
    """Walk-forward OOS predictions tagged with UTC hour and trailing volatility."""
    dts, ch, ups = load_span(asset, interval)
    tr, te = WINDOWS[interval]
    res, _ = walk_forward(ch, ups, tr, te, n_lags=12, models=models)
    idx = res["ising"]["idx"].astype(int)
    p = res["ising"]["probs"]
    y = res["ising"]["actuals"].astype(int)
    hours = np.array([int(dts[i][11:13]) for i in idx])
    # trailing realized vol (std of prior VOL_WIN returns), causal
    vol = np.array([ch[max(0, i - VOL_WIN):i].std() if i >= 10 else np.nan for i in idx])
    return p, y, hours, vol


def auc_safe(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 and len(y) > 50 else float("nan")


def pool(assets):
    P, Y, H, V = [], [], [], []
    for a in assets:
        p, y, h, v = collect(a)
        # asset-relative volatility terciles (so terciles mean the same across assets)
        ok = np.isfinite(v)
        vt = np.full(len(v), -1)
        q1, q2 = np.nanquantile(v[ok], [1 / 3, 2 / 3])
        vt[ok] = np.where(v[ok] <= q1, 0, np.where(v[ok] <= q2, 1, 2))
        P.append(p); Y.append(y); H.append(h); V.append(vt)
    return (np.concatenate(P), np.concatenate(Y), np.concatenate(H), np.concatenate(V))


def analyze(label, assets):
    p, y, h, vt = pool(assets)
    overall = auc_safe(y, p)
    by_hour = {}
    for hr in range(24):
        m = h == hr
        by_hour[hr] = {"auc": auc_safe(y[m], p[m]), "n": int(m.sum())}
    by_vol = {}
    for t, name in [(0, "low"), (1, "mid"), (2, "high")]:
        m = vt == t
        by_vol[name] = {"auc": auc_safe(y[m], p[m]), "n": int(m.sum())}
    print(f"\n[{label}]  overall AUC={overall:.4f}  n={len(y)}")
    print("  vol regime:  " + "  ".join(
        f"{k}={by_vol[k]['auc']:.4f}(n={by_vol[k]['n']})" for k in ("low", "mid", "high")))
    # session summary: UTC hour buckets
    def blk(lo, hi):
        m = (h >= lo) & (h < hi)
        return auc_safe(y[m], p[m]), int(m.sum())
    for name, lo, hi in [("Asia 00-08", 0, 8), ("EU 08-14", 8, 14),
                         ("US 14-21", 14, 21), ("late 21-24", 21, 24)]:
        a, n = blk(lo, hi)
        print(f"  {name}: AUC={a:.4f} (n={n})")
    return {"label": label, "overall_auc": overall, "by_hour": by_hour, "by_vol": by_vol}


def main():
    out = {"crypto": analyze("crypto 15m", CRYPTO_15M),
           "traditional": analyze("traditional 15m", TRADITIONAL)}
    (OUT / "mechanism.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT/'mechanism.json'}")


if __name__ == "__main__":
    main()
