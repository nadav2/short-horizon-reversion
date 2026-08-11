"""Perp replication and basis/funding conditioning.

(a) Does the reversal replicate on the USDT-M perpetual tape? (It should:
    perps carry most crypto volume and arbitrage ties them to spot.)
(b) Is the model's edge conditional on the spot-perp basis being stretched
    (perp-spot arbitrage channel) or on the funding rate (carry channel)?
    The funding-hour null (paper.mechanism) predicts both conditionings are
    flat; confirming that isolates the spot-flow account.

    uv run --active python -m paper.perp_test
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .compare_markets import SPAN, WINDOWS
from .models import IsingLogit
from .walkforward import walk_forward

HERE = Path(__file__).resolve().parent
PERP = HERE / "perp_data"
OUT = HERE / "out"
COINS = ["btc", "eth", "sol", "xrp"]


def load(coin):
    raw = [d for d in json.loads((PERP / f"{coin}-perp-15m.json").read_text())
           if SPAN[0] <= d["datetime"][:10] <= SPAN[1]]
    raw.sort(key=lambda d: d["timestamp"])
    ts = np.array([d["timestamp"] for d in raw], np.int64)
    ch = np.array([d["change"] for d in raw])
    ups = np.array([bool(d["up"]) for d in raw])
    pclose = np.array([d["close"] for d in raw])

    spot = {d["timestamp"]: d["close"]
            for d in json.loads((PERP / f"{coin}-spot-close.json").read_text())}
    sclose = np.array([spot.get(int(t), np.nan) for t in ts])
    basis = pclose / sclose - 1.0

    fr = json.loads((PERP / f"{coin}-funding.json").read_text())
    fr.sort(key=lambda d: d["fundingTime"])
    ftimes = np.array([d["fundingTime"] for d in fr], np.int64)
    frates = np.array([d["fundingRate"] for d in fr])
    # funding applicable at bar t: most recent settlement at or before t
    j = np.searchsorted(ftimes, ts, side="right") - 1
    fund = np.where(j >= 0, frates[np.clip(j, 0, None)], np.nan)
    return ch, ups, basis, fund


def bucket_auc(y, p, key, q=(1 / 3, 2 / 3), absval=False):
    k = np.abs(key) if absval else key
    ok = np.isfinite(k)
    lo, hi = np.nanquantile(k[ok], q)
    out = {}
    for name, mask in (("low", k <= lo), ("mid", (k > lo) & (k <= hi)), ("high", k > hi)):
        m = mask & ok
        out[name] = {"auc": float(roc_auc_score(y[m], p[m])) if m.sum() > 100
                     and len(np.unique(y[m])) > 1 else float("nan"),
                     "n": int(m.sum())}
    return out


def main():
    summary = {}
    P, Y, B, F = [], [], [], []
    for coin in COINS:
        ch, ups, basis, fund = load(coin)
        tr, te = WINDOWS["15m"]
        res, _ = walk_forward(ch, ups, tr, te, n_lags=12,
                              models=lambda: [IsingLogit(n_lags=12)])
        r = res["ising"]
        idx = r["idx"].astype(int)
        y, p = r["actuals"].astype(int), r["probs"]
        b_prev = basis[idx - 1]          # lagged, causal
        f_prev = fund[idx - 1]
        auc = roc_auc_score(y, p)
        A = float(np.mean([fp["A"] for fp in r["fold_params"]]))
        by_b = bucket_auc(y, p, b_prev, absval=True)
        by_f = bucket_auc(y, p, f_prev)
        print(f"  {coin:4s} perp AUC={auc:.4f} A={A:+.3f}  "
              f"|basis| terciles: " + " ".join(f"{by_b[k]['auc']:.4f}" for k in ("low", "mid", "high"))
              + "   funding terciles: " + " ".join(f"{by_f[k]['auc']:.4f}" for k in ("low", "mid", "high")),
              flush=True)
        summary[coin] = {"auc_perp": float(auc), "A_perp": A,
                         "by_abs_basis": by_b, "by_funding": by_f}
        P.append(p); Y.append(y); B.append(b_prev); F.append(f_prev)

    # pooled conditionings (per-coin terciles would differ; use per-coin
    # standardized ranks so buckets mean the same thing across coins)
    def ranks(x):
        r = np.full(len(x), np.nan)
        ok = np.isfinite(x)
        r[ok] = np.argsort(np.argsort(np.abs(x[ok]))) / max(1, ok.sum() - 1)
        return r

    yy, pp = np.concatenate(Y), np.concatenate(P)
    rb = np.concatenate([ranks(b) for b in B])
    rf = np.concatenate([ranks(f) for f in F])
    pooled_b = bucket_auc(yy, pp, rb)
    pooled_f = bucket_auc(yy, pp, rf)
    print("  pooled |basis| terciles: "
          + " ".join(f"{pooled_b[k]['auc']:.4f}" for k in ("low", "mid", "high")))
    print("  pooled |funding| terciles: "
          + " ".join(f"{pooled_f[k]['auc']:.4f}" for k in ("low", "mid", "high")))
    summary["pooled"] = {"by_abs_basis_rank": pooled_b, "by_abs_funding_rank": pooled_f}

    OUT.mkdir(exist_ok=True)
    (OUT / "perp_test.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'perp_test.json'}")


if __name__ == "__main__":
    main()
