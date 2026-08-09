"""Regular-trading-hours robustness for the WIDE stock/ETF universe (referee
point 4).

The focal RTH check (`paper.rth`) covers only the 8 focal US-listed
instruments. A referee can object that the wide-universe crypto-vs-stock
contrast compares 24/7 crypto bars against 24-hour Alpaca bars whose thin
overnight session is not comparable. This script closes that gap: it re-runs
the matched 15m walk-forward on RTH-only bars (09:30-16:00 America/New_York,
DST-aware) for every stock/ETF in the frozen wide universe, and reports the
class-mean AUC and both-model significant fraction for direct comparison with
the 24/7 crypto numbers (which are unaffected, crypto having no session).

    uv run --active python -m paper.wide_rth
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

from .models import ARLogit, IsingLogit
from .walkforward import walk_forward

BULK = Path(__file__).resolve().parent / "bulk_data"
OUT = Path(__file__).resolve().parent / "out"
NY = ZoneInfo("America/New_York")
BARS_PER_DAY = 26
TRAIN, TEST = 60 * BARS_PER_DAY, 10 * BARS_PER_DAY
BLOCK, N_BOOT, SEED = 4 * BARS_PER_DAY, 300, 5


def load_rth(asset: str):
    raw = json.loads((BULK / f"{asset}-15m.json").read_text())
    raw.sort(key=lambda d: d["timestamp"])
    keep = []
    for d in raw:
        t = datetime.fromtimestamp(d["timestamp"], timezone.utc).astimezone(NY)
        if t.weekday() < 5 and (9, 30) <= (t.hour, t.minute) < (16, 0):
            keep.append(d)
    ch = np.array([d.get("change", 0.0) for d in keep], float)
    ups = np.array([bool(d["up"]) for d in keep], bool)
    return ch, ups


def block_idx(n, rng):
    nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, max(1, n - BLOCK + 1), size=nb)
    return np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]


def models():
    return [IsingLogit(n_lags=12), ARLogit("free", n_lags=12)]


def main():
    wide = json.loads((OUT / "wide.json").read_text())
    stocks = [r["asset"] for r in wide if r["class"] == "stock"]
    print(f"{len(stocks)} wide stocks/ETFs; re-running RTH-only matched walk-forward")
    rows = []
    for i, asset in enumerate(stocks):
        try:
            ch, ups = load_rth(asset)
        except FileNotFoundError:
            continue
        if len(ch) < TRAIN + TEST:
            continue
        try:
            res, nf = walk_forward(ch, ups, TRAIN, TEST, n_lags=12, models=models)
        except Exception as e:
            print(f"  {asset}: FAILED {e}")
            continue
        y = res["ising"]["actuals"].astype(int)
        if len(y) < 500 or len(np.unique(y)) < 2:
            continue
        p_is, p_fr = res["ising"]["probs"], res["free"]["probs"]
        rng = np.random.default_rng(SEED)
        ge_is = ge_fr = 0
        for _ in range(N_BOOT):
            idx = block_idx(len(y), rng)
            yb = y[idx]
            if len(np.unique(yb)) < 2:
                continue
            ge_is += roc_auc_score(yb, p_is[idx]) <= 0.5
            ge_fr += roc_auc_score(yb, p_fr[idx]) <= 0.5
        rows.append({"asset": asset, "n_rth_bars": int(len(ch)), "n_oos": int(len(y)),
                     "auc_ising": float(roc_auc_score(y, p_is)),
                     "auc_free": float(roc_auc_score(y, p_fr)),
                     "p_ising": ge_is / N_BOOT, "p_free": ge_fr / N_BOOT,
                     "A": float(np.mean([fp["A"] for fp in res["ising"]["fold_params"]]))})
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(stocks)}] kept {len(rows)}", flush=True)

    aucs = np.array([r["auc_ising"] for r in rows])
    sig_both = np.mean([r["p_ising"] < 0.05 and r["p_free"] < 0.05 for r in rows])
    As = np.array([r["A"] for r in rows])
    summary = {"n_assets": len(rows),
               "mean_auc_ising": float(aucs.mean()),
               "median_auc_ising": float(np.median(aucs)),
               "frac_auc_gt05": float(np.mean(aucs > 0.5)),
               "frac_sig_both": float(sig_both),
               "mean_A": float(As.mean()), "frac_A_neg": float(np.mean(As < 0))}
    (OUT / "wide_rth.json").write_text(json.dumps({"summary": summary, "assets": rows}, indent=2))
    print(f"\n=== WIDE RTH-ONLY STOCKS (n={summary['n_assets']}) ===")
    print(f"  mean Ising AUC={summary['mean_auc_ising']:.4f} "
          f"median={summary['median_auc_ising']:.4f} frac>0.5={summary['frac_auc_gt05']:.2f}")
    print(f"  both-model significant: {summary['frac_sig_both']*100:.1f}%")
    print(f"  coupling A mean={summary['mean_A']:+.3f} frac<0={summary['frac_A_neg']:.2f}")
    print(f"Wrote {OUT/'wide_rth.json'}")


if __name__ == "__main__":
    main()
