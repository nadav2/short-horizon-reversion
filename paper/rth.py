"""Regular-trading-hours robustness for the traditional instruments.

The focal study uses 24-hour bars for stocks/ETFs, which include thin pre/post-
market candles. Stale or bounce-dominated after-hours quotes could in principle
*create* spurious structure (or mask real structure). This check re-runs the
matched 15m walk-forward on RTH bars only (09:30-16:00 America/New_York, DST-
aware, bar start times). Window geometry is calendar-matched: 60 trading days of
training (60 x 26 RTH bars), 10 days of testing, mirroring the 60/10 calendar-day
geometry of the 24h study.

    uv run --active python -m paper.rth
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

from .common import DATA_DIR
from .compare_markets import (ASSET_CLASS, N_BOOT, SEED, SPAN, TRADITIONAL,
                              autocorr1, block_boot_idx)
from .walkforward import walk_forward
from .models import ARLogit, IsingLogit

OUT = Path(__file__).resolve().parent / "out"
NY = ZoneInfo("America/New_York")
BARS_PER_DAY = 26                      # 09:30..15:45 starts at 15m
TRAIN, TEST = 60 * BARS_PER_DAY, 10 * BARS_PER_DAY
BLOCK = 4 * BARS_PER_DAY               # ~4 trading days


def load_rth(asset: str):
    raw = json.loads((DATA_DIR / f"{asset}-15m.json").read_text())
    raw = [d for d in raw if SPAN[0] <= d["datetime"][:10] <= SPAN[1]]
    raw.sort(key=lambda d: d["timestamp"])
    keep = []
    for d in raw:
        t = datetime.fromtimestamp(d["timestamp"], timezone.utc).astimezone(NY)
        if t.weekday() < 5 and (9, 30) <= (t.hour, t.minute) < (16, 0):
            keep.append(d)
    ch = np.array([d.get("change", 0.0) for d in keep], float)
    ups = np.array([bool(d["up"]) for d in keep], bool)
    return ch, ups


def models():
    return [IsingLogit(n_lags=12), ARLogit("free", n_lags=12)]


def main():
    rows = []
    us_listed = [a for a in TRADITIONAL if ASSET_CLASS.get(a) != "fx"]
    for asset in us_listed:
        try:
            ch, ups = load_rth(asset)
        except FileNotFoundError:
            continue
        if len(ch) < TRAIN + TEST:
            print(f"  {asset}: only {len(ch)} RTH bars, skipped")
            continue
        res, nf = walk_forward(ch, ups, TRAIN, TEST, n_lags=12, models=models)
        y = res["ising"]["actuals"].astype(int)
        if len(np.unique(y)) < 2:
            continue
        rng = np.random.default_rng(SEED)
        row = {"asset": asset, "n_rth_bars": int(len(ch)), "n_oos": int(len(y)),
               "n_folds": nf, "ac1_rth": autocorr1(ch)}
        for key in ("ising", "free"):
            p = res[key]["probs"]
            ge = n_eff = 0
            for _ in range(N_BOOT):
                idx = block_boot_idx(len(y), BLOCK, rng)
                yb = y[idx]
                if yb.min() == yb.max():
                    continue
                n_eff += 1
                ge += roc_auc_score(yb, p[idx]) <= 0.5
            row[f"{key}_auc"] = float(roc_auc_score(y, p))
            row[f"{key}_p_auc_gt05"] = ge / max(n_eff, 1)
        row["A"] = float(np.mean([fp["A"] for fp in res["ising"]["fold_params"]]))
        rows.append(row)
        print(f"  {asset:5s} RTH bars={row['n_rth_bars']:5d}  "
              f"AUC ising={row['ising_auc']:.4f} (p={row['ising_p_auc_gt05']:.3f})  "
              f"free={row['free_auc']:.4f} (p={row['free_p_auc_gt05']:.3f})  A={row['A']:+.3f}")

    n_sig_both = sum(1 for r in rows
                     if r["ising_p_auc_gt05"] < 0.05 and r["free_p_auc_gt05"] < 0.05)
    summary = {"n_assets": len(rows), "sig_both": n_sig_both,
               "mean_ising_auc": float(np.mean([r["ising_auc"] for r in rows]))}
    (OUT / "rth.json").write_text(json.dumps({"summary": summary, "assets": rows}, indent=2))
    print(f"\nRTH summary: mean Ising AUC={summary['mean_ising_auc']:.4f}, "
          f"both-model significant {n_sig_both}/{len(rows)}")
    print(f"Wrote {OUT/'rth.json'}")


if __name__ == "__main__":
    main()
