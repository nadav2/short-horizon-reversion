"""Paired block-bootstrap CIs for the wrapper-minus-underlying AUC gap.

wrapper_events.py and panel.py score each leg on its own tape, so their
per-leg bootstrap CIs describe each AUC separately and say nothing about the
uncertainty of the *difference* the event-study figure plots. This module
computes that difference properly:

  * wrapper and underlying are intersected on their 15m timestamps first, so
    both legs run on exactly the same bars (the ETF tape is the binding one:
    it is missing ~4% of the underlying's RTH slots);
  * each leg then gets the identical RTH walk-forward (train 60 / test 10
    trading days), which on equal-length inputs yields bar-for-bar aligned
    out-of-sample series;
  * a moving-block bootstrap resamples block indices ONCE per replicate and
    applies them to both legs, so the resampled gap preserves the pairing.

Reported per cell: the gap on the shared bars and its 95% percentile CI.
Reads wrapper_data/, multiyear_data/ and DATA_DIR; writes
out/wrapper_gap_boot.json.

    uv run --active python -m paper.wrapper_gap_boot
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

from .common import DATA_DIR
from .compare_markets import N_BOOT, SEED, block_boot_idx
from .models import IsingLogit
from .walkforward import walk_forward

HERE = Path(__file__).resolve().parent
WRAP = HERE / "wrapper_data"
MULTI = HERE / "multiyear_data"
OUT = HERE / "out"
NY = ZoneInfo("America/New_York")
BARS = 26                                   # RTH 15m bars per day
TRAIN, TEST, BLOCK = 60 * BARS, 10 * BARS, 4 * BARS

PANEL_SPAN = ("2025-01-01", "2026-02-11")

# (cell, wrapper, underlying, start, end, group)
CELLS = [
    ("ibit_birth", "ibit", "btc", "2024-01-11", "2024-07-11", "launch"),
    ("fbtc_birth", "fbtc", "btc", "2024-01-11", "2024-07-11", "launch"),
    ("gbtc_birth", "gbtc", "btc", "2024-01-11", "2024-07-11", "launch"),
    ("etha_birth", "etha", "eth", "2024-07-23", "2025-01-23", "launch"),
    ("feth_birth", "feth", "eth", "2024-07-23", "2025-01-23", "launch"),
    ("ethe_birth", "ethe", "eth", "2024-07-23", "2025-01-23", "launch"),
    ("bito_early", "bito", "btc", "2021-10-19", "2023-12-31", "futures"),
    ("bito_late",  "bito", "btc", "2024-01-11", "2026-02-11", "futures"),
    ("bitu_panel", "bitu", "btc", *PANEL_SPAN, "leveraged"),
    ("ethu_panel", "ethu", "eth", *PANEL_SPAN, "leveraged"),
    ("ibit_full",  "ibit", "btc", "2024-01-11", "2026-02-11", "full"),
    ("fbtc_full",  "fbtc", "btc", "2024-01-11", "2026-02-11", "full"),
    ("gbtc_etf",   "gbtc", "btc", "2024-01-11", "2026-02-11", "full"),
    ("ethe_etf",   "ethe", "eth", "2024-07-23", "2026-02-11", "full"),
]


def _path(sid: str) -> Path:
    """Wrapper tapes live in wrapper_data/, underlyings in multiyear_data/;
    the leveraged funds only exist in the shared crypto-data directory."""
    for cand in (WRAP / f"{sid}-15m.json", MULTI / f"{sid}-15m.json",
                 DATA_DIR / f"{sid}-15m.json"):
        if cand.exists():
            return cand
    raise FileNotFoundError(sid)


def load_rth(sid: str, start: str, end: str):
    """RTH 15m bars as (timestamps, changes, ups)."""
    raw = [d for d in json.loads(_path(sid).read_text())
           if start <= d["datetime"][:10] <= end]
    raw.sort(key=lambda d: d["timestamp"])
    ts, ch, up = [], [], []
    for d in raw:
        t = datetime.fromtimestamp(d["timestamp"], timezone.utc).astimezone(NY)
        if t.weekday() < 5 and (9, 30) <= (t.hour, t.minute) < (16, 0):
            ts.append(int(d["timestamp"]))
            ch.append(float(d.get("change", 0.0)))
            up.append(bool(d["up"]))
    return np.array(ts), np.array(ch, float), np.array(up, bool)


def oos(ch, ups):
    """Concatenated OOS probabilities and actuals from the RTH walk-forward."""
    res, _ = walk_forward(ch, ups, TRAIN, TEST, n_lags=12,
                          models=lambda: [IsingLogit(n_lags=12)])
    r = res["ising"]
    return r["probs"], r["actuals"].astype(int)


def main() -> None:
    rows = []
    for cell, wsid, usid, start, end, group in CELLS:
        w_ts, w_ch, w_up = load_rth(wsid, start, end)
        u_ts, u_ch, u_up = load_rth(usid, start, end)

        # Intersect on timestamps so both legs run on identical bars.
        keep = np.intersect1d(w_ts, u_ts)
        wi = np.isin(w_ts, keep)
        ui = np.isin(u_ts, keep)
        w_ch, w_up, u_ch, u_up = w_ch[wi], w_up[wi], u_ch[ui], u_up[ui]
        if len(w_ch) < TRAIN + TEST:
            print(f"  {cell}: too few shared bars ({len(w_ch)})")
            continue

        pw, yw = oos(w_ch, w_up)
        pu, yu = oos(u_ch, u_up)
        assert len(yw) == len(yu), (cell, len(yw), len(yu))

        auc_w, auc_u = roc_auc_score(yw, pw), roc_auc_score(yu, pu)
        rng = np.random.default_rng(SEED)
        diffs = []
        for _ in range(N_BOOT):
            bi = block_boot_idx(len(yw), BLOCK, rng)          # one draw, both legs
            yb_w, yb_u = yw[bi], yu[bi]
            if yb_w.min() == yb_w.max() or yb_u.min() == yb_u.max():
                continue
            diffs.append(roc_auc_score(yb_w, pw[bi]) - roc_auc_score(yb_u, pu[bi]))
        diffs = np.array(diffs)

        row = {
            "cell": cell, "wrapper": wsid, "underlying": usid, "group": group,
            "start": start, "end": end,
            "n_shared_bars": int(len(w_ch)), "n_oos": int(len(yw)),
            "auc_wrapper": float(auc_w), "auc_underlying": float(auc_u),
            "gap": float(auc_w - auc_u),
            "gap_ci": [float(np.percentile(diffs, 2.5)),
                       float(np.percentile(diffs, 97.5))],
            "p_two_sided": float(2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))),
        }
        rows.append(row)
        print(f"  {cell:11s} n_oos={row['n_oos']:6d}  w={auc_w:.4f} u={auc_u:.4f}  "
              f"gap={row['gap']:+.4f} CI[{row['gap_ci'][0]:+.4f},"
              f"{row['gap_ci'][1]:+.4f}]  p={row['p_two_sided']:.2f}", flush=True)

    gaps = np.array([r["gap"] for r in rows])
    summary = {
        "n_cells": len(rows),
        "mean_abs_gap": float(np.mean(np.abs(gaps))),
        "n_ci_covering_zero": int(sum(r["gap_ci"][0] <= 0 <= r["gap_ci"][1]
                                      for r in rows)),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "wrapper_gap_boot.json").write_text(json.dumps(
        {"config": {"train": TRAIN, "test": TEST, "block": BLOCK,
                    "n_boot": N_BOOT, "seed": SEED,
                    "note": "legs intersected on timestamps; blocks resampled "
                            "once per replicate and applied to both legs"},
         "summary": summary, "rows": rows}, indent=2))
    print(f"\n{summary}\nwrote {OUT / 'wrapper_gap_boot.json'}")


if __name__ == "__main__":
    main()
