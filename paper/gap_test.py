"""Skip-one-bar (gap) robustness test against stale-price / bid-ask-bounce artifacts.

The classic microstructure worry: with last-trade candles, a stale close (or a
close at the bid/ask) mechanically produces one-bar reversal -- exactly the
negative-coupling signature this paper reports. Genuine multi-lag mean
reversion with a 3-6 candle memory kernel should survive *skipping the most
recent candle*; pure bounce/staleness, which lives entirely in the boundary
between adjacent bars, should not.

Implementation: the feature series is shifted by one candle
(``ch_gap[t] = ch[t-1]``), so the model predicts the direction of candle ``t``
from candles ``t-2, ..., t-N-1`` -- a full one-bar gap between the last
feature and the label. Protocol (windows, models, alpha grid, bootstrap) is
otherwise identical to ``wide.py``; the no-gap baseline is read from
``out/wide.json``.

    uv run --active python -m paper.gap_test            # crypto pairs only
    uv run --active python -m paper.gap_test --all      # include stocks
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .fetch_bulk_stocks import UNIVERSE
from .models import ARLogit, IsingLogit
from .walkforward import walk_forward

BULK = Path(__file__).resolve().parent / "bulk_data"
OUT = Path(__file__).resolve().parent / "out"
TRAIN, TEST = 5760, 960          # matched 15m geometry (as in wide.py)
BLOCK, N_BOOT, SEED = 384, 300, 5
STOCK_SET = {s.lower().replace(".", "-") for s in UNIVERSE}


def block_idx(n, rng):
    nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, max(1, n - BLOCK + 1), size=nb)
    return np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]


def evaluate_gap(path_str: str):
    path = Path(path_str)
    asset = path.name[: -len("-15m.json")]
    raw = json.loads(path.read_text())
    raw.sort(key=lambda d: d["timestamp"])
    ch = np.array([d.get("change", 0.0) for d in raw], float)
    ups = np.array([bool(d["up"]) for d in raw], bool)
    if len(ch) < TRAIN + TEST + 2000:
        return None
    # one-bar gap: feature series lags the label series by one extra candle
    ch_gap = np.concatenate([[0.0], ch[:-1]])

    def factory():
        return [IsingLogit(n_lags=12), ARLogit("free", n_lags=12)]

    res, _ = walk_forward(ch_gap, ups, TRAIN, TEST, n_lags=12, models=factory)
    y = res["ising"]["actuals"].astype(int)
    if len(y) < 2000 or len(np.unique(y)) < 2:
        return None
    p_is, p_fr = res["ising"]["probs"], res["free"]["probs"]
    A = float(np.mean([fp["A"] for fp in res["ising"]["fold_params"]]))
    rng = np.random.default_rng(SEED)
    ge_is = ge_fr = 0
    for _ in range(N_BOOT):
        idx = block_idx(len(y), rng)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        ge_is += roc_auc_score(yb, p_is[idx]) <= 0.5
        ge_fr += roc_auc_score(yb, p_fr[idx]) <= 0.5
    return {"asset": asset,
            "class": "stock" if asset in STOCK_SET else "crypto",
            "auc_gap_ising": float(roc_auc_score(y, p_is)),
            "auc_gap_free": float(roc_auc_score(y, p_fr)),
            "p_gap_ising": ge_is / N_BOOT, "p_gap_free": ge_fr / N_BOOT,
            "A_gap": A, "n_oos": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include stocks/ETFs")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    base = {r["asset"]: r for r in json.loads((OUT / "wide.json").read_text())}
    files = []
    for f in sorted(BULK.glob("*-15m.json")):
        asset = f.name[: -len("-15m.json")]
        if asset not in base:
            continue
        if not args.all and asset in STOCK_SET:
            continue
        files.append(str(f))
    print(f"gap test on {len(files)} assets, {args.workers} workers")

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(evaluate_gap, f): f for f in files}
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            if r is None:
                continue
            b = base[r["asset"]]
            r["auc_ising"] = b["auc_ising"]
            r["p_ising"] = b["p_ising"]
            r["p_free"] = b["p_free"]
            r["A"] = b["A"]
            rows.append(r)
            if (i + 1) % 20 == 0:
                el = time.time() - t0
                print(f"  [{i+1}/{len(files)}] {el:.0f}s elapsed")

    (OUT / "gap_test.json").write_text(json.dumps(rows, indent=2))

    for cls in ("crypto", "stock"):
        sub = [r for r in rows if r["class"] == cls]
        if not sub:
            continue
        a0 = np.array([r["auc_ising"] for r in sub])
        ag = np.array([r["auc_gap_ising"] for r in sub])
        sig_g = np.mean([r["p_gap_ising"] < 0.05 and r["p_gap_free"] < 0.05 for r in sub])
        sig_0 = np.mean([r["p_ising"] < 0.05 and r["p_free"] < 0.05 for r in sub])
        Ag = np.array([r["A_gap"] for r in sub])
        print(f"\n{cls.upper()} (n={len(sub)})")
        print(f"  AUC: no-gap mean={a0.mean():.4f}  gap mean={ag.mean():.4f}  "
              f"(retained {100*(ag.mean()-0.5)/(a0.mean()-0.5):.0f}% of the edge)")
        print(f"  two-model significant: no-gap {sig_0*100:.0f}%  gap {sig_g*100:.0f}%")
        print(f"  coupling A (gap): mean={Ag.mean():+.3f}  frac<0={np.mean(Ag<0):.2f}")
    print(f"\nWrote {OUT/'gap_test.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
