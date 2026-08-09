"""Run the full empirical pipeline and persist results.

    uv run --active python -m paper.run

Outputs (in paper/out/):
  results.json   — all summary metrics, fitted Ising params, lag-N sweep
  oos_<cell>.npz — concatenated OOS probabilities/actuals for representative cells
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .common import classification_metrics, load_merged, power_law_field, spins
from .models import ARLogit, IsingLogit, _nll, build_models
from .trading import trading_metrics
from .walkforward import walk_forward, window_candles

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

COINS = ["btc", "eth", "sol", "xrp"]
INTERVALS = ["5m", "15m", "1h", "4h"]
FEE = 0.02                       # round-trip transaction cost, $ per $1 staked
REP_CELLS = [("btc", "15m"), ("btc", "5m"), ("eth", "15m"), ("btc", "1h")]
LAG_SWEEP_CELLS = [("btc", "15m"), ("btc", "5m"), ("eth", "1h")]
LAG_NS = [2, 4, 6, 8, 12, 16, 24, 36, 48]


def run_cell(coin, interval):
    try:
        dts, ch, ups = load_merged(coin, interval)
    except FileNotFoundError:
        return None
    train_size, test_size = window_candles(interval)
    if len(ch) < train_size + test_size:
        return None
    t0 = time.time()
    res, n_folds = walk_forward(ch, ups, train_size, test_size, n_lags=12)
    if len(res["ising"]["probs"]) == 0:
        return None

    models = {}
    for short, r in res.items():
        pm = classification_metrics(r["probs"], r["actuals"])
        tm = trading_metrics(r["probs"], r["actuals"], fee=FEE)
        models[short] = {
            "name": r["name"], "pred": pm,
            "trade_all": tm["trade_all"], "thresholded": tm["thresholded"],
        }

    # Ising fitted parameters across folds (for physical interpretation + figures).
    ip = res["ising"]["fold_params"]
    ising_params = {
        "alpha": [p["alpha"] for p in ip],
        "A": [p["A"] for p in ip],
        "C": [p["C"] for p in ip],
    }
    # Mean per-lag effective weight (Ising) vs mean free AR coefficient, for the cell.
    n_lags = 12
    mean_alpha = float(np.mean(ising_params["alpha"]))
    mean_A = float(np.mean(ising_params["A"]))
    k = np.arange(1, n_lags + 1)
    ising_kernel = (mean_A / (k ** mean_alpha)).tolist()

    info = {
        "coin": coin, "interval": interval, "n_total": len(ch),
        "span": [dts[0][:10], dts[-1][:10]],
        "train_candles": train_size, "test_candles": test_size,
        "n_folds": n_folds, "n_oos": int(len(res["ising"]["probs"])),
        "base_rate_up": float(ups.mean()),
        "elapsed": round(time.time() - t0, 1),
    }
    # Persist OOS arrays for representative cells (for calibration/equity figures).
    if (coin, interval) in REP_CELLS:
        np.savez(OUT / f"oos_{coin}_{interval}.npz",
                 **{f"p_{s}": r["probs"] for s, r in res.items()},
                 actual=res["ising"]["actuals"])
    return {"info": info, "models": models, "ising_params": ising_params,
            "ising_kernel": ising_kernel}


def lag_sweep(coin, interval):
    """Train vs OOS log-loss as the lag order N grows, for the free AR-logit, the
    L2-penalized AR-logit, and the 3-parameter Ising model. Demonstrates the
    overfitting (variance) the structural prior controls."""
    dts, ch, ups = load_merged(coin, interval)
    train_size, test_size = window_candles(interval)
    n = len(ch)
    rows = {N: {} for N in LAG_NS}

    def eval_model(build, N):
        tr_ll, te_ll, te_p, te_y, np_list = [], [], [], [], []
        start = 0
        while start + train_size < n:
            tlo, thi = start, start + train_size
            xlo, xhi = thi, min(thi + test_size, n)
            if xhi <= xlo:
                break
            m = build(N)
            m.fit(ch, ups, tlo, thi)
            ptr = m.predict_series(ch, ups, max(tlo, N), thi)
            ytr = ups.astype(int)[max(tlo, N):thi]
            tr_ll.append(_nll(ptr, ytr))
            pte = m.predict_series(ch, ups, xlo, xhi)
            te_p.append(pte); te_y.append(ups.astype(int)[xlo:xhi])
            np_list.append(m.params().get("n_params", N + 1))
            start += test_size
        te_p = np.concatenate(te_p); te_y = np.concatenate(te_y)
        return {"train_ll": float(np.mean(tr_ll)),
                "test_ll": float(_nll(te_p, te_y)),
                "test_acc": float(np.mean((te_p > 0.5) == (te_y > 0.5))),
                "n_params": int(np.mean(np_list))}

    for N in LAG_NS:
        rows[N]["free"] = eval_model(lambda n: ARLogit("free", n_lags=n), N)
        rows[N]["l2"] = eval_model(lambda n: ARLogit("l2", n_lags=n), N)
        rows[N]["ising"] = eval_model(lambda n: IsingLogit(n_lags=n), N)
    return rows


def main():
    out = {"config": {"coins": COINS, "intervals": INTERVALS, "fee": FEE,
                      "lag_ns": LAG_NS}, "cells": {}, "lag_sweep": {}}
    for coin in COINS:
        for interval in INTERVALS:
            key = f"{coin}-{interval}"
            r = run_cell(coin, interval)
            if r is None:
                print(f"  {key:10s}  skipped (insufficient data)")
                continue
            out["cells"][key] = r
            i = r["info"]
            isg = r["models"]["ising"]["pred"]
            fre = r["models"]["free"]["pred"]
            print(f"  {key:10s}  folds={i['n_folds']:3d} oos={i['n_oos']:7d}  "
                  f"ising acc={isg['accuracy']*100:5.2f}% auc={isg['auc']:.3f} ll={isg['log_loss']:.4f} | "
                  f"free acc={fre['accuracy']*100:5.2f}% ll={fre['log_loss']:.4f}  ({i['elapsed']}s)")

    for coin, interval in LAG_SWEEP_CELLS:
        print(f"  lag-sweep {coin}-{interval} ...", flush=True)
        out["lag_sweep"][f"{coin}-{interval}"] = lag_sweep(coin, interval)

    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT/'results.json'}")


if __name__ == "__main__":
    main()
