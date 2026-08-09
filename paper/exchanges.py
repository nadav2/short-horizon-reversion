"""Cross-exchange replication of the 15m focal-coin result.

Runs the identical matched walk-forward (train 5760 / test 960 candles, N=12,
lambda=150, Ising + free logit + base rate, moving-block bootstrap) on 15-minute
candles for BTC/ETH/SOL/XRP from four venues (Binance, Coinbase, OKX, Bybit;
data via paper.fetch_exchanges), plus an aggregated composite per coin: the
median close across all four venues on the intersected 15m grid, with
close-to-close returns and labels.

Output: paper/out/exchanges.json

    uv run --active python -m paper.fetch_exchanges   # once
    uv run --active python -m paper.exchanges
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .compare_markets import BLOCK, N_BOOT, SEED, WINDOWS, autocorr1, block_boot_idx
from .models import ARLogit, BaseRate, IsingLogit
from .walkforward import walk_forward

DATA = Path(__file__).resolve().parent / "exchange_data"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

VENUES = ["binance", "coinbase", "okx", "bybit"]
COINS = ["btc", "eth", "sol", "xrp"]


def score_series(ch: np.ndarray, ups: np.ndarray, interval: str = "15m",
                 label: str = "") -> dict | None:
    """The matched walk-forward + two-model block-bootstrap scorer of
    paper.compare_markets, for an arbitrary (returns, labels) series."""
    from sklearn.metrics import roc_auc_score

    tr, te = WINDOWS[interval]
    if len(ch) < tr + te:
        return None
    res, n_folds = walk_forward(
        ch, ups, tr, te, n_lags=12,
        models=lambda: [IsingLogit(n_lags=12), ARLogit("free", n_lags=12), BaseRate()])
    y = res["ising"]["actuals"].astype(int)
    if len(y) == 0 or y.min() == y.max():
        return None
    p_is, p_fr = res["ising"]["probs"], res["free"]["probs"]

    rng = np.random.default_rng(SEED)
    boot_is, boot_fr = [], []
    for _ in range(N_BOOT):
        idx = block_boot_idx(len(y), BLOCK[interval], rng)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        boot_is.append(roc_auc_score(yb, p_is[idx]))
        boot_fr.append(roc_auc_score(yb, p_fr[idx]))
    boot_is, boot_fr = np.array(boot_is), np.array(boot_fr)

    return {
        "label": label, "n_oos": int(len(y)), "n_folds": n_folds,
        "base_rate_up": float(y.mean()), "ac1": autocorr1(ch),
        "ising_auc": float(roc_auc_score(y, p_is)),
        "ising_auc_ci": [float(np.percentile(boot_is, 2.5)),
                         float(np.percentile(boot_is, 97.5))],
        "ising_auc_p_gt05": float(np.mean(boot_is <= 0.5)),
        "free_auc": float(roc_auc_score(y, p_fr)),
        "free_auc_p_gt05": float(np.mean(boot_fr <= 0.5)),
        "A": float(np.mean([p["A"] for p in res["ising"]["fold_params"]])),
        "alpha": float(np.mean([p["alpha"] for p in res["ising"]["fold_params"]])),
    }


def load_rows(venue: str, coin: str):
    path = DATA / f"{venue}-{coin}-15m.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def oc_series(rows):
    """Baseline intra-bar open-to-close convention."""
    ch = np.array([r["change"] for r in rows], dtype=float)
    ups = np.array([bool(r["up"]) for r in rows], dtype=bool)
    return ch, ups


def composite_series(coin: str):
    """Median close across all four venues on the intersected grid; close-to-close."""
    per_venue = {}
    for v in VENUES:
        rows = load_rows(v, coin)
        if rows is None:
            return None, 0
        per_venue[v] = {r["timestamp"]: r["close"] for r in rows}
    common = sorted(set.intersection(*(set(m) for m in per_venue.values())))
    closes = np.array([[per_venue[v][ts] for v in VENUES] for ts in common])
    med = np.median(closes, axis=1)
    ch = np.zeros(len(med))
    ch[1:] = med[1:] / med[:-1] - 1.0
    return (ch, ch > 0), len(common)


def main():
    results = {"venues": {}, "composite": {}}
    print("=== per-venue, baseline open-to-close convention ===")
    for venue in VENUES:
        for coin in COINS:
            rows = load_rows(venue, coin)
            if rows is None:
                print(f"  {venue}-{coin}: no data, skipped")
                continue
            ch, ups = oc_series(rows)
            r = score_series(ch, ups, label=f"{venue}-{coin}")
            if r is None:
                print(f"  {venue}-{coin}: series too short, skipped")
                continue
            r["n_bars"] = len(rows)
            results["venues"][f"{venue}-{coin}"] = r
            both = "*" if max(r["ising_auc_p_gt05"], r["free_auc_p_gt05"]) < 0.05 else " "
            print(f"  {venue:9s}-{coin}: AUC={r['ising_auc']:.4f} "
                  f"[{r['ising_auc_ci'][0]:.3f},{r['ising_auc_ci'][1]:.3f}] "
                  f"pI={r['ising_auc_p_gt05']:.3f} pF={r['free_auc_p_gt05']:.3f}{both} "
                  f"A={r['A']:+.3f} rho1={r['ac1']:+.3f} n={r['n_oos']}")

    print("=== composite (median close across venues, close-to-close) ===")
    for coin in COINS:
        series, n_common = composite_series(coin)
        if series is None:
            print(f"  composite-{coin}: missing venue data, skipped")
            continue
        ch, ups = series
        r = score_series(ch, ups, label=f"composite-{coin}")
        if r is None:
            continue
        r["n_bars"] = n_common
        results["composite"][coin] = r
        both = "*" if max(r["ising_auc_p_gt05"], r["free_auc_p_gt05"]) < 0.05 else " "
        print(f"  composite-{coin}: AUC={r['ising_auc']:.4f} "
              f"[{r['ising_auc_ci'][0]:.3f},{r['ising_auc_ci'][1]:.3f}] "
              f"pI={r['ising_auc_p_gt05']:.3f} pF={r['free_auc_p_gt05']:.3f}{both} "
              f"A={r['A']:+.3f} n={r['n_oos']}")

    (OUT / "exchanges.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT/'exchanges.json'}")


if __name__ == "__main__":
    main()
