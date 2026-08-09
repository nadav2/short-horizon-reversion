"""Alternative return definitions for the focal coins (candle-convention test).

The paper's baseline label is the intra-bar open-to-close return. This module
re-runs the identical matched 15m walk-forward on the same Binance bars under
three return definitions:

  * oc   — intra-bar open-to-close (baseline): r_t = close_t/open_t - 1
  * cc   — close-to-close:                     r_t = close_t/close_{t-1} - 1
  * vwap — VWAP-to-VWAP: per-bar VWAP = quote_volume / base_volume,
                                               r_t = vwap_t/vwap_{t-1} - 1

(The fourth definition requested by referees, mid-to-mid on quote-midprice
bars, lives in paper.midprice because it requires a quote-based source.)

Requires paper/exchange_data/binance-{coin}-15m.json from paper.fetch_exchanges
(full OHLCV + quote-volume schema). Output: paper/out/return_defs.json

    uv run --active python -m paper.return_defs
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .exchanges import COINS, load_rows, score_series

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def series_oc(rows):
    return np.array([r["change"] for r in rows], dtype=float)


def series_cc(rows):
    c = np.array([r["close"] for r in rows], dtype=float)
    r = np.zeros(len(c))
    r[1:] = c[1:] / c[:-1] - 1.0
    return r


def series_vwap(rows):
    v = np.array([r["volume"] for r in rows], dtype=float)
    q = np.array([r["qvol"] for r in rows], dtype=float)
    c = np.array([r["close"] for r in rows], dtype=float)
    vwap = np.where(v > 0, q / np.maximum(v, 1e-12), np.nan)
    # zero-volume bars (none expected for focal coins): carry the close
    vwap = np.where(np.isnan(vwap), c, vwap)
    r = np.zeros(len(vwap))
    r[1:] = vwap[1:] / vwap[:-1] - 1.0
    return r


DEFS = {"oc": series_oc, "cc": series_cc, "vwap": series_vwap}


def main():
    results = {}
    for coin in COINS:
        rows = load_rows("binance", coin)
        if rows is None:
            print(f"  {coin}: no binance data (run paper.fetch_exchanges), skipped")
            continue
        nzero = sum(1 for r in rows if r["volume"] == 0)
        results[coin] = {"n_bars": len(rows), "n_zero_volume": nzero}
        for name, fn in DEFS.items():
            r = fn(rows)
            sc = score_series(r, r > 0, label=f"{coin}-{name}")
            results[coin][name] = sc
            if sc:
                both = "*" if max(sc["ising_auc_p_gt05"], sc["free_auc_p_gt05"]) < 0.05 else " "
                print(f"  {coin}-{name:4s}: AUC={sc['ising_auc']:.4f} "
                      f"[{sc['ising_auc_ci'][0]:.3f},{sc['ising_auc_ci'][1]:.3f}] "
                      f"pI={sc['ising_auc_p_gt05']:.3f} pF={sc['free_auc_p_gt05']:.3f}{both} "
                      f"A={sc['A']:+.3f} rho1={sc['ac1']:+.3f} n={sc['n_oos']}")

    (OUT / "return_defs.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT/'return_defs.json'}")


if __name__ == "__main__":
    main()
