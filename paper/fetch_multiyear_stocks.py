"""Fetch a multi-year 15m history for the focal US-listed instruments (Alpaca).

Extends the equity side of the regime-stability study: the matched
cross-market window is a single ~14-month regime, so "absent in equities"
deserves the same 2021-2026 per-year scoring the crypto side gets
(paper.multiyear / paper.multiyear_stocks). Same {timestamp, datetime,
change, up} schema, written to paper/multiyear_data/{id}-15m.json.

    cd scripts/data/py && dotenvx run -f ../../../.env -- \
        uv run --active python ../../strategies/paper/fetch_multiyear_stocks.py
(run from the py env that has alpaca-py; keys are decrypted by dotenvx)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent / "multiyear_data"

TICKERS = {
    "spx": "SPY", "qqq": "QQQ", "iwm": "IWM",
    "aapl": "AAPL", "nvda": "NVDA", "tsla": "TSLA",
    "gld": "GLD", "tlt": "TLT",
}
START, END = "2021-01-01", "2026-02-11"


def fetch_15m(ticker):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"],
                                       os.environ["ALPACA_SECRET_KEY"])
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(15, TimeFrameUnit.Minute),
        start=datetime.fromisoformat(f"{START}T00:00:00"),
        end=datetime.fromisoformat(f"{END}T23:59:59"),
    )
    bars = client.get_stock_bars(req)
    rows = []
    for b in bars[ticker]:
        ts = int(b.timestamp.timestamp())
        o, c = float(b.open), float(b.close)
        rows.append({"timestamp": ts,
                     "datetime": b.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                     "up": c > o, "change": (c - o) / o if o else 0.0})
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def main():
    OUTDIR.mkdir(exist_ok=True)
    for sid, ticker in TICKERS.items():
        out = OUTDIR / f"{sid}-15m.json"
        if out.exists():
            print(f"{sid}: exists, skipping")
            continue
        try:
            rows = fetch_15m(ticker)
        except Exception as e:
            print(f"{sid} ({ticker}) FAILED: {e}")
            continue
        out.write_text(json.dumps(rows))
        print(f"{sid:5s} ({ticker:4s}) n={len(rows)} "
              f"({rows[0]['datetime'][:10]} → {rows[-1]['datetime'][:10]})")
    print(f"done → {OUTDIR}")


if __name__ == "__main__":
    main()
