"""Fetch additional traditional-market assets from Alpaca for the cross-market study.

Mirrors scripts/data/py/download_snp.py's request (same feed/timeframe, so coverage
matches the existing SPY/QQQ files), but writes bundled JSON directly to
data/{id}-{interval}.json with the same {timestamp, datetime,
change, up} schema the paper pipeline consumes, and resamples 15m -> 1h, 4h.

    cd scripts/data/py && dotenvx run -f ../../../.env -- \
        uv run --active python /abs/path/paper/fetch_markets.py
(run from the py env that has alpaca-py; keys are decrypted by dotenvx)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

OUT_DIR = Path(os.environ.get("PAPER_DATA_DIR",
                              Path(__file__).resolve().parent.parent / "data"))

# id -> Alpaca ticker. Diverse traditional markets: index, small-cap, single stocks,
# commodity (gold), long bonds.
TICKERS = {
    "spx": "SPY",   # S&P 500 proxy
    "qqq": "QQQ",   # Nasdaq-100
    "iwm": "IWM",    # Russell 2000 small-cap (more retail than large-cap index)
    "aapl": "AAPL",  # mega-cap, very liquid single stock
    "nvda": "NVDA",  # high-volatility momentum single stock
    "tsla": "TSLA",  # retail-heavy, high-volatility single stock
    "gld": "GLD",    # gold (commodity)
    "tlt": "TLT",    # 20y+ Treasuries (rates/bonds)
}
START, END = "2025-01-01", "2026-02-11"
RESAMPLE = {"1h": 3600, "4h": 14400}


def fetch_15m(ticker):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(15, TimeFrameUnit.Minute),
        start=datetime.fromisoformat(f"{START}T00:00:00"),
        end=datetime.fromisoformat(f"{END}T23:59:59"),
    )
    bars = client.get_stock_bars(req)
    rows = []
    for b in bars[ticker]:
        rows.append({"timestamp": int(b.timestamp.timestamp()),
                     "datetime": b.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                     "open": float(b.open), "close": float(b.close)})
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def to_record(open_, close_, ts, dt):
    return {"timestamp": ts, "datetime": dt,
            "up": close_ > open_,
            "change": (close_ - open_) / open_ if open_ else 0.0}


def resample(rows, seconds):
    """Aggregate 15m open/close rows into buckets aligned to `seconds`."""
    buckets: dict[int, list] = {}
    for r in rows:
        key = r["timestamp"] - (r["timestamp"] % seconds)
        buckets.setdefault(key, []).append(r)
    out = []
    for key in sorted(buckets):
        grp = sorted(buckets[key], key=lambda r: r["timestamp"])
        o, c = grp[0]["open"], grp[-1]["close"]
        out.append(to_record(o, c, key, grp[0]["datetime"]))
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sid, ticker in TICKERS.items():
        try:
            rows = fetch_15m(ticker)
        except Exception as e:
            print(f"  {sid} ({ticker}) FAILED: {e}")
            continue
        if not rows:
            print(f"  {sid} ({ticker}): no data"); continue
        recs15 = [to_record(r["open"], r["close"], r["timestamp"], r["datetime"]) for r in rows]
        (OUT_DIR / f"{sid}-15m.json").write_text(json.dumps(recs15))
        msg = f"  {sid:5s} ({ticker:4s}) 15m n={len(recs15)} {recs15[0]['datetime'][:10]}→{recs15[-1]['datetime'][:10]}"
        for iv, sec in RESAMPLE.items():
            rs = resample(rows, sec)
            (OUT_DIR / f"{sid}-{iv}.json").write_text(json.dumps(rs))
            msg += f" | {iv} n={len(rs)}"
        print(msg)


if __name__ == "__main__":
    main()
