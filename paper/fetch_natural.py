"""Fetch the same-asset / different-market-structure instruments.

Every cross-market comparison in the main study confounds the ASSET with the
MARKET STRUCTURE it trades in: crypto assets on crypto venues versus equity
assets on equity venues. The interpretation in the Discussion is that structure,
not asset class, drives the ordering -- but nothing in the design separates them.

These two pairs do separate them, by holding the underlying asset fixed and
varying only the venue's structure:

  GOLD    XAUUSD  spot gold, Dukascopy, 24/5 OTC dealer-intermediated
          GLD     the listed ETF on the same metal (already fetched)

  BITCOIN btc     Binance spot, 24/7, no obligated market makers
          IBIT    iShares Bitcoin Trust -- listed spot-BTC ETF
          FBTC    Fidelity Wise Origin Bitcoin Fund -- listed spot-BTC ETF
          COIN    Coinbase Global -- listed equity, crypto-revenue proxy
          MSTR    MicroStrategy/Strategy -- listed equity, levered BTC holder

IBIT/FBTC are the sharpest case: their NAV is (essentially) spot BTC, so the
underlying price process is the SAME one that shows AUC 0.533 on Binance. If the
reversal is a property of the asset it should survive the wrapper; if it is a
property of the venue it should not.

Alpaca leg (keys via dotenvx):
    cd scripts/data/py && env -u VIRTUAL_ENV dotenvx run -f ../../../.env -- \
        uv run python ../../strategies/paper/fetch_natural.py --alpaca

Dukascopy leg (run from paper/):
    npx dukascopy-node -i xauusd -from 2025-01-01 -to 2026-02-12 -t m15 -f json -dir natural_raw
    uv run python -m paper.fetch_natural --fx
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import DATA_DIR

HERE = Path(__file__).resolve().parent
RAW = HERE / "natural_raw"

# Listed wrappers around an underlying that also trades in a 24/7 or OTC venue.
ALPACA_TICKERS = {
    "ibit": "IBIT",   # spot-BTC ETF
    "fbtc": "FBTC",   # spot-BTC ETF
    "coin": "COIN",   # crypto-exchange equity
    "mstr": "MSTR",   # levered BTC treasury equity
}
FX_PAIRS = {"xauusd": "xauusd"}   # spot gold, 24/5 OTC
START, END = "2025-01-01", "2026-02-11"


def to_record(open_, close_, ts, dt):
    return {"timestamp": ts, "datetime": dt,
            "up": close_ > open_,
            "change": (close_ - open_) / open_ if open_ else 0.0}


def fetch_alpaca():
    import os
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"],
                                       os.environ["ALPACA_SECRET_KEY"])
    for sid, ticker in ALPACA_TICKERS.items():
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            start=datetime.fromisoformat(f"{START}T00:00:00"),
            end=datetime.fromisoformat(f"{END}T23:59:59"),
        )
        try:
            bars = client.get_stock_bars(req)[ticker]
        except Exception as e:
            print(f"  {sid} ({ticker}) FAILED: {e}")
            continue
        recs = [to_record(float(b.open), float(b.close), int(b.timestamp.timestamp()),
                          b.timestamp.strftime("%Y-%m-%d %H:%M:%S")) for b in bars]
        recs.sort(key=lambda r: r["timestamp"])
        (DATA_DIR / f"{sid}-15m.json").write_text(json.dumps(recs))
        flat = sum(1 for r in recs if r["change"] == 0) / max(1, len(recs))
        print(f"  {sid:5s} ({ticker:4s}) 15m n={len(recs):6d} flat={flat*100:4.1f}% "
              f"{recs[0]['datetime'][:10]}→{recs[-1]['datetime'][:10]}")


def fetch_fx():
    for sid, pair in FX_PAIRS.items():
        files = sorted(RAW.glob(f"{pair}-m15-*.json"))
        if not files:
            print(f"  {sid}: no raw file in {RAW} — run the dukascopy-node step first")
            continue
        rows = []
        for r in json.loads(files[-1].read_text()):
            if r["high"] == r["low"]:        # closed-market bar (weekend / holiday)
                continue
            ts = int(r["timestamp"] // 1000)
            rows.append(to_record(float(r["open"]), float(r["close"]), ts,
                                  datetime.fromtimestamp(ts, timezone.utc)
                                  .strftime("%Y-%m-%d %H:%M:%S")))
        rows.sort(key=lambda r: r["timestamp"])
        rows = [r for r in rows if START <= r["datetime"][:10] <= END]
        (DATA_DIR / f"{sid}-15m.json").write_text(json.dumps(rows))
        flat = sum(1 for r in rows if r["change"] == 0) / max(1, len(rows))
        print(f"  {sid:7s} 15m n={len(rows):6d} flat={flat*100:4.1f}% "
              f"{rows[0]['datetime'][:10]}→{rows[-1]['datetime'][:10]}")


def main():
    if "--alpaca" in sys.argv:
        fetch_alpaca()
    elif "--fx" in sys.argv:
        fetch_fx()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
