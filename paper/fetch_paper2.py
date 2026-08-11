"""Fetch the extended wrapper-panel instruments for the process-vs-venue paper.

Three legs:
  --panel    New 15m wrappers over the paper span (2025-01-01..2026-02-11), to
             the standard DATA_DIR so paper.panel can score them: the remaining
             US spot-BTC ETFs (BITB, ARKB, HODL, BTCO, EZBC), the CME-futures
             wrapper BITO, the 2x leveraged wrappers BITU/ETHU, and the FX ETFs
             FXE/FXY/UUP (dose test: spot FX carries a faint reversal copy).
  --history  Long-span 15m series to wrapper_data/: GBTC and ETHE from 2021
             (closed-end-trust era through ETF conversion; pre-conversion OTC
             prints may be missing on the IEX feed - fetch reports actual
             coverage), BITO from its 2021-10 launch, and the four spot ETFs
             from their launch dates (inheritance-at-birth).
  --leadlag  1m bars for the transmission-channel test: IBIT (Alpaca) and BTC
             (Binance) over 2025-08-11..2026-02-11, to wrapper_data/.

Alpaca legs need keys:
    cd scripts/data/py && env -u VIRTUAL_ENV dotenvx run -f ../../../.env -- \
        uv run python -m paper.fetch_paper2 --panel --history --leadlag
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent.parent / "gui" / "public" / "crypto-data"
WRAP = HERE / "wrapper_data"

SPAN_PANEL = ("2025-01-01", "2026-02-11")
PANEL_TICKERS = {
    "bitb": "BITB", "arkb": "ARKB", "hodl": "HODL", "btco": "BTCO",
    "ezbc": "EZBC", "bito": "BITO", "bitu": "BITU", "ethu": "ETHU",
    "fxe": "FXE", "fxy": "FXY", "uup": "UUP",
}
HISTORY_TICKERS = {   # sid: (ticker, start)
    "gbtc": ("GBTC", "2021-01-01"),
    "ethe": ("ETHE", "2021-01-01"),
    "bito": ("BITO", "2021-10-19"),
    "ibit": ("IBIT", "2024-01-11"),
    "fbtc": ("FBTC", "2024-01-11"),
    "etha": ("ETHA", "2024-07-23"),
    "feth": ("FETH", "2024-07-23"),
}
HIST_END = "2026-02-11"
LEADLAG_SPAN = ("2025-08-11", "2026-02-11")


def to_record(open_, close_, ts, dt):
    return {"timestamp": ts, "datetime": dt,
            "up": close_ > open_,
            "change": (close_ - open_) / open_ if open_ else 0.0}


def alpaca_client():
    import os
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(os.environ["ALPACA_API_KEY"],
                                     os.environ["ALPACA_SECRET_KEY"])


def fetch_alpaca(client, ticker, start, end, minutes=15):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
        start=datetime.fromisoformat(f"{start}T00:00:00"),
        end=datetime.fromisoformat(f"{end}T23:59:59"),
    )
    bars = client.get_stock_bars(req)[ticker]
    recs = [to_record(float(b.open), float(b.close), int(b.timestamp.timestamp()),
                      b.timestamp.strftime("%Y-%m-%d %H:%M:%S")) for b in bars]
    recs.sort(key=lambda r: r["timestamp"])
    return recs


def report(sid, recs):
    flat = sum(1 for r in recs if r["change"] == 0) / max(1, len(recs))
    print(f"  {sid:5s} n={len(recs):6d} flat={flat * 100:4.1f}% "
          f"{recs[0]['datetime'][:10]} -> {recs[-1]['datetime'][:10]}", flush=True)


def leg_panel(client):
    print("panel wrappers ->", DATA_DIR)
    for sid, ticker in PANEL_TICKERS.items():
        out = DATA_DIR / f"{sid}-15m.json"
        if out.exists():
            print(f"  {sid} exists, skipping")
            continue
        try:
            recs = fetch_alpaca(client, ticker, *SPAN_PANEL)
        except Exception as e:
            print(f"  {sid} ({ticker}) FAILED: {e}")
            continue
        out.write_text(json.dumps(recs))
        report(sid, recs)


def leg_history(client):
    print("long-span wrappers ->", WRAP)
    for sid, (ticker, start) in HISTORY_TICKERS.items():
        out = WRAP / f"{sid}-15m.json"
        if out.exists():
            print(f"  {sid} exists, skipping")
            continue
        try:
            recs = fetch_alpaca(client, ticker, start, HIST_END)
        except Exception as e:
            print(f"  {sid} ({ticker}) FAILED: {e}")
            continue
        out.write_text(json.dumps(recs))
        report(sid, recs)


BINANCE = ["https://api.binance.com", "https://data-api.binance.vision"]


def binance_1m(symbol, start, end):
    def get(params):
        q = "&".join(f"{k}={v}" for k, v in params.items())
        last = None
        for base in BINANCE:
            for _ in range(3):
                try:
                    req = urllib.request.Request(f"{base}/api/v3/klines?{q}",
                                                 headers={"User-Agent": "research"})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        return json.loads(r.read())
                except Exception as e:
                    last = e
                    time.sleep(0.5)
        raise last

    to_ms = lambda d: int(datetime.fromisoformat(f"{d}T00:00:00+00:00").timestamp() * 1000)
    rows, cur, end_ms = [], to_ms(start), to_ms(end)
    while cur < end_ms:
        kl = get({"symbol": symbol, "interval": "1m", "startTime": cur,
                  "endTime": end_ms, "limit": 1000})
        if not kl:
            break
        for k in kl:
            o, c = float(k[1]), float(k[4])
            ts = int(k[0] // 1000)
            rows.append(to_record(o, c, ts,
                        datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")))
        cur = kl[-1][0] + 1
        time.sleep(0.05)
        if len(kl) < 1000:
            break
    return rows


def leg_leadlag(client):
    print("lead-lag 1m ->", WRAP)
    out = WRAP / "ibit-1m.json"
    if not out.exists():
        recs = fetch_alpaca(client, "IBIT", *LEADLAG_SPAN, minutes=1)
        out.write_text(json.dumps(recs))
        report("ibit1", recs)
    out = WRAP / "btc-1m.json"
    if not out.exists():
        recs = binance_1m("BTCUSDT", *LEADLAG_SPAN)
        out.write_text(json.dumps(recs))
        report("btc1", recs)


def main():
    WRAP.mkdir(exist_ok=True)
    client = alpaca_client()
    if "--panel" in sys.argv:
        leg_panel(client)
    if "--history" in sys.argv:
        leg_history(client)
    if "--leadlag" in sys.argv:
        leg_leadlag(client)
    if len(sys.argv) == 1:
        print(__doc__)


if __name__ == "__main__":
    main()
