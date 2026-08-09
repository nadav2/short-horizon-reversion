"""Fetch the focal-coin candles (BTC/ETH/SOL/XRP at 5m/15m/1h/4h) from the
public Binance klines API, in the standard {timestamp, datetime, change, up}
schema the paper pipeline consumes.

In the monorepo these files are normally produced by the TypeScript data
pipeline; this script exists so the replication package can rebuild them from
public data with no other tooling. Existing files are never overwritten.

    uv run --active python -m paper.fetch_focal_crypto [--start 2024-03-01] [--end 2026-06-01]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone

from .common import DATA_DIR

BASES = ["https://api.binance.com", "https://data-api.binance.vision"]
COINS = {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT", "xrp": "XRPUSDT"}
INTERVALS = ["5m", "15m", "1h", "4h"]


def _get(path, params):
    q = "&".join(f"{k}={v}" for k, v in params.items())
    last = None
    for base in BASES:
        url = f"{base}{path}?{q}"
        for _ in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "research"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    return json.loads(r.read())
            except Exception as e:
                last = e
                time.sleep(0.5)
    raise last


def to_ms(date):
    return int(datetime.fromisoformat(f"{date}T00:00:00+00:00").timestamp() * 1000)


def fetch_klines(symbol, interval, start_ms, end_ms):
    rows = []
    cur = start_ms
    while cur < end_ms:
        kl = _get("/api/v3/klines", {"symbol": symbol, "interval": interval,
                                     "startTime": cur, "endTime": end_ms, "limit": 1000})
        if not kl:
            break
        for k in kl:
            o, c = float(k[1]), float(k[4])
            ts = int(k[0] // 1000)
            rows.append({"timestamp": ts,
                         "datetime": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                         "up": c > o, "change": (c - o) / o if o else 0.0})
        cur = kl[-1][0] + 1
        time.sleep(0.05)
        if len(kl) < 1000:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-03-01")
    ap.add_argument("--end", default="2026-06-01")
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    start_ms, end_ms = to_ms(args.start), to_ms(args.end)

    for coin, symbol in COINS.items():
        for interval in INTERVALS:
            out = DATA_DIR / f"{coin}-{interval}.json"
            if out.exists():
                print(f"  {coin}-{interval} exists, skipping")
                continue
            rows = fetch_klines(symbol, interval, start_ms, end_ms)
            out.write_text(json.dumps(rows))
            print(f"  {coin}-{interval} n={len(rows)} "
                  f"({rows[0]['datetime'][:10]} -> {rows[-1]['datetime'][:10]})")
    print(f"done -> {DATA_DIR}")


if __name__ == "__main__":
    main()
