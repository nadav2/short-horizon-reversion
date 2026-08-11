"""Fetch focal-coin 15m klines KEEPING the per-bar flow fields Binance provides
(base volume, trade count, taker-buy base volume), which the standard pipeline
discards. These enable order-flow-imbalance mechanism tests (paper.flow_test)
without any order-book data: taker-buy volume identifies the aggressor side of
every trade, so per-bar signed flow is imbalance = 2*tbv/vol - 1.

Output: flow_data/{coin}-15m-flow.json, standard schema plus vol/ntr/tbv.

    uv run --active python -m paper.fetch_flow [--start 2025-01-01] [--end 2026-02-12]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASES = ["https://api.binance.com", "https://data-api.binance.vision"]
COINS = {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT", "xrp": "XRPUSDT"}
INTERVAL = "15m"
FLOW_DIR = Path(__file__).resolve().parent / "flow_data"


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


def fetch_klines(symbol, start_ms, end_ms):
    rows = []
    cur = start_ms
    while cur < end_ms:
        kl = _get("/api/v3/klines", {"symbol": symbol, "interval": INTERVAL,
                                     "startTime": cur, "endTime": end_ms, "limit": 1000})
        if not kl:
            break
        for k in kl:
            o, c = float(k[1]), float(k[4])
            ts = int(k[0] // 1000)
            rows.append({"timestamp": ts,
                         "datetime": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                         "up": c > o, "change": (c - o) / o if o else 0.0,
                         "vol": float(k[5]), "ntr": int(k[8]), "tbv": float(k[9])})
        cur = kl[-1][0] + 1
        time.sleep(0.05)
        if len(kl) < 1000:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-02-12")
    args = ap.parse_args()
    FLOW_DIR.mkdir(parents=True, exist_ok=True)
    start_ms, end_ms = to_ms(args.start), to_ms(args.end)

    for coin, symbol in COINS.items():
        out = FLOW_DIR / f"{coin}-{INTERVAL}-flow.json"
        if out.exists():
            print(f"  {coin} exists, skipping")
            continue
        rows = fetch_klines(symbol, start_ms, end_ms)
        out.write_text(json.dumps(rows))
        print(f"  {coin} n={len(rows)} "
              f"({rows[0]['datetime'][:10]} -> {rows[-1]['datetime'][:10]})")
    print(f"done -> {FLOW_DIR}")


if __name__ == "__main__":
    main()
