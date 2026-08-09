"""Fetch a multi-year 15m history for the focal coins (regime-stability study).

Same Binance source and {timestamp, datetime, change, up} schema as the wide
universe, but spanning 2021-01-01 onward so the walk-forward can be scored per
calendar year. Writes to paper/multiyear_data/{sym}-15m.json.

    uv run --active python -m paper.fetch_multiyear
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .fetch_bulk_crypto import _get, to_ms

OUTDIR = Path(__file__).resolve().parent / "multiyear_data"
START, END = "2021-01-01", "2026-02-11"
INTERVAL = "15m"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


def fetch_klines(symbol: str, start: str, end: str):
    start_ms, end_ms = to_ms(start), to_ms(end)
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
                         "up": c > o, "change": (c - o) / o if o else 0.0})
        cur = kl[-1][0] + 1
        time.sleep(0.05)
        if len(kl) < 1000:
            break
    return rows


def main():
    OUTDIR.mkdir(exist_ok=True)
    for sym in SYMBOLS:
        sid = sym[:-4].lower()
        out = OUTDIR / f"{sid}-{INTERVAL}.json"
        if out.exists():
            print(f"{sym}: exists, skipping")
            continue
        rows = fetch_klines(sym, START, END)
        out.write_text(json.dumps(rows))
        print(f"{sym}: n={len(rows)} ({rows[0]['datetime'][:10]} → {rows[-1]['datetime'][:10]})")
    print(f"done → {OUTDIR}")


if __name__ == "__main__":
    main()
