"""Bulk-fetch a wide universe of Binance USDT spot pairs at 15m (matched span),
for the hundreds-of-assets cross-market study and trading simulation.

Writes one file per symbol to paper/bulk_data/{sym}-15m.json with the standard
{timestamp, datetime, change, up} schema (change=(close-open)/open, up=close>open).

    uv run --active python -m paper.fetch_bulk_crypto --top 200
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASES = ["https://api.binance.com", "https://data-api.binance.vision"]
OUTDIR = Path(__file__).resolve().parent / "bulk_data"
START, END = "2025-01-01", "2026-02-11"
INTERVAL = "15m"
EXCLUDE_QUOTE_STABLES = {"USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "EURUSDT",
                         "USDPUSDT", "DAIUSDT", "USDTUSDT", "PAXGUSDT"}


def _get(path, params):
    q = "&".join(f"{k}={v}" for k, v in params.items())
    last = None
    for base in BASES:
        url = f"{base}{path}?{q}" if q else f"{base}{path}"
        for _ in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "research"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    return json.loads(r.read())
            except Exception as e:
                last = e
                time.sleep(0.5)
    raise last


def top_usdt_symbols(n):
    data = _get("/api/v3/ticker/24hr", {})
    usdt = [d for d in data if d["symbol"].endswith("USDT")
            and d["symbol"] not in EXCLUDE_QUOTE_STABLES
            and not any(x in d["symbol"] for x in ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"))]
    usdt.sort(key=lambda d: float(d.get("quoteVolume", 0)), reverse=True)
    return [d["symbol"] for d in usdt[:n]]


def to_ms(date):
    return int(datetime.fromisoformat(f"{date}T00:00:00+00:00").timestamp() * 1000)


def fetch_klines(symbol):
    start_ms, end_ms = to_ms(START), to_ms(END)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--symbols", nargs="*", help="explicit symbols (test mode)")
    args = ap.parse_args()
    OUTDIR.mkdir(exist_ok=True)

    symbols = args.symbols or top_usdt_symbols(args.top)
    print(f"fetching {len(symbols)} symbols → {OUTDIR}")
    done = 0
    for i, sym in enumerate(symbols):
        sid = sym[:-4].lower()                         # strip USDT
        out = OUTDIR / f"{sid}-15m.json"
        if out.exists():
            done += 1
            continue
        try:
            rows = fetch_klines(sym)
        except Exception as e:
            print(f"  [{i+1}/{len(symbols)}] {sym} FAILED: {e}")
            continue
        if len(rows) < 5000:                           # require reasonable coverage
            print(f"  [{i+1}/{len(symbols)}] {sym} skipped (only {len(rows)} candles)")
            continue
        out.write_text(json.dumps(rows))
        done += 1
        if (i + 1) % 20 == 0 or i < 3:
            print(f"  [{i+1}/{len(symbols)}] {sym} n={len(rows)} ({rows[0]['datetime'][:10]}→{rows[-1]['datetime'][:10]})")
    print(f"done: {done} symbols written to {OUTDIR}")


if __name__ == "__main__":
    main()
