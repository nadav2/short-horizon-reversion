"""Fetch USDT-M perpetual-futures 15m klines, funding-rate history, and spot
close prices for the focal coins over the paper span. Enables (a) replicating
the reversal on the perp tape, and (b) conditioning it on the spot-perp basis
and the funding rate (paper.perp_test).

Output: perp_data/{coin}-perp-15m.json  (standard schema + close),
        perp_data/{coin}-spot-close.json ({timestamp, close}),
        perp_data/{coin}-funding.json    ({fundingTime, fundingRate}).

    uv run --active python -m paper.fetch_perp
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "perp_data"
SPOT = ["https://api.binance.com", "https://data-api.binance.vision"]
FAPI = ["https://fapi.binance.com"]
COINS = {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT", "xrp": "XRPUSDT"}
START, END = "2025-01-01", "2026-02-12"


def _get(bases, path, params):
    q = "&".join(f"{k}={v}" for k, v in params.items())
    last = None
    for base in bases:
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


def klines(bases, path, symbol, keep_close):
    rows, cur, end_ms = [], to_ms(START), to_ms(END)
    while cur < end_ms:
        kl = _get(bases, path, {"symbol": symbol, "interval": "15m",
                                "startTime": cur, "endTime": end_ms, "limit": 1000})
        if not kl:
            break
        for k in kl:
            o, c = float(k[1]), float(k[4])
            ts = int(k[0] // 1000)
            r = {"timestamp": ts,
                 "datetime": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                 "up": c > o, "change": (c - o) / o if o else 0.0}
            if keep_close:
                r["close"] = c
            rows.append(r)
        cur = kl[-1][0] + 1
        time.sleep(0.05)
        if len(kl) < 1000:
            break
    return rows


def funding(symbol):
    rows, cur, end_ms = [], to_ms(START), to_ms(END)
    while cur < end_ms:
        fr = _get(FAPI, "/fapi/v1/fundingRate",
                  {"symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": 1000})
        if not fr:
            break
        rows.extend({"fundingTime": int(r["fundingTime"] // 1000),
                     "fundingRate": float(r["fundingRate"])} for r in fr)
        cur = fr[-1]["fundingTime"] + 1
        time.sleep(0.1)
        if len(fr) < 1000:
            break
    return rows


def main():
    OUTDIR.mkdir(exist_ok=True)
    for coin, sym in COINS.items():
        out = OUTDIR / f"{coin}-perp-15m.json"
        if not out.exists():
            rows = klines(FAPI, "/fapi/v1/klines", sym, keep_close=True)
            out.write_text(json.dumps(rows))
            print(f"  {coin} perp n={len(rows)}", flush=True)
        out = OUTDIR / f"{coin}-spot-close.json"
        if not out.exists():
            rows = klines(SPOT, "/api/v3/klines", sym, keep_close=True)
            out.write_text(json.dumps([{"timestamp": r["timestamp"], "close": r["close"]}
                                       for r in rows]))
            print(f"  {coin} spot-close n={len(rows)}", flush=True)
        out = OUTDIR / f"{coin}-funding.json"
        if not out.exists():
            rows = funding(sym)
            out.write_text(json.dumps(rows))
            print(f"  {coin} funding n={len(rows)}", flush=True)
    print(f"done -> {OUTDIR}")


if __name__ == "__main__":
    main()
