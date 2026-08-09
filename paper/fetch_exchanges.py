"""Fetch 15m candles for the focal coins from independent venues (cross-exchange
replication study) plus Binance with full OHLCV fields (return-definition study).

Venues and public endpoints (all keyless):
  * coinbase — Coinbase Exchange  GET /products/{id}/candles   (300 bars/req)
  * okx      — OKX                GET /api/v5/market/history-candles (100 bars/req)
  * bybit    — Bybit              GET /v5/market/kline          (1000 bars/req)
  * binance  — Binance            GET /api/v3/klines            (1000 bars/req)

Kraken is omitted: its public OHLC endpoint returns only the most recent 720
bars (7.5 days at 15m), so the matched 2025-01-01..2026-02-11 span cannot be
rebuilt from its REST API.

Writes paper/exchange_data/{venue}-{sym}-15m.json with rows
  {timestamp, datetime, open, close, volume, qvol, change, up}
where change = (close-open)/open and up = close > open (the paper's baseline
convention); qvol is the quote-currency volume (null where the venue does not
report it). All timestamps are UTC bucket-start seconds on the shared 15m grid.

    uv run --active python -m paper.fetch_exchanges
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent / "exchange_data"
START, END = "2025-01-01", "2026-02-11"
COINS = ["btc", "eth", "sol", "xrp"]
STEP_S = 900


def to_s(date: str) -> int:
    return int(datetime.fromisoformat(f"{date}T00:00:00+00:00").timestamp())


def _get(url: str, retries: int = 5):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            last = e
            time.sleep(1.0 + i)
    raise last


def row(ts: int, o: float, c: float, vol: float, qvol):
    return {"timestamp": ts,
            "datetime": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "open": o, "close": c, "volume": vol, "qvol": qvol,
            "change": (c - o) / o if o else 0.0, "up": c > o}


# ── per-venue fetchers (each returns a ts-ascending list of rows) ─────────────

def fetch_coinbase(coin: str):
    pid = f"{coin.upper()}-USD"
    lo, hi = to_s(START), to_s(END)
    rows = []
    cur = lo
    while cur < hi:
        chunk_end = min(cur + 300 * STEP_S, hi)
        s = datetime.fromtimestamp(cur, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        e = datetime.fromtimestamp(chunk_end, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        data = _get(f"https://api.exchange.coinbase.com/products/{pid}/candles"
                    f"?granularity=900&start={s}&end={e}")
        for k in sorted(data, key=lambda k: k[0]):       # [time, low, high, open, close, volume]
            ts, o, c, v = int(k[0]), float(k[3]), float(k[4]), float(k[5])
            if lo <= ts < hi:
                rows.append(row(ts, o, c, v, None))
        cur = chunk_end
        time.sleep(0.15)
    return rows


def fetch_okx(coin: str):
    inst = f"{coin.upper()}-USDT"
    lo_ms, hi_ms = to_s(START) * 1000, to_s(END) * 1000
    rows = []
    after = hi_ms                                         # walk backwards
    while True:
        data = _get(f"https://www.okx.com/api/v5/market/history-candles"
                    f"?instId={inst}&bar=15m&after={after}&limit=100")
        kl = data.get("data", [])
        if not kl:
            break
        for k in kl:                                      # [ts,o,h,l,c,vol,volCcy,...] desc
            ts = int(k[0]) // 1000
            if ts * 1000 < lo_ms:
                continue
            rows.append(row(ts, float(k[1]), float(k[4]), float(k[5]), float(k[6])))
        oldest = int(kl[-1][0])
        if oldest <= lo_ms:
            break
        after = oldest
        time.sleep(0.12)
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def fetch_bybit(coin: str):
    sym = f"{coin.upper()}USDT"
    lo_ms, hi_ms = to_s(START) * 1000, to_s(END) * 1000
    rows = []
    cur = lo_ms
    while cur < hi_ms:
        data = _get(f"https://api.bybit.com/v5/market/kline?category=spot"
                    f"&symbol={sym}&interval=15&start={cur}&limit=1000")
        kl = data.get("result", {}).get("list", [])
        if not kl:
            break
        kl.sort(key=lambda k: int(k[0]))                  # [ts,o,h,l,c,vol,turnover] desc
        for k in kl:
            ts = int(k[0]) // 1000
            if lo_ms <= ts * 1000 < hi_ms:
                rows.append(row(ts, float(k[1]), float(k[4]), float(k[5]), float(k[6])))
        nxt = int(kl[-1][0]) + STEP_S * 1000
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.1)
    return rows


def fetch_binance(coin: str):
    sym = f"{coin.upper()}USDT"
    lo_ms, hi_ms = to_s(START) * 1000, to_s(END) * 1000
    rows = []
    cur = lo_ms
    while cur < hi_ms:
        data = _get(f"https://api.binance.com/api/v3/klines?symbol={sym}"
                    f"&interval=15m&startTime={cur}&endTime={hi_ms}&limit=1000")
        if not data:
            break
        for k in data:                                    # [openT,o,h,l,c,vol,closeT,qvol,...]
            ts = int(k[0]) // 1000
            rows.append(row(ts, float(k[1]), float(k[4]), float(k[5]), float(k[7])))
        cur = data[-1][0] + 1
        time.sleep(0.05)
        if len(data) < 1000:
            break
    return rows


VENUES = {"coinbase": fetch_coinbase, "okx": fetch_okx,
          "bybit": fetch_bybit, "binance": fetch_binance}


def main():
    OUTDIR.mkdir(exist_ok=True)
    expected = (to_s(END) - to_s(START)) // STEP_S
    for venue, fn in VENUES.items():
        for coin in COINS:
            out = OUTDIR / f"{venue}-{coin}-15m.json"
            if out.exists():
                print(f"{venue}-{coin}: exists, skipping")
                continue
            t0 = time.time()
            try:
                rows = fn(coin)
            except Exception as e:
                print(f"{venue}-{coin}: FAILED ({e})")
                continue
            # dedup on the grid, keep first occurrence
            seen, ded = set(), []
            for r in rows:
                if r["timestamp"] not in seen:
                    seen.add(r["timestamp"])
                    ded.append(r)
            out.write_text(json.dumps(ded))
            miss = expected - len(ded)
            print(f"{venue}-{coin}: n={len(ded)} (missing {miss}/{expected} grid slots, "
                  f"{100*miss/expected:.2f}%)  [{time.time()-t0:.0f}s]")
    print(f"done → {OUTDIR}")


if __name__ == "__main__":
    main()
