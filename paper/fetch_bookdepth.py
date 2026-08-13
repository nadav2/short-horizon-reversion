"""Fetch USDT-M perp order-book depth for the focal coins over the paper span.

Binance Vision publishes free daily `bookDepth` archives for UM perps
(2023-01-01 onward): snapshots every ~25s of CUMULATIVE bid/ask notional
within 1..5% of mid. This is the book-state data the paper's Discussion
names as its sharpest open test ("reversal absent after depth-consuming
moves") -- and the perp tape is where perp_test already replicated the
reversal, so no spot L2 purchase is needed.

Each day is aggregated on the fly to per-15m-bar features (first / last /
min / mean notional within 1% and 2% of mid, per side) and cached to
depth_data/daily/{coin}-{date}.json; a merge step then writes
depth_data/{coin}-depth-15m.json. Also fetches perp 15m klines KEEPING the
flow fields (vol, ntr, taker-buy volume) so paper.depth_test can condition
on aggressor flow from the SAME tape as the book states:
depth_data/{coin}-perp-flow-15m.json.

    uv run --active python -m paper.fetch_bookdepth [--start 2025-01-01]
        [--end 2026-02-11] [--coins btc,eth,sol,xrp]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "depth_data"
DAILY = OUTDIR / "daily"
VISION = "https://data.binance.vision/data/futures/um/daily/bookDepth"
FAPI = ["https://fapi.binance.com"]
COINS = {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT", "xrp": "XRPUSDT"}
BAR = 900  # 15m
# (json key prefix, percentage column value): cumulative notional within
# 1% / 2% of mid, bid and ask side
LEVELS = {"b1": "-1", "a1": "1", "b2": "-2", "a2": "2"}


def _download(url, retries=4):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
        except Exception as e:
            last = e
        time.sleep(1.0 + i)
    raise last


def aggregate_day(blob):
    """bookDepth daily CSV -> {bar_ts: features}. Snapshots arrive as ~10
    rows (percentage levels) per timestamp, ~3.5k timestamps per day."""
    zf = zipfile.ZipFile(io.BytesIO(blob))
    text = io.TextIOWrapper(zf.open(zf.namelist()[0]), encoding="utf-8")
    reader = csv.reader(text)
    header = next(reader)
    ci = {c: header.index(c) for c in ("timestamp", "percentage", "notional")}

    ts_cache = {}
    snaps = {}  # bar_ts -> key -> [notional in snapshot order]
    for row in reader:
        key = None
        for k, pct in LEVELS.items():
            if float(row[ci["percentage"]]) == float(pct):
                key = k
                break
        if key is None:
            continue
        raw = row[ci["timestamp"]]
        ts = ts_cache.get(raw)
        if ts is None:
            ts = int(datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
                     .replace(tzinfo=timezone.utc).timestamp())
            ts_cache[raw] = ts
        bar = ts - ts % BAR
        snaps.setdefault(bar, {}).setdefault(key, []).append(
            float(row[ci["notional"]]))

    out = []
    for bar in sorted(snaps):
        rec = {"timestamp": bar, "n": max(len(v) for v in snaps[bar].values())}
        for k in LEVELS:
            v = snaps[bar].get(k)
            if not v:
                continue
            rec[f"{k}_first"] = round(v[0], 2)
            rec[f"{k}_last"] = round(v[-1], 2)
            rec[f"{k}_min"] = round(min(v), 2)
            rec[f"{k}_mean"] = round(sum(v) / len(v), 2)
        out.append(rec)
    return out


def fetch_depth(coin, symbol, days):
    missing = []
    for d in days:
        cache = DAILY / f"{coin}-{d}.json"
        if cache.exists():
            continue
        blob = _download(f"{VISION}/{symbol}/{symbol}-bookDepth-{d}.zip")
        if blob is None:
            print(f"  {coin} {d}: MISSING (404)", flush=True)
            missing.append(d)
            cache.write_text("[]")
            continue
        cache.write_text(json.dumps(aggregate_day(blob)))
        time.sleep(0.1)
    rows = []
    for d in days:
        rows.extend(json.loads((DAILY / f"{coin}-{d}.json").read_text()))
    rows.sort(key=lambda r: r["timestamp"])
    merged = OUTDIR / f"{coin}-depth-15m.json"
    merged.write_text(json.dumps(rows))
    print(f"  {coin} depth: {len(rows)} bars over {len(days)} days "
          f"({len(missing)} missing) -> {merged.name}", flush=True)


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


def fetch_perp_flow(coin, symbol, start, end):
    out = OUTDIR / f"{coin}-perp-flow-15m.json"
    if out.exists():
        print(f"  {coin} perp-flow exists, skipping", flush=True)
        return
    to_ms = lambda d: int(datetime.fromisoformat(f"{d}T00:00:00+00:00").timestamp() * 1000)
    rows, cur, end_ms = [], to_ms(start), to_ms(end) + 86_400_000
    while cur < end_ms:
        kl = _get(FAPI, "/fapi/v1/klines", {"symbol": symbol, "interval": "15m",
                                            "startTime": cur, "endTime": end_ms,
                                            "limit": 1000})
        if not kl:
            break
        for k in kl:
            o, c = float(k[1]), float(k[4])
            ts = int(k[0] // 1000)
            rows.append({"timestamp": ts,
                         "datetime": datetime.fromtimestamp(ts, timezone.utc)
                         .strftime("%Y-%m-%d %H:%M:%S"),
                         "up": c > o, "change": (c - o) / o if o else 0.0,
                         "vol": float(k[5]), "ntr": int(k[8]), "tbv": float(k[9])})
        cur = kl[-1][0] + 1
        time.sleep(0.05)
        if len(kl) < 1000:
            break
    out.write_text(json.dumps(rows))
    print(f"  {coin} perp-flow n={len(rows)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-02-11")  # inclusive, = SPAN end
    ap.add_argument("--coins", default="btc,eth,sol,xrp")
    args = ap.parse_args()
    DAILY.mkdir(parents=True, exist_ok=True)

    d0, d1 = date.fromisoformat(args.start), date.fromisoformat(args.end)
    days = [(d0 + timedelta(days=i)).isoformat()
            for i in range((d1 - d0).days + 1)]
    coins = [c.strip() for c in args.coins.split(",")]

    for coin in coins:
        fetch_perp_flow(coin, COINS[coin], args.start, args.end)
        fetch_depth(coin, COINS[coin], days)
    print(f"done -> {OUTDIR}")


if __name__ == "__main__":
    main()
