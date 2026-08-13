"""Fetch forced-liquidation orders and matched perp klines for the
liquidation-archive window (identification design).

Binance Vision publishes historical forced-order (liquidation) snapshots for
the COIN-margined perps only, 2023-06-25..2024-10-14 (the USDT-M archive was
never published, and the CM one stops there). Liquidations are mechanically
triggered by margin breaches: forced demand for immediacy that carries no
private information. That makes them the identifying variation the mechanism
section lacks: if reversal is compensation for supplying immediacy, it should
be at least as strong after forced flow as after ordinary aggressor flow of
the same size, and increase in the forced intensity holding size and flow
fixed (the intraday analogue of fire-sale identification).

Caveats encoded downstream: the archive carries Binance's throttled
forced-order stream (at most one order per second per symbol), so per-bar
sums are an INTENSITY measure, not a complete forced-volume census; rows
duplicate and are deduplicated on all columns. Days with no liquidations
have no file (404 = zero, recorded as such).

Output (liq_data/): {coin}-liq-15m.json   per-bar forced-order aggregates
                    {coin}-cm-15m.json    CM perp klines + flow fields
                    {coin}-um-15m.json    USDT-M perp klines + flow fields

    uv run --active python -m paper.fetch_liq
"""

from __future__ import annotations

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
OUTDIR = HERE / "liq_data"
DAILY = OUTDIR / "daily"
VISION = "https://data.binance.vision/data/futures/cm/daily/liquidationSnapshot"
DAPI = ["https://dapi.binance.com"]
FAPI = ["https://fapi.binance.com"]
START, END = "2023-06-25", "2024-10-14"  # the full CM liquidation archive
BAR = 900
# CM inverse contracts: notional = contracts x contract size (USD)
COINS = {"btc": ("BTCUSD_PERP", "BTCUSDT", 100.0),
         "eth": ("ETHUSD_PERP", "ETHUSDT", 10.0),
         "sol": ("SOLUSD_PERP", "SOLUSDT", 10.0),
         "xrp": ("XRPUSD_PERP", "XRPUSDT", 10.0)}


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


def aggregate_day(blob, csize):
    """liquidationSnapshot CSV -> {bar_ts: aggregates}. side=SELL is a long
    position force-closed (forced selling); BUY a short (forced buying)."""
    zf = zipfile.ZipFile(io.BytesIO(blob))
    text = io.TextIOWrapper(zf.open(zf.namelist()[0]), encoding="utf-8")
    reader = csv.reader(text)
    header = next(reader)
    ci = {c: header.index(c) for c in ("time", "side", "average_price",
                                       "accumulated_fill_quantity")}
    seen = set()
    bars = {}
    for row in reader:
        key = tuple(row)
        if key in seen:  # archive rows duplicate
            continue
        seen.add(key)
        ts = int(row[ci["time"]]) // 1000
        bar = ts - ts % BAR
        side = row[ci["side"]].lower()
        notional = float(row[ci["accumulated_fill_quantity"]]) * csize
        rec = bars.setdefault(bar, {"sell_notional": 0.0, "sell_n": 0,
                                    "buy_notional": 0.0, "buy_n": 0})
        rec[f"{side}_notional"] += notional
        rec[f"{side}_n"] += 1
    return [{"timestamp": b, **{k: (round(v, 2) if isinstance(v, float) else v)
                                for k, v in rec.items()}}
            for b, rec in sorted(bars.items())]


def fetch_liq(coin, symbol, csize, days):
    empty = 0
    for d in days:
        cache = DAILY / f"{coin}-liq-{d}.json"
        if cache.exists():
            continue
        blob = _download(f"{VISION}/{symbol}/{symbol}-liquidationSnapshot-{d}.zip")
        if blob is None:  # no liquidations that day
            cache.write_text("[]")
            empty += 1
            continue
        cache.write_text(json.dumps(aggregate_day(blob, csize)))
        time.sleep(0.1)
    rows = []
    for d in days:
        rows.extend(json.loads((DAILY / f"{coin}-liq-{d}.json").read_text()))
    rows.sort(key=lambda r: r["timestamp"])
    merged = OUTDIR / f"{coin}-liq-15m.json"
    merged.write_text(json.dumps(rows))
    print(f"  {coin} liq: {len(rows)} bars with forced orders "
          f"({empty} empty days this run) -> {merged.name}", flush=True)


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


def fetch_klines(out, bases, path, symbol, limit):
    if out.exists():
        print(f"  {out.name} exists, skipping", flush=True)
        return
    to_ms = lambda d: int(datetime.fromisoformat(f"{d}T00:00:00+00:00").timestamp() * 1000)
    rows, cur, end_ms = [], to_ms(START), to_ms(END) + 86_400_000
    while cur < end_ms:
        kl = _get(bases, path, {"symbol": symbol, "interval": "15m",
                                "startTime": cur, "endTime": end_ms,
                                "limit": limit})
        if not kl:
            break
        for k in kl:
            o, c = float(k[1]), float(k[4])
            ts = int(k[0] // 1000)
            rows.append({"timestamp": ts,
                         "datetime": datetime.fromtimestamp(ts, timezone.utc)
                         .strftime("%Y-%m-%d %H:%M:%S"),
                         "up": c > o, "change": (c - o) / o if o else 0.0,
                         "close": c,
                         "vol": float(k[5]), "ntr": int(k[8]), "tbv": float(k[9])})
        cur = kl[-1][0] + 1
        time.sleep(0.05)
        if len(kl) < limit:
            break
    out.write_text(json.dumps(rows))
    print(f"  {out.name} n={len(rows)}", flush=True)


def main():
    DAILY.mkdir(parents=True, exist_ok=True)
    d0, d1 = date.fromisoformat(START), date.fromisoformat(END)
    days = [(d0 + timedelta(days=i)).isoformat()
            for i in range((d1 - d0).days + 1)]
    for coin, (cm_sym, um_sym, csize) in COINS.items():
        fetch_klines(OUTDIR / f"{coin}-um-15m.json", FAPI, "/fapi/v1/klines",
                     um_sym, 1000)
        fetch_klines(OUTDIR / f"{coin}-cm-15m.json", DAPI, "/dapi/v1/klines",
                     cm_sym, 1000)
        fetch_liq(coin, cm_sym, csize, days)
    print(f"done -> {OUTDIR}")


if __name__ == "__main__":
    main()
