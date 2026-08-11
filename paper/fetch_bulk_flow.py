"""Refetch the wide crypto universe keeping the flow fields (base volume,
trade count, taker-buy base volume) that fetch_bulk_crypto discarded. The
symbol list is read from the existing bulk_data/ crypto files so the universe
is identical to the paper's 183-pair cross-section (stocks are skipped by
probing Binance for the symbol).

Output: bulk_flow/{sid}-15m-flow.json over the identical span.

    uv run --active python -m paper.fetch_bulk_flow
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BULK = HERE / "bulk_data"
OUTDIR = HERE / "bulk_flow"
BASES = ["https://api.binance.com", "https://data-api.binance.vision"]
INTERVAL = "15m"


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


def crypto_sids():
    """Bulk files that are Binance USDT pairs (probe once against exchangeInfo)."""
    info = _get("/api/v3/exchangeInfo", {"permissions": "SPOT"})
    listed = {s["symbol"] for s in info["symbols"]}
    sids = []
    for f in sorted(BULK.glob("*-15m.json")):
        sid = f.name[:-len("-15m.json")]
        if f"{sid.upper()}USDT" in listed:
            sids.append(sid)
    return sids


def fetch(symbol, start_ms, end_ms):
    rows, cur = [], start_ms
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
                         "vol": float(k[5]), "qvol": float(k[7]),
                         "ntr": int(k[8]), "tbv": float(k[9])})
        cur = kl[-1][0] + 1
        time.sleep(0.05)
        if len(kl) < 1000:
            break
    return rows


def main():
    OUTDIR.mkdir(exist_ok=True)
    sids = crypto_sids()
    print(f"{len(sids)} crypto sids from bulk_data")
    for i, sid in enumerate(sids):
        out = OUTDIR / f"{sid}-15m-flow.json"
        if out.exists():
            continue
        # identical span to the existing bulk file
        old = json.loads((BULK / f"{sid}-15m.json").read_text())
        if not old:
            continue
        start_ms = old[0]["timestamp"] * 1000
        end_ms = old[-1]["timestamp"] * 1000 + 900_000
        try:
            rows = fetch(f"{sid.upper()}USDT", start_ms, end_ms)
        except Exception as e:
            print(f"  [{i+1}/{len(sids)}] {sid} FAILED: {e}")
            continue
        out.write_text(json.dumps(rows))
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(sids)}] {sid} n={len(rows)}", flush=True)
    print(f"done -> {OUTDIR}")


if __name__ == "__main__":
    main()
