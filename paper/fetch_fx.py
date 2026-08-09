"""Convert Dukascopy FX candles (fx_raw/, fetched with dukascopy-node) into the
standard {timestamp, datetime, up, change} schema used by the focal study, and
resample to 1h / 4h. Bars with high == low are dropped: at 15m granularity these
occur only when the market is closed (weekends, holiday gaps), so this filter
removes the non-trading bars without touching live ones.

Fetch step (run once; writes fx_raw/{pair}-m15-bid-*.json):
    npx dukascopy-node -i eurusd -from 2025-01-01 -to 2026-02-12 -t m15 -f json -dir fx_raw

Convert step:
    uv run --active python -m paper.fetch_fx
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .common import DATA_DIR
from .fetch_markets import resample, to_record

RAW = Path(__file__).resolve().parent / "fx_raw"
PAIRS = ["eurusd", "usdjpy", "gbpusd"]
RESAMPLE = {"1h": 3600, "4h": 14400}


def load_raw(pair: str):
    files = sorted(RAW.glob(f"{pair}-m15-*.json"))
    if not files:
        raise FileNotFoundError(f"no raw file for {pair} in {RAW}")
    raw = json.loads(files[-1].read_text())
    rows = []
    for r in raw:
        if r["high"] == r["low"]:        # closed-market bar
            continue
        ts = int(r["timestamp"] // 1000)
        rows.append({"timestamp": ts,
                     "datetime": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                     "open": float(r["open"]), "close": float(r["close"])})
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def main():
    for pair in PAIRS:
        rows = load_raw(pair)
        recs15 = [to_record(r["open"], r["close"], r["timestamp"], r["datetime"]) for r in rows]
        (DATA_DIR / f"{pair}-15m.json").write_text(json.dumps(recs15))
        msg = (f"  {pair:7s} 15m n={len(recs15)} "
               f"{recs15[0]['datetime'][:10]}→{recs15[-1]['datetime'][:10]}")
        for iv, sec in RESAMPLE.items():
            rs = resample(rows, sec)
            (DATA_DIR / f"{pair}-{iv}.json").write_text(json.dumps(rs))
            msg += f" | {iv} n={len(rs)}"
        print(msg)


if __name__ == "__main__":
    main()
