"""DFA-1 on session-restricted stock series (resolves the diurnal-crossover caveat).

The DFA exhibit fits a single slope over 8-512 candles, a range that straddles the
diurnal scale (96 candles = one day). Stocks on 24-hour bars carry a session
periodicity that 24/7 crypto lacks; periodicity puts a crossover in the
fluctuation function, and a single-slope fit through a crossover is biased --
asymmetrically across the classes, in the direction that widens the measured H
gap. The proposed fix, never previously run: re-estimate stock H on RTH-only
series (09:30-16:00 New York, weekdays), where the overnight/weekend cycle is
spliced out. RTH stock series carry ~26 bars/day, so the diurnal scale moves to
26 candles; scales are the standard grid capped at the shorter series' half.

Output: per-stock H on 24h vs RTH bars, class means, and the delta -- the size of
the session-periodicity bias in the stock leg of the DFA row.

    uv run --active python -m paper.dfa_rth
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .dfa import SCALES, dfa1

BULK = Path(__file__).resolve().parent / "bulk_data"
OUT = Path(__file__).resolve().parent / "out"
NY = ZoneInfo("America/New_York")


def load(asset: str, rth: bool):
    raw = json.loads((BULK / f"{asset}-15m.json").read_text())
    raw.sort(key=lambda d: d["timestamp"])
    if rth:
        keep = []
        for d in raw:
            t = datetime.fromtimestamp(d["timestamp"], timezone.utc).astimezone(NY)
            if t.weekday() < 5 and (9, 30) <= (t.hour, t.minute) < (16, 0):
                keep.append(d)
        raw = keep
    return np.array([d.get("change", 0.0) for d in raw], float)


def main():
    wide = json.loads((OUT / "wide.json").read_text())
    stocks = [r["asset"] for r in wide if r["class"] == "stock"]
    rows = []
    for a in stocks:
        f = BULK / f"{a}-15m.json"
        if not f.exists():
            continue
        ch24 = load(a, rth=False)
        chrth = load(a, rth=True)
        if len(chrth) < 2 * SCALES[-1]:
            continue
        rows.append({"asset": a,
                     "H_24h": dfa1(ch24, SCALES),
                     "H_rth": dfa1(chrth, SCALES),
                     "n_24h": len(ch24), "n_rth": len(chrth)})
    h24 = np.array([r["H_24h"] for r in rows])
    hrt = np.array([r["H_rth"] for r in rows])
    res = {"n_stocks": len(rows),
           "scales": [int(s) for s in SCALES],
           "mean_H_24h": float(h24.mean()), "median_H_24h": float(np.median(h24)),
           "mean_H_rth": float(hrt.mean()), "median_H_rth": float(np.median(hrt)),
           "mean_delta_rth_minus_24h": float((hrt - h24).mean()),
           "frac_below_half_24h": float((h24 < 0.5).mean()),
           "frac_below_half_rth": float((hrt < 0.5).mean())}
    (OUT / "dfa_rth.json").write_text(json.dumps({**res, "rows": rows}, indent=2))
    print(f"stocks n={len(rows)}: H 24h mean {res['mean_H_24h']:.4f} "
          f"-> RTH mean {res['mean_H_rth']:.4f} "
          f"(delta {res['mean_delta_rth_minus_24h']:+.4f}); "
          f"below 1/2: {res['frac_below_half_24h']*100:.0f}% -> "
          f"{res['frac_below_half_rth']*100:.0f}%")


if __name__ == "__main__":
    main()
