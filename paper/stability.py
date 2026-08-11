"""Calendar-time stability of the MODEL-FREE sign statistic, 2021--2026.

The multi-year walk-forward (multiyear.py) shows the fitted model's 15m edge
is significant in all 24 coin-years. This module asks the same stability
question with nothing fitted: the kernel-weighted sign correlation R of
signlag.py (alpha = 1 fixed a priori, 12 lags), computed per calendar QUARTER
per asset on the 2021-01 -> 2026-02 multiyear 15m histories (4 focal coins,
8 US-listed instruments). If the crypto reversal were a regime artifact of
the 14-month wide window, R would fade somewhere in five years; it does not.

Writes out/stability.json:
  {"quarters": ["2021Q1", ...],
   "assets": {asset: {"class": ..., "R": {quarter: R}}},
   "summary": {per-class mean/min/max per quarter + headline counts}}

    uv run python -m paper.stability
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

from .signlag import N_LAGS, corr, kernel_field, sign_series

MY = Path(__file__).resolve().parent / "multiyear_data"
OUT = Path(__file__).resolve().parent / "out"
CRYPTO = ["btc", "eth", "sol", "xrp"]
TRAD = ["aapl", "gld", "iwm", "nvda", "qqq", "spx", "tlt", "tsla"]
MIN_BARS_PER_Q = 1500          # a stock quarter has ~4000 15m RTH bars


def quarter_key(ts: int) -> str:
    d = datetime.date.fromtimestamp(ts)
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def quarterly_R(name: str) -> dict[str, float]:
    raw = sorted(json.loads((MY / f"{name}-15m.json").read_text()),
                 key=lambda d: d["timestamp"])
    ch = np.array([d.get("change", 0.0) for d in raw], float)
    qs = np.array([quarter_key(d["timestamp"]) for d in raw])
    out = {}
    for q in np.unique(qs):
        m = qs == q
        if m.sum() < MIN_BARS_PER_Q:
            continue
        sig = sign_series(ch[m])
        field = kernel_field(sig)
        out[str(q)] = corr(sig[N_LAGS:], field[N_LAGS:])
    return out


def main():
    assets = {}
    for a in CRYPTO + TRAD:
        assets[a] = {"class": "crypto" if a in CRYPTO else "traditional",
                     "R": quarterly_R(a)}
        print(f"  {a}: {len(assets[a]['R'])} quarters")

    quarters = sorted(set().union(*[set(v["R"]) for v in assets.values()]))
    summary = {}
    for cls, names in (("crypto", CRYPTO), ("traditional", TRAD)):
        per_q = {q: [assets[a]["R"][q] for a in names if q in assets[a]["R"]]
                 for q in quarters}
        vals = [v for q in quarters for v in per_q[q]]
        summary[cls] = {
            "mean": {q: float(np.mean(v)) for q, v in per_q.items() if v},
            "lo": {q: float(np.min(v)) for q, v in per_q.items() if v},
            "hi": {q: float(np.max(v)) for q, v in per_q.items() if v},
            "n_asset_quarters": len(vals),
            "n_negative": int(np.sum(np.array(vals) < 0)),
            "mean_range": [float(np.min([np.mean(v) for v in per_q.values() if v])),
                           float(np.max([np.mean(v) for v in per_q.values() if v]))],
        }

    (OUT / "stability.json").write_text(json.dumps(
        {"quarters": quarters, "assets": assets, "summary": summary}, indent=2))

    for cls in ("crypto", "traditional"):
        s = summary[cls]
        print(f"  {cls:12s} R<0 in {s['n_negative']}/{s['n_asset_quarters']} "
              f"asset-quarters; class mean range [{s['mean_range'][0]:+.4f}, "
              f"{s['mean_range'][1]:+.4f}]")
    print(f"Wrote {OUT/'stability.json'}")


if __name__ == "__main__":
    main()
