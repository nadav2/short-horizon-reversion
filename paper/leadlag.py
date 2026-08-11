"""Spot -> wrapper transmission at 1-minute resolution.

If listed wrappers inherit the underlying's price process through the NAV
link, Binance BTC returns should LEAD IBIT returns intraday, and not much
the other way round. Cross-correlations of 1m returns on aligned RTH
minutes, plus one-lag predictive R^2 in both directions, with a moving-block
bootstrap for the asymmetry.

    uv run --active python -m paper.leadlag
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
WRAP = HERE / "wrapper_data"
OUT = HERE / "out"
NY = ZoneInfo("America/New_York")
MAXLAG = 10
BLOCK = 390          # one RTH day of minutes
N_BOOT = 1000
SEED = 7
# NYSE early closes (13:00 ET) inside the 1m span; the weekday filter alone
# would otherwise admit post-close extended-hours minutes on these dates.
HALF_DAYS = {"2025-11-28", "2025-12-24"}


def load_1m(name):
    raw = json.loads((WRAP / f"{name}-1m.json").read_text())
    out = {}
    for d in raw:
        t = datetime.fromtimestamp(d["timestamp"], timezone.utc).astimezone(NY)
        close = (13, 0) if t.strftime("%Y-%m-%d") in HALF_DAYS else (16, 0)
        if t.weekday() < 5 and (9, 30) <= (t.hour, t.minute) < close:
            out[d["timestamp"]] = d["change"]
    return out


def xcorr(a, b, k):
    """corr(a[t-k], b[t]) for k >= 0."""
    if k == 0:
        return float(np.corrcoef(a, b)[0, 1])
    return float(np.corrcoef(a[:-k], b[k:])[0, 1])


def main():
    btc, ibit = load_1m("btc"), load_1m("ibit")
    common = sorted(set(btc) & set(ibit))
    # require consecutive minutes for lagged pairs: build runs
    ts = np.array(common, np.int64)
    rb = np.array([btc[t] for t in common])
    ri = np.array([ibit[t] for t in common])
    # break at gaps > 60s so lags never straddle a session boundary
    breaks = np.where(np.diff(ts) != 60)[0] + 1
    runs = np.split(np.arange(len(ts)), breaks)
    runs = [r for r in runs if len(r) > MAXLAG + 1]
    print(f"aligned RTH minutes: {len(ts)}  in {len(runs)} consecutive runs")

    def run_xcorr(k, lead, lag):
        num, cnt = [], 0
        for r in runs:
            a, b = lead[r], lag[r]
            if k > 0:
                a, b = a[:-k], b[k:]
            num.append(np.column_stack([a, b]))
        m = np.vstack(num)
        return float(np.corrcoef(m[:, 0], m[:, 1])[0, 1]), len(m)

    rows = {"btc_leads_ibit": {}, "ibit_leads_btc": {}}
    for k in range(0, MAXLAG + 1):
        c1, n = run_xcorr(k, rb, ri)   # btc[t-k] vs ibit[t]
        c2, _ = run_xcorr(k, ri, rb)   # ibit[t-k] vs btc[t]
        rows["btc_leads_ibit"][k] = c1
        rows["ibit_leads_btc"][k] = c2
        print(f"  k={k:2d}  corr(btc[t-k], ibit[t])={c1:+.4f}   "
              f"corr(ibit[t-k], btc[t])={c2:+.4f}")

    lead_sum = sum(rows["btc_leads_ibit"][k] for k in range(1, MAXLAG + 1))
    lag_sum = sum(rows["ibit_leads_btc"][k] for k in range(1, MAXLAG + 1))
    print(f"  asymmetry: sum_k>0 btc->ibit {lead_sum:+.4f}  vs  "
          f"ibit->btc {lag_sum:+.4f}")

    # block bootstrap over runs for the asymmetry
    rng = np.random.default_rng(SEED)
    diffs = []
    ridx = np.arange(len(runs))
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(runs), len(runs))
        sel = [runs[i] for i in pick]
        ls = lg = 0.0
        for k in range(1, MAXLAG + 1):
            m1, m2 = [], []
            for r in sel:
                m1.append(np.column_stack([rb[r][:-k], ri[r][k:]]))
                m2.append(np.column_stack([ri[r][:-k], rb[r][k:]]))
            a = np.vstack(m1); b = np.vstack(m2)
            ls += np.corrcoef(a[:, 0], a[:, 1])[0, 1]
            lg += np.corrcoef(b[:, 0], b[:, 1])[0, 1]
        diffs.append(ls - lg)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  asymmetry diff CI: [{lo:+.4f}, {hi:+.4f}]")

    OUT.mkdir(exist_ok=True)
    (OUT / "leadlag.json").write_text(json.dumps(
        {"n_minutes": int(len(ts)), "n_runs": len(runs), "xcorr": rows,
         "asym_btc_to_ibit": lead_sum, "asym_ibit_to_btc": lag_sum,
         "asym_diff_ci": [float(lo), float(hi)]}, indent=2))
    print(f"wrote {OUT / 'leadlag.json'}")


if __name__ == "__main__":
    main()
