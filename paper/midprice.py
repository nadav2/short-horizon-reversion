"""Quote-midprice-bar replication for BTC and ETH.

Dukascopy distributes separate bid-side and ask-side 15m candles for its crypto
instruments (the same source used for the paper's spot FX panel). Averaging the
two sides bar-by-bar gives quote-midprice bars with no last-trade bid/ask
bounce: mid_open = (bid_open + ask_open)/2, mid_close = (bid_close +
ask_close)/2, both sampled at the same bar boundary. We run the identical
matched walk-forward on two return conventions over these bars —
intra-bar open-to-close (the paper's baseline) and close-to-close — and report
the average relative quoted spread for scale.

Fetch step (run once; writes mid_raw/{inst}-m15-{bid,ask}-*.json):
    npx dukascopy-node -i btcusd -from 2025-01-01 -to 2026-02-12 -t m15 -f json -p bid -dir mid_raw
    npx dukascopy-node -i btcusd -from 2025-01-01 -to 2026-02-12 -t m15 -f json -p ask -dir mid_raw
    (same for ethusd)

Output: paper/out/midprice.json

    uv run --active python -m paper.midprice
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .exchanges import score_series

PAPER = Path(__file__).resolve().parent
RAW_CANDIDATES = [PAPER / "mid_raw", PAPER.parent / "mid_raw"]
OUT = PAPER / "out"
OUT.mkdir(exist_ok=True)

INSTRUMENTS = ["btcusd", "ethusd"]


def load_side(inst: str, side: str):
    for raw in RAW_CANDIDATES:
        files = sorted(raw.glob(f"{inst}-m15-{side}-*.json"))
        if files:
            rows = json.loads(files[-1].read_text())
            return {int(r["timestamp"] // 1000): (float(r["open"]), float(r["close"]),
                                                  float(r["high"]), float(r["low"]))
                    for r in rows}
    raise FileNotFoundError(f"no {side} file for {inst} in {RAW_CANDIDATES}")


def mid_bars(inst: str):
    bid, ask = load_side(inst, "bid"), load_side(inst, "ask")
    common = sorted(set(bid) & set(ask))
    ts, mo, mc, spread_rel, dropped = [], [], [], [], 0
    for t in common:
        bo, bc, bh, bl = bid[t]
        ao, ac, ah, al = ask[t]
        if bh == bl and ah == al:          # market-closed / dead bar
            dropped += 1
            continue
        m_open, m_close = (bo + ao) / 2.0, (bc + ac) / 2.0
        ts.append(t)
        mo.append(m_open)
        mc.append(m_close)
        spread_rel.append((ac - bc) / m_close)
    return (np.array(ts), np.array(mo), np.array(mc),
            np.array(spread_rel), dropped, len(common))


def main():
    results = {}
    for inst in INSTRUMENTS:
        ts, mo, mc, spr, dropped, n_raw = mid_bars(inst)
        oc = (mc - mo) / mo
        cc = np.zeros(len(mc))
        cc[1:] = mc[1:] / mc[:-1] - 1.0
        entry = {
            "n_bars": int(len(ts)), "n_dropped_dead": dropped,
            "span": [str(np.datetime64(int(ts[0]), "s")), str(np.datetime64(int(ts[-1]), "s"))],
            "median_rel_spread_bp": float(np.median(spr) * 1e4),
            "mean_rel_spread_bp": float(np.mean(spr) * 1e4),
        }
        for name, r in (("mid_oc", oc), ("mid_cc", cc)):
            sc = score_series(r, r > 0, label=f"{inst}-{name}")
            entry[name] = sc
            both = "*" if sc and max(sc["ising_auc_p_gt05"], sc["free_auc_p_gt05"]) < 0.05 else " "
            if sc:
                print(f"  {inst}-{name}: AUC={sc['ising_auc']:.4f} "
                      f"[{sc['ising_auc_ci'][0]:.3f},{sc['ising_auc_ci'][1]:.3f}] "
                      f"pI={sc['ising_auc_p_gt05']:.3f} pF={sc['free_auc_p_gt05']:.3f}{both} "
                      f"A={sc['A']:+.3f} n={sc['n_oos']}")
        print(f"  {inst}: {len(ts)} bars ({dropped} dead dropped of {n_raw}), "
              f"median spread {entry['median_rel_spread_bp']:.2f}bp")
        results[inst] = entry

    (OUT / "midprice.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT/'midprice.json'}")


if __name__ == "__main__":
    main()
