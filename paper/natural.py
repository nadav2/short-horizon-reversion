"""Same-asset / different-market-structure natural experiments.

Every other comparison in the paper varies the asset and the venue together, so
"crypto reverts, equities do not" is observationally consistent with either an
asset story or a market-structure story. These two experiments hold the
UNDERLYING FIXED and vary only the structure it trades in:

  GOLD     xauusd  24/5 OTC spot, dealer-intermediated, no consolidated tape
           gld     the listed ETF on the same metal

  BITCOIN  btc     Binance spot: 24/7, no obligated market makers, retail-heavy
           ibit    iShares Bitcoin Trust    -- listed spot-BTC ETF
           fbtc    Fidelity Wise Origin     -- listed spot-BTC ETF
           coin    Coinbase Global          -- listed crypto-revenue equity
           mstr    Strategy (MicroStrategy) -- listed levered BTC holder

IBIT/FBTC are the decisive case: their NAV tracks the same spot BTC process that
scores AUC ~0.533 on Binance, but they trade inside the US equity market
structure. Asset-based accounts predict the reversal survives the wrapper;
structure-based accounts predict it does not.

Three readings per instrument, all on the identical 15m walk-forward, models and
moving-block bootstrap used for the headline result:

  * 24h bars      -- the paper's baseline convention;
  * non-flat bars -- flat-bar-robust AUC (the paper's primary metric);
  * RTH-matched   -- listed instruments restricted to 09:30-16:00 New York AND
                     btc/xauusd restricted to the SAME slots, so the comparison
                     is like-for-like in session terms rather than comparing a
                     24/7 tape against a 6.5h one.

    uv run python -m paper.natural
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

from .common import DATA_DIR
from .compare_markets import N_BOOT, SEED, SPAN, WINDOWS, autocorr1, block_boot_idx
from .models import ARLogit, IsingLogit
from .walkforward import walk_forward

OUT = Path(__file__).resolve().parent / "out"
NY = ZoneInfo("America/New_York")
INTERVAL = "15m"
BLOCK = 384
BARS_PER_DAY_RTH = 26
RTH_TRAIN, RTH_TEST, RTH_BLOCK = 60 * BARS_PER_DAY_RTH, 10 * BARS_PER_DAY_RTH, 4 * BARS_PER_DAY_RTH

# (label, underlying, [instruments]) -- first entry of each list is the
# round-the-clock / OTC leg, the rest are the listed wrappers.
EXPERIMENTS = [
    {"experiment": "gold", "underlying": "gold",
     "legs": [("xauusd", "OTC spot 24/5"), ("gld", "listed ETF")]},
    {"experiment": "bitcoin", "underlying": "BTC",
     "legs": [("btc", "crypto spot 24/7"), ("ibit", "listed spot-BTC ETF"),
              ("fbtc", "listed spot-BTC ETF"), ("coin", "listed equity proxy"),
              ("mstr", "listed levered proxy")]},
]


def load(asset: str, rth: bool = False):
    raw = [d for d in json.loads((DATA_DIR / f"{asset}-{INTERVAL}.json").read_text())
           if SPAN[0] <= d["datetime"][:10] <= SPAN[1]]
    raw.sort(key=lambda d: d["timestamp"])
    if rth:
        keep = []
        for d in raw:
            t = datetime.fromtimestamp(d["timestamp"], timezone.utc).astimezone(NY)
            if t.weekday() < 5 and (9, 30) <= (t.hour, t.minute) < (16, 0):
                keep.append(d)
        raw = keep
    ch = np.array([d.get("change", 0.0) for d in raw], float)
    ups = np.array([bool(d["up"]) for d in raw], bool)
    return ch, ups


def models():
    return [IsingLogit(n_lags=12), ARLogit("free", n_lags=12)]


def evaluate(asset: str, rth: bool = False, gap: bool = False) -> dict | None:
    ch, ups = load(asset, rth=rth)
    tr, te = (RTH_TRAIN, RTH_TEST) if rth else WINDOWS[INTERVAL]
    block = RTH_BLOCK if rth else BLOCK
    if len(ch) < tr + te:
        return None
    if gap:
        # one-candle gap between the last predictor and the label: predict candle
        # t from t-2..t-13. Pure bid-ask bounce / staleness lives at the boundary
        # between adjacent bars and should not survive; genuine multi-lag reversion
        # should. Same control as Sec. "Skipping the most recent bar".
        ch = np.concatenate([[0.0], ch[:-1]])
    res, n_folds = walk_forward(ch, ups, tr, te, n_lags=12, models=models)
    y = res["ising"]["actuals"].astype(int)
    if len(y) == 0 or y.min() == y.max():
        return None
    p_is, p_fr = res["ising"]["probs"], res["free"]["probs"]
    idx = res["ising"]["idx"]

    auc_is, auc_fr = roc_auc_score(y, p_is), roc_auc_score(y, p_fr)
    rng = np.random.default_rng(SEED)
    b_is, b_fr = [], []
    for _ in range(N_BOOT):
        bi = block_boot_idx(len(y), block, rng)
        yb = y[bi]
        if yb.min() == yb.max():
            continue
        b_is.append(roc_auc_score(yb, p_is[bi]))
        b_fr.append(roc_auc_score(yb, p_fr[bi]))
    b_is, b_fr = np.array(b_is), np.array(b_fr)
    p_gt_is, p_gt_fr = float(np.mean(b_is <= 0.5)), float(np.mean(b_fr <= 0.5))

    # flat-bar-robust AUC: score only bars with r_t != 0
    nonflat = ch[idx] != 0
    auc_nf = (float(roc_auc_score(y[nonflat], p_is[nonflat]))
              if nonflat.sum() > 100 and len(np.unique(y[nonflat])) > 1 else float("nan"))

    # split-half stability: is the effect carried by the whole span or a few folds?
    h = len(y) // 2
    halves = []
    for sl in (slice(None, h), slice(h, None)):
        yy = y[sl]
        halves.append(float(roc_auc_score(yy, p_is[sl]))
                      if len(np.unique(yy)) > 1 else float("nan"))

    return {
        "asset": asset, "rth": rth, "gap": gap,
        "auc_half1": halves[0], "auc_half2": halves[1],
        "n_bars": int(len(ch)), "n_oos": int(len(y)),
        "n_folds": n_folds, "flat_frac": float(np.mean(ch == 0.0)),
        "ac1": autocorr1(ch), "base_rate_up": float(y.mean()),
        "auc_ising": float(auc_is),
        "auc_ci": [float(np.percentile(b_is, 2.5)), float(np.percentile(b_is, 97.5))],
        "auc_free": float(auc_fr),
        "p_ising": p_gt_is, "p_free": p_gt_fr,
        "conj_p": max(p_gt_is, p_gt_fr),
        "auc_nonflat": auc_nf,
        "A": float(np.mean([p["A"] for p in res["ising"]["fold_params"]])),
    }


def main():
    rows = []
    for exp in EXPERIMENTS:
        print(f"\n=== {exp['experiment']}: same underlying ({exp['underlying']}), "
              f"different market structure ===")
        for asset, structure in exp["legs"]:
            for rth, gap in ((False, False), (True, False), (False, True), (True, True)):
                r = evaluate(asset, rth=rth, gap=gap)
                tag = ("RTH" if rth else "24h") + ("+gap" if gap else "")
                if r is None:
                    print(f"  {asset:7s} {tag:8s} skipped (insufficient bars)")
                    continue
                r.update({"experiment": exp["experiment"], "structure": structure,
                          "underlying": exp["underlying"]})
                rows.append(r)
                sig = "*" if r["conj_p"] < 0.05 else " "
                print(f"  {asset:7s} {tag:8s} {structure:22s} "
                      f"n={r['n_oos']:6d} AUC={r['auc_ising']:.4f} "
                      f"[{r['auc_ci'][0]:.3f},{r['auc_ci'][1]:.3f}] "
                      f"halves={r['auc_half1']:.3f}/{r['auc_half2']:.3f} "
                      f"nonflat={r['auc_nonflat']:.4f} p={r['conj_p']:.3f}{sig} "
                      f"A={r['A']:+.3f}", flush=True)

    (OUT / "natural.json").write_text(json.dumps(
        {"config": {"interval": INTERVAL, "span": SPAN, "n_boot": N_BOOT,
                    "block": BLOCK, "rth_block": RTH_BLOCK, "seed": SEED},
         "experiments": EXPERIMENTS, "rows": rows}, indent=2))
    print(f"\nWrote {OUT/'natural.json'}")


if __name__ == "__main__":
    main()
