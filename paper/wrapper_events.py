"""Within-instrument wrapper event studies.

Three designs, all on session-matched RTH 15m bars with the panel's RTH
walk-forward geometry and block bootstrap:

1. CONVERSION SWITCH. GBTC traded as a closed-end trust at large premiums or
   discounts to NAV (NAV-arbitrage link OFF: no creation/redemption) until it
   converted to a spot ETF on 2024-01-11; ETHE converted 2024-07-23. Same
   ticker, same underlying -- only the NAV link changed. Inheritance predicts
   the wrapper's reversal snaps to the underlying's after conversion and
   tracks it loosely before.

2. INHERITANCE AT BIRTH. IBIT and FBTC launched 2024-01-11 with no trading
   history. Do their first six months already carry the underlying's
   coupling? Instant inheritance is hard for a venue-learns-its-own-
   microstructure account to mimic.

3. THE FUTURES WRAPPER'S LONG HISTORY. BITO (CME-futures-based) has traded
   since 2021-10-19, before any spot ETF existed. Its early era tests
   inheritance through a futures NAV link, in a different regime.

Each wrapper era is paired with its underlying (multiyear_data BTC/ETH)
scored on the identical NY 09:30-16:00 slots over the identical dates.

    uv run --active python -m paper.wrapper_events
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

from .compare_markets import N_BOOT, SEED, block_boot_idx
from .models import ARLogit, IsingLogit
from .walkforward import walk_forward

HERE = Path(__file__).resolve().parent
WRAP = HERE / "wrapper_data"
MULTI = HERE / "multiyear_data"
OUT = HERE / "out"
NY = ZoneInfo("America/New_York")
BARS = 26                                   # RTH 15m bars per day
TRAIN, TEST, BLOCK = 60 * BARS, 10 * BARS, 4 * BARS

# (label, wrapper_file, underlying_file, start, end)
ERAS = [
    # NOTE: Alpaca's feed has no OTC prints, so GBTC/ETHE begin at their ETF
    # conversion (2024-01-11 / 2024-07-23); the trust era is not observable
    # here and the conversion test reduces to inheritance-from-uplisting.
    ("gbtc_etf",   "gbtc", "btc", "2024-01-11", "2026-02-11"),
    ("gbtc_birth", "gbtc", "btc", "2024-01-11", "2024-07-11"),
    ("ethe_etf",   "ethe", "eth", "2024-07-23", "2026-02-11"),
    ("ethe_birth", "ethe", "eth", "2024-07-23", "2025-01-23"),
    ("ibit_birth", "ibit", "btc", "2024-01-11", "2024-07-11"),
    ("fbtc_birth", "fbtc", "btc", "2024-01-11", "2024-07-11"),
    ("etha_birth", "etha", "eth", "2024-07-23", "2025-01-23"),
    ("feth_birth", "feth", "eth", "2024-07-23", "2025-01-23"),
    ("ibit_full",  "ibit", "btc", "2024-01-11", "2026-02-11"),
    ("fbtc_full",  "fbtc", "btc", "2024-01-11", "2026-02-11"),
    ("bito_early", "bito", "btc", "2021-10-19", "2023-12-31"),
    ("bito_late",  "bito", "btc", "2024-01-11", "2026-02-11"),
]


def load_rth(path, start, end):
    raw = [d for d in json.loads(path.read_text())
           if start <= d["datetime"][:10] <= end]
    raw.sort(key=lambda d: d["timestamp"])
    keep = []
    for d in raw:
        t = datetime.fromtimestamp(d["timestamp"], timezone.utc).astimezone(NY)
        if t.weekday() < 5 and (9, 30) <= (t.hour, t.minute) < (16, 0):
            keep.append(d)
    ch = np.array([d.get("change", 0.0) for d in keep], float)
    ups = np.array([bool(d["up"]) for d in keep], bool)
    return ch, ups


def models():
    return [IsingLogit(n_lags=12), ARLogit("free", n_lags=12)]


def evaluate(ch, ups):
    if len(ch) < TRAIN + TEST:
        return None
    res, nf = walk_forward(ch, ups, TRAIN, TEST, n_lags=12, models=models)
    y = res["ising"]["actuals"].astype(int)
    if len(y) == 0 or y.min() == y.max():
        return None
    p_is, p_fr = res["ising"]["probs"], res["free"]["probs"]
    rng = np.random.default_rng(SEED)
    b_is, b_fr = [], []
    for _ in range(N_BOOT):
        bi = block_boot_idx(len(y), BLOCK, rng)
        yb = y[bi]
        if yb.min() == yb.max():
            continue
        b_is.append(roc_auc_score(yb, p_is[bi]))
        b_fr.append(roc_auc_score(yb, p_fr[bi]))
    b_is, b_fr = np.array(b_is), np.array(b_fr)
    return {
        "n_bars": int(len(ch)), "n_oos": int(len(y)), "n_folds": nf,
        "flat_frac": float(np.mean(ch == 0.0)),
        "auc_ising": float(roc_auc_score(y, p_is)),
        "auc_ci": [float(np.percentile(b_is, 2.5)), float(np.percentile(b_is, 97.5))],
        "auc_free": float(roc_auc_score(y, p_fr)),
        "p_ising": float(np.mean(b_is <= 0.5)), "p_free": float(np.mean(b_fr <= 0.5)),
        "conj_p": float(max(np.mean(b_is <= 0.5), np.mean(b_fr <= 0.5))),
        "A": float(np.mean([p["A"] for p in res["ising"]["fold_params"]])),
    }


def main():
    rows = []
    for label, wsid, usid, start, end in ERAS:
        wpath = WRAP / f"{wsid}-15m.json"
        if not wpath.exists():
            print(f"  {label}: no data file, skipping")
            continue
        w_ch, w_ups = load_rth(wpath, start, end)
        u_ch, u_ups = load_rth(MULTI / f"{usid}-15m.json", start, end)
        rw, ru = evaluate(w_ch, w_ups), evaluate(u_ch, u_ups)
        if rw is None:
            print(f"  {label}: insufficient wrapper bars "
                  f"(n={len(w_ch)}, need {TRAIN + TEST})")
            continue
        row = {"era": label, "wrapper": wsid, "underlying": usid,
               "start": start, "end": end, "w": rw, "u": ru,
               "gap": (rw["auc_ising"] - ru["auc_ising"]) if ru else None}
        rows.append(row)
        us = (f"underlying AUC={ru['auc_ising']:.4f} A={ru['A']:+.3f}" if ru
              else "underlying n/a")
        sig = "*" if rw["conj_p"] < 0.05 else " "
        print(f"  {label:11s} {start}..{end}  n_oos={rw['n_oos']:6d} "
              f"flat={rw['flat_frac'] * 100:4.1f}%  "
              f"AUC={rw['auc_ising']:.4f} [{rw['auc_ci'][0]:.3f},{rw['auc_ci'][1]:.3f}] "
              f"p={rw['conj_p']:.3f}{sig} A={rw['A']:+.3f}   {us}", flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "wrapper_events.json").write_text(json.dumps(
        {"config": {"train": TRAIN, "test": TEST, "block": BLOCK,
                    "n_boot": N_BOOT, "seed": SEED}, "rows": rows}, indent=2))
    print(f"\nwrote {OUT / 'wrapper_events.json'}")


if __name__ == "__main__":
    main()
