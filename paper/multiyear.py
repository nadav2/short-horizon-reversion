"""Multi-year regime stability of the crypto 15m edge (2021 - 2026).

The matched cross-market window is a single ~14-month regime. Here the focal
coins' 15m history is extended back to 2021 (multiyear_data/, Binance) and the
standard walk-forward is scored *per calendar year*: out-of-sample AUC with a
moving-block bootstrap CI, the free-logit corroboration, and the fitted coupling.
This answers two referee questions at once: is the edge specific to the 2025-26
window, and has it decayed as the market matured?

    uv run --active python -m paper.multiyear
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .compare_markets import block_boot_idx
from .models import ARLogit, IsingLogit
from .walkforward import walk_forward

DATA = Path(__file__).resolve().parent / "multiyear_data"
OUT = Path(__file__).resolve().parent / "out"
COINS = ["btc", "eth", "sol", "xrp"]
TRAIN, TEST = 5760, 960          # matched 15m geometry (compare_markets)
BLOCK, N_BOOT, SEED = 384, 1000, 7


def models():
    return [IsingLogit(n_lags=12), ARLogit("free", n_lags=12)]


def score_asset(name: str, min_year_candles: int = 3000) -> list[dict]:
    """Per-calendar-year walk-forward scoring for one multiyear_data series."""
    raw = json.loads((DATA / f"{name}-15m.json").read_text())
    raw.sort(key=lambda d: d["timestamp"])
    ch = np.array([d.get("change", 0.0) for d in raw], float)
    ups = np.array([bool(d["up"]) for d in raw], bool)
    ts = np.array([d["timestamp"] for d in raw], np.int64)

    res, nf = walk_forward(ch, ups, TRAIN, TEST, n_lags=12, models=models)
    y = res["ising"]["actuals"].astype(int)
    idx = res["ising"]["idx"].astype(int)
    years = np.array([datetime.fromtimestamp(t, timezone.utc).year for t in ts[idx]])
    # fold-level A, assigned to the year of the fold's first test candle
    fold_years = years[::TEST][:nf] if len(years) else []
    fold_A = np.array([fp["A"] for fp in res["ising"]["fold_params"]])

    rows = []
    for yr in sorted(set(years.tolist())):
        sel = years == yr
        if sel.sum() < min_year_candles:
            continue
        yy = y[sel]
        if yy.min() == yy.max():
            continue
        p_is, p_fr = res["ising"]["probs"][sel], res["free"]["probs"][sel]
        rng = np.random.default_rng(SEED)
        ge_is = ge_fr = n_eff = 0
        boot = []
        for _ in range(N_BOOT):
            bidx = block_boot_idx(len(yy), BLOCK, rng)
            yb = yy[bidx]
            if yb.min() == yb.max():
                continue
            n_eff += 1
            a = roc_auc_score(yb, p_is[bidx])
            boot.append(a)
            ge_is += a <= 0.5
            ge_fr += roc_auc_score(yb, p_fr[bidx]) <= 0.5
        A_yr = float(fold_A[np.asarray(fold_years[:len(fold_A)]) == yr].mean()) \
            if len(fold_A) and (np.asarray(fold_years[:len(fold_A)]) == yr).any() else float("nan")
        rows.append({
            "coin": name, "year": int(yr), "n_oos": int(sel.sum()),
            "auc_ising": float(roc_auc_score(yy, p_is)),
            "auc_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p_ising": ge_is / max(n_eff, 1),
            "auc_free": float(roc_auc_score(yy, p_fr)),
            "p_free": ge_fr / max(n_eff, 1),
            "A": A_yr,
        })
        r = rows[-1]
        both = "*" if (r["p_ising"] < 0.05 and r["p_free"] < 0.05) else " "
        print(f"  {name:4s} {yr}  n={r['n_oos']:6d}  AUC={r['auc_ising']:.4f} "
              f"[{r['auc_ci'][0]:.3f},{r['auc_ci'][1]:.3f}]  free={r['auc_free']:.4f}  "
              f"A={r['A']:+.3f}  both-sig{both}")
    return rows


def main():
    rows = []
    for coin in COINS:
        rows.extend(score_asset(coin))
    (OUT / "multiyear.json").write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {OUT/'multiyear.json'}")


if __name__ == "__main__":
    main()
