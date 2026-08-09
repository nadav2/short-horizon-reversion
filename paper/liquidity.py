"""Liquidity stratification of the wide-universe result + flat-bar robustness.

Two microstructure integrity checks on the 183-pair crypto claim:

1. **Liquidity stratification.** If the reported predictability were a
   stale-price / bid-ask-bounce artifact, it would concentrate in the thinnest
   pairs. We fetch each pair's average daily quote volume over the matched
   span (Binance daily klines, public) and report AUC / significance /
   coupling by volume quintile, plus rank correlations of AUC with volume and
   with per-pair thinness proxies (fraction of flat candles, median |return|).

2. **Flat-bar-robust AUC.** A stale bar (change == 0) is labeled "down" by the
   ``up = close > open`` convention, so a reversal model can score it
   "correctly" with zero P&L. We recompute each asset's OOS AUC excluding
   flat candles; if the headline AUC moves, it was partly staleness.

    uv run --active python -m paper.liquidity
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .fetch_bulk_crypto import _get, to_ms, START, END
from .fetch_bulk_stocks import UNIVERSE

BULK = Path(__file__).resolve().parent / "bulk_data"
OUT = Path(__file__).resolve().parent / "out"
STOCK_SET = {s.lower().replace(".", "-") for s in UNIVERSE}
N_QUINTILES = 5


def fetch_volumes(assets: list[str]) -> dict[str, float]:
    """Mean daily quote volume (USDT) per pair over the matched span; cached."""
    cache = OUT / "volumes.json"
    vols = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [a for a in assets if a not in vols]
    for i, a in enumerate(todo):
        sym = a.upper().replace("-", "") + "USDT"
        try:
            kl = _get("/api/v3/klines", {"symbol": sym, "interval": "1d",
                                         "startTime": to_ms(START),
                                         "endTime": to_ms(END), "limit": 1000})
            qv = [float(k[7]) for k in kl]
            vols[a] = float(np.mean(qv)) if qv else None
        except Exception as e:
            print(f"  {a} ({sym}): volume fetch failed: {e}")
            vols[a] = None
        if (i + 1) % 25 == 0:
            print(f"  volumes [{i+1}/{len(todo)}]")
            cache.write_text(json.dumps(vols))
    cache.write_text(json.dumps(vols))
    return vols


def asset_micro(asset: str) -> dict | None:
    """Thinness proxies + flat-bar-robust OOS AUC for one asset."""
    bulk = BULK / f"{asset}-15m.json"
    dump = OUT / "wide_oos" / f"{asset}.npz"
    if not (bulk.exists() and dump.exists()):
        return None
    raw = json.loads(bulk.read_text())
    raw.sort(key=lambda rr: rr["timestamp"])
    ts_all = np.array([rr["timestamp"] for rr in raw], np.int64)
    ch_all = np.array([rr.get("change", 0.0) for rr in raw], float)
    d = np.load(dump)
    pos = np.searchsorted(ts_all, d["ts"])
    if not np.array_equal(ts_all[pos], d["ts"]):
        return None
    r = ch_all[pos]
    y = d["actual"].astype(int)
    out = {"zero_frac": float(np.mean(ch_all == 0.0)),
           "med_abs_change": float(np.median(np.abs(ch_all[ch_all != 0.0])))}
    nz = r != 0.0
    for m in ("ising", "free"):
        p = d[f"p_{m}"].astype(float)
        ynz = y[nz]
        out[f"auc_nz_{m}"] = (float(roc_auc_score(ynz, p[nz]))
                              if len(np.unique(ynz)) > 1 and nz.sum() > 2000 else None)
    return out


def main():
    wide = json.loads((OUT / "wide.json").read_text())
    crypto = [r for r in wide if r["class"] == "crypto"]
    print(f"{len(crypto)} crypto pairs")
    vols = fetch_volumes([r["asset"] for r in crypto])

    rows = []
    for r in wide:
        m = asset_micro(r["asset"])
        if m is None:
            continue
        rows.append({"asset": r["asset"], "class": r["class"],
                     "auc_ising": r["auc_ising"], "auc_free": r["auc_free"],
                     "p_ising": r["p_ising"], "p_free": r["p_free"], "A": r["A"],
                     "volume": vols.get(r["asset"]), **m})

    cr = [r for r in rows if r["class"] == "crypto" and r["volume"]]
    cr.sort(key=lambda r: r["volume"])
    qs = np.array_split(np.arange(len(cr)), N_QUINTILES)
    quintiles = []
    for qi, idx in enumerate(qs):
        sub = [cr[i] for i in idx]
        aucs = np.array([r["auc_ising"] for r in sub])
        quintiles.append({
            "quintile": qi + 1,
            "n": len(sub),
            "vol_range_musd": [round(sub[0]["volume"] / 1e6, 1),
                               round(sub[-1]["volume"] / 1e6, 1)],
            "mean_auc": float(aucs.mean()),
            "median_auc": float(np.median(aucs)),
            "frac_sig_both": float(np.mean([r["p_ising"] < 0.05 and r["p_free"] < 0.05
                                            for r in sub])),
            "mean_A": float(np.mean([r["A"] for r in sub])),
            "mean_zero_frac": float(np.mean([r["zero_frac"] for r in sub])),
        })
        q = quintiles[-1]
        print(f"  Q{q['quintile']} vol {q['vol_range_musd'][0]}-{q['vol_range_musd'][1]}M$:"
              f" AUC {q['mean_auc']:.4f}  sig {q['frac_sig_both']*100:.0f}%"
              f"  A {q['mean_A']:+.3f}  zero {q['mean_zero_frac']*100:.2f}%")

    logv = np.log10([r["volume"] for r in cr])
    auc = np.array([r["auc_ising"] for r in cr])
    zf = np.array([r["zero_frac"] for r in cr])
    sp_vol = spearmanr(logv, auc)
    sp_zero = spearmanr(zf, auc)
    nz_pairs = [(r["auc_ising"], r["auc_nz_ising"]) for r in cr
                if r["auc_nz_ising"] is not None]
    d_nz = np.array([a - b for a, b in nz_pairs])
    summary = {
        "spearman_logvol_auc": {"rho": float(sp_vol.statistic), "p": float(sp_vol.pvalue)},
        "spearman_zerofrac_auc": {"rho": float(sp_zero.statistic), "p": float(sp_zero.pvalue)},
        "flatbar": {"n": len(nz_pairs),
                    "mean_auc_minus_aucnz": float(d_nz.mean()),
                    "max_abs_diff": float(np.abs(d_nz).max()),
                    "frac_within_0005": float(np.mean(np.abs(d_nz) < 0.005))},
        "quintiles": quintiles,
    }
    print(f"  Spearman(log vol, AUC) rho={sp_vol.statistic:+.3f} p={sp_vol.pvalue:.3f}")
    print(f"  Spearman(zero frac, AUC) rho={sp_zero.statistic:+.3f} p={sp_zero.pvalue:.3f}")
    print(f"  flat-bar AUC shift: mean {d_nz.mean():+.5f}, max |diff| {np.abs(d_nz).max():.4f}")

    (OUT / "liquidity.json").write_text(json.dumps(
        {"summary": summary, "assets": rows}, indent=2))
    print(f"Wrote {OUT/'liquidity.json'}")


if __name__ == "__main__":
    main()
