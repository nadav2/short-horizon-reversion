"""Ex-ante universe construction: is the wide-universe gap an artifact of
selecting high-volume Binance pairs *after* the sample (2026-06-08)?

The wide universe (Section sec:wide) was frozen on a post-sample volume
ranking. This module tests whether the class-mean AUC gap survives when the
crypto universe is instead selected on information available at the sample
START: each pair's quote volume over the first month of the sample window
(2025-01-01 .. 2025-01-31). Two concerns are separated:

  (1) ex-post VOLUME conditioning -- pairs that only became high-volume late
      in the window. We rank the existing crypto pairs by their Jan-2025
      (ex-ante) quote volume and recompute the gap on the ex-ante top-N and
      across ex-ante volume strata. If the gap is unchanged, ranking on
      ex-post volume did not manufacture it.

  (2) late LISTING -- pairs with no Jan-2025 history at all (listed after the
      sample start). We recompute the gap on the subset that already traded
      ex ante. If unchanged, late-listed survivors do not drive it.

Full survivorship (pairs delisted before the freeze date) cannot be repaired
from the public API and remains a disclosed limitation; this module addresses
the volume-conditioning and late-listing components, which are mechanical.

Reuses the per-asset OOS dumps (out/wide_oos/) and the joint dependence-aware
block bootstrap of paper.dependence, restricted to each sub-universe.

    uv run --active python -m paper.exante
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .dependence import OOS, SLOT, BLOCK, MIN_OBS, fast_auc
from .fetch_bulk_crypto import _get, to_ms

OUT = Path(__file__).resolve().parent / "out"
VOL_CACHE = OUT / "exante_volume.json"
EXANTE_START, EXANTE_END = "2025-01-01", "2025-01-31"   # first month of the sample
N_BOOT, SEED = 1000, 7


def fetch_exante_volume(symbols: list[str]) -> dict[str, float]:
    """Summed quote volume over the first sample month, per crypto pair.

    Returns {symbol: quote_volume}; pairs with no klines in the window
    (listed later) are absent from the dict."""
    if VOL_CACHE.exists():
        return json.loads(VOL_CACHE.read_text())
    start_ms, end_ms = to_ms(EXANTE_START), to_ms(EXANTE_END)
    vol = {}
    for i, sym in enumerate(symbols):
        binance_sym = sym.upper() + "USDT"
        cur, q = start_ms, 0.0
        n = 0
        try:
            while cur < end_ms:
                kl = _get("/api/v3/klines", {"symbol": binance_sym, "interval": "15m",
                                             "startTime": cur, "endTime": end_ms, "limit": 1000})
                if not kl:
                    break
                for k in kl:
                    q += float(k[7])        # field 7 = quote-asset volume
                    n += 1
                cur = kl[-1][0] + 1
                time.sleep(0.04)
                if len(kl) < 1000:
                    break
        except Exception as e:
            print(f"  {sym}: fetch failed ({e})")
            continue
        if n > 0:
            vol[sym] = q
        if (i + 1) % 40 == 0:
            print(f"  fetched {i+1}/{len(symbols)} ex-ante volumes")
    VOL_CACHE.write_text(json.dumps(vol, indent=2))
    return vol


def load_oos(rows):
    """Load per-asset OOS dumps + slot LUTs over the shared grid (as in dependence.py)."""
    data = {}
    for r in rows:
        f = OOS / f"{r['asset']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        data[r["asset"]] = {"cls": r["class"], "slot": z["ts"] // SLOT,
                            "y": z["actual"].astype(np.int8), "p": z["p_ising"]}
    lo = min(int(d["slot"].min()) for d in data.values())
    hi = max(int(d["slot"].max()) for d in data.values())
    T = hi - lo + 1
    for d in data.values():
        lut = np.full(T, -1, np.int32)
        lut[d["slot"] - lo] = np.arange(len(d["slot"]), dtype=np.int32)
        d["lut"] = lut
    return data, lo, T


def asset_auc(d, take_slots, lo):
    """Resampled (or full, take_slots=None) AUC for one asset; nan if too few obs."""
    if take_slots is None:
        return fast_auc(d["y"], d["p"])
    obs = d["lut"][take_slots - lo]
    obs = obs[obs >= 0]
    if len(obs) < MIN_OBS:
        return float("nan")
    return fast_auc(d["y"][obs], d["p"][obs])


def main():
    rows = json.loads((OUT / "wide.json").read_text())
    crypto = [r["asset"] for r in rows if r["class"] == "crypto"]
    stocks = [r["asset"] for r in rows if r["class"] == "stock"]

    print(f"=== fetching ex-ante (Jan-2025) volume for {len(crypto)} crypto pairs ===")
    vol = fetch_exante_volume(crypto)
    exante_existing = [a for a in crypto if a in vol]
    late_listed = [a for a in crypto if a not in vol]
    ranked = sorted(exante_existing, key=lambda a: vol[a], reverse=True)
    print(f"  {len(exante_existing)} pairs traded ex ante; {len(late_listed)} listed after "
          f"{EXANTE_START} ({', '.join(late_listed[:12])}{'...' if len(late_listed)>12 else ''})")

    data, lo, T = load_oos(rows)
    crypto = [a for a in crypto if a in data]
    stocks = [a for a in stocks if a in data]
    ranked = [a for a in ranked if a in data]

    # sub-universe membership (crypto only; stocks unchanged throughout)
    universes = {
        "full": crypto,                                  # all crypto (baseline = sec:wide)
        "exante_existing": [a for a in ranked],          # excludes late-listed pairs
        "exante_top100": ranked[:100],
        "exante_top150": ranked[:150],
    }
    # ex-ante volume quintiles (within the ex-ante-existing pairs, high->low)
    q_edges = np.linspace(0, len(ranked), 6).astype(int)
    quintiles = [ranked[q_edges[i]:q_edges[i + 1]] for i in range(5)]

    def class_means(take_slots):
        stock_aucs = [a for a in (asset_auc(data[s], take_slots, lo) for s in stocks) if np.isfinite(a)]
        crypto_auc = {a: asset_auc(data[a], take_slots, lo) for a in crypto}
        return float(np.mean(stock_aucs)) if stock_aucs else float("nan"), crypto_auc

    def mean_over(crypto_auc, names):
        v = [crypto_auc[a] for a in names if a in crypto_auc and np.isfinite(crypto_auc[a])]
        return float(np.mean(v)) if v else float("nan")

    # observed (full-sample) values
    stock_mean0, crypto_auc0 = class_means(None)
    obs = {u: mean_over(crypto_auc0, names) - stock_mean0 for u, names in universes.items()}
    obs_quint = [mean_over(crypto_auc0, q) for q in quintiles]
    spearman = _spearman([vol[a] for a in ranked], [crypto_auc0[a] for a in ranked])

    # joint dependence-preserving block bootstrap, all sub-universes in one pass
    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(T / BLOCK))
    boot = {u: [] for u in universes}
    for b in range(N_BOOT):
        starts = rng.integers(0, max(1, T - BLOCK + 1), size=nb)
        slots = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:T] + lo
        sm, cauc = class_means(slots)
        if not np.isfinite(sm):
            continue
        for u, names in universes.items():
            cm = mean_over(cauc, names)
            if np.isfinite(cm):
                boot[u].append(cm - sm)
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b+1}/{N_BOOT}")

    result = {
        "n_crypto_total": len(crypto), "n_exante_existing": len(exante_existing),
        "n_late_listed": len(late_listed), "late_listed": late_listed,
        "n_stocks": len(stocks), "stock_mean_auc": stock_mean0,
        "exante_window": [EXANTE_START, EXANTE_END], "n_boot": N_BOOT,
        "spearman_exante_vol_vs_auc": spearman,
        "universes": {}, "quintiles": [],
    }
    for u in universes:
        g = np.array(boot[u])
        result["universes"][u] = {
            "n_crypto": len(universes[u]), "obs_gap": obs[u],
            "gap_ci": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
            "p_gap_le0": float(np.mean(g <= 0)),
        }
    for i, q in enumerate(quintiles):
        result["quintiles"].append({"quintile": i + 1, "n": len(q),
                                    "mean_crypto_auc": obs_quint[i],
                                    "vol_range": [vol[q[-1]], vol[q[0]]] if q else None})

    (OUT / "exante.json").write_text(json.dumps(result, indent=2))

    print("\n=== ex-ante universe: class-mean AUC gap (joint block bootstrap) ===")
    print(f"  stock mean AUC = {stock_mean0:.4f}")
    for u in universes:
        r = result["universes"][u]
        print(f"  {u:16s} (n={r['n_crypto']:3d}): gap={r['obs_gap']:+.4f} "
              f"CI=[{r['gap_ci'][0]:+.4f},{r['gap_ci'][1]:+.4f}] p(gap<=0)={r['p_gap_le0']:.4f}")
    print("  ex-ante volume quintiles (high->low), crypto mean AUC:")
    print("   ", "  ".join(f"Q{q['quintile']}={q['mean_crypto_auc']:.4f}" for q in result["quintiles"]))
    print(f"  Spearman(ex-ante volume, per-asset AUC) = {spearman:+.3f}")
    print(f"\nWrote {OUT/'exante.json'}")


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


if __name__ == "__main__":
    main()
