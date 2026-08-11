"""Joint moving-block bootstrap of the class-mean AUC gap, all bars and
non-flat bars only, at B=5,000.

The primary gap (`paper.dependence`) is bootstrapped at B=1,000 on all bars.
Section 3 designates the flat-bar-excluded rescoring as the paper's preferred
artifact-robust effect size, so that estimate needs an interval from the same
dependence-preserving procedure. This module supplies both:

  allbars  -- the primary gap re-bootstrapped at B=5,000 (a convergence check
              on `paper.dependence`; the point estimates are identical by
              construction, only the percentile interval is re-drawn);
  nonflat  -- the same statistic with flat bars (change == 0) dropped from every
              asset's out-of-sample series before each AUC is computed.

Identical geometry to `paper.dependence` throughout: BLOCK=384 slots, SEED=7,
MIN_OBS=1000, percentile CI, one set of blocks per replicate applied to every
asset on the shared 15m grid. The AUC kernel is algebraically the same
Mann-Whitney statistic, evaluated from a per-asset pre-sorted order so that a
bootstrap subset costs O(n) instead of a fresh sort; it reproduces
`paper.dependence.fast_auc` exactly.

Requires the per-asset OOS dumps from `wide.py --dump-oos`.

    uv run --active python -m paper.nonflat_gap
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BULK = Path(__file__).resolve().parent / "bulk_data"
OUT = Path(__file__).resolve().parent / "out"
OOS = OUT / "wide_oos"
SLOT = 900
BLOCK = 384
SEED = 7
MIN_OBS = 1000
N_BOOT = 5000


def load() -> dict:
    rows = json.loads((OUT / "wide.json").read_text())
    data = {}
    for r in rows:
        f = OOS / f"{r['asset']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        slot = (z["ts"] // SLOT).astype(np.int64)
        y = z["actual"].astype(np.int8)
        raw = json.loads((BULK / f"{r['asset']}-15m.json").read_text())
        ch = {d["timestamp"] // SLOT: d.get("change", 0.0) for d in raw}
        d = {"cls": r["class"], "slot": slot,
             "nonflat": np.array([ch.get(int(s), 0.0) != 0.0 for s in slot], bool)}
        for model, key in (("p_is", "p_ising"), ("p_fr", "p_free")):
            d[model] = {"order": np.argsort(z[key], kind="stable")}
        data[r["asset"]] = d
    return data


def index(data: dict) -> int:
    lo = min(int(d["slot"].min()) for d in data.values())
    hi = max(int(d["slot"].max()) for d in data.values())
    T = hi - lo + 1
    for d in data.values():
        lut = np.full(T, -1, np.int32)
        lut[d["slot"] - lo] = np.arange(len(d["slot"]), dtype=np.int32)
        d["lut"] = lut
        for model in ("p_is", "p_fr"):
            order = d[model]["order"]
            inv = np.empty(len(order), np.int32)
            inv[order] = np.arange(len(order), dtype=np.int32)
            d[model]["inv"] = inv
            d[model]["nonflat_sorted"] = d["nonflat"][order]
    return T


def auc_sorted(ys: np.ndarray, mask: np.ndarray) -> float:
    """Mann-Whitney AUC over the masked subset of a pre-sorted score order."""
    yy = ys[mask]
    n1 = int(yy.sum())
    n0 = len(yy) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    neg_cum = np.cumsum(1 - yy.astype(np.int64))
    return float(neg_cum[yy == 1].sum() / (n0 * n1))


def class_means(data, T, take_slots, nonflat_only):
    means = {}
    for model in ("p_is", "p_fr"):
        per = {"crypto": [], "stock": []}
        for d in data.values():
            n = len(d["slot"])
            if take_slots is None:
                mask = np.ones(n, bool)
            else:
                obs = d["lut"][take_slots]
                obs = obs[obs >= 0]
                mask = np.zeros(n, bool)
                mask[d[model]["inv"][obs]] = True
            if nonflat_only:
                mask &= d[model]["nonflat_sorted"]
            if take_slots is not None and int(mask.sum()) < MIN_OBS:
                continue
            a = auc_sorted(d[model]["ys"], mask)
            if np.isfinite(a):
                per[d["cls"]].append(a)
        means[model] = {c: float(np.mean(v)) for c, v in per.items() if v}
    return means


def run(data, T, nonflat_only, label):
    obs = class_means(data, T, None, nonflat_only)
    obs_gap = {m: obs[m]["crypto"] - obs[m]["stock"] for m in obs}
    rng = np.random.default_rng(SEED)
    gaps = {"p_is": [], "p_fr": []}
    nb = int(np.ceil(T / BLOCK))
    for b in range(N_BOOT):
        starts = rng.integers(0, max(1, T - BLOCK + 1), size=nb)
        slots = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:T]
        m = class_means(data, T, slots, nonflat_only)
        for model in gaps:
            if "crypto" in m[model] and "stock" in m[model]:
                gaps[model].append(m[model]["crypto"] - m[model]["stock"])
        if (b + 1) % 250 == 0:
            print(f"  [{label}] {b+1}/{N_BOOT}", flush=True)
    out = {"n_boot": N_BOOT, "block_slots": BLOCK, "n_assets_used": len(data)}
    for model, name in (("p_is", "ising"), ("p_fr", "free")):
        g = np.array(gaps[model])
        out[name] = {
            "obs_gap": obs_gap[model],
            "gap_ci": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
            "p_gap_le0": float(np.mean(g <= 0)),
            "crypto_mean_auc": obs[model]["crypto"],
            "stock_mean_auc": obs[model]["stock"],
        }
        r = out[name]
        print(f"  [{label}] {name}: gap={r['obs_gap']:+.4f} "
              f"CI=[{r['gap_ci'][0]:+.4f},{r['gap_ci'][1]:+.4f}] "
              f"p={r['p_gap_le0']:.4f} (crypto {r['crypto_mean_auc']:.4f} vs "
              f"stocks {r['stock_mean_auc']:.4f})", flush=True)
    return out


def main():
    data = load()
    print(f"loaded {len(data)} assets")
    T = index(data)
    for d in data.values():
        for model in ("p_is", "p_fr"):
            d[model]["ys"] = None
    # attach sorted labels (kept out of load() so the dict stays small there)
    rows = {r["asset"]: r for r in json.loads((OUT / "wide.json").read_text())}
    for asset, d in data.items():
        z = np.load(OOS / f"{asset}.npz")
        y = z["actual"].astype(np.int8)
        for model in ("p_is", "p_fr"):
            d[model]["ys"] = y[d[model]["order"]]
    res = {"allbars": run(data, T, False, "all"),
           "nonflat": run(data, T, True, "nonflat")}
    (OUT / "nonflat_gap.json").write_text(json.dumps(res, indent=1))
    print(f"\nWrote {OUT/'nonflat_gap.json'}")


if __name__ == "__main__":
    main()
