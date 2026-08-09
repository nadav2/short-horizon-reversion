"""Detrended fluctuation analysis (DFA-1) across the full wide universe.

Bridges the conditional Ising measurement to the unconditional scaling toolkit of
the econophysics literature: per asset we estimate the Hurst exponent H of the
15m signed-return series (DFA-1, scales 8-512 candles, log-spaced). H < 0.5 is
antipersistence -- the model-free counterpart of an antiferromagnetic coupling.
Results are merged with the wide-study Ising AUC and coupling A so the paper can
report both the class-level contrast in H and the cross-asset co-variation of
(H, A, AUC).

    uv run --active python -m paper.dfa
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BULK = Path(__file__).resolve().parent / "bulk_data"
OUT = Path(__file__).resolve().parent / "out"
SCALES = np.unique(np.logspace(np.log10(8), np.log10(512), 12).astype(int))
MIN_CANDLES = 8000

# Joint (dependence-preserving) bootstrap of the class-mean H gap, mirroring the
# AUC-gap treatment in dependence.py. Block-resampling destroys correlation
# structure at lags beyond the block length, so the bootstrap re-estimates H on
# scales capped at BLOCK/2 -- the observed gap is recomputed on the same capped
# scale set so the point estimate and the CI refer to the same statistic.
SLOT = 900
BLOCK = 384
BOOT_SCALES = np.unique(np.logspace(np.log10(8), np.log10(BLOCK // 2), 10).astype(int))
N_BOOT, SEED = 1000, 7
MIN_OBS = 4000


def dfa1(x: np.ndarray, scales=SCALES) -> float:
    """DFA-1 scaling exponent of series x (linear detrending per window)."""
    x = np.asarray(x, float)
    y = np.cumsum(x - x.mean())
    pts = []
    for s in scales:
        n = len(y) // s
        if n < 8:
            continue
        seg = y[: n * s].reshape(n, s)
        t = np.arange(s)
        tm = t - t.mean()
        beta = (seg * tm).sum(axis=1) / (tm ** 2).sum()
        resid = seg - (seg.mean(axis=1)[:, None] + beta[:, None] * tm[None, :])
        pts.append((np.log(s), np.log(np.sqrt((resid ** 2).mean()))))
    a = np.array(pts)
    return float(np.polyfit(a[:, 0], a[:, 1], 1)[0])


def joint_H_bootstrap(assets: dict[str, str]) -> dict:
    """Joint moving-block bootstrap of the class-mean Hurst gap.

    Time blocks are drawn ONCE on the shared 15m grid and applied to every asset
    simultaneously, so each resample preserves the cross-sectional dependence that
    makes per-asset tallies non-independent. This is the model-free counterpart of
    the AUC-gap bootstrap: it uses no fitted model at all."""
    data = {}
    for a, cls in assets.items():
        raw = sorted(json.loads((BULK / f"{a}-15m.json").read_text()),
                     key=lambda d: d["timestamp"])
        ch = np.array([d.get("change", 0.0) for d in raw], float)
        if len(ch) < MIN_CANDLES:
            continue
        data[a] = {"cls": cls, "slot": np.array([d["timestamp"] // SLOT for d in raw], np.int64),
                   "ch": ch}

    lo = min(int(d["slot"].min()) for d in data.values())
    hi = max(int(d["slot"].max()) for d in data.values())
    T = hi - lo + 1
    for d in data.values():
        lut = np.full(T, -1, np.int32)
        lut[d["slot"] - lo] = np.arange(len(d["ch"]), dtype=np.int32)
        d["lut"] = lut

    def class_means(take_slots):
        per = {"crypto": [], "stock": []}
        for d in data.values():
            if take_slots is None:
                x = d["ch"]
            else:
                obs = d["lut"][take_slots]
                obs = obs[obs >= 0]
                if len(obs) < MIN_OBS:
                    continue
                x = d["ch"][obs]
            per[d["cls"]].append(dfa1(x, BOOT_SCALES))
        return {c: float(np.mean(v)) for c, v in per.items() if v}

    obs = class_means(None)
    obs_gap = obs["crypto"] - obs["stock"]

    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(T / BLOCK))
    gaps = []
    for b in range(N_BOOT):
        starts = rng.integers(0, max(1, T - BLOCK + 1), size=nb)
        slots = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:T]
        m = class_means(slots)
        if "crypto" in m and "stock" in m:
            gaps.append(m["crypto"] - m["stock"])
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b+1}/{N_BOOT}", flush=True)
    g = np.array(gaps)
    return {
        "n_assets_used": len(data), "n_boot": N_BOOT, "block_slots": BLOCK,
        "boot_scales": [int(s) for s in BOOT_SCALES],
        "crypto_mean_H": obs["crypto"], "stock_mean_H": obs["stock"],
        "obs_gap": obs_gap,
        "gap_ci": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
        "p_gap_ge0": float(np.mean(g >= 0)),
    }


def main():
    wide = {r["asset"]: r for r in json.loads((OUT / "wide.json").read_text())}
    rows = []
    files = sorted(BULK.glob("*-15m.json"))
    for i, f in enumerate(files):
        asset = f.name[: -len("-15m.json")]
        raw = json.loads(f.read_text())
        ch = np.array([d.get("change", 0.0) for d in raw], float)
        if len(ch) < MIN_CANDLES:
            continue
        w = wide.get(asset)
        rows.append({
            "asset": asset,
            "class": w["class"] if w else None,
            "H": dfa1(ch),
            "n": int(len(ch)),
            "A": w["A"] if w else None,
            "auc_ising": w["auc_ising"] if w else None,
        })
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(files)}]")

    summary = {}
    matched = [r for r in rows if r["class"] is not None]
    for c in ("crypto", "stock"):
        hs = np.array([r["H"] for r in matched if r["class"] == c])
        summary[c] = {"n": int(len(hs)), "mean_H": float(hs.mean()),
                      "median_H": float(np.median(hs)),
                      "frac_H_below_05": float(np.mean(hs < 0.5))}
    H = np.array([r["H"] for r in matched])
    A = np.array([r["A"] for r in matched])
    auc = np.array([r["auc_ising"] for r in matched])
    is_c = np.array([r["class"] == "crypto" for r in matched])
    summary["corr_H_A"] = float(np.corrcoef(H, A)[0, 1])
    summary["corr_H_auc"] = float(np.corrcoef(H, auc)[0, 1])
    summary["within_crypto_corr_H_A"] = float(np.corrcoef(H[is_c], A[is_c])[0, 1])
    summary["within_crypto_corr_H_auc"] = float(np.corrcoef(H[is_c], auc[is_c])[0, 1])

    print("\n=== joint (dependence-preserving) block bootstrap of the mean-H gap ===")
    summary["joint_gap"] = joint_H_bootstrap(
        {r["asset"]: r["class"] for r in matched})

    (OUT / "dfa.json").write_text(json.dumps({"summary": summary, "assets": rows}, indent=2))
    print("\n=== DFA-1 Hurst exponents (15m returns) ===")
    for c in ("crypto", "stock"):
        s = summary[c]
        print(f"  {c:6s} n={s['n']:4d}  mean H={s['mean_H']:.4f}  median={s['median_H']:.4f}  "
              f"frac H<0.5 = {s['frac_H_below_05']*100:.0f}%")
    print(f"  corr(H, A)   = {summary['corr_H_A']:+.3f}   (within crypto {summary['within_crypto_corr_H_A']:+.3f})")
    print(f"  corr(H, AUC) = {summary['corr_H_auc']:+.3f}   (within crypto {summary['within_crypto_corr_H_auc']:+.3f})")
    jg = summary["joint_gap"]
    print(f"  mean-H gap   = {jg['obs_gap']:+.4f}  CI=[{jg['gap_ci'][0]:+.4f}, {jg['gap_ci'][1]:+.4f}]  "
          f"p(gap>=0)={jg['p_gap_ge0']:.4f}   (crypto {jg['crypto_mean_H']:.4f} vs stocks {jg['stock_mean_H']:.4f})")
    print(f"Wrote {OUT/'dfa.json'}")


if __name__ == "__main__":
    main()
