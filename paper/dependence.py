"""Cross-sectional dependence in the wide universe, and a dependence-aware
population test.

Crypto pairs co-move strongly (alt-coins track BTC), so the per-asset
significance counts of the wide study are not 183 independent confirmations.
This script quantifies the dependence and provides the inference that respects it:

  (a) the full within-class pairwise correlation matrix of 15m returns and the
      implied effective number of independent assets,
      N_eff = N / (1 + (N-1) * rho_bar)  (equicorrelation approximation);

  (b) a JOINT moving-block bootstrap of the class-mean AUC gap: time blocks are
      resampled once on the shared 15m grid and applied to every asset
      simultaneously, so each resample preserves the full cross-sectional
      dependence structure. The resulting CI / p-value for
      gap = mean AUC(crypto) - mean AUC(stocks) is the honest population-level
      statement. Requires the per-asset OOS dumps from `wide.py --dump-oos`.

    uv run --active python -m paper.dependence
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BULK = Path(__file__).resolve().parent / "bulk_data"
OUT = Path(__file__).resolve().parent / "out"
OOS = OUT / "wide_oos"
SLOT = 900                      # 15m in seconds
BLOCK = 384                     # ~4 days of 15m slots, as elsewhere
N_BOOT, SEED = 1000, 7
MIN_OBS = 1000


def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC via ranks (ties broken arbitrarily; p is continuous)."""
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ── (a) pairwise return correlations ─────────────────────────────────────────

def pairwise_mean_corr(assets: list[str]) -> dict:
    """Mean/median pairwise correlation of 15m returns within a class, on the
    pair-specific common support (masked matrix algebra, all pairs)."""
    slots_all = {}
    for a in assets:
        raw = json.loads((BULK / f"{a}-15m.json").read_text())
        slots_all[a] = {d["timestamp"] // SLOT: d.get("change", 0.0) for d in raw}
    lo = min(min(s) for s in slots_all.values())
    hi = max(max(s) for s in slots_all.values())
    T = hi - lo + 1
    X = np.zeros((len(assets), T), np.float32)
    M = np.zeros((len(assets), T), np.float32)
    for i, a in enumerate(assets):
        for sl, ch in slots_all[a].items():
            X[i, sl - lo] = ch
            M[i, sl - lo] = 1.0
    n = M @ M.T
    Sx = X @ M.T
    Sxx = (X * X) @ M.T
    Sxy = X @ X.T
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = Sxy - Sx * Sx.T / n
        var_i = Sxx - Sx ** 2 / n
        corr = cov / np.sqrt(var_i * var_i.T)
    iu = np.triu_indices(len(assets), k=1)
    vals = corr[iu]
    ok = np.isfinite(vals) & (n[iu] > 5000)
    vals = vals[ok]
    N = len(assets)
    rho = float(np.mean(vals))
    return {"n_assets": N, "n_pairs": int(len(vals)),
            "mean_corr": rho, "median_corr": float(np.median(vals)),
            "n_eff": float(N / (1 + (N - 1) * rho))}


# ── (b) joint moving-block bootstrap of the class-mean AUC gap ───────────────

def joint_gap_bootstrap(rows: list[dict]) -> dict:
    data = {}
    for r in rows:
        f = OOS / f"{r['asset']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        data[r["asset"]] = {"cls": r["class"], "slot": z["ts"] // SLOT,
                            "y": z["actual"].astype(np.int8),
                            "p_is": z["p_ising"], "p_fr": z["p_free"]}
    lo = min(int(d["slot"].min()) for d in data.values())
    hi = max(int(d["slot"].max()) for d in data.values())
    T = hi - lo + 1
    # slot -> observation-index lookup per asset (-1 = no observation)
    for d in data.values():
        lut = np.full(T, -1, np.int32)
        lut[d["slot"] - lo] = np.arange(len(d["slot"]), dtype=np.int32)
        d["lut"] = lut

    def class_means(take_slots: np.ndarray | None):
        means = {}
        for model in ("p_is", "p_fr"):
            per_cls = {"crypto": [], "stock": []}
            for d in data.values():
                if take_slots is None:
                    y, p = d["y"], d[model]
                else:
                    obs = d["lut"][take_slots]
                    obs = obs[obs >= 0]
                    if len(obs) < MIN_OBS:
                        continue
                    y, p = d["y"][obs], d[model][obs]
                a = fast_auc(y, p)
                if np.isfinite(a):
                    per_cls[d["cls"]].append(a)
            means[model] = {c: float(np.mean(v)) for c, v in per_cls.items() if v}
        return means

    obs_means = class_means(None)
    obs_gap = {m: obs_means[m]["crypto"] - obs_means[m]["stock"] for m in obs_means}

    rng = np.random.default_rng(SEED)
    gaps = {"p_is": [], "p_fr": []}
    nb = int(np.ceil(T / BLOCK))
    for b in range(N_BOOT):
        starts = rng.integers(0, max(1, T - BLOCK + 1), size=nb)
        slots = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:T]
        m = class_means(slots)
        for model in gaps:
            if "crypto" in m[model] and "stock" in m[model]:
                gaps[model].append(m[model]["crypto"] - m[model]["stock"])
        if (b + 1) % 50 == 0:
            print(f"  bootstrap {b+1}/{N_BOOT}")

    out = {"n_assets_used": len(data), "n_boot": N_BOOT, "block_slots": BLOCK}
    for model, label in (("p_is", "ising"), ("p_fr", "free")):
        g = np.array(gaps[model])
        out[label] = {
            "obs_gap": obs_gap[model],
            "gap_ci": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
            "p_gap_le0": float(np.mean(g <= 0)),
            "crypto_mean_auc": obs_means[model]["crypto"],
            "stock_mean_auc": obs_means[model]["stock"],
            "gap_samples": [float(x) for x in g],
        }
    return out


def main():
    rows = json.loads((OUT / "wide.json").read_text())
    res = {}
    print("=== pairwise 15m return correlations (full within-class matrices) ===")
    for cls in ("crypto", "stock"):
        assets = [r["asset"] for r in rows if r["class"] == cls
                  and (BULK / f"{r['asset']}-15m.json").exists()]
        res[f"corr_{cls}"] = pairwise_mean_corr(assets)
        c = res[f"corr_{cls}"]
        print(f"  {cls:6s} N={c['n_assets']}  mean rho={c['mean_corr']:.3f} "
              f"median={c['median_corr']:.3f}  ->  N_eff={c['n_eff']:.1f}")

    print("\n=== joint (dependence-preserving) block bootstrap of the AUC gap ===")
    res["joint_gap"] = joint_gap_bootstrap(rows)
    for label in ("ising", "free"):
        g = res["joint_gap"][label]
        print(f"  {label:5s}: gap={g['obs_gap']:+.4f}  CI=[{g['gap_ci'][0]:+.4f}, "
              f"{g['gap_ci'][1]:+.4f}]  p(gap<=0)={g['p_gap_le0']:.4f} "
              f"(crypto {g['crypto_mean_auc']:.4f} vs stocks {g['stock_mean_auc']:.4f})")

    (OUT / "dependence.json").write_text(json.dumps(res, indent=2))
    print(f"\nWrote {OUT/'dependence.json'}")


if __name__ == "__main__":
    main()
