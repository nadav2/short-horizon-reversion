"""The N = 1 baseline: does the twelve-lag kernel beat betting against the
last candle?

Roughly two thirds of the kernel-weighted signal sits at lag 1 (fig:signlag),
so the natural null for the constrained logit is its own one-lag special case
P(up_t) = sigmoid(C + A s_{t-1}) -- the probabilistic form of "predict the
opposite of the previous candle's sign" (for A < 0). This module reruns the
identical wide-universe walk-forward with the constrained model at
N in {1, 2, 3, 6, 12} and reports, per class, mean out-of-sample AUC,
log-loss, and Brier, so the kernel's marginal value over the one-lag rule is
stated rather than left for a referee to compute.

Inference on the increment: for every asset the N=12 and N=1 OOS series live
on the same candles, so the class-mean paired difference
mean_i[AUC_i(12) - AUC_i(1)] is bootstrapped JOINTLY on the shared 15m time
grid (one set of moving blocks per replicate, applied to every asset and both
lag orders at once), exactly as in paper.dependence -- cross-sectional
dependence and the pairing are both preserved.

    uv run --active python -m paper.nbaseline

Writes out/nbaseline.json (tab:nbaseline / app:kernel).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .fetch_bulk_stocks import UNIVERSE
from .models import ALPHA_GRID, IsingLogit
from .walkforward import walk_forward
from .wide import TRAIN, TEST, load

BULK = Path(__file__).resolve().parent / "bulk_data"
OUT = Path(__file__).resolve().parent / "out"
INTERVAL = "15m"
LAG_NS = (1, 2, 3, 6, 12)
SLOT = 900                      # 15m in seconds
BLOCK, N_BOOT, SEED = 384, 1000, 11
STOCK_SET = {s.lower().replace(".", "-") for s in UNIVERSE}


def evaluate(path):
    """Constrained-logit walk-forward at every N in LAG_NS on one asset.

    Returns (per_n_metrics, oos) where oos maps N -> (slots, y, p) aligned
    series for the joint bootstrap. alpha is irrelevant at N=1 (k^-alpha == 1),
    so the grid collapses to a point there; other N use the standard grid.
    """
    ch, ups, ts = load(path)
    if len(ch) < TRAIN + TEST + 2000:
        return None, None
    per_n, oos = {}, {}
    for n_lags in LAG_NS:
        grid = (1.0,) if n_lags == 1 else None
        def factory(n=n_lags, g=grid):
            return [IsingLogit(n_lags=n, alpha_grid=g) if g else IsingLogit(n_lags=n)]
        res, _ = walk_forward(ch, ups, TRAIN, TEST, n_lags=n_lags, models=factory)
        y = res["ising"]["actuals"].astype(int)
        if len(y) < 2000 or len(np.unique(y)) < 2:
            return None, None
        p = np.asarray(res["ising"]["probs"], float)
        pc = np.clip(p, 1e-15, 1 - 1e-15)
        alphas = np.array([fp["alpha"] for fp in res["ising"]["fold_params"]], float)
        per_n[n_lags] = {
            "auc": float(roc_auc_score(y, p)),
            "log_loss": float(log_loss(y, pc, labels=[0, 1])),
            "brier": float(brier_score_loss(y, pc)),
            "A": float(np.mean([fp["A"] for fp in res["ising"]["fold_params"]])),
            # alpha diagnostics: is the decay exponent identified at this N?
            # A short window cannot separate k^-alpha from a flat kernel, so a
            # weakly identified alpha shows up as fold-to-fold instability and
            # as selections pinned to the ends of the search grid.
            "alpha": float(np.mean(alphas)),
            "alpha_fold_sd": float(np.std(alphas, ddof=1)) if len(alphas) > 1 else float("nan"),
            "alpha_edge_frac": float(np.mean((alphas <= ALPHA_GRID[0] + 1e-9)
                                             | (alphas >= ALPHA_GRID[-1] - 1e-9))),
            "n_oos": int(len(y)),
        }
        oos[n_lags] = (ts[res["ising"]["idx"].astype(int)] // SLOT,
                       y.astype(np.int8), p.astype(np.float32))
    return per_n, oos


def fast_auc(y, p):
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def joint_paired_gain(crypto_oos, n_hi=12, n_lo=1):
    """Joint moving-block bootstrap (shared 15m grid) of the crypto class-mean
    paired AUC difference AUC(N=n_hi) - AUC(N=n_lo)."""
    lo = min(int(s.min()) for oos in crypto_oos for s in (oos[n_hi][0],))
    hi = max(int(s.max()) for oos in crypto_oos for s in (oos[n_hi][0],))
    T = hi - lo + 1
    # dense per-asset series on the shared grid (N=1 and N=12 share candles)
    dense = []
    for oos in crypto_oos:
        s12, y12, p12 = oos[n_hi]
        s1, y1, p1 = oos[n_lo]
        row_y = np.full(T, -1, np.int8)
        row_p12 = np.full(T, np.nan, np.float32)
        row_p1 = np.full(T, np.nan, np.float32)
        row_y[s12 - lo] = y12
        row_p12[s12 - lo] = p12
        row_p1[s1 - lo] = p1
        dense.append((row_y, row_p12, row_p1))

    def class_gain(sel):
        gains = []
        for row_y, row_p12, row_p1 in dense:
            y = row_y[sel]
            ok = (y >= 0) & ~np.isnan(row_p12[sel]) & ~np.isnan(row_p1[sel])
            if ok.sum() < 500 or len(np.unique(y[ok])) < 2:
                continue
            gains.append(fast_auc(y[ok], row_p12[sel][ok])
                         - fast_auc(y[ok], row_p1[sel][ok]))
        return float(np.mean(gains)) if gains else float("nan")

    obs = class_gain(np.arange(T))
    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(T / BLOCK))
    boots = []
    for _ in range(N_BOOT):
        starts = rng.integers(0, max(1, T - BLOCK + 1), size=nb)
        sel = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:T]
        g = class_gain(sel)
        if g == g:
            boots.append(g)
    boots = np.array(boots)
    return {"obs_gain": obs,
            "ci": [float(np.percentile(boots, 2.5)),
                   float(np.percentile(boots, 97.5))],
            "p_le0": float(np.mean(boots <= 0.0)),
            "n_boot": int(len(boots))}


def main():
    files = sorted(BULK.glob(f"*-{INTERVAL}.json"))
    print(f"{len(files)} assets in {BULK}")
    rows, crypto_oos = [], []
    for i, f in enumerate(files):
        base = f.name[:-(len(INTERVAL) + 6)]
        cls = "stock" if base in STOCK_SET else "crypto"
        try:
            per_n, oos = evaluate(f)
        except Exception as e:
            print(f"  {base}: FAILED {e}")
            continue
        if per_n is None:
            continue
        rows.append({"asset": base, "class": cls,
                     "per_n": {str(n): m for n, m in per_n.items()}})
        if cls == "crypto":
            crypto_oos.append(oos)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(files)}] processed; kept {len(rows)}", flush=True)

    summary = {}
    for cls in ("crypto", "stock"):
        sub = [r for r in rows if r["class"] == cls]
        summary[cls] = {"n_assets": len(sub)}
        for n in LAG_NS:
            ms = [r["per_n"][str(n)] for r in sub]
            summary[cls][str(n)] = {
                "mean_auc": float(np.mean([m["auc"] for m in ms])),
                "mean_log_loss": float(np.mean([m["log_loss"] for m in ms])),
                "mean_brier": float(np.mean([m["brier"] for m in ms])),
                "frac_A_neg": float(np.mean([m["A"] < 0 for m in ms])),
            }
            if n > 1:      # alpha is not a free parameter at N=1 (k^-alpha == 1)
                al = np.array([m["alpha"] for m in ms], float)
                summary[cls][str(n)].update({
                    "alpha_median": float(np.median(al)),
                    "alpha_iqr": [float(np.percentile(al, 25)),
                                  float(np.percentile(al, 75))],
                    "alpha_cross_asset_sd": float(np.std(al, ddof=1)),
                    "mean_alpha_fold_sd": float(np.nanmean([m["alpha_fold_sd"] for m in ms])),
                    "mean_alpha_edge_frac": float(np.mean([m["alpha_edge_frac"] for m in ms])),
                })
        # per-asset paired gains vs N=1, descriptive
        for n in LAG_NS[1:]:
            d = [r["per_n"][str(n)]["auc"] - r["per_n"]["1"]["auc"] for r in sub]
            summary[cls][f"gain_auc_{n}_vs_1"] = {
                "mean": float(np.mean(d)),
                "frac_pos": float(np.mean(np.array(d) > 0)),
            }

    print("joint paired bootstrap of the crypto N=12 vs N=1 AUC gain ...", flush=True)
    joint = joint_paired_gain(crypto_oos)

    out = {"config": {"interval": INTERVAL, "train": TRAIN, "test": TEST,
                      "lag_ns": list(LAG_NS), "block": BLOCK,
                      "n_boot": N_BOOT, "seed": SEED},
           "summary": summary, "joint_gain_12_vs_1": joint, "assets": rows}
    (OUT / "nbaseline.json").write_text(json.dumps(out, indent=2))

    print("\n=== N-LAG BASELINE (15m, constrained logit) ===")
    for cls in ("crypto", "stock"):
        print(f"\n{cls.upper()} (n={summary[cls]['n_assets']})")
        for n in LAG_NS:
            s = summary[cls][str(n)]
            extra = ""
            if n > 1:
                extra = (f"  alpha med {s['alpha_median']:.2f} "
                         f"IQR [{s['alpha_iqr'][0]:.2f},{s['alpha_iqr'][1]:.2f}] "
                         f"fold-SD {s['mean_alpha_fold_sd']:.2f} "
                         f"grid-edge {s['mean_alpha_edge_frac']*100:.0f}%")
            print(f"  N={n:>2}: AUC {s['mean_auc']:.4f}  logloss {s['mean_log_loss']:.5f}  "
                  f"brier {s['mean_brier']:.5f}{extra}")
    g = joint
    print(f"\ncrypto paired AUC gain N=12 vs N=1: {g['obs_gain']:+.4f} "
          f"CI [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}]  p(<=0)={g['p_le0']:.3f}")
    print(f"\nWrote {OUT/'nbaseline.json'}")


if __name__ == "__main__":
    main()
