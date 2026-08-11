"""Negative controls: can the pipeline manufacture AUC > 0.5 from nothing?

Three controls, each run through the IDENTICAL 15m walk-forward + moving-block
bootstrap used for the headline cross-market result (compare_markets.py), on
every focal instrument. They have complementary artifact coverage:

  * shuffle — the label series is randomly permuted while the feature series is
    untouched. All feature-label dependence is destroyed, so any apparent skill
    indicts the scoring / bootstrap machinery itself (e.g. a biased AUC
    estimator or p-value).
  * phase — an IAAFT surrogate of the return series (iterated amplitude-adjusted
    Fourier transform: preserves the power spectrum and the exact marginal
    distribution; destroys all nonlinear temporal dependence). Labels are
    recomputed from the surrogate (up = r > 0), so the deterministic
    contemporaneous link between label and return is PRESERVED: a look-ahead or
    alignment bug that leaks candle t into its own features would still show
    skill here. Genuine conditional structure should not survive.
  * reverse — the price path is reversed in time (r -> -r reversed), labels
    recomputed. Not a strict null: statistically time-reversible dependence
    survives by construction, while any directional pipeline artifact tied to
    the arrow of the loop (e.g. an off-by-one in fold construction) changes.
    Reported as a structural diagnostic alongside the two nulls.

    uv run --active python -m paper.negative_controls [--quick]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .compare_markets import (BLOCK, CRYPTO, N_BOOT, TRADITIONAL, WINDOWS,
                              block_boot_idx, load_span)
from .models import ARLogit, IsingLogit
from .walkforward import walk_forward

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

INTERVAL = "15m"
SEED = 7
IAAFT_ITER = 100


def iaaft(x: np.ndarray, rng: np.random.Generator, n_iter: int = IAAFT_ITER) -> np.ndarray:
    """Iterated amplitude-adjusted Fourier-transform surrogate (Schreiber &
    Schmitz 1996): alternate imposing the original power spectrum and the
    original marginal (by rank remapping) until both approximately hold."""
    x = np.asarray(x, float)
    n = len(x)
    target_amp = np.abs(np.fft.rfft(x))
    sorted_x = np.sort(x)
    y = rng.permutation(x)
    for _ in range(n_iter):
        spec = np.fft.rfft(y)
        phases = np.angle(spec)
        y = np.fft.irfft(target_amp * np.exp(1j * phases), n)
        ranks = np.argsort(np.argsort(y))
        y = sorted_x[ranks]
    return y


def make_control(ch: np.ndarray, ups: np.ndarray, kind: str,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if kind == "shuffle":
        return ch, rng.permutation(ups)
    if kind == "reverse":
        ch_rev = -ch[::-1].copy()
        return ch_rev, ch_rev > 0
    if kind == "phase":
        ch_sur = iaaft(ch, rng)
        return ch_sur, ch_sur > 0
    if kind == "signrand":
        # Hold the |r| path fixed (volatility clustering and the diurnal profile
        # survive EXACTLY) and randomize only the signs, iid with the empirical
        # up-probability. Destroys precisely the sign dependence under test;
        # flat bars stay flat. The clean counterpart of the IAAFT surrogate,
        # which destroys volatility clustering and is anti-conservative on
        # session-heteroskedastic series.
        nz = ch != 0.0
        p_up = float(np.mean(ch[nz] > 0)) if nz.any() else 0.5
        signs = np.where(rng.random(len(ch)) < p_up, 1.0, -1.0)
        ch_sur = np.abs(ch) * signs
        return ch_sur, ch_sur > 0
    raise ValueError(kind)


def evaluate_control(asset: str, kind: str, rng: np.random.Generator,
                     n_boot: int = N_BOOT) -> dict | None:
    dts, ch, ups = load_span(asset, INTERVAL)
    tr, te = WINDOWS[INTERVAL]
    if len(ch) < tr + te:
        return None
    ch_c, ups_c = make_control(ch, ups, kind, rng)

    factory = lambda: [IsingLogit(n_lags=12), ARLogit("free", n_lags=12)]
    res, n_folds = walk_forward(ch_c, ups_c, tr, te, n_lags=12, models=factory)
    y = res["ising"]["actuals"].astype(int)
    if len(y) == 0 or y.min() == y.max():
        return None
    p_is, p_fr = res["ising"]["probs"], res["free"]["probs"]
    auc_is = roc_auc_score(y, p_is)
    auc_fr = roc_auc_score(y, p_fr)

    boot_is, boot_fr = [], []
    brng = np.random.default_rng(SEED)
    n = len(y)
    for _ in range(n_boot):
        idx = block_boot_idx(n, BLOCK[INTERVAL], brng)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        boot_is.append(roc_auc_score(yb, p_is[idx]))
        boot_fr.append(roc_auc_score(yb, p_fr[idx]))
    boot_is, boot_fr = np.array(boot_is), np.array(boot_fr)
    p_is_gt05 = float(np.mean(boot_is <= 0.5))
    p_fr_gt05 = float(np.mean(boot_fr <= 0.5))
    A = float(np.mean([p["A"] for p in res["ising"]["fold_params"]]))

    return {
        "asset": asset, "control": kind, "n_oos": int(n), "n_folds": n_folds,
        "base_rate_up": float(y.mean()),
        "ising_auc": float(auc_is),
        "ising_auc_ci": [float(np.percentile(boot_is, 2.5)),
                         float(np.percentile(boot_is, 97.5))],
        "ising_auc_p_gt05": p_is_gt05,
        "free_auc": float(auc_fr),
        "free_auc_p_gt05": p_fr_gt05,
        "conj_p": max(p_is_gt05, p_fr_gt05),
        "A": A,
    }


def main():
    quick = "--quick" in sys.argv
    assets = ["btc"] if quick else CRYPTO + TRADITIONAL
    n_boot = 100 if quick else N_BOOT
    controls = ["shuffle", "phase", "reverse"]
    rng = np.random.default_rng(SEED)

    rows = []
    for kind in controls:
        for asset in assets:
            t0 = time.time()
            r = evaluate_control(asset, kind, rng, n_boot=n_boot)
            if r is None:
                print(f"  {kind:8s} {asset:7s} skipped", flush=True)
                continue
            rows.append(r)
            sig = "*" if r["conj_p"] < 0.05 else " "
            print(f"  {kind:8s} {asset:7s} n={r['n_oos']:6d}  "
                  f"AUC_ising={r['ising_auc']:.4f} [{r['ising_auc_ci'][0]:.3f},"
                  f"{r['ising_auc_ci'][1]:.3f}] p={r['ising_auc_p_gt05']:.3f}  "
                  f"AUC_free={r['free_auc']:.4f} p={r['free_auc_p_gt05']:.3f}  "
                  f"conj={r['conj_p']:.3f}{sig}  A={r['A']:+.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    summary = {}
    for kind in controls:
        sel = [r for r in rows if r["control"] == kind]
        if not sel:
            continue
        crypto_sel = [r for r in sel if r["asset"] in CRYPTO]
        summary[kind] = {
            "n_assets": len(sel),
            "mean_ising_auc": float(np.mean([r["ising_auc"] for r in sel])),
            "mean_free_auc": float(np.mean([r["free_auc"] for r in sel])),
            "min_max_ising_auc": [float(min(r["ising_auc"] for r in sel)),
                                  float(max(r["ising_auc"] for r in sel))],
            "n_two_model_sig": sum(1 for r in sel if r["conj_p"] < 0.05),
            "crypto_mean_ising_auc": (float(np.mean([r["ising_auc"] for r in crypto_sel]))
                                      if crypto_sel else None),
            "crypto_two_model_sig": [r["asset"] for r in crypto_sel if r["conj_p"] < 0.05],
        }

    out = {
        "config": {"interval": INTERVAL, "windows": WINDOWS[INTERVAL],
                   "block": BLOCK[INTERVAL], "n_boot": n_boot, "seed": SEED,
                   "iaaft_iter": IAAFT_ITER, "quick": quick,
                   "models": ["ising", "free"], "n_lags": 12},
        "rows": rows,
        "summary": summary,
    }
    path = OUT / ("negative_controls_quick.json" if quick else "negative_controls.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {path}")
    for kind, s in summary.items():
        print(f"  {kind:8s}: mean AUC ising={s['mean_ising_auc']:.4f} "
              f"free={s['mean_free_auc']:.4f}  two-model sig {s['n_two_model_sig']}"
              f"/{s['n_assets']}")


if __name__ == "__main__":
    main()
