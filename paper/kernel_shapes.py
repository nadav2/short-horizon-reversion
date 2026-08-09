"""Kernel-shape identifiability study.

The Ising prior ties the AR-logit weights to a power-law kernel w_k = A k^-alpha.
This script asks what the data actually identify: the *decay* of the kernel, or
its specific *functional form*? Three 3-parameter families are compared under the
identical walk-forward + validation-tail protocol used everywhere else:

    power-law    w_k = k^-alpha          alpha on the standard grid (0..3)
    exponential  w_k = exp(-(k-1)/tau)   tau on a geometric grid (0.3..30)
    flat         w_k = 1                 (2 parameters: no shape to select)

For every cell we report OOS AUC / log-loss per family plus paired moving-block
bootstrap deltas (power minus exponential, power minus flat). The honest headline:
if power and exponential are indistinguishable, the power-law form is a choice of
parameterization (motivated by the statistical-mechanics lineage), not a finding.

    uv run --active python -m paper.kernel_shapes
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .common import load_merged, spins
from .compare_markets import CRYPTO, TRADITIONAL, WINDOWS, load_span
from .models import ALPHA_GRID, N_LAGS, SPIN_SCALE, VAL_FRAC, _nll, _sigmoid
from .walkforward import window_candles

OUT = Path(__file__).resolve().parent / "out"
TAU_GRID = tuple(np.round(np.geomspace(0.3, 30.0, 31), 4))
BLOCK = {"5m": 1152, "15m": 384, "1h": 96, "4h": 30}    # ~4 days, as elsewhere
N_BOOT, SEED = 300, 11

KS = np.arange(1, N_LAGS + 1)
FAMILIES = {
    "powerlaw": [("alpha", a, KS ** -float(a)) for a in ALPHA_GRID],
    "exponential": [("tau", t, np.exp(-(KS - 1) / float(t))) for t in TAU_GRID],
    "flat": [("flat", None, np.ones(N_LAGS))],
}


def kernel_field(sp: np.ndarray, w: np.ndarray) -> np.ndarray:
    """field_t = sum_{k=1..N} w_k * s_{t-k} (partial sums for t < N, as in
    common.power_law_field)."""
    n = len(sp)
    field = np.zeros(n)
    for k in range(1, len(w) + 1):
        field[k:] += w[k - 1] * sp[: n - k]
    return field


def fit_CA(field: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Unpenalized logistic MLE of (C, A) on the single field feature (Newton)."""
    w = np.zeros(2)
    X = np.column_stack([np.ones_like(field), field])
    for _ in range(50):
        p = _sigmoid(X @ w)
        g = X.T @ (p - y)
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])           # (C, A)


def run_family(fields: list[np.ndarray], y: np.ndarray, train: int, test: int):
    """Walk-forward for one kernel family. fields[i] is the full-series field of
    candidate i; the candidate is selected per fold on the validation tail (same
    protocol as models.IsingLogit). Returns (probs, actuals, params_per_fold)."""
    n = len(y)
    probs, acts, sel = [], [], []
    start = 0
    while start + train < n:                  # mirror walkforward.walk_forward
        lo, hi = start, start + train
        te_hi = min(hi + test, n)
        if te_hi <= hi:
            break
        cut = lo + int(round(train * (1 - VAL_FRAC)))
        best = (np.inf, 0)
        if len(fields) > 1:
            yf = y[lo:cut]
            if yf.min() != yf.max():
                for i, f in enumerate(fields):
                    C, A = fit_CA(f[lo:cut], yf)
                    ll = _nll(_sigmoid(C + A * f[cut:hi]), y[cut:hi])
                    if ll < best[0]:
                        best = (ll, i)
        i = best[1]
        C, A = fit_CA(fields[i][lo:hi], y[lo:hi])
        probs.append(_sigmoid(C + A * fields[i][hi:te_hi]))
        acts.append(y[hi:te_hi])
        sel.append({"i": i, "C": C, "A": A})
        start += test
    return np.concatenate(probs), np.concatenate(acts), sel


def block_boot_deltas(y, p_a, p_b, block, rng, n_boot=N_BOOT):
    """Paired moving-block bootstrap of AUC(a)-AUC(b) and logloss(b)-logloss(a)."""
    n = len(y)
    d_auc, d_ll = [], []
    eps = 1e-15

    def ll_vec(p):
        pc = np.clip(p, eps, 1 - eps)
        return -(y_b * np.log(pc) + (1 - y_b) * np.log(1 - pc))

    for _ in range(n_boot):
        nb = int(np.ceil(n / block))
        starts = rng.integers(0, max(1, n - block + 1), size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        y_b = y[idx]
        if y_b.min() == y_b.max():
            continue
        d_auc.append(roc_auc_score(y_b, p_a[idx]) - roc_auc_score(y_b, p_b[idx]))
        d_ll.append(np.mean(ll_vec(p_b[idx])) - np.mean(ll_vec(p_a[idx])))
    return np.array(d_auc), np.array(d_ll)


def evaluate_cell(cell, ch, ups, train, test, interval):
    sp = spins(ch, SPIN_SCALE)
    y = ups.astype(int)
    fields = {fam: [kernel_field(sp, w) for (_, _, w) in grid]
              for fam, grid in FAMILIES.items()}
    out = {"cell": cell, "interval": interval, "n_total": int(len(y))}
    series = {}
    for fam, grid in FAMILIES.items():
        p, a, sel = run_family(fields[fam], y, train, test)
        series[fam] = (p, a)
        pc = np.clip(p, 1e-15, 1 - 1e-15)
        out[fam] = {
            "auc": float(roc_auc_score(a, p)),
            "logloss": float(-np.mean(a * np.log(pc) + (1 - a) * np.log(1 - pc))),
            "mean_param": (float(np.mean([grid[s["i"]][1] for s in sel]))
                           if grid[0][1] is not None else None),
            "mean_A": float(np.mean([s["A"] for s in sel])),
            "n_folds": len(sel),
        }
    out["n_oos"] = int(len(series["powerlaw"][1]))
    rng = np.random.default_rng(SEED)
    y_oos = series["powerlaw"][1]
    for tag, other in (("pl_minus_exp", "exponential"), ("pl_minus_flat", "flat")):
        d_auc, d_ll = block_boot_deltas(y_oos, series["powerlaw"][0], series[other][0],
                                        BLOCK[interval], rng)
        out[tag] = {
            "d_auc_mean": float(np.mean(d_auc)),
            "d_auc_ci": [float(np.percentile(d_auc, 2.5)), float(np.percentile(d_auc, 97.5))],
            "d_ll_mean_mnats": float(np.mean(d_ll) * 1000),
            "d_ll_ci_mnats": [float(np.percentile(d_ll, 2.5) * 1000),
                              float(np.percentile(d_ll, 97.5) * 1000)],
            "p_pl_better_auc": float(np.mean(d_auc <= 0)),     # one-sided
        }
    # store fitted kernels for the figure (mean A * w_k per family)
    out["kernels"] = {}
    for fam, grid in FAMILIES.items():
        p_mean = out[fam]["mean_param"]
        if fam == "powerlaw":
            w = KS ** -(p_mean if p_mean is not None else 1.0)
        elif fam == "exponential":
            w = np.exp(-(KS - 1) / (p_mean if p_mean else 1.0))
        else:
            w = np.ones(N_LAGS)
        out["kernels"][fam] = (out[fam]["mean_A"] * w).tolist()
    return out


def main():
    OUT.mkdir(exist_ok=True)
    cells = []

    # within-crypto deep cells (same data + geometry as run.py)
    for coin in ("btc", "eth", "sol", "xrp"):
        for interval in ("5m", "15m", "1h", "4h"):
            try:
                _, ch, ups = load_merged(coin, interval)
            except FileNotFoundError:
                print(f"  {coin}-{interval}: no data, skipped")
                continue
            tr, te = window_candles(interval)
            if len(ch) < tr + te:
                print(f"  {coin}-{interval}: too short, skipped")
                continue
            r = evaluate_cell(f"{coin}-{interval}", ch, ups, tr, te, interval)
            r["study"] = "within"
            cells.append(r)
            print(f"  {r['cell']:10s} AUC pl={r['powerlaw']['auc']:.4f} "
                  f"exp={r['exponential']['auc']:.4f} flat={r['flat']['auc']:.4f}  "
                  f"dAUC(pl-exp)={r['pl_minus_exp']['d_auc_mean']:+.4f} "
                  f"CI{r['pl_minus_exp']['d_auc_ci']}")

    # focal cross-market 15m cells (matched span + geometry, as compare_markets)
    tr, te = WINDOWS["15m"]
    for asset in CRYPTO + TRADITIONAL:
        try:
            _, ch, ups = load_span(asset, "15m")
        except FileNotFoundError:
            continue
        if len(ch) < tr + te:
            continue
        r = evaluate_cell(f"{asset}-15m-focal", ch, ups, tr, te, "15m")
        r["study"] = "focal"
        cells.append(r)
        print(f"  {r['cell']:16s} AUC pl={r['powerlaw']['auc']:.4f} "
              f"exp={r['exponential']['auc']:.4f} flat={r['flat']['auc']:.4f}")

    # summary
    within = [c for c in cells if c["study"] == "within"]
    summary = {
        "n_within_cells": len(within),
        "pl_beats_exp_auc": sum(c["powerlaw"]["auc"] > c["exponential"]["auc"] for c in within),
        "pl_beats_flat_auc": sum(c["powerlaw"]["auc"] > c["flat"]["auc"] for c in within),
        "pl_exp_indistinct": sum(c["pl_minus_exp"]["d_auc_ci"][0] <= 0 <= c["pl_minus_exp"]["d_auc_ci"][1]
                                 for c in within),
        "pl_flat_sig": sum(c["pl_minus_flat"]["p_pl_better_auc"] < 0.05 for c in within),
    }
    (OUT / "kernel_shape.json").write_text(json.dumps({"summary": summary, "cells": cells}, indent=2))
    print(f"\nsummary: {summary}")
    print(f"Wrote {OUT/'kernel_shape.json'}")


if __name__ == "__main__":
    main()
