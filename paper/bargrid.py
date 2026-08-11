"""Bar-grid alignment: is the reversal tied to the quarter-hour clock?

Every 15m result in the paper is computed on the conventional grid, with bars
opening at :00/:15/:30/:45. Quarter-hour clock effects -- order-flow, quoting,
or settlement seasonality concentrated on those boundaries -- could in principle
manufacture structure that is specific to that grid rather than a property of
the tape. If so, rebuilding the bars on an off-grid clock should weaken or
destroy the signal.

This module rebuilds 15-minute bars directly from the cached 1-second klines
with the bar opening shifted to +1, +2, +5, and +7 minutes past the quarter
hour, and recomputes the parameter-free kernel-weighted sign correlation

    R = corr( sigma_t , sum_{k=1..12} sigma_{t-k} / k^alpha ),  alpha = 1

of `signlag.py` (Eq. R in the paper) at each offset. R is used rather than the
walk-forward AUC because it takes no fitted parameter and no train/test split,
so a grid shift changes exactly one thing: where the bars open.

The reconstruction is deliberately like-for-like -- the on-grid (+0) row is
rebuilt from the SAME 1-second archive under the SAME completeness rules as the
off-grid rows, so the comparison is not contaminated by a change of data source.
Bars are open-to-close (the paper's return convention, see `common.py`); a bar
is kept only if the archive covers it densely (>= MIN_TICKS of its 900 seconds,
with ticks near both edges), and a bar enters the statistic only when its own
twelve predecessors are all present and contiguous, so a gap in the archive
cannot splice unrelated stretches of tape into one kernel window.

Coverage is the three focal coins with a 1-second archive (SOL has none). This
is a focal-panel check, not a population-scale one; the paper says so.

    uv run --active python -m paper.bargrid
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .signlag import ALPHA_FIXED, N_LAGS, corr, kernel_field, sign_series

ONE_S = Path(os.environ.get(
    "PAPER_1S_DIR", Path(__file__).resolve().parents[3] / "data"))
OUT = Path(__file__).resolve().parent / "out"

COINS = ["btc", "eth", "xrp"]
OFFSETS_MIN = [0, 1, 2, 5, 7]      # minutes past the quarter hour
SLOT = 900                         # 15m in seconds
MIN_TICKS = 800                    # of 900 seconds; tolerates thin-second gaps
EDGE_TOL = 30                      # first/last tick must sit within 30s of the edge
BLOCK, N_BOOT, SEED = 384, 1000, 7  # matching the signlag / dependence bootstrap
MIN_BARS = 1000


def load_1s(coin: str) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the daily 1-second archives for `coin`.

    Returns (timestamps, prices) sorted by time and de-duplicated. Files hold
    ``[[open_timestamp_seconds, close_price], ...]`` for one UTC day
    (see scripts/data/py/download_1s.py)."""
    d = ONE_S / f"{coin}_1s"
    if not d.is_dir():
        raise FileNotFoundError(f"no 1-second archive at {d}")
    ts, px = [], []
    for f in sorted(d.glob("*.json")):
        for t, p in json.loads(f.read_text()):
            ts.append(int(t)); px.append(float(p))
    t = np.asarray(ts, np.int64); p = np.asarray(px, float)
    order = np.argsort(t, kind="stable")
    t, p = t[order], p[order]
    keep = np.append(np.diff(t) > 0, True)      # keep the last tick of each second
    return t[keep], p[keep]


def build_bars(t: np.ndarray, p: np.ndarray, offset_min: int) -> tuple[np.ndarray, np.ndarray]:
    """15m open-to-close returns on a grid shifted `offset_min` minutes.

    Returns (slot, ret) for the bars that pass the completeness rules, where
    `slot` is the bar's index on the shifted grid (so gaps are detectable as
    jumps of more than one)."""
    off = offset_min * 60
    slot_of = (t - off) // SLOT
    # segment boundaries: first/last tick index of each occupied slot
    edges = np.flatnonzero(np.append(np.diff(slot_of) != 0, True))
    starts = np.append(0, edges[:-1] + 1)
    slots, rets = [], []
    for i0, i1 in zip(starts, edges):
        n = i1 - i0 + 1
        if n < MIN_TICKS:
            continue
        s = int(slot_of[i0])
        bar_open_ts = s * SLOT + off
        if t[i0] - bar_open_ts > EDGE_TOL or (bar_open_ts + SLOT - 1) - t[i1] > EDGE_TOL:
            continue
        o, c = p[i0], p[i1]
        if o <= 0:
            continue
        slots.append(s); rets.append((c - o) / o)
    return np.asarray(slots, np.int64), np.asarray(rets, float)


def kernel_pairs(slot: np.ndarray, ret: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sigma_t, field_t) for every bar whose twelve predecessors are all present
    and contiguous on the grid. Missing bars break the kernel window rather than
    being silently skipped over."""
    lo, hi = int(slot[0]), int(slot[-1])
    T = hi - lo + 1
    present = np.zeros(T, bool)
    dense = np.zeros(T, float)
    present[slot - lo] = True
    dense[slot - lo] = ret
    sig = sign_series(dense)
    sig[~present] = 0.0
    field = kernel_field(sig, ALPHA_FIXED, N_LAGS)
    # valid_t: bar t and each of its N_LAGS predecessors present
    ok = present.copy()
    for k in range(1, N_LAGS + 1):
        ok[k:] &= present[:T - k]
    ok[:N_LAGS] = False
    return sig[ok], field[ok]


def block_bootstrap_ci(sig: np.ndarray, field: np.ndarray, seed: int = SEED) -> list[float]:
    """Moving-block percentile CI for R, blocks of BLOCK consecutive bars."""
    n = len(sig)
    if n <= BLOCK:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, size=(N_BOOT, n_blocks))
    idx_in = np.arange(BLOCK)
    draws = np.empty(N_BOOT, float)
    for b in range(N_BOOT):
        take = (starts[b][:, None] + idx_in).ravel()[:n]
        draws[b] = corr(sig[take], field[take])
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main() -> None:
    results: dict[str, dict] = {}
    for coin in COINS:
        try:
            t, p = load_1s(coin)
        except FileNotFoundError as e:
            print(f"  {coin}: skipped ({e})")
            continue
        span = [str(np.datetime64(int(t[0]), "s"))[:10],
                str(np.datetime64(int(t[-1]), "s"))[:10]]
        per_offset = {}
        for off in OFFSETS_MIN:
            slot, ret = build_bars(t, p, off)
            if len(slot) < MIN_BARS:
                print(f"  {coin} +{off}m: only {len(slot)} bars, skipped")
                continue
            sig, field = kernel_pairs(slot, ret)
            per_offset[str(off)] = {
                "offset_min": off,
                "n_bars": int(len(slot)),
                "n_obs": int(len(sig)),
                "flat_frac": float(np.mean(ret == 0.0)),
                "R_kernel": corr(sig, field),
                "R_ci": block_bootstrap_ci(sig, field),
            }
        results[coin] = {"span": span, "n_ticks": int(len(t)), "offsets": per_offset}

    summary = {}
    for coin, r in results.items():
        offs = r["offsets"]
        if "0" not in offs:
            continue
        on = offs["0"]["R_kernel"]
        off_vals = [v["R_kernel"] for k, v in offs.items() if k != "0"]
        summary[coin] = {
            "R_on_grid": on,
            "R_off_grid_min": float(min(off_vals)),   # most negative
            "R_off_grid_max": float(max(off_vals)),   # least negative
            "all_negative": bool(on < 0 and all(v < 0 for v in off_vals)),
            "on_grid_largest_magnitude": bool(abs(on) >= max(abs(v) for v in off_vals)),
        }

    payload = {"config": {"offsets_min": OFFSETS_MIN, "slot_seconds": SLOT,
                          "n_lags": N_LAGS, "alpha": ALPHA_FIXED,
                          "min_ticks": MIN_TICKS, "edge_tol_s": EDGE_TOL,
                          "block": BLOCK, "n_boot": N_BOOT, "seed": SEED},
               "summary": summary, "coins": results}
    OUT.mkdir(exist_ok=True)
    (OUT / "bargrid.json").write_text(json.dumps(payload, indent=2))

    print("\n=== Bar-grid alignment: R (Eq. R) by bar-opening offset, 15m from 1s klines ===")
    for coin, r in results.items():
        print(f"\n  {coin.upper()}  {r['span'][0]}..{r['span'][1]}")
        for k in map(str, OFFSETS_MIN):
            if k not in r["offsets"]:
                continue
            v = r["offsets"][k]
            tag = "on-grid " if k == "0" else f"+{k:>2s}m    "
            print(f"    {tag} R={v['R_kernel']:+.4f} "
                  f"CI=[{v['R_ci'][0]:+.4f},{v['R_ci'][1]:+.4f}]  "
                  f"bars={v['n_bars']:5d} obs={v['n_obs']:5d} flat={v['flat_frac']*100:.2f}%")
    print(f"\nWrote {OUT/'bargrid.json'}")


if __name__ == "__main__":
    main()
