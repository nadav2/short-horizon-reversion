"""Exact permutation null for the class-mean AUC gap (referee response, Tier 1 #1).

The headline gap is reported with a percentile statement from the joint moving-block
bootstrap ("p < 1e-3"), which is (i) a coverage statement rather than a tail
probability under a null and (ii) bounded below by 1/B, so the quoted value is the
bootstrap's resolution floor rather than a measured tail.

This module builds the missing null directly. Under H0 the fitted score carries no
information about the label, so we destroy score-label alignment while preserving
everything else:

  * a single random circular shift of the label series is applied to EVERY asset on
    the shared 15m slot grid, so within-asset label autocorrelation and the full
    cross-sectional label dependence are preserved exactly;
  * shifts are drawn at least one block (384 slots) away from zero and from T, so no
    replicate is near-aligned;
  * class-mean AUCs and the gap are recomputed with the identical fast_auc, MIN_OBS
    and class-membership rules used by dependence.py.

The resulting p is an exact one-sided permutation p-value for
H0: gap <= 0, namely (1 + #{gap_null >= gap_obs}) / (1 + n_perm).

    uv run --active python -m paper.permutation_null
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dependence import BLOCK, MIN_OBS, OOS, OUT, SLOT, fast_auc

N_PERM, SEED = 1000, 20260811


def _load() -> dict:
    rows = json.loads((OUT / "wide.json").read_text())
    data = {}
    for r in rows:
        f = OOS / f"{r['asset']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        data[r["asset"]] = {
            "cls": r["class"],
            "slot": z["ts"] // SLOT,
            "y": z["actual"].astype(np.int8),
            "p_is": z["p_ising"],
            "p_fr": z["p_free"],
        }
    return data


def run(n_perm: int = N_PERM) -> dict:
    data = _load()
    lo = min(int(d["slot"].min()) for d in data.values())
    hi = max(int(d["slot"].max()) for d in data.values())
    T = hi - lo + 1

    # slot -> observation index per asset, so a shift on the shared grid maps to
    # each asset's own (irregular, gap-carrying) observation sequence.
    for d in data.values():
        lut = np.full(T, -1, np.int32)
        lut[d["slot"] - lo] = np.arange(len(d["slot"]), dtype=np.int32)
        d["lut"] = lut
        d["pos"] = d["slot"] - lo

    def class_gap(shift: int | None) -> dict:
        out = {}
        for model in ("p_is", "p_fr"):
            per_cls: dict[str, list[float]] = {"crypto": [], "stock": []}
            for d in data.values():
                if shift is None:
                    y, p = d["y"], d[model]
                else:
                    # label at slot s is paired with the score at slot s-shift
                    src = d["lut"][(d["pos"] - shift) % T]
                    keep = src >= 0
                    if keep.sum() < MIN_OBS:
                        continue
                    y, p = d["y"][keep], d[model][src[keep]]
                if len(np.unique(y)) < 2:
                    continue
                a = fast_auc(y, p)
                if np.isfinite(a):
                    per_cls[d["cls"]].append(a)
            if per_cls["crypto"] and per_cls["stock"]:
                c, s = float(np.mean(per_cls["crypto"])), float(np.mean(per_cls["stock"]))
                out[model] = {"crypto": c, "stock": s, "gap": c - s}
        return out

    obs = class_gap(None)
    rng = np.random.default_rng(SEED)
    # Shifts are WHOLE-DAY multiples (96 slots of 15m). Listed instruments cover only
    # 68 of the 96 slots-of-day while crypto covers all 96, so an arbitrary shift would
    # map session labels onto hours where a stock has no score, dropping observations
    # and changing class composition replicate to replicate. A day-aligned shift maps
    # every asset's coverage pattern exactly onto itself, so the only thing destroyed
    # is score-label alignment.
    day = 96
    max_d = (T - BLOCK) // day
    min_d = max(1, BLOCK // day)
    cand = np.arange(min_d, max_d + 1) * day
    shifts = rng.choice(cand, size=min(n_perm, len(cand)), replace=False)
    n_perm = len(shifts)

    null = {"p_is": [], "p_fr": []}
    for i, sh in enumerate(shifts):
        g = class_gap(int(sh))
        for m in null:
            if m in g:
                null[m].append(g[m]["gap"])
        if (i + 1) % 100 == 0:
            print(f"  permutation {i+1}/{n_perm}")

    res = {"n_perm": n_perm, "n_assets_used": len(data), "block_min_shift": BLOCK,
           "T_slots": int(T), "shift_grid": "whole-day multiples (96 slots)",
           "n_distinct_shifts_available": int(len(cand))}
    for m, label in (("p_is", "ising"), ("p_fr", "free")):
        g = np.array(null[m])
        o = obs[m]["gap"]
        res[label] = {
            "obs_gap": o,
            "obs_crypto_mean": obs[m]["crypto"],
            "obs_stock_mean": obs[m]["stock"],
            "null_mean": float(g.mean()),
            "null_sd": float(g.std(ddof=1)),
            "null_q025": float(np.percentile(g, 2.5)),
            "null_q975": float(np.percentile(g, 97.5)),
            "null_max": float(g.max()),
            "n_null_ge_obs": int((g >= o).sum()),
            "perm_p_one_sided": float((1 + (g >= o).sum()) / (1 + len(g))),
            "z_vs_null": float((o - g.mean()) / g.std(ddof=1)),
        }
    return res


if __name__ == "__main__":
    out = run()
    (OUT / "permutation_null.json").write_text(json.dumps(out, indent=2))
    for m in ("ising", "free"):
        r = out[m]
        print(f"\n{m}: observed gap {r['obs_gap']:+.5f}")
        print(f"  null mean {r['null_mean']:+.5f} sd {r['null_sd']:.5f} "
              f"[{r['null_q025']:+.5f},{r['null_q975']:+.5f}] max {r['null_max']:+.5f}")
        print(f"  permutation p = {r['perm_p_one_sided']:.4g}   z = {r['z_vs_null']:.1f}")
