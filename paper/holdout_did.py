"""Holdout attenuation: decay or composition? (referee response, Tier 1 #5).

The primary gap (+0.031) is measured on 183 crypto vs 187 stocks; the holdout gap
(+0.020) is measured on the subset that clears the holdout's relaxed bar cuts --
173 crypto and only 52 stocks, and those 52 are the longest-history names, a
non-random selection. The reported 35% attenuation is therefore a compound of true
decay and a universe change, in unknown proportion.

This module separates them by recomputing the PRIMARY-window gap restricted to
exactly the assets the holdout scored, from the frozen per-asset OOS dumps with no
refits. The difference-in-differences

    (gap_primary_full - gap_primary_matched)   <- composition
    (gap_primary_matched - gap_holdout)        <- decay on a fixed universe

attributes the attenuation.

    uv run --active python -m paper.holdout_did
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dependence import BLOCK, MIN_OBS, OOS, OUT, SLOT, fast_auc

N_BOOT, SEED = 1000, 11


def _holdout_scored() -> dict[str, set[str]]:
    """Assets with a usable model leg in the holdout, by class."""
    h = json.loads((OUT / "holdout.json").read_text())
    got: dict[str, set[str]] = {"crypto": set(), "stock": set()}
    for r in h["rows"]:
        if r.get("auc_ising") is None or r.get("n_folds") in (None, 0):
            continue
        got[r["class"]].add(r["asset"])
    return got


def gap_on(subset: dict[str, set[str]] | None, n_boot: int = N_BOOT) -> dict:
    rows = json.loads((OUT / "wide.json").read_text())
    data = {}
    for r in rows:
        if subset is not None and r["asset"] not in subset[r["class"]]:
            continue
        f = OOS / f"{r['asset']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        data[r["asset"]] = {"cls": r["class"], "slot": z["ts"] // SLOT,
                            "y": z["actual"].astype(np.int8), "p": z["p_ising"]}
    if not data:
        return {}
    lo = min(int(d["slot"].min()) for d in data.values())
    hi = max(int(d["slot"].max()) for d in data.values())
    T = hi - lo + 1
    for d in data.values():
        lut = np.full(T, -1, np.int32)
        lut[d["slot"] - lo] = np.arange(len(d["slot"]), dtype=np.int32)
        d["lut"] = lut

    def means(take):
        per = {"crypto": [], "stock": []}
        for d in data.values():
            if take is None:
                y, p = d["y"], d["p"]
            else:
                obs = d["lut"][take]
                obs = obs[obs >= 0]
                if len(obs) < MIN_OBS:
                    continue
                y, p = d["y"][obs], d["p"][obs]
            if len(np.unique(y)) < 2:
                continue
            a = fast_auc(y, p)
            if np.isfinite(a):
                per[d["cls"]].append(a)
        if not per["crypto"] or not per["stock"]:
            return None
        return float(np.mean(per["crypto"])), float(np.mean(per["stock"]))

    obs = means(None)
    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(T / BLOCK))
    gaps = []
    for _ in range(n_boot):
        st = rng.integers(0, max(1, T - BLOCK + 1), size=nb)
        slots = np.concatenate([np.arange(s, s + BLOCK) for s in st])[:T]
        m = means(slots)
        if m:
            gaps.append(m[0] - m[1])
    g = np.array(gaps)
    return {
        "n_crypto": sum(1 for d in data.values() if d["cls"] == "crypto"),
        "n_stock": sum(1 for d in data.values() if d["cls"] == "stock"),
        "crypto_mean": obs[0], "stock_mean": obs[1], "gap": obs[0] - obs[1],
        "gap_ci": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
        "gap_sd": float(g.std(ddof=1)),
    }


if __name__ == "__main__":
    sub = _holdout_scored()
    full = gap_on(None)
    matched = gap_on(sub)
    h = json.loads((OUT / "holdout.json").read_text())["summary"]["joint_gap"]
    h_gap = h["ising"]["obs_gap"] if "ising" in h else h.get("obs_gap")

    comp = full["gap"] - matched["gap"]
    decay = matched["gap"] - h_gap
    res = {
        "primary_full": full,
        "primary_matched_to_holdout_universe": matched,
        "holdout_gap": h_gap,
        "total_attenuation": full["gap"] - h_gap,
        "attributable_to_composition": comp,
        "attributable_to_decay": decay,
        "composition_share": comp / (full["gap"] - h_gap),
        "holdout_universe": {"n_crypto": len(sub["crypto"]), "n_stock": len(sub["stock"])},
    }
    (OUT / "holdout_did.json").write_text(json.dumps(res, indent=2))
    print(f"primary, full universe      ({full['n_crypto']}c/{full['n_stock']}s): gap {full['gap']:+.5f} {full['gap_ci']}")
    print(f"primary, holdout universe   ({matched['n_crypto']}c/{matched['n_stock']}s): gap {matched['gap']:+.5f} {matched['gap_ci']}")
    print(f"holdout window                                    : gap {h_gap:+.5f}")
    print()
    print(f"total attenuation {res['total_attenuation']:+.5f}")
    print(f"  composition {comp:+.5f} ({res['composition_share']*100:.0f}%)   decay {decay:+.5f}")
