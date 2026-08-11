"""Per-asset bootstrap under data-driven block lengths (resolves the block-count caveat).

The per-asset test fixes the block at 384 slots for every asset, which gives the
median crypto pair ~86 blocks but the median stock only ~16 -- a thinly resampled
equity leg. This rerun scales the block length at the standard n^(1/3) rate,
anchored so the median crypto pair keeps its original 384:

    b_i = round(384 * (n_i / 33,217)^(1/3)),  floored at 64 slots (16 hours),

so the median stock uses ~217-slot blocks (~28 blocks) instead of 16. Everything
else is identical and nothing is refit: frozen per-asset scores from wide_oos/,
one-sided p for AUC > 0.5 per model at B = 1,000, the conjunction, and joint
Benjamini-Hochberg across all 370. Output: the FDR counts under scaled blocks
next to the frozen counts.

    uv run --active python -m paper.blocklen_sensitivity
"""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .dependence import OOS, OUT, fast_auc
from .fdr import bh_mask

N_BOOT, SEED = 1000, 20260812
ANCHOR_N, ANCHOR_B, FLOOR = 33217, 384, 64


def block_len(n: int) -> int:
    return max(FLOOR, round(ANCHOR_B * (n / ANCHOR_N) ** (1 / 3)))


def one_asset(args):
    asset, cls = args
    f = OOS / f"{asset}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    y, p_is, p_fr = z["actual"].astype(np.int8), z["p_ising"], z["p_free"]
    n = len(y)
    if len(np.unique(y)) < 2:
        return None
    b = block_len(n)
    nb = int(np.ceil(n / b))
    rng = np.random.default_rng([SEED, hash(asset) & 0x7FFFFFFF])
    ge_is = ge_fr = used = 0
    for _ in range(N_BOOT):
        starts = rng.integers(0, max(1, n - b + 1), size=nb)
        idx = np.concatenate([np.arange(s, s + b) for s in starts])[:n]
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        used += 1
        ge_is += fast_auc(yb, p_is[idx]) <= 0.5
        ge_fr += fast_auc(yb, p_fr[idx]) <= 0.5
    if used == 0:
        return None
    return {"asset": asset, "class": cls, "n_oos": n, "block": b,
            "n_blocks": round(n / b, 1),
            "p_ising": ge_is / used, "p_free": ge_fr / used,
            "conj_p": max(ge_is, ge_fr) / used}


def main():
    wide = json.loads((OUT / "wide.json").read_text())
    args = [(r["asset"], r["class"]) for r in wide]
    with Pool(8) as pool:
        rows = [r for r in pool.imap_unordered(one_asset, args, chunksize=8) if r]
    p = np.array([r["conj_p"] for r in rows])
    keep = bh_mask(p, 0.05)
    for r, k in zip(rows, keep):
        r["bh_sig"] = bool(k)

    res = {"n_assets": len(rows), "n_boot": N_BOOT,
           "rule": f"b = round({ANCHOR_B}*(n/{ANCHOR_N})^(1/3)), floor {FLOOR}"}
    frozen = {"crypto": (164, 183), "stock": (5, 187)}
    for cls in ("crypto", "stock"):
        sel = [r for r in rows if r["class"] == cls]
        sig = sum(r["bh_sig"] for r in sel)
        res[cls] = {"n": len(sel), "bh_sig": sig,
                    "bh_frac": sig / len(sel),
                    "median_block": int(np.median([r["block"] for r in sel])),
                    "median_n_blocks": float(np.median([r["n_blocks"] for r in sel])),
                    "frozen_bh_sig": frozen[cls][0], "frozen_n": frozen[cls][1]}
        print(f"{cls:7s} block med {res[cls]['median_block']:3d} "
              f"(~{res[cls]['median_n_blocks']:.0f} blocks): "
              f"BH-sig {sig}/{len(sel)}  (frozen {frozen[cls][0]}/{frozen[cls][1]})")
    (OUT / "blocklen_sensitivity.json").write_text(
        json.dumps({**res, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
