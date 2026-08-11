"""The artifact channels applied JOINTLY, not one at a time (referee response, Tier 1 #4).

Table `tab:ledger` varies one artifact channel at a time. Two of the rows each remove
roughly a third of the crypto excess -- dropping flat bars (0.531 -> 0.520) and putting
a one-bar gap between features and label (0.531 -> 0.522) -- and they are largely
independent: one is a labelling convention, the other is boundary microstructure. Their
conjunction is never reported, so a reader cannot tell whether the effect survives both
at once.

This module scores the 2x2 cell directly. Features are shifted by one candle exactly as
in `gap_test.py` (so the model predicts candle t from t-2 ... t-N-1), the walk-forward is
refit under the identical protocol, and the resulting out-of-sample series is scored both
on all bars and with flat bars dropped. The class gap in each of the four cells is then
taken under the same joint moving-block bootstrap on the shared 15m grid used everywhere
else, so all four intervals are comparable.

    uv run --active python -m paper.joint_artifact
"""

from __future__ import annotations

import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .models import ARLogit, IsingLogit
from .walkforward import walk_forward

BULK = Path(__file__).resolve().parent / "bulk_data"
OUT = Path(__file__).resolve().parent / "out"
DUMP = OUT / "gap_oos"
SLOT = 900
TRAIN, TEST = 5760, 960
BLOCK, N_BOOT, SEED = 384, 1000, 7
MIN_OBS = 1000


def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _score_gapped(path_str: str):
    """Refit the walk-forward with a one-bar feature/label gap and dump the OOS series."""
    path = Path(path_str)
    asset = path.name[: -len("-15m.json")]
    out = DUMP / f"{asset}.npz"
    if out.exists():
        return asset
    raw = json.loads(path.read_text())
    raw.sort(key=lambda d: d["timestamp"])
    ch = np.array([d.get("change", 0.0) for d in raw], float)
    ups = np.array([bool(d["up"]) for d in raw], bool)
    ts = np.array([d["timestamp"] for d in raw], np.int64)
    if len(ch) < TRAIN + TEST + 2000:
        return None
    ch_gap = np.concatenate([[0.0], ch[:-1]])

    def factory():
        return [IsingLogit(n_lags=12), ARLogit("free", n_lags=12)]

    res, _ = walk_forward(ch_gap, ups, TRAIN, TEST, n_lags=12, models=factory)
    y = res["ising"]["actuals"].astype(int)
    if len(y) < 2000 or len(np.unique(y)) < 2:
        return None
    # walk_forward scores the tail of the series; align ts and the raw change series
    n = len(y)
    DUMP.mkdir(exist_ok=True)
    np.savez_compressed(
        out,
        ts=ts[-n:],
        actual=y.astype(np.int8),
        p_ising=res["ising"]["probs"].astype(np.float32),
        p_free=res["free"]["probs"].astype(np.float32),
        change=ch[-n:].astype(np.float32),
    )
    return asset


def joint_gap(dump_dir: Path, nonflat: bool, model: str = "p_ising") -> dict:
    rows = json.loads((OUT / "wide.json").read_text())
    cls = {r["asset"]: r["class"] for r in rows}
    data = {}
    for f in sorted(dump_dir.glob("*.npz")):
        a = f.stem
        if a not in cls:
            continue
        z = np.load(f)
        y, p, slot = z["actual"].astype(np.int8), z[model], z["ts"] // SLOT
        if nonflat:
            keep = z["change"] != 0.0
            y, p, slot = y[keep], p[keep], slot[keep]
        if len(y) < MIN_OBS or len(np.unique(y)) < 2:
            continue
        data[a] = {"cls": cls[a], "slot": slot, "y": y, "p": p}
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
    for _ in range(N_BOOT):
        st = rng.integers(0, max(1, T - BLOCK + 1), size=nb)
        slots = np.concatenate([np.arange(s, s + BLOCK) for s in st])[:T]
        m = means(slots)
        if m:
            gaps.append(m[0] - m[1])
    g = np.array(gaps)
    return {"n_crypto": sum(1 for d in data.values() if d["cls"] == "crypto"),
            "n_stock": sum(1 for d in data.values() if d["cls"] == "stock"),
            "crypto_mean": obs[0], "stock_mean": obs[1], "gap": obs[0] - obs[1],
            "gap_ci": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]}


def main():
    files = [str(p) for p in sorted(BULK.glob("*-15m.json"))]
    workers = max(2, (os.cpu_count() or 4) - 2)
    print(f"refitting {len(files)} assets with a one-bar gap on {workers} workers ...")
    with Pool(workers) as pool:
        done = [a for a in pool.imap_unordered(_score_gapped, files, chunksize=4) if a]
    print(f"  scored {len(done)}")

    res = {}
    for nf in (False, True):
        key = "gap_nonflat" if nf else "gap_allbars"
        res[key] = joint_gap(DUMP, nonflat=nf)
        print(f"{key}: gap {res[key]['gap']:+.5f} {res[key]['gap_ci']}"
              f"  (crypto {res[key]['crypto_mean']:.4f} / stock {res[key]['stock_mean']:.4f})")
    (OUT / "joint_artifact.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
