"""Wide study at 1h (horizon robustness at the population level).

Resamples each bulk 15m series to 1h (compounding the four sub-candle returns;
exact for continuous markets where open_{i+1}=close_i) and reruns the matched
predictability test across the full universe, to check whether the crypto-vs-stock
contrast persists one horizon up from 15m.

    uv run --active python -m paper.wide_horizon
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .models import ARLogit, IsingLogit
from .wide import BULK, STOCK_SET, OUT
from .walkforward import walk_forward

TRAIN, TEST, BLOCK, N_BOOT, SEED = 2160, 360, 96, 300, 5     # 1h geometry


def load_1h(path):
    raw = json.loads(path.read_text())
    raw.sort(key=lambda d: d["timestamp"])
    buckets: dict[int, list] = {}
    for d in raw:
        buckets.setdefault(d["timestamp"] // 3600 * 3600, []).append(d)
    ch, ups = [], []
    for key in sorted(buckets):
        gross = 1.0
        for d in sorted(buckets[key], key=lambda x: x["timestamp"]):
            gross *= (1.0 + d.get("change", 0.0))
        c = gross - 1.0
        ch.append(c); ups.append(c > 0)
    return np.array(ch), np.array(ups, bool)


def block_idx(n, rng):
    nb = int(np.ceil(n / BLOCK))
    s = rng.integers(0, max(1, n - BLOCK + 1), size=nb)
    return np.concatenate([np.arange(x, x + BLOCK) for x in s])[:n]


def evaluate(path):
    ch, ups = load_1h(path)
    if len(ch) < TRAIN + TEST + 1000:
        return None
    res, _ = walk_forward(ch, ups, TRAIN, TEST, n_lags=12,
                          models=lambda: [IsingLogit(n_lags=12), ARLogit("free", n_lags=12)])
    y = res["ising"]["actuals"].astype(int)
    if len(y) < 1000 or len(np.unique(y)) < 2:
        return None
    p_is, p_fr = res["ising"]["probs"], res["free"]["probs"]
    auc = roc_auc_score(y, p_is)
    A = float(np.mean([fp["A"] for fp in res["ising"]["fold_params"]]))
    rng = np.random.default_rng(SEED)
    gi = gf = 0
    for _ in range(N_BOOT):
        idx = block_idx(len(y), rng); yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        gi += roc_auc_score(yb, p_is[idx]) <= 0.5
        gf += roc_auc_score(yb, p_fr[idx]) <= 0.5
    return {"auc_ising": float(auc), "A": A, "p_ising": gi / N_BOOT, "p_free": gf / N_BOOT}


def main():
    files = sorted(BULK.glob("*-15m.json"))
    rows = []
    for i, f in enumerate(files):
        base = f.name[:-9]
        cls = "stock" if base in STOCK_SET else "crypto"
        try:
            r = evaluate(f)
        except Exception:
            continue
        if r:
            r.update(asset=base, **{"class": cls}); rows.append(r)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(files)}] kept {len(rows)}")
    (OUT / "wide_1h.json").write_text(json.dumps(rows, indent=2))
    print("\n=== WIDE 1h SUMMARY ===")
    for cls in ("crypto", "stock"):
        sub = [r for r in rows if r["class"] == cls]
        if not sub:
            continue
        aucs = np.array([r["auc_ising"] for r in sub])
        sig_both = np.mean([r["p_ising"] < 0.05 and r["p_free"] < 0.05 for r in sub])
        print(f"{cls.upper()} (n={len(sub)}): mean AUC={aucs.mean():.4f}, frac>0.5={np.mean(aucs>0.5):.2f}, "
              f"both-sig={sig_both*100:.0f}%, frac A<0={np.mean([r['A']<0 for r in sub]):.2f}")
    print(f"Wrote {OUT/'wide_1h.json'}")


if __name__ == "__main__":
    main()
