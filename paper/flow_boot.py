"""Block-bootstrap inference for the model-free flow-conditioned flip rates
(upgrades the naive binomial z of paper.flow_test): per coin and pooled,
moving-block bootstrap CI for flip(driven) - flip(opposed), and for the
dose-response slope of flip rate on |imbalance| quintile within driven bars.

    uv run --active python -m paper.flow_boot
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .flow_test import COINS, load
from .compare_markets import block_boot_idx

OUT = Path(__file__).resolve().parent / "out"
BLOCK, N_BOOT, SEED = 384, 2000, 7


def series(coin):
    ch, ups, imb = load(coin)
    s = np.sign(ch)
    t = np.arange(len(ch) - 1)
    ok = (s[t] != 0) & (s[t + 1] != 0) & np.isfinite(imb[t]) & (imb[t] != 0)
    t = t[ok]
    return (s[t] * imb[t] > 0), (s[t + 1] == -s[t]), np.abs(imb[t])


def stats(driven, flip, absimb, q):
    d = float(flip[driven].mean() - flip[~driven].mean())
    bins = np.digitize(absimb, q)
    m = [flip[driven & (bins == b)].mean() if (driven & (bins == b)).sum() > 30
         else np.nan for b in range(5)]
    slope = np.polyfit(np.arange(5)[np.isfinite(m)], np.array(m)[np.isfinite(m)], 1)[0] \
        if np.isfinite(m).sum() >= 3 else np.nan
    return d, float(slope)


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    allD, allF, allI = [], [], []
    for coin in COINS:
        driven, flip, absimb = series(coin)
        q = np.nanquantile(absimb, [0.2, 0.4, 0.6, 0.8])
        d0, s0 = stats(driven, flip, absimb, q)
        bd, bs = [], []
        for _ in range(N_BOOT):
            bi = block_boot_idx(len(flip), BLOCK, rng)
            d, s = stats(driven[bi], flip[bi], absimb[bi], q)
            bd.append(d); bs.append(s)
        ci_d = np.percentile(bd, [2.5, 97.5])
        ci_s = np.percentile([x for x in bs if np.isfinite(x)], [2.5, 97.5])
        p_d = float(np.mean(np.array(bd) <= 0))
        print(f"  {coin}: delta_flip={d0:+.4f} CI [{ci_d[0]:+.4f},{ci_d[1]:+.4f}] "
              f"p(<=0)={p_d:.4f}   dose slope={s0:+.5f} CI [{ci_s[0]:+.5f},{ci_s[1]:+.5f}]")
        out[coin] = {"delta_flip": d0, "ci": ci_d.tolist(), "p_le_0": p_d,
                     "dose_slope": s0, "slope_ci": ci_s.tolist()}
        allD.append(driven); allF.append(flip); allI.append(absimb)

    # pooled: concatenate, bootstrap within-coin blocks jointly
    driven = np.concatenate(allD); flip = np.concatenate(allF)
    absimb = np.concatenate(allI)
    q = np.nanquantile(absimb, [0.2, 0.4, 0.6, 0.8])
    d0, s0 = stats(driven, flip, absimb, q)
    lens = [len(x) for x in allD]
    offs = np.cumsum([0] + lens[:-1])
    bd, bs = [], []
    for _ in range(N_BOOT):
        bi = np.concatenate([o + block_boot_idx(n, BLOCK, rng)
                             for o, n in zip(offs, lens)])
        d, s = stats(driven[bi], flip[bi], absimb[bi], q)
        bd.append(d); bs.append(s)
    ci_d = np.percentile(bd, [2.5, 97.5]); ci_s = np.percentile(bs, [2.5, 97.5])
    print(f"  pooled: delta_flip={d0:+.4f} CI [{ci_d[0]:+.4f},{ci_d[1]:+.4f}] "
          f"p(<=0)={float(np.mean(np.array(bd) <= 0)):.4f}   "
          f"dose slope={s0:+.5f} CI [{ci_s[0]:+.5f},{ci_s[1]:+.5f}]")
    out["pooled"] = {"delta_flip": d0, "ci": ci_d.tolist(),
                     "p_le_0": float(np.mean(np.array(bd) <= 0)),
                     "dose_slope": s0, "slope_ci": ci_s.tolist()}
    (OUT / "flow_boot.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT / 'flow_boot.json'}")


if __name__ == "__main__":
    main()
