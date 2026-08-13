"""Depth-conditioned reversal on the perp tape (the paper's named open test).

Reads the Binance Vision bookDepth features (paper.fetch_bookdepth): cumulative
bid/ask notional within 1% (primary) and 2% (robustness) of mid, ~25s
snapshots aggregated to 15m bars. A bar's MOVE-SIDE CONSUMPTION is
c_t = log(D_open / D_close) of the notional on the side price moved into
(asks for an up bar, bids for a down bar): positive when the move ate more
depth than was replenished. The liquidity-provision account predicts next-bar
reversal concentrates after depth-consuming moves; the sharpest falsification
in sec:discussion is reversal ABSENT after them.

1. MODEL-FREE FLIP RATES. flip(consumed) - flip(replenished), a dose-response
   over signed-consumption quintiles, the same contrast inside |r| quintiles
   (size-matched: consumption correlates with move size, and large moves flip
   more), and the 2x2 with aggressor flow (flow-driven AND depth-consuming
   should revert hardest). Moving-block bootstrap CIs per coin and pooled.
2. CONDITIONAL AUC. Walk-forward Ising OOS predictions bucketed by prev-bar
   consumption terciles, and driven&consumed vs driven&replenished.
3. BOOK STATE. Flip rate by pre-bar move-side depth tercile (thin book =>
   more reversal?) and by impact ratio |r|/depth quintile (Kyle-lambda flavor).
4. ROBUSTNESS. Headline contrast at the 2% band (less sensitive to the
   moving-mid band migration caveat), with consumption residualized on
   time-of-day (depth and volatility both have strong intraday seasonality),
   and with SWEEP depth (open-to-minimum depletion, catching a
   sweep-and-refill that nets out in first/last; above-median split).

Caveat carried into the paper: bands are percentages of the CURRENT mid, so
within-bar consumption conflates order removal with band migration as mid
moves; the 2% robustness bounds that, tick-level L2 would resolve it.

    uv run --active python -m paper.depth_test
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .compare_markets import SPAN, WINDOWS, block_boot_idx
from .models import IsingLogit
from .perp_test import bucket_auc
from .walkforward import walk_forward

HERE = Path(__file__).resolve().parent
DEPTH = HERE / "depth_data"
OUT = HERE / "out"
COINS = ["btc", "eth", "sol", "xrp"]
N_LAGS = 12
BLOCK, N_BOOT, SEED = 384, 2000, 7


def _move_side(s, bid, ask):
    """Move-side series: asks for up bars, bids for down bars, NaN on flat."""
    out = np.where(s > 0, ask, bid)
    return np.where(s == 0, np.nan, out)


def load(coin):
    raw = [d for d in json.loads((DEPTH / f"{coin}-perp-flow-15m.json").read_text())
           if SPAN[0] <= d["datetime"][:10] <= SPAN[1]]
    raw.sort(key=lambda d: d["timestamp"])
    ts = np.array([d["timestamp"] for d in raw], np.int64)
    ch = np.array([d["change"] for d in raw])
    ups = np.array([bool(d["up"]) for d in raw])
    vol = np.array([d["vol"] for d in raw])
    tbv = np.array([d["tbv"] for d in raw])
    with np.errstate(divide="ignore", invalid="ignore"):
        imb = np.where(vol > 0, 2 * tbv / vol - 1, np.nan)

    depth = {r["timestamp"]: r
             for r in json.loads((DEPTH / f"{coin}-depth-15m.json").read_text())}

    def col(name):
        return np.array([depth.get(int(t), {}).get(name, np.nan) for t in ts])

    s = np.sign(ch)

    def consumption(lvl):
        first = _move_side(s, col(f"b{lvl}_first"), col(f"a{lvl}_first"))
        last = _move_side(s, col(f"b{lvl}_last"), col(f"a{lvl}_last"))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((first > 0) & (last > 0), np.log(first / last), np.nan)

    cons1, cons2 = consumption(1), consumption(2)

    # sweep depth: open-to-minimum depletion within the bar (catches a
    # sweep-and-refill that nets out in first/last); always >= 0, so the
    # binary contrast uses an above-median split (centering on the median)
    first1 = _move_side(s, col("b1_first"), col("a1_first"))
    min1 = _move_side(s, col("b1_min"), col("a1_min"))
    with np.errstate(divide="ignore", invalid="ignore"):
        sweep = np.where((first1 > 0) & (min1 > 0), np.log(first1 / min1), np.nan)
    sweep = sweep - np.nanmedian(sweep)
    predepth = first1
    with np.errstate(divide="ignore", invalid="ignore"):
        impact = np.where(predepth > 0, np.abs(ch) / predepth, np.nan)

    # time-of-day residualization: subtract the per-slot median consumption
    slot = (ts % 86_400) // 900
    cons_adj = cons1.copy()
    for sl in range(96):
        m = (slot == sl) & np.isfinite(cons1)
        if m.sum() > 30:
            cons_adj[m] = cons1[m] - np.median(cons1[m])
    return ch, ups, imb, cons1, cons2, cons_adj, sweep, predepth, impact


# ------------------------------------------------------------ model-free
def flip_series(ch, cons, extra=()):
    """Valid transitions t -> t+1 with finite consumption at t. Returns
    (consumed, flip, cons_t, |r_t|, *extra_at_t)."""
    s = np.sign(ch)
    t = np.arange(len(ch) - 1)
    ok = (s[t] != 0) & (s[t + 1] != 0) & np.isfinite(cons[t])
    t = t[ok]
    base = (cons[t] > 0, s[t + 1] == -s[t], cons[t], np.abs(ch[t]))
    return base + tuple(x[t] for x in extra), s[t], t


def rate(flip, mask):
    n = int(mask.sum())
    p = float(flip[mask].mean()) if n else float("nan")
    se = float(np.sqrt(p * (1 - p) / n)) if n else float("nan")
    return {"rate": p, "n": n, "se": se}


def delta_stats(consumed, flip, cons, absr, q_cons, q_absr):
    """(headline delta, size-matched stratified delta, dose slope)."""
    d = float(flip[consumed].mean() - flip[~consumed].mean())

    strata = np.digitize(absr, q_absr)
    num = den = 0.0
    for k in range(5):
        a, b = consumed & (strata == k), ~consumed & (strata == k)
        if a.sum() > 30 and b.sum() > 30:
            w = a.sum() + b.sum()
            num += w * (flip[a].mean() - flip[b].mean())
            den += w
    d_strat = float(num / den) if den else float("nan")

    bins = np.digitize(cons, q_cons)
    m = [flip[bins == b].mean() if (bins == b).sum() > 30 else np.nan
         for b in range(5)]
    ok = np.isfinite(m)
    slope = float(np.polyfit(np.arange(5)[ok], np.array(m)[ok], 1)[0]) \
        if ok.sum() >= 3 else float("nan")
    return d, d_strat, slope


def boot(consumed, flip, cons, absr, rng, lens=None):
    q_cons = np.nanquantile(cons, [0.2, 0.4, 0.6, 0.8])
    q_absr = np.nanquantile(absr, [0.2, 0.4, 0.6, 0.8])
    d0 = delta_stats(consumed, flip, cons, absr, q_cons, q_absr)
    lens = lens or [len(flip)]
    offs = np.cumsum([0] + lens[:-1])
    reps = []
    for _ in range(N_BOOT):
        bi = np.concatenate([o + block_boot_idx(n, BLOCK, rng)
                             for o, n in zip(offs, lens)])
        reps.append(delta_stats(consumed[bi], flip[bi], cons[bi], absr[bi],
                                q_cons, q_absr))
    reps = np.array(reps)
    out = {}
    for i, name in enumerate(("delta_flip", "delta_flip_sizematched", "dose_slope")):
        col = reps[:, i][np.isfinite(reps[:, i])]
        lo, hi = np.percentile(col, [2.5, 97.5])
        out[name] = {"value": d0[i], "ci": [float(lo), float(hi)],
                     "p_le_0": float(np.mean(col <= 0))}
    return out


# --------------------------------------------------------- conditional AUC
def conditional_auc(ch, ups, imb, cons):
    tr, te = WINDOWS["15m"]
    res, _ = walk_forward(ch, ups, tr, te, n_lags=N_LAGS,
                          models=lambda: [IsingLogit(n_lags=N_LAGS)])
    r = res["ising"]
    idx = r["idx"].astype(int)
    y, p = r["actuals"].astype(int), r["probs"]
    prev = idx - 1
    c_prev = cons[prev]
    s_prev = np.sign(ch[prev])
    driven = (s_prev != 0) & np.isfinite(imb[prev]) & (s_prev * imb[prev] > 0)

    def auc(mask):
        m = mask & np.isfinite(c_prev)
        return (float(roc_auc_score(y[m], p[m]))
                if m.sum() > 100 and len(np.unique(y[m])) > 1 else float("nan"),
                int(m.sum()))

    a_dc, n_dc = auc(driven & (c_prev > 0))
    a_dr, n_dr = auc(driven & (c_prev <= 0))
    return {"auc_all": float(roc_auc_score(y, p)),
            "by_cons_tercile": bucket_auc(y, p, c_prev),
            "auc_driven_consumed": a_dc, "n_driven_consumed": n_dc,
            "auc_driven_replenished": a_dr, "n_driven_replenished": n_dr,
            "y": y, "p": p, "c_prev": c_prev, "driven": driven}


def main():
    rng = np.random.default_rng(SEED)
    summary = {}
    pool = {"consumed": [], "flip": [], "cons": [], "absr": [], "driven": []}
    pool_auc = {"y": [], "p": [], "rank_c": [], "driven": [], "consumed": []}
    lens = []

    for coin in COINS:
        ch, ups, imb, cons1, cons2, cons_adj, sweep, predepth, impact = load(coin)
        cov = float(np.isfinite(cons1).mean())
        print(f"\n=== {coin} (n={len(ch)}, depth coverage {cov:.1%}) ===")

        (consumed, flip, c, absr, imb_t, pre_t, imp_t), s_t, _ = \
            flip_series(ch, cons1, extra=(imb, predepth, impact))
        # diagnostics for the confound story
        diag = {"corr_cons_absr": float(np.corrcoef(c, absr)[0, 1]),
                "corr_cons_signedimb": float(np.corrcoef(
                    c[np.isfinite(imb_t)], (s_t * imb_t)[np.isfinite(imb_t)])[0, 1])}

        b = boot(consumed, flip, c, absr, rng)
        r_c, r_r = rate(flip, consumed), rate(flip, ~consumed)
        print(f"  flip| consumed {r_c['rate']:.4f} (n={r_c['n']})  "
              f"replenished {r_r['rate']:.4f} (n={r_r['n']})")
        print(f"  delta={b['delta_flip']['value']:+.4f} CI {b['delta_flip']['ci']}  "
              f"size-matched={b['delta_flip_sizematched']['value']:+.4f} "
              f"CI {b['delta_flip_sizematched']['ci']}  "
              f"dose slope={b['dose_slope']['value']:+.5f}")

        # 2x2 with aggressor flow
        okf = np.isfinite(imb_t) & (imb_t != 0)
        driven = okf & (s_t * imb_t > 0)
        two = {f"{f}_{d}": rate(flip, fm & dm)
               for f, fm in (("driven", driven), ("opposed", okf & ~driven))
               for d, dm in (("consumed", consumed), ("replenished", ~consumed))}
        print("  2x2 flip: " + "  ".join(f"{k}={v['rate']:.4f}"
                                         for k, v in two.items()))

        # book state
        q_pre = np.nanquantile(pre_t, [1 / 3, 2 / 3])
        by_pre = [rate(flip, np.digitize(pre_t, q_pre) == k) for k in range(3)]
        q_imp = np.nanquantile(imp_t, [0.2, 0.4, 0.6, 0.8])
        by_imp = [rate(flip, np.digitize(imp_t, q_imp) == k) for k in range(5)]
        print("  flip by pre-depth tercile (thin->deep): "
              + " ".join(f"{r['rate']:.4f}" for r in by_pre))
        print("  flip by |r|/depth quintile: "
              + " ".join(f"{r['rate']:.4f}" for r in by_imp))

        # robustness: 2% band, time-of-day residualized
        robust = {}
        for name, series in (("band2", cons2), ("tod_adj", cons_adj),
                             ("sweep", sweep)):
            (cc, ff, cx, ar), _, _ = flip_series(ch, series)
            robust[name] = boot(cc, ff, cx, ar, rng)
            print(f"  robustness {name}: delta="
                  f"{robust[name]['delta_flip']['value']:+.4f} "
                  f"CI {robust[name]['delta_flip']['ci']}")

        ca = conditional_auc(ch, ups, imb, cons1)
        t3 = ca["by_cons_tercile"]
        print(f"  ising OOS AUC {ca['auc_all']:.4f} | prev-bar cons terciles: "
              + " ".join(f"{t3[k]['auc']:.4f}" for k in ("low", "mid", "high"))
              + f"   driven&consumed {ca['auc_driven_consumed']:.4f} "
              f"vs driven&replenished {ca['auc_driven_replenished']:.4f}")

        summary[coin] = {
            "n_bars": len(ch), "depth_coverage": cov, "diag": diag,
            "flip_consumed": r_c, "flip_replenished": r_r, "boot": b,
            "flow_by_depth_2x2": two,
            "flip_by_predepth_tercile": by_pre,
            "flip_by_impact_quintile": by_imp,
            "robust": robust,
            "auc": {k: ca[k] for k in ("auc_all", "by_cons_tercile",
                                       "auc_driven_consumed", "n_driven_consumed",
                                       "auc_driven_replenished",
                                       "n_driven_replenished")},
        }
        for k, v in zip(pool, (consumed, flip, c, absr, driven)):
            pool[k].append(v)
        lens.append(len(flip))
        okc = np.isfinite(ca["c_prev"])
        rank_c = np.full(len(ca["c_prev"]), np.nan)
        rank_c[okc] = np.argsort(np.argsort(ca["c_prev"][okc])) / max(1, okc.sum() - 1)
        pool_auc["y"].append(ca["y"]); pool_auc["p"].append(ca["p"])
        pool_auc["rank_c"].append(rank_c)
        pool_auc["driven"].append(ca["driven"])
        pool_auc["consumed"].append(ca["c_prev"] > 0)

    # ------------------------------------------------------------- pooled
    print("\n=== pooled (4 coins) ===")
    P = {k: np.concatenate(v) for k, v in pool.items()}
    b = boot(P["consumed"], P["flip"], P["cons"], P["absr"], rng, lens=lens)
    print(f"  delta={b['delta_flip']['value']:+.4f} CI {b['delta_flip']['ci']} "
          f"p(<=0)={b['delta_flip']['p_le_0']:.4f}   "
          f"size-matched={b['delta_flip_sizematched']['value']:+.4f} "
          f"CI {b['delta_flip_sizematched']['ci']} "
          f"p(<=0)={b['delta_flip_sizematched']['p_le_0']:.4f}   "
          f"dose slope={b['dose_slope']['value']:+.5f} "
          f"CI {b['dose_slope']['ci']}")

    y = np.concatenate(pool_auc["y"]); p = np.concatenate(pool_auc["p"])
    rc = np.concatenate(pool_auc["rank_c"])
    dv = np.concatenate(pool_auc["driven"])
    cs = np.concatenate(pool_auc["consumed"])
    pooled_terc = bucket_auc(y, p, rc)
    ok = np.isfinite(rc)
    a_dc = float(roc_auc_score(y[ok & dv & cs], p[ok & dv & cs]))
    a_dr = float(roc_auc_score(y[ok & dv & ~cs], p[ok & dv & ~cs]))
    print("  pooled AUC | prev-bar cons-rank terciles: "
          + " ".join(f"{pooled_terc[k]['auc']:.4f}" for k in ("low", "mid", "high"))
          + f"   driven&consumed {a_dc:.4f} vs driven&replenished {a_dr:.4f}")

    summary["pooled"] = {"boot": b, "auc_by_cons_rank_tercile": pooled_terc,
                         "auc_driven_consumed": a_dc,
                         "auc_driven_replenished": a_dr,
                         "n_driven_consumed": int((ok & dv & cs).sum()),
                         "n_driven_replenished": int((ok & dv & ~cs).sum())}
    OUT.mkdir(exist_ok=True)
    (OUT / "depth_test.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'depth_test.json'}")


if __name__ == "__main__":
    main()
