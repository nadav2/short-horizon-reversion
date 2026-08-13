"""Forced-flow identification: reversal after liquidation-driven moves.

The mechanism section's flow conditioning cannot separate compensated
liquidity provision from informational stories, because ordinary aggressor
flow chooses to trade. Forced liquidation orders do not: they are triggered
mechanically by margin breaches, carry no private information beyond the
price path that triggered them, and demand immediacy unconditionally. The
identifying contrast (the intraday analogue of fire-sale identification,
Coval-Stafford style): among flow-driven bars of MATCHED move size and
aggressor imbalance, the informational content of the move decreases in its
forced share while the immediacy demanded is constant, so

  - liquidity-provision compensation predicts next-bar reversal at least as
    strong, and increasing in forced intensity;
  - information-based accounts predict the concession after forced moves to
    be smaller or absent (nothing was learned, nothing needs unwinding), so
    reversal FLAT or decreasing in forced share after matching.

Tests, per coin and pooled, on both tapes (CM: same-tape forced orders;
UM: the paper's main tape conditioned on the same-minute CM cascades):

1. REPLICATION: flip(driven)-flip(opposed) inside the archive window.
2. FIRST STAGE: forced orders predict move-aligned aggressor flow.
3. GRADIENT: among flow-driven bars, flip rate by forced-intensity class
   (none / below-median / above-median), matched within |r| x |imb| cells;
   weighted delta with moving-block bootstrap CIs.
4. PLACEBO: forced orders OPPOSING the move (no immediacy consumed in the
   move direction) should carry no extra reversal.
5. MAGNITUDE: next-bar counter-move return, -s_t r_{t+1}, by forced class.
6. ROBUSTNESS: trailing-volatility strata replacing |imb| strata, and the
   headline gradient excluding the top-1% |r| days (cascade outliers).
7. MODEL: walk-forward constrained-logit OOS AUC by prior-bar class.

Data limits stated where used: the archive is Binance's throttled
forced-order stream (<=1 order/s/symbol), so intensity is a lower-bound
measure; the archive exists for COIN-margined perps only and ends
2024-10-14, so the window predates the paper span (the phenomenon is
stable across 2021-2026; Appendix multiyear).

    uv run --active python -m paper.liq_test
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .compare_markets import block_boot_idx
from .models import IsingLogit
from .walkforward import walk_forward

HERE = Path(__file__).resolve().parent
LIQ = HERE / "liq_data"
OUT = HERE / "out"
COINS = ["btc", "eth", "sol", "xrp"]
CSIZE = {"btc": 100.0, "eth": 10.0, "sol": 10.0, "xrp": 10.0}
N_LAGS = 12
BLOCK, N_BOOT, SEED = 384, 2000, 7
WINDOWS_15M = (5760, 960)


def load(coin, tape):
    raw = json.loads((LIQ / f"{coin}-{tape}-15m.json").read_text())
    raw.sort(key=lambda d: d["timestamp"])
    ts = np.array([d["timestamp"] for d in raw], np.int64)
    ch = np.array([d["change"] for d in raw])
    ups = np.array([bool(d["up"]) for d in raw])
    vol = np.array([d["vol"] for d in raw])
    tbv = np.array([d["tbv"] for d in raw])
    close = np.array([d["close"] for d in raw])
    with np.errstate(divide="ignore", invalid="ignore"):
        imb = np.where(vol > 0, 2 * tbv / vol - 1, np.nan)
    dollar_vol = vol * CSIZE[coin] if tape == "cm" else vol * close

    liq = {r["timestamp"]: r
           for r in json.loads((LIQ / f"{coin}-liq-15m.json").read_text())}
    sell = np.array([liq.get(int(t), {}).get("sell_notional", 0.0) for t in ts])
    buy = np.array([liq.get(int(t), {}).get("buy_notional", 0.0) for t in ts])

    s = np.sign(ch)
    # forced flow aligned with the move: forced selling in a down bar,
    # forced buying in an up bar; and the opposing side
    F = np.where(s < 0, sell, np.where(s > 0, buy, 0.0))
    O = np.where(s < 0, buy, np.where(s > 0, sell, 0.0))
    return ch, ups, imb, F, O, dollar_vol


def transitions(ch, imb, F, O):
    """Valid t -> t+1 transitions restricted to FLOW-DRIVEN bars (the
    conditioning the identification refines)."""
    s = np.sign(ch)
    t = np.arange(len(ch) - 1)
    ok = (s[t] != 0) & (s[t + 1] != 0) & np.isfinite(imb[t]) & (imb[t] != 0)
    t = t[ok]
    driven = s[t] * imb[t] > 0
    flip = (s[t + 1] == -s[t])
    counter = -s[t] * ch[t + 1]  # next-bar return against the move
    return t, driven, flip, counter, np.abs(ch[t]), np.abs(imb[t]), F[t], O[t]


def forced_class(F, med):
    """0 = no forced orders, 1 = below-median intensity, 2 = above."""
    cls = np.zeros(len(F), dtype=int)
    cls[(F > 0) & (F <= med)] = 1
    cls[F > med] = 2
    return cls


def cell_ids(absr, absimb, q_r, q_i):
    return np.digitize(absr, q_r) * 8 + np.digitize(absimb, q_i)


def matched_delta(flip, cls, cells, hi_cls):
    """Weighted flip-rate difference, class hi_cls vs class 0, matched
    within cells; weight = per-cell harmonic arm size."""
    num = den = 0.0
    for c in np.unique(cells):
        a = (cells == c) & (cls == hi_cls)
        b = (cells == c) & (cls == 0)
        na, nb = a.sum(), b.sum()
        if na >= 20 and nb >= 20:
            w = 2 * na * nb / (na + nb)
            num += w * (flip[a].mean() - flip[b].mean())
            den += w
    return float(num / den) if den else float("nan")


def gradient_stats(flip, cls, cells):
    d1 = matched_delta(flip, cls, cells, 1)
    d2 = matched_delta(flip, cls, cells, 2)
    return d1, d2, (d2 - d1 if np.isfinite(d1) and np.isfinite(d2)
                    else float("nan"))


def boot_gradient(flip, cls, cells, rng, lens=None):
    d0 = gradient_stats(flip, cls, cells)
    lens = lens or [len(flip)]
    offs = np.cumsum([0] + lens[:-1])
    reps = []
    for _ in range(N_BOOT):
        bi = np.concatenate([o + block_boot_idx(n, BLOCK, rng)
                             for o, n in zip(offs, lens)])
        reps.append(gradient_stats(flip[bi], cls[bi], cells[bi]))
    reps = np.array(reps, dtype=float)
    out = {}
    for i, name in enumerate(("delta_low", "delta_high", "monotone_step")):
        col = reps[:, i][np.isfinite(reps[:, i])]
        if len(col) < 100:
            out[name] = {"value": d0[i], "ci": None, "p_le_0": None}
            continue
        lo, hi = np.percentile(col, [2.5, 97.5])
        out[name] = {"value": d0[i], "ci": [float(lo), float(hi)],
                     "p_le_0": float(np.mean(col <= 0))}
    return out


def rate(x, mask):
    n = int(mask.sum())
    return {"rate": float(x[mask].mean()) if n else float("nan"), "n": n}


def analyze_tape(coin, tape, label=""):
    ch, ups, imb, F, O, dvol = load(coin, tape)
    t, driven, flip, counter, absr, absimb, F_t, O_t = transitions(ch, imb, F, O)

    # 1. replication inside the window
    repl = {"flip_driven": rate(flip, driven), "flip_opposed": rate(flip, ~driven)}

    # 2. first stage: forced orders align aggressor flow with the move
    liq_any = F_t > 0
    first = {"p_driven_given_liq": rate(driven.astype(float), liq_any),
             "p_driven_given_noliq": rate(driven.astype(float), ~liq_any)}

    # 3. matched gradient among driven bars
    d = driven
    med = np.median(F_t[d & (F_t > 0)]) if (d & (F_t > 0)).sum() > 50 else np.inf
    cls = forced_class(F_t, med)
    q_r = np.quantile(absr[d], [0.2, 0.4, 0.6, 0.8])
    q_i = np.quantile(absimb[d], [0.2, 0.4, 0.6, 0.8])
    cells = cell_ids(absr, absimb, q_r, q_i)
    rng = np.random.default_rng(SEED)
    grad = boot_gradient(flip[d], cls[d], cells[d], rng)
    raw_rates = [rate(flip, d & (cls == k)) for k in (0, 1, 2)]

    # 4. placebo: forced orders against the move, none with it
    pl_mask = d & (O_t > 0) & (F_t == 0)
    pl_cells = cells.copy()
    pl_cls = np.where(pl_mask, 2, np.where(d & (F_t == 0) & (O_t == 0), 0, -1))
    placebo = matched_delta(flip[d], pl_cls[d], pl_cells[d], 2)

    # 5. magnitude in return units (bp of counter-move next bar)
    mag = [float(counter[d & (cls == k)].mean() * 1e4) for k in (0, 1, 2)]

    # 6a. robustness: trailing-vol strata instead of |imb|
    trail = np.full(len(ch), np.nan)
    w = 96
    absch = np.abs(ch)
    c = np.cumsum(np.insert(absch, 0, 0.0))
    trail[w:] = (c[w:-1] - c[:-w-1]) / w if len(c) > w + 1 else np.nan
    tr_t = trail[t]
    okv = np.isfinite(tr_t)
    q_v = np.quantile(tr_t[d & okv], [1/3, 2/3])
    cells_v = np.digitize(absr, q_r) * 4 + np.digitize(tr_t, q_v)
    grad_vol = boot_gradient(flip[d & okv], cls[d & okv], cells_v[d & okv],
                             np.random.default_rng(SEED))

    # 6b. robustness: drop top-1% |r| days (cascade outliers)
    day = t // 96
    bigday = np.unique(day[absr >= np.quantile(absr, 0.99)])
    keep = ~np.isin(day, bigday)
    grad_trim = boot_gradient(flip[d & keep], cls[d & keep], cells[d & keep],
                              np.random.default_rng(SEED))

    return {
        "label": label or tape, "n_transitions": int(len(t)),
        "n_driven": int(d.sum()),
        "n_liq_driven": int((d & liq_any).sum()),
        "replication": repl, "first_stage": first,
        "flip_by_forced_class_raw": raw_rates,
        "matched_gradient": grad,
        "placebo_opposed_liq": placebo,
        "countermove_bp_by_class": mag,
        "gradient_volstrata": grad_vol,
        "gradient_extrim": grad_trim,
    }, (flip[d], cls[d], cells[d])


def model_auc(coin):
    """Walk-forward constrained logit on the UM tape; OOS AUC by prior-bar
    class (driven+forced / driven ordinary / opposed)."""
    ch, ups, imb, F, O, _ = load(coin, "um")
    tr, te = WINDOWS_15M
    res, _ = walk_forward(ch, ups, tr, te, n_lags=N_LAGS,
                          models=lambda: [IsingLogit(n_lags=N_LAGS)])
    r = res["ising"]
    idx = r["idx"].astype(int)
    y, p = r["actuals"].astype(int), r["probs"]
    prev = idx - 1
    s_prev = np.sign(ch[prev])
    ok = (s_prev != 0) & np.isfinite(imb[prev]) & (imb[prev] != 0)
    driven = ok & (s_prev * imb[prev] > 0)
    liq = F[prev] > 0

    def auc(m):
        return (float(roc_auc_score(y[m], p[m]))
                if m.sum() > 100 and len(np.unique(y[m])) > 1 else float("nan"),
                int(m.sum()))

    a_lf, n_lf = auc(driven & liq)
    a_or, n_or = auc(driven & ~liq)
    a_op, n_op = auc(ok & ~driven)
    return {"auc_driven_forced": a_lf, "n_driven_forced": n_lf,
            "auc_driven_ordinary": a_or, "n_driven_ordinary": n_or,
            "auc_opposed": a_op, "n_opposed": n_op}


def main():
    summary = {}
    pooled = {"cm": [], "um": []}
    for coin in COINS:
        summary[coin] = {}
        print(f"\n=== {coin} ===")
        # same-tape design (CM forced orders on the CM tape)
        cm_res, cm_pool = analyze_tape(coin, "cm")
        summary[coin]["cm"] = cm_res
        pooled["cm"].append(cm_pool)
        # cross-tape: UM tape, conditioned on the same-minute CM cascade
        # (load() aligns the CM-sourced liq bars onto the UM grid by timestamp)
        um_res, um_pool = analyze_tape(coin, "um", label="um|cm-cascade")
        summary[coin]["um"] = um_res
        pooled["um"].append(um_pool)
        summary[coin]["model_auc_um"] = model_auc(coin)

        for tape in ("cm", "um"):
            r = summary[coin][tape]
            g = r["matched_gradient"]
            print(f"  [{r['label']}] driven={r['n_driven']} "
                  f"liq-driven={r['n_liq_driven']}  "
                  f"repl dflip={r['replication']['flip_driven']['rate'] - r['replication']['flip_opposed']['rate']:+.4f}")
            print(f"    flip by forced class: "
                  + " ".join(f"{x['rate']:.4f}(n={x['n']})" for x in r["flip_by_forced_class_raw"])
                  + f"  matched d_low={g['delta_low']['value']:+.4f} "
                  f"d_high={g['delta_high']['value']:+.4f} "
                  f"CI_high {g['delta_high']['ci']}")
            print(f"    counter-move bp by class: "
                  + " ".join(f"{x:+.2f}" for x in r["countermove_bp_by_class"])
                  + f"   placebo={r['placebo_opposed_liq']:+.4f}")
        m = summary[coin]["model_auc_um"]
        print(f"    UM model AUC | driven+forced {m['auc_driven_forced']:.4f} "
              f"(n={m['n_driven_forced']})  driven ordinary "
              f"{m['auc_driven_ordinary']:.4f}  opposed {m['auc_opposed']:.4f}")

    # pooled gradients with within-coin blocks
    print("\n=== pooled ===")
    for tape in ("cm", "um"):
        fl = np.concatenate([p[0] for p in pooled[tape]])
        cl = np.concatenate([p[1] for p in pooled[tape]])
        # cells offset per coin so matching never crosses coins
        ce = np.concatenate([p[2] + 100 * i for i, p in enumerate(pooled[tape])])
        lens = [len(p[0]) for p in pooled[tape]]
        g = boot_gradient(fl, cl, ce, np.random.default_rng(SEED), lens=lens)
        summary[f"pooled_{tape}"] = g
        print(f"  [{tape}] matched d_low={g['delta_low']['value']:+.4f} "
              f"CI {g['delta_low']['ci']}  d_high={g['delta_high']['value']:+.4f} "
              f"CI {g['delta_high']['ci']} p(<=0)={g['delta_high']['p_le_0']}")

    OUT.mkdir(exist_ok=True)
    (OUT / "liq_test.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'liq_test.json'}")


if __name__ == "__main__":
    main()
