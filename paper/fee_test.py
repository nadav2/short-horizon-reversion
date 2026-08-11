"""Zero-fee natural experiment: Binance eliminated spot trading fees on 13 BTC
pairs (including BTC/USDT) from 2022-07-08 14:00 UTC to 2023-03-22 00:00 UTC,
while ETH/SOL/XRP pairs kept normal fees throughout. If the reversal is a
residual priced just below the marginal arbitrageur's round-trip cost
(limits-to-arbitrage), removing the fee leg should compress BTC's edge during
the window relative to the controls; if zero fees instead amplified the
uninformed-flow share (zero-fee pairs reached ~85% of volume), the edge could
widen. The sign of the difference-in-differences is therefore informative
about which force dominates.

Design: standard matched walk-forward (5760/960) over 2021-01-01..2024-03-22
per coin; OOS predictions bucketed into PRE / ZERO / POST by bar timestamp;
per coin x period: Ising OOS AUC, free-logit AUC, fitted coupling A (fold
level), and the model-free lag-1 sign flip rate. DiD on excess AUC:
    effect_c = e_zero - (e_pre + e_post)/2,   DiD = effect_btc - mean(controls)
with a JOINT moving-block bootstrap (same time blocks applied to every coin,
preserving cross-coin dependence) for the CI.

    uv run --active python -m paper.fee_test
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .compare_markets import block_boot_idx
from .models import ARLogit, IsingLogit
from .walkforward import walk_forward

DATA = Path(__file__).resolve().parent / "multiyear_data"
OUT = Path(__file__).resolve().parent / "out"
COINS = ["btc", "eth", "sol", "xrp"]
CONTROLS = ["eth", "sol", "xrp"]
TRAIN, TEST = 5760, 960
BLOCK, N_BOOT, SEED = 384, 1000, 7

T_ZERO_START = int(datetime(2022, 7, 8, 14, tzinfo=timezone.utc).timestamp())
T_ZERO_END = int(datetime(2023, 3, 22, tzinfo=timezone.utc).timestamp())
T_POST_END = int(datetime(2024, 3, 22, tzinfo=timezone.utc).timestamp())
PERIODS = ["pre", "zero", "post"]


def period_of(ts):
    ts = np.asarray(ts)
    out = np.full(ts.shape, "post", dtype=object)
    out[ts < T_ZERO_END] = "zero"
    out[ts < T_ZERO_START] = "pre"
    return out


def models():
    return [IsingLogit(n_lags=12), ARLogit("free", n_lags=12)]


def score_coin(coin):
    raw = json.loads((DATA / f"{coin}-15m.json").read_text())
    raw = [d for d in raw if d["timestamp"] < T_POST_END]
    raw.sort(key=lambda d: d["timestamp"])
    ch = np.array([d.get("change", 0.0) for d in raw], float)
    ups = np.array([bool(d["up"]) for d in raw], bool)
    ts = np.array([d["timestamp"] for d in raw], np.int64)

    res, nf = walk_forward(ch, ups, TRAIN, TEST, n_lags=12, models=models)
    idx = res["ising"]["idx"].astype(int)
    oos_ts = ts[idx]
    per = period_of(oos_ts)
    fold_A = np.array([fp["A"] for fp in res["ising"]["fold_params"]])
    fold_per = per[::TEST][:len(fold_A)]

    # model-free lag-1 flip rate on the raw sign series, by period
    s = np.sign(ch)
    t = np.arange(len(s) - 1)
    ok = (s[t] != 0) & (s[t + 1] != 0)
    flip_per = period_of(ts[t[ok]])
    flips = (s[t + 1] == -s[t])[ok]

    rows = {}
    for p in PERIODS:
        sel = per == p
        y = res["ising"]["actuals"][sel].astype(int)
        rows[p] = {
            "n_oos": int(sel.sum()),
            "auc_ising": float(roc_auc_score(y, res["ising"]["probs"][sel])),
            "auc_free": float(roc_auc_score(y, res["free"]["probs"][sel])),
            "A": float(fold_A[fold_per == p].mean()),
            "flip": float(flips[flip_per == p].mean()),
            "n_flip": int((flip_per == p).sum()),
        }
    return {"rows": rows,
            "oos": {"ts": oos_ts, "per": per,
                    "y": res["ising"]["actuals"].astype(int),
                    "p": res["ising"]["probs"]}}


def did(excess):
    """excess: {coin: {period: e}} -> DiD point estimate."""
    eff = {c: excess[c]["zero"] - (excess[c]["pre"] + excess[c]["post"]) / 2
           for c in COINS}
    return eff["btc"] - np.mean([eff[c] for c in CONTROLS]), eff


def main():
    results, oos = {}, {}
    for coin in COINS:
        r = score_coin(coin)
        results[coin] = r["rows"]
        oos[coin] = r["oos"]
        row = r["rows"]
        print(f"  {coin:4s} " + "  ".join(
            f"{p}: AUC={row[p]['auc_ising']:.4f} free={row[p]['auc_free']:.4f} "
            f"A={row[p]['A']:+.3f} flip={row[p]['flip']:.4f}" for p in PERIODS))

    # ---- align coins on common OOS timestamps for the joint bootstrap
    common = set(oos[COINS[0]]["ts"].tolist())
    for c in COINS[1:]:
        common &= set(oos[c]["ts"].tolist())
    common = np.array(sorted(common), np.int64)
    aligned = {}
    for c in COINS:
        pos = {t: i for i, t in enumerate(oos[c]["ts"].tolist())}
        sel = np.array([pos[t] for t in common.tolist()])
        aligned[c] = {"y": oos[c]["y"][sel], "p": oos[c]["p"][sel]}
    per_common = period_of(common)
    print(f"\n  joint bootstrap on {len(common)} common OOS bars "
          f"({', '.join(f'{p}:{int((per_common == p).sum())}' for p in PERIODS)})")

    excess = {c: {p: results[c][p]["auc_ising"] - 0.5 for p in PERIODS}
              for c in COINS}
    d0, eff = did(excess)
    print("  effect (zero - mean(pre,post), excess AUC): "
          + "  ".join(f"{c}={eff[c]:+.4f}" for c in COINS))
    print(f"  DiD (btc - mean controls) = {d0:+.4f}")

    rng = np.random.default_rng(SEED)
    pidx = {p: np.where(per_common == p)[0] for p in PERIODS}
    boots = []
    for _ in range(N_BOOT):
        exb = {c: {} for c in COINS}
        okdraw = True
        for p in PERIODS:
            ii = pidx[p]
            bi = ii[block_boot_idx(len(ii), BLOCK, rng)]
            for c in COINS:
                yb = aligned[c]["y"][bi]
                if yb.min() == yb.max():
                    okdraw = False
                    break
                exb[c][p] = roc_auc_score(yb, aligned[c]["p"][bi]) - 0.5
            if not okdraw:
                break
        if okdraw:
            boots.append(did(exb)[0])
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_two = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    print(f"  DiD bootstrap: mean {boots.mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]"
          f"  two-sided p={p_two:.3f}  (n_eff={len(boots)})")

    # model-free corroboration: DiD on flip rates (higher flip = more reversal)
    flip_excess = {c: {p: results[c][p]["flip"] - 0.5 for p in PERIODS} for c in COINS}
    dflip, eff_flip = did(flip_excess)
    print("  flip-rate effect: " + "  ".join(f"{c}={eff_flip[c]:+.4f}" for c in COINS))
    print(f"  flip-rate DiD = {dflip:+.4f}")

    summary = {"periods": {"zero_start_utc": "2022-07-08T14:00Z",
                           "zero_end_utc": "2023-03-22T00:00Z",
                           "post_end_utc": "2024-03-22T00:00Z"},
               "per_coin": results,
               "excess_auc_effects": eff,
               "did_auc": {"point": float(d0), "boot_mean": float(boots.mean()),
                           "ci": [float(lo), float(hi)], "p": float(p_two)},
               "flip_effects": eff_flip, "did_flip": float(dflip)}
    OUT.mkdir(exist_ok=True)
    (OUT / "fee_test.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'fee_test.json'}")


if __name__ == "__main__":
    main()
