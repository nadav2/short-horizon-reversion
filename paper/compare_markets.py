"""Crypto vs. equity-index comparison for the reframed paper.

Question: is short-horizon *directional predictability* — and specifically the
power-law mean-reversion memory captured by the kinetic Ising model — systematically
STRONGER in crypto than in (continuously-traded) equity indices?

Fairness controls:
  * Same calendar span for every asset (the overlap window where all have data).
  * Same walk-forward window geometry (train/test candle counts) per interval for
    every asset, so each model sees equal information.
  * Same models, same MLE fitting.

Significance: a moving-block bootstrap (block length respects residual autocorrelation)
on the concatenated out-of-sample series gives CIs for AUC and for the log-loss skill
over the base rate, plus a paired Ising-minus-free test.

    uv run --active python -m paper.compare_markets
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import DATA_DIR, classification_metrics
from .models import BaseRate, ARLogit, IsingLogit, MarkovModel
from .walkforward import walk_forward

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

CRYPTO = ["btc", "eth", "sol", "xrp"]
FX = ["eurusd", "usdjpy", "gbpusd"]
TRADITIONAL = ["spx", "qqq", "iwm", "aapl", "nvda", "tsla", "gld", "tlt"] + FX
EQUITY = TRADITIONAL                        # alias kept for back-compat
ASSET_CLASS = {
    "btc": "crypto", "eth": "crypto", "sol": "crypto", "xrp": "crypto",
    "spx": "index", "qqq": "index", "iwm": "index",
    "aapl": "stock", "nvda": "stock", "tsla": "stock",
    "gld": "commodity", "tlt": "bond",
    "eurusd": "fx", "usdjpy": "fx", "gbpusd": "fx",
}
INTERVALS = ["15m", "1h", "4h"]            # no 5m for traditional assets
# Common calendar span (the equity window); restricts every asset to the same regime.
SPAN = ("2025-01-01", "2026-02-11")
# Matched walk-forward geometry per interval, sized so the shortest (equity-4h) series
# still yields several folds. (train candles, test candles).
WINDOWS = {"15m": (5760, 960), "1h": (2160, 360), "4h": (720, 120)}
BLOCK = {"15m": 384, "1h": 96, "4h": 30}   # ~4 trading days, for the block bootstrap
N_BOOT = 1000
SEED = 7


def load_span(asset, interval, span=SPAN):
    """Load a single source (base file) restricted to the common span."""
    path = DATA_DIR / f"{asset}-{interval}.json"
    raw = json.loads(path.read_text())
    raw = [d for d in raw if span[0] <= d["datetime"][:10] <= span[1]]
    raw.sort(key=lambda d: d["timestamp"])
    dts = [d["datetime"] for d in raw]
    ch = np.array([d.get("change", 0.0) for d in raw], dtype=float)
    ups = np.array([bool(d["up"]) for d in raw], dtype=bool)
    return dts, ch, ups


def models():
    return [IsingLogit(n_lags=12), ARLogit("free", n_lags=12),
            MarkovModel(order=1), BaseRate()]


def block_boot_idx(n, block, rng):
    """Indices for one moving-block bootstrap resample of length ~n."""
    nblocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=nblocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return idx


def autocorr1(x):
    """Lag-1 autocorrelation of the return series (model-free mean-reversion proxy).
    Negative => anti-persistent / mean-reverting."""
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = np.sum(x * x)
    return float(np.sum(x[1:] * x[:-1]) / denom) if denom > 0 else 0.0


def evaluate(asset, interval):
    dts, ch, ups = load_span(asset, interval)
    tr, te = WINDOWS[interval]
    if len(ch) < tr + te:
        return None
    ac1 = autocorr1(ch)
    res, n_folds = walk_forward(ch, ups, tr, te, n_lags=12, models=models)
    if len(res["ising"]["probs"]) == 0:
        return None

    y = res["ising"]["actuals"].astype(int)
    p_is = res["ising"]["probs"]
    p_fr = res["free"]["probs"]
    p_base = res["base"]["probs"]
    eps = 1e-15

    def ll_vec(p):
        pc = np.clip(p, eps, 1 - eps)
        return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))

    ll_is, ll_fr, ll_base = ll_vec(p_is), ll_vec(p_fr), ll_vec(p_base)
    skill = ll_base - ll_is          # >0 means Ising beats the base rate (per obs, nats)
    skill_free = ll_base - ll_fr     # >0 means the FREE logit beats the base rate
    diff_isf = ll_fr - ll_is         # >0 means Ising beats the free logit (per obs)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, p_is)
    auc_free = roc_auc_score(y, p_fr)

    rng = np.random.default_rng(SEED)
    n = len(y)
    boot_auc, boot_auc_free, boot_skill, boot_skill_free, boot_diff = [], [], [], [], []
    for _ in range(N_BOOT):
        idx = block_boot_idx(n, BLOCK[interval], rng)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        boot_auc.append(roc_auc_score(yb, p_is[idx]))
        boot_auc_free.append(roc_auc_score(yb, p_fr[idx]))
        boot_skill.append(skill[idx].mean())
        boot_skill_free.append(skill_free[idx].mean())
        boot_diff.append(diff_isf[idx].mean())
    boot_auc, boot_auc_free, boot_skill, boot_skill_free, boot_diff = \
        map(np.array, (boot_auc, boot_auc_free, boot_skill, boot_skill_free, boot_diff))

    isg = classification_metrics(p_is, y)
    fr = classification_metrics(p_fr, y)
    A = float(np.mean([p["A"] for p in res["ising"]["fold_params"]]))
    alpha = float(np.mean([p["alpha"] for p in res["ising"]["fold_params"]]))

    return {
        "asset": asset, "interval": interval,
        "class": "crypto" if asset in CRYPTO else "traditional",
        "asset_class": ASSET_CLASS.get(asset, "?"), "ac1": ac1,
        "n_oos": int(n), "n_folds": n_folds, "base_rate_up": float(y.mean()),
        "span": [dts[0][:10], dts[-1][:10]],
        "ising_auc": float(auc), "ising_auc_ci": [float(np.percentile(boot_auc, 2.5)),
                                                  float(np.percentile(boot_auc, 97.5))],
        "ising_acc": isg["accuracy"], "free_acc": fr["accuracy"],
        "ising_logloss": isg["log_loss"], "free_logloss": fr["log_loss"],
        "skill_vs_base_nats": float(skill.mean() * 1000),   # in milli-nats
        "skill_ci": [float(np.percentile(boot_skill, 2.5) * 1000),
                     float(np.percentile(boot_skill, 97.5) * 1000)],
        "skill_p_gt0": float(np.mean(boot_skill <= 0)),     # bootstrap p that skill <= 0
        "free_auc": float(auc_free),
        "free_auc_p_gt05": float(np.mean(boot_auc_free <= 0.5)),   # one-sided p that free AUC <= 0.5
        "ising_auc_p_gt05": float(np.mean(boot_auc <= 0.5)),
        "free_skill_vs_base_nats": float(skill_free.mean() * 1000),
        "free_skill_p_gt0": float(np.mean(boot_skill_free <= 0)),
        "ising_minus_free_nats": float(diff_isf.mean() * 1000),
        "isf_p_gt0": float(np.mean(boot_diff <= 0)),
        "A": A, "alpha": alpha,
    }


def main():
    rows = []
    for asset in CRYPTO + EQUITY:
        for interval in INTERVALS:
            r = evaluate(asset, interval)
            if r is None:
                print(f"  {asset}-{interval:4s} skipped"); continue
            rows.append(r)
            sig = "*" if r["skill_p_gt0"] < 0.05 else " "
            fsig = "*" if r["free_skill_p_gt0"] < 0.05 else " "
            print(f"  {asset}-{interval:4s} [{r['asset_class']:9s}] n={r['n_oos']:6d}  "
                  f"AUC={r['ising_auc']:.4f} [{r['ising_auc_ci'][0]:.3f},{r['ising_auc_ci'][1]:.3f}]  "
                  f"isingP={r['skill_p_gt0']:.3f}{sig} freeP={r['free_skill_p_gt0']:.3f}{fsig}  "
                  f"A={r['A']:+.3f} α={r['alpha']:.2f}")

    (OUT / "markets.json").write_text(json.dumps(rows, indent=2))

    # Aggregate crypto vs equity
    def agg(sel, label):
        if not sel:
            return
        auc = np.mean([r["ising_auc"] for r in sel])
        skill = np.mean([r["skill_vs_base_nats"] for r in sel])
        A = np.mean([r["A"] for r in sel])
        alpha = np.mean([r["alpha"] for r in sel])
        ac1 = np.mean([r["ac1"] for r in sel])
        nsig = sum(1 for r in sel if r["skill_p_gt0"] < 0.05)
        print(f"  {label:22s}: AUC={auc:.4f}  skill={skill:+.3f}mn  A={A:+.3f}  "
              f"α={alpha:.2f}  ac1={ac1:+.3f}  ({nsig}/{len(sel)} skill p<0.05)")

    print("\n=== crypto vs traditional, by interval ===")
    for interval in INTERVALS + ["ALL"]:
        for cls in ("crypto", "traditional"):
            sel = [r for r in rows if r["class"] == cls and (interval == "ALL" or r["interval"] == interval)]
            agg(sel, f"{interval} {cls}")

    print("\n=== by asset class (15m only) ===")
    for ac in ("crypto", "index", "stock", "commodity", "bond", "fx"):
        agg([r for r in rows if r["asset_class"] == ac and r["interval"] == "15m"], f"15m {ac}")

    print("\n=== model-agnostic corroboration (15m): AUC>0.5 significance (one-sided p<0.05) ===")
    for cls in ("crypto", "traditional"):
        sel = [r for r in rows if r["class"] == cls and r["interval"] == "15m"]
        ni = sum(1 for r in sel if r["ising_auc_p_gt05"] < 0.05)
        nf = sum(1 for r in sel if r["free_auc_p_gt05"] < 0.05)
        nb = sum(1 for r in sel if r["ising_auc_p_gt05"] < 0.05 and r["free_auc_p_gt05"] < 0.05)
        print(f"  {cls:12s}: Ising AUC>0.5 {ni}/{len(sel)}, free-logit AUC>0.5 {nf}/{len(sel)}, "
              f"BOTH {nb}/{len(sel)}")
    print(f"\nWrote {OUT/'markets.json'}")


if __name__ == "__main__":
    main()
