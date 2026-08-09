"""Wide cross-market study over HUNDREDS of assets (bulk_data/).

Replaces the 4-vs-8 comparison with the full universe: ~200 Binance crypto pairs vs
~250 liquid US stocks/ETFs, matched span and walk-forward. For each asset we compute
the 15m out-of-sample Ising AUC and a moving-block-bootstrap one-sided p that AUC>0.5,
plus the free-logit AUC for model-agnostic corroboration. We then report, per class,
the fraction of assets with significant predictability and the AUC distribution.

    uv run --active python -m paper.wide
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .fetch_bulk_stocks import UNIVERSE
from .models import ARLogit, IsingLogit
from .simulate import simulate
from .walkforward import walk_forward

BULK = Path(__file__).resolve().parent / "bulk_data"
OUT = Path(__file__).resolve().parent / "out"
INTERVAL = "15m"
TRAIN, TEST = 5760, 960          # matched 15m geometry (as in compare_markets)
BLOCK, N_BOOT, SEED = 384, 300, 5
IDX_PER_YEAR = 96 * 365          # 15m candles per year
STOCK_SET = {s.lower().replace(".", "-") for s in UNIVERSE}


def load(path):
    raw = json.loads(path.read_text())
    raw.sort(key=lambda d: d["timestamp"])
    ch = np.array([d.get("change", 0.0) for d in raw], float)
    ups = np.array([bool(d["up"]) for d in raw], bool)
    ts = np.array([d["timestamp"] for d in raw], np.int64)
    return ch, ups, ts


def block_idx(n, rng):
    nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, max(1, n - BLOCK + 1), size=nb)
    return np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]


def evaluate(path, dump_dir: Path | None = None, asset: str = ""):
    ch, ups, ts = load(path)
    if len(ch) < TRAIN + TEST + 2000:
        return None
    # ising + free AR(12) + AR(1) market reference (distinct short key), one walk-forward
    def factory():
        ar1 = ARLogit("free", n_lags=1); ar1.short = "ar1"; ar1.name = "AR(1) ref"
        return [IsingLogit(n_lags=12), ARLogit("free", n_lags=12), ar1]
    res, nf = walk_forward(ch, ups, TRAIN, TEST, n_lags=12, models=factory)
    y = res["ising"]["actuals"].astype(int)
    if len(y) < 2000 or len(np.unique(y)) < 2:
        return None
    p_is, p_fr = res["ising"]["probs"], res["free"]["probs"]
    if dump_dir is not None:
        # per-asset OOS series keyed by candle timestamp, for the joint
        # (dependence-aware) bootstrap in paper.dependence
        np.savez_compressed(dump_dir / f"{asset}.npz",
                            ts=ts[res["ising"]["idx"].astype(int)],
                            actual=y.astype(np.int8),
                            p_ising=p_is.astype(np.float32),
                            p_free=p_fr.astype(np.float32))
    auc_is, auc_fr = roc_auc_score(y, p_is), roc_auc_score(y, p_fr)
    A = float(np.mean([fp["A"] for fp in res["ising"]["fold_params"]]))
    rng = np.random.default_rng(SEED)
    ge_is = ge_fr = 0
    for _ in range(N_BOOT):
        idx = block_idx(len(y), rng)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        ge_is += roc_auc_score(yb, p_is[idx]) <= 0.5
        ge_fr += roc_auc_score(yb, p_fr[idx]) <= 0.5
    # trading sim vs a semi-efficient market that prices the AR(1) signal (rho=1, hardest)
    ref = res["ar1"]["probs"]
    m = ref
    sim_is, _ = simulate(p_is, y, m, IDX_PER_YEAR)
    sim_fr, _ = simulate(p_fr, y, m, IDX_PER_YEAR)
    return {"auc_ising": float(auc_is), "auc_free": float(auc_fr), "A": A,
            "p_ising": ge_is / N_BOOT, "p_free": ge_fr / N_BOOT,
            "sharpe_ising": sim_is["sharpe"], "sharpe_free": sim_fr["sharpe"],
            "annret_ising": sim_is["ann_return"], "annret_free": sim_fr["ann_return"],
            "n_oos": int(len(y)), "n_folds": nf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-oos", action="store_true",
                    help="save per-asset OOS prediction series to out/wide_oos/")
    args = ap.parse_args()
    dump_dir = None
    if args.dump_oos:
        dump_dir = OUT / "wide_oos"
        dump_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(BULK.glob(f"*-{INTERVAL}.json"))
    print(f"{len(files)} assets in {BULK}")
    rows = []
    for i, f in enumerate(files):
        base = f.name[:-(len(INTERVAL) + 6)]            # strip '-15m.json'
        cls = "stock" if base in STOCK_SET else "crypto"
        try:
            r = evaluate(f, dump_dir=dump_dir, asset=base)
        except Exception as e:
            print(f"  {base}: FAILED {e}"); continue
        if r is None:
            continue
        r.update(asset=base, **{"class": cls})
        rows.append(r)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(files)}] processed; kept {len(rows)}")
    (OUT / "wide.json").write_text(json.dumps(rows, indent=2))

    print("\n=== WIDE CROSS-MARKET SUMMARY (15m) ===")
    for cls in ("crypto", "stock"):
        sub = [r for r in rows if r["class"] == cls]
        if not sub:
            continue
        aucs = np.array([r["auc_ising"] for r in sub])
        As = np.array([r["A"] for r in sub])
        sig_is = np.mean([r["p_ising"] < 0.05 for r in sub])
        sig_both = np.mean([r["p_ising"] < 0.05 and r["p_free"] < 0.05 for r in sub])
        print(f"\n{cls.upper()}  (n={len(sub)} assets)")
        print(f"  Ising AUC: mean={aucs.mean():.4f} median={np.median(aucs):.4f} "
              f"[Q1={np.percentile(aucs,25):.4f}, Q3={np.percentile(aucs,75):.4f}]  "
              f"frac AUC>0.5={np.mean(aucs>0.5):.2f}")
        print(f"  significant AUC>0.5: Ising {sig_is*100:.0f}% of assets, "
              f"both-models {sig_both*100:.0f}%")
        print(f"  coupling A: mean={As.mean():+.3f}  frac A<0 (mean-reverting)={np.mean(As<0):.2f}")
        sh_is = np.array([r["sharpe_ising"] for r in sub if r["sharpe_ising"] == r["sharpe_ising"]])
        sh_fr = np.array([r["sharpe_free"] for r in sub if r["sharpe_free"] == r["sharpe_free"]])
        win = np.mean([r["sharpe_ising"] > r["sharpe_free"] for r in sub
                       if r["sharpe_ising"] == r["sharpe_ising"] and r["sharpe_free"] == r["sharpe_free"]])
        print(f"  bot Sharpe (ρ=1): Ising median={np.median(sh_is):.2f}, free median={np.median(sh_fr):.2f}; "
              f"Ising>free in {win*100:.0f}% of assets")
    print(f"\nWrote {OUT/'wide.json'}")


if __name__ == "__main__":
    main()
