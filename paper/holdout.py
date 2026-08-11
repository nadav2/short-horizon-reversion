"""Post-sample holdout: the frozen primary pipeline on data after Feb 2026.

The primary sample ends 2026-02-11. Everything here scores the identical
frozen universe (the 370 assets of out/wide.json) on the untouched period
that accrued afterwards, with the identical models, walk-forward geometry,
and per-asset bootstrap as wide.py. Because the holdout is ~6 months, the
walk-forward yields ~11 OOS folds per crypto pair but only ~1--2k OOS bars
per stock; the only deviations from wide.py are correspondingly relaxed
minimum-bar cuts (disclosed in the output). The model-free kernel sign
statistic R (alpha == 1, nothing fitted) is computed on every holdout bar
as the estimator-free companion.

Three stages:

  # crypto klines (public Binance API; no keys)
  uv run --active python -m paper.holdout --fetch-crypto
  # stock bars (needs alpaca-py + ALPACA_* env; run from scripts/data/py)
  cd scripts/data/py && dotenvx run -f ../../../.env -- \
      uv run --active python -m paper.holdout --fetch-stocks
  # evaluation + aggregation -> out/holdout.json
  uv run --active python -m paper.holdout --eval [--workers 8]

Writes holdout_data/{sid}-15m.json and out/holdout.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "holdout_data"
OUT = HERE / "out"

START, END = "2026-02-12", "2026-08-08"     # exclusive end, UTC midnight
INTERVAL = "15m"
TRAIN, TEST = 5760, 960                     # identical to wide.py
BLOCK, N_BOOT, SEED = 384, 300, 5           # identical to wide.py
MIN_BARS = TRAIN + TEST + 240               # relaxed from wide.py's +2000
MIN_OOS = 480                               # relaxed from wide.py's 2000
SLOT = 900
GAP_BLOCK, GAP_BOOT, GAP_SEED = 384, 1000, 7
GAP_MIN_OBS = 300                           # relaxed from dependence.py's 1000
P_FLOOR = 1.0 / (N_BOOT + 1)
N_LAGS_R = 12


def universe():
    rows = json.loads((OUT / "wide.json").read_text())
    return {r["asset"]: r["class"] for r in rows}


# ── stage 1: crypto fetch (public Binance) ───────────────────────────────────

def fetch_crypto():
    from .fetch_bulk_crypto import _get, to_ms   # reuse endpoints/retry logic

    DATA.mkdir(exist_ok=True)
    sids = sorted(a for a, c in universe().items() if c == "crypto")
    start_ms = to_ms(START)
    end_ms = to_ms(END)
    done = missing = 0
    for i, sid in enumerate(sids):
        out = DATA / f"{sid}-15m.json"
        if out.exists():
            done += 1
            continue
        sym = sid.upper() + "USDT"
        rows, cur = [], start_ms
        try:
            while cur < end_ms:
                kl = _get("/api/v3/klines",
                          {"symbol": sym, "interval": INTERVAL,
                           "startTime": cur, "endTime": end_ms, "limit": 1000})
                if not kl:
                    break
                for k in kl:
                    o, c = float(k[1]), float(k[4])
                    ts = int(k[0] // 1000)
                    rows.append({"timestamp": ts,
                                 "datetime": datetime.fromtimestamp(
                                     ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                                 "up": c > o, "change": (c - o) / o if o else 0.0})
                cur = kl[-1][0] + 1
                time.sleep(0.05)
                if len(kl) < 1000:
                    break
        except Exception as e:
            print(f"  [{i+1}/{len(sids)}] {sym} FAILED: {e}")
            missing += 1
            continue
        if len(rows) < 1000:
            print(f"  [{i+1}/{len(sids)}] {sym}: only {len(rows)} bars "
                  f"(delisted since selection?)")
            missing += 1
            continue
        out.write_text(json.dumps(rows))
        done += 1
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(sids)}] written {done}")
    print(f"crypto: {done} written, {missing} unavailable")


# ── stage 2: stock fetch (Alpaca; run under scripts/data/py env) ─────────────

def fetch_stocks():
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"],
                                       os.environ["ALPACA_SECRET_KEY"])
    DATA.mkdir(exist_ok=True)
    sids = sorted(a for a, c in universe().items() if c == "stock")
    todo = [s for s in sids if not (DATA / f"{s}-15m.json").exists()]
    print(f"stocks: {len(todo)} of {len(sids)} to fetch")
    BATCH = 40
    done = 0
    for b in range(0, len(todo), BATCH):
        batch = todo[b:b + BATCH]
        syms = [s.upper().replace("-", ".") for s in batch]
        req = StockBarsRequest(
            symbol_or_symbols=syms,
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            start=datetime.fromisoformat(f"{START}T00:00:00"),
            end=datetime.fromisoformat(f"{END}T00:00:00"))
        bars = client.get_stock_bars(req)
        for sid, sym in zip(batch, syms):
            data = bars.data.get(sym, [])
            if len(data) < 1000:
                print(f"  {sym}: only {len(data)} bars, skipped")
                continue
            rows = []
            for bar in sorted(data, key=lambda x: x.timestamp):
                o, c = float(bar.open), float(bar.close)
                ts = int(bar.timestamp.timestamp())
                rows.append({"timestamp": ts,
                             "datetime": bar.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                             "up": c > o, "change": (c - o) / o if o else 0.0})
            (DATA / f"{sid}-15m.json").write_text(json.dumps(rows))
            done += 1
        print(f"  batch {b}-{b+len(batch)}: total written {done}")
    print(f"stocks: {done} written")


# ── stage 3: evaluation ──────────────────────────────────────────────────────

def kernel_R(ch: np.ndarray) -> float:
    """Model-free kernel sign correlation, alpha == 1 fixed a priori."""
    s = np.sign(ch)
    w = 1.0 / np.arange(1, N_LAGS_R + 1)
    n = len(s)
    if n < N_LAGS_R + 100:
        return float("nan")
    F = np.zeros(n - N_LAGS_R)
    for k in range(1, N_LAGS_R + 1):
        F += w[k - 1] * s[N_LAGS_R - k:n - k]
    tgt = s[N_LAGS_R:]
    if tgt.std() == 0 or F.std() == 0:
        return float("nan")
    return float(np.corrcoef(tgt, F)[0, 1])


def eval_asset(args):
    sid, cls = args
    from sklearn.metrics import roc_auc_score
    from .models import ARLogit, IsingLogit
    from .walkforward import walk_forward

    raw = json.loads((DATA / f"{sid}-15m.json").read_text())
    raw.sort(key=lambda d: d["timestamp"])
    ch = np.array([d.get("change", 0.0) for d in raw], float)
    ups = np.array([bool(d["up"]) for d in raw], bool)
    ts = np.array([d["timestamp"] for d in raw], np.int64)

    row = {"asset": sid, "class": cls, "n_bars": int(len(ch)),
           "R": kernel_R(ch), "flip": float(np.mean(np.sign(ch[1:]) ==
                                                    -np.sign(ch[:-1])))}
    if len(ch) < MIN_BARS:
        row["skipped"] = f"bars<{MIN_BARS}"
        return row

    res, nf = walk_forward(ch, ups, TRAIN, TEST, n_lags=12,
                           models=lambda: [IsingLogit(n_lags=12),
                                           ARLogit("free", n_lags=12)])
    y = res["ising"]["actuals"].astype(int)
    if len(y) < MIN_OOS or len(np.unique(y)) < 2:
        row["skipped"] = f"oos<{MIN_OOS}"
        return row
    p_is, p_fr = res["ising"]["probs"], res["free"]["probs"]
    rng = np.random.default_rng(SEED)
    ge_is = ge_fr = 0
    for _ in range(N_BOOT):
        nb = int(np.ceil(len(y) / BLOCK))
        starts = rng.integers(0, max(1, len(y) - BLOCK + 1), size=nb)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:len(y)]
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        ge_is += roc_auc_score(yb, p_is[idx]) <= 0.5
        ge_fr += roc_auc_score(yb, p_fr[idx]) <= 0.5
    row.update(
        auc_ising=float(roc_auc_score(y, p_is)),
        auc_free=float(roc_auc_score(y, p_fr)),
        A=float(np.mean([fp["A"] for fp in res["ising"]["fold_params"]])),
        p_ising=ge_is / N_BOOT, p_free=ge_fr / N_BOOT,
        n_oos=int(len(y)), n_folds=nf,
        oos_slots=(ts[res["ising"]["idx"].astype(int)] // SLOT).tolist(),
        oos_y=y.astype(int).tolist(),
        oos_p_is=np.round(p_is, 5).tolist(),
        oos_p_fr=np.round(p_fr, 5).tolist(),
    )
    return row


def joint_gap(rows):
    """Dependence-preserving joint block bootstrap of the class-mean AUC gap,
    same construction as dependence.joint_gap_bootstrap on in-memory series."""
    from .dependence import fast_auc

    data = [r for r in rows if "auc_ising" in r]
    lo = min(min(r["oos_slots"]) for r in data)
    hi = max(max(r["oos_slots"]) for r in data)
    T = hi - lo + 1
    prep = []
    for r in data:
        sl = np.array(r["oos_slots"], np.int64) - lo
        lut = np.full(T, -1, np.int32)
        lut[sl] = np.arange(len(sl), dtype=np.int32)
        prep.append({"cls": r["class"], "lut": lut,
                     "y": np.array(r["oos_y"], np.int8),
                     "p_is": np.array(r["oos_p_is"], np.float32),
                     "p_fr": np.array(r["oos_p_fr"], np.float32)})

    def class_means(take):
        means = {}
        for model in ("p_is", "p_fr"):
            per = {"crypto": [], "stock": []}
            for d in prep:
                if take is None:
                    y, p = d["y"], d[model]
                else:
                    obs = d["lut"][take]
                    obs = obs[obs >= 0]
                    if len(obs) < GAP_MIN_OBS:
                        continue
                    y, p = d["y"][obs], d[model][obs]
                if y.min() == y.max():
                    continue
                a = fast_auc(y, p)
                if np.isfinite(a):
                    per[d["cls"]].append(a)
            means[model] = {c: float(np.mean(v)) for c, v in per.items() if v}
        return means

    obs = class_means(None)
    rng = np.random.default_rng(GAP_SEED)
    nb = int(np.ceil(T / GAP_BLOCK))
    gaps = {"p_is": [], "p_fr": []}
    for b in range(GAP_BOOT):
        starts = rng.integers(0, max(1, T - GAP_BLOCK + 1), size=nb)
        slots = np.concatenate([np.arange(s, s + GAP_BLOCK) for s in starts])[:T]
        m = class_means(slots)
        for model in gaps:
            if "crypto" in m[model] and "stock" in m[model]:
                gaps[model].append(m[model]["crypto"] - m[model]["stock"])
        if (b + 1) % 100 == 0:
            print(f"  joint bootstrap {b+1}/{GAP_BOOT}")
    out = {}
    for model, label in (("p_is", "ising"), ("p_fr", "free")):
        g = np.array(gaps[model])
        out[label] = {
            "obs_gap": obs[model]["crypto"] - obs[model]["stock"],
            "crypto_mean_auc": obs[model]["crypto"],
            "stock_mean_auc": obs[model]["stock"],
            "gap_ci": [float(np.percentile(g, 2.5)),
                       float(np.percentile(g, 97.5))],
            "p_gap_le0": float(np.mean(g <= 0)),
        }
    return out


def evaluate(workers: int):
    from .fdr import bh_mask

    uni = universe()
    jobs = [(sid, cls) for sid, cls in sorted(uni.items())
            if (DATA / f"{sid}-15m.json").exists()]
    print(f"{len(jobs)} of {len(uni)} frozen assets have holdout data")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, row in enumerate(ex.map(eval_asset, jobs, chunksize=4)):
            rows.append(row)
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(jobs)}] evaluated", flush=True)

    scored = [r for r in rows if "auc_ising" in r]
    p_is = np.maximum([r["p_ising"] for r in scored], P_FLOOR)
    p_fr = np.maximum([r["p_free"] for r in scored], P_FLOOR)
    m_conj = bh_mask(np.maximum(p_is, p_fr), 0.05)
    cls = np.array([r["class"] for r in scored])

    summary = {"span": [START, END],
               "deviations": f"min bars {MIN_BARS} (wide.py: {TRAIN+TEST+2000}); "
                             f"min OOS {MIN_OOS} (wide.py: 2000); "
                             f"joint-bootstrap min obs {GAP_MIN_OBS} "
                             "(dependence.py: 1000)",
               "n_frozen": len(uni), "n_with_data": len(jobs),
               "n_scored": len(scored), "classes": {}}
    exante = set(json.loads((OUT / "exante_volume.json").read_text()))
    for c in ("crypto", "stock"):
        sel = cls == c
        sub = [r for r in scored if r["class"] == c]
        aucs = np.array([r["auc_ising"] for r in sub])
        A = np.array([r["A"] for r in sub])
        Rs = np.array([r["R"] for r in rows
                       if r["class"] == c and np.isfinite(r["R"])])
        summary["classes"][c] = {
            "n_scored": len(sub),
            "mean_auc_ising": float(aucs.mean()),
            "median_auc_ising": float(np.median(aucs)),
            "frac_auc_gt_half": float(np.mean(aucs > 0.5)),
            "mean_auc_free": float(np.mean([r["auc_free"] for r in sub])),
            "raw_both_sig": int(sum(r["p_ising"] < 0.05 and r["p_free"] < 0.05
                                    for r in sub)),
            "bh_conj_sig": int(np.sum(m_conj & sel)),
            "mean_A": float(A.mean()), "frac_A_neg": float(np.mean(A < 0)),
            "mean_oos": float(np.mean([r["n_oos"] for r in sub])),
            "R_mean": float(Rs.mean()), "R_frac_neg": float(np.mean(Rs < 0)),
            "R_n": int(len(Rs)),
        }
        if c == "crypto":
            ex = aucs[[r["asset"] in exante for r in sub]]
            summary["classes"][c]["exante_n"] = int(len(ex))
            summary["classes"][c]["exante_mean_auc"] = float(ex.mean())

    print("computing joint dependence-preserving gap bootstrap ...")
    summary["joint_gap"] = joint_gap(scored)

    slim = [{k: v for k, v in r.items() if not k.startswith("oos_")}
            for r in rows]
    (OUT / "holdout.json").write_text(json.dumps(
        {"summary": summary, "rows": slim}, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {OUT / 'holdout.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-crypto", action="store_true")
    ap.add_argument("--fetch-stocks", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.fetch_crypto:
        fetch_crypto()
    if args.fetch_stocks:
        fetch_stocks()
    if args.eval:
        evaluate(args.workers)
    if not (args.fetch_crypto or args.fetch_stocks or args.eval):
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
