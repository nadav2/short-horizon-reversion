"""Selective prediction: coverage-accuracy-edge curves and break-even costs.

Re-expresses the ranking skill (AUC) as the economics a strategy can access:
a bet is placed only when the model's confidence clears a fixed, ex-ante
threshold ``|p - 0.5| >= tau``. As tau rises, trades get rarer but each carries
more edge. For every threshold we report coverage (fraction of candles
traded), directional accuracy on the traded subset, and the mean gross edge
per trade in basis points of notional, ``E[sign(p - 1/2) * r_t | traded]`` --
the number that must clear a round-trip transaction cost for the signal to be
economically exploitable on the spot market.

This is NOT new evidence of predictability (AUC already summarizes ranking
skill across all thresholds); it is the bridge from statistical to economic
significance, plus a calibration check: a well-calibrated model's accuracy
must rise monotonically with its own confidence.

Inputs are the persisted OOS dumps (no walk-forward re-runs):
  out/oos_{coin}_{interval}.npz   focal within-crypto cells (run.py)
  out/wide_oos/{asset}.npz        wide universe (wide.py --dump-oos)

    uv run --active python -m paper.selective
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import CANDLES_PER_DAY, load_merged
from .walkforward import window_candles

OUT = Path(__file__).resolve().parent / "out"
BULK = Path(__file__).resolve().parent / "bulk_data"

# Fixed, ex-ante confidence thresholds (NOT tuned on OOS data).
TAU_GRID = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10)
FOCAL_CELLS = [("btc", "15m"), ("eth", "15m"), ("btc", "5m"), ("btc", "1h")]
MODELS = ("ising", "free")
N_BOOT, SEED = 1000, 11
BLOCK_DAYS = 4          # moving-block bootstrap block length in calendar days
N_DECILES = 10
WIDE_TAUS = (0.0, 0.02)         # thresholds reported for the wide universe
COST_BANDS_BP = (5.0, 10.0, 20.0)  # round-trip cost bands (bp of notional)


def block_idx(n: int, block: int, rng) -> np.ndarray:
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=nb)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def stats_at_tau(p: np.ndarray, y: np.ndarray, r: np.ndarray, tau: float) -> dict | None:
    """Coverage / accuracy / gross edge per trade (bp) at a fixed threshold.

    ``acc_nonzero`` excludes flat (change == 0) candles: a stale bar is labeled
    "down" by the ``up = change > 0`` convention, so a reversal model gets it
    "right" with zero P&L -- accuracy on thin instruments is inflated by
    staleness while the signed-return edge is not."""
    trade = np.abs(p - 0.5) >= tau
    n = int(trade.sum())
    if n < 50:
        return None
    side = np.sign(p[trade] - 0.5)
    acc = float(np.mean((p[trade] > 0.5) == (y[trade] > 0.5)))
    edge_bp = float(np.mean(side * r[trade]) * 1e4)
    nz = trade & (r != 0.0)
    acc_nz = float(np.mean((p[nz] > 0.5) == (y[nz] > 0.5))) if nz.sum() >= 50 else None
    return {"tau": tau, "n_trades": n, "coverage": float(n / len(p)),
            "accuracy": acc, "acc_nonzero": acc_nz,
            "frac_zero": float(np.mean(r[trade] == 0.0)), "edge_bp": edge_bp}


def boot_ci(p, y, r, tau, block, rng):
    """Moving-block bootstrap CIs for accuracy and edge_bp at fixed tau.

    Blocks are resampled on the OOS time grid; the trade mask is recomputed on
    each resample so the selection (which clusters in volatile stretches)
    travels with the candles rather than being treated as i.i.d. trades."""
    accs, edges = [], []
    n = len(p)
    for _ in range(N_BOOT):
        idx = block_idx(n, block, rng)
        pb, yb, rb = p[idx], y[idx], r[idx]
        trade = np.abs(pb - 0.5) >= tau
        if trade.sum() < 50:
            continue
        side = np.sign(pb[trade] - 0.5)
        accs.append(np.mean((pb[trade] > 0.5) == (yb[trade] > 0.5)))
        edges.append(np.mean(side * rb[trade]) * 1e4)
    if len(accs) < 100:
        return None
    return {"acc_ci": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
            "edge_ci": [float(np.percentile(edges, 2.5)), float(np.percentile(edges, 97.5))],
            "p_acc_le_half": float(np.mean(np.asarray(accs) <= 0.5)),
            "p_edge_le_zero": float(np.mean(np.asarray(edges) <= 0.0))}


def decile_conditioning(p, y, r) -> list[dict]:
    """Accuracy and edge per confidence decile (descriptive conditioning)."""
    conf = np.abs(p - 0.5)
    qs = np.quantile(conf, np.linspace(0, 1, N_DECILES + 1))
    qs[-1] = np.inf
    rows = []
    for d in range(N_DECILES):
        sel = (conf >= qs[d]) & (conf < qs[d + 1])
        if sel.sum() < 50:
            continue
        side = np.sign(p[sel] - 0.5)
        rows.append({"decile": d + 1, "n": int(sel.sum()),
                     "conf_mid": float(np.median(conf[sel])),
                     "accuracy": float(np.mean((p[sel] > 0.5) == (y[sel] > 0.5))),
                     "edge_bp": float(np.mean(side * r[sel]) * 1e4)})
    return rows


def vol_conditioning(p, y, r, ch_full, idx, cpd) -> list[dict]:
    """Top-quintile-confidence accuracy within trailing-volatility terciles."""
    # causal trailing 1-day realized vol per candle
    c2 = np.concatenate([[0.0], np.cumsum(ch_full ** 2)])
    lo = np.maximum(idx - cpd, 0)
    rv = np.sqrt((c2[idx] - c2[lo]) / np.maximum(idx - lo, 1))
    terc = np.quantile(rv, [1 / 3, 2 / 3])
    conf = np.abs(p - 0.5)
    top = conf >= np.quantile(conf, 0.8)
    rows = []
    for t, (vlo, vhi) in enumerate([(-np.inf, terc[0]), (terc[0], terc[1]),
                                    (terc[1], np.inf)]):
        sel = (rv >= vlo) & (rv < vhi)
        st = sel & top
        if st.sum() < 50:
            continue
        rows.append({"tercile": t + 1, "n_top": int(st.sum()),
                     "acc_all": float(np.mean((p[sel] > 0.5) == (y[sel] > 0.5))),
                     "acc_top": float(np.mean((p[st] > 0.5) == (y[st] > 0.5)))})
    return rows


def focal():
    cells = {}
    for coin, iv in FOCAL_CELLS:
        d = np.load(OUT / f"oos_{coin}_{iv}.npz")
        y = d["actual"].astype(int)
        dts, ch, ups = load_merged(coin, iv)
        tr, _ = window_candles(iv)
        idx = np.arange(tr, tr + len(y))
        assert np.array_equal(ups[idx].astype(int), y), f"alignment broke for {coin}-{iv}"
        r = ch[idx]
        cpd = CANDLES_PER_DAY[iv]
        block = BLOCK_DAYS * cpd
        cell = {"n_oos": int(len(y)), "models": {}}
        for m in MODELS:
            p = d[f"p_{m}"].astype(float)
            rng = np.random.default_rng(SEED)
            curve = []
            for tau in TAU_GRID:
                s = stats_at_tau(p, y, r, tau)
                if s is None:
                    continue
                ci = boot_ci(p, y, r, tau, block, rng)
                if ci:
                    s.update(ci)
                curve.append(s)
            cell["models"][m] = {"curve": curve,
                                 "deciles": decile_conditioning(p, y, r)}
            if iv == "15m":
                cell["models"][m]["vol"] = vol_conditioning(p, y, r, ch, idx, cpd)
        cells[f"{coin}-{iv}"] = cell
        a0 = cell["models"]["ising"]["curve"][0]
        at = [c for c in cell["models"]["ising"]["curve"] if c["tau"] == 0.02]
        msg = f"  {coin}-{iv}: tau=0 acc={a0['accuracy']*100:.2f}% edge={a0['edge_bp']:.2f}bp"
        if at:
            msg += (f" | tau=0.02 cov={at[0]['coverage']*100:.0f}% "
                    f"acc={at[0]['accuracy']*100:.2f}% edge={at[0]['edge_bp']:.2f}bp")
        print(msg)
    return cells


def wide():
    """Per-asset selective stats over the wide universe (no per-asset CIs;
    inference lives in the cross-asset distribution + the focal MBB CIs)."""
    from .fetch_bulk_stocks import UNIVERSE
    stock_set = {s.lower().replace(".", "-") for s in UNIVERSE}
    rows = []
    for f in sorted((OUT / "wide_oos").glob("*.npz")):
        asset = f.stem
        bulk = BULK / f"{asset}-15m.json"
        if not bulk.exists():
            continue
        d = np.load(f)
        raw = json.loads(bulk.read_text())
        raw.sort(key=lambda rr: rr["timestamp"])
        ts_all = np.array([rr["timestamp"] for rr in raw], np.int64)
        ch_all = np.array([rr.get("change", 0.0) for rr in raw], float)
        up_all = np.array([bool(rr["up"]) for rr in raw], bool)
        pos = np.searchsorted(ts_all, d["ts"])
        if not (np.array_equal(ts_all[pos], d["ts"])
                and np.array_equal(up_all[pos].astype(int), d["actual"].astype(int))):
            print(f"  {asset}: alignment failed, skipped")
            continue
        y = d["actual"].astype(int)
        r = ch_all[pos]
        row = {"asset": asset,
               "class": "stock" if asset in stock_set else "crypto"}
        for m in MODELS:
            p = d[f"p_{m}"].astype(float)
            for tau in WIDE_TAUS:
                s = stats_at_tau(p, y, r, tau)
                key = f"{m}_tau{tau:g}"
                row[key] = ({"coverage": s["coverage"], "accuracy": s["accuracy"],
                             "acc_nonzero": s["acc_nonzero"], "frac_zero": s["frac_zero"],
                             "edge_bp": s["edge_bp"], "n_trades": s["n_trades"]}
                            if s else None)
        rows.append(row)
    print(f"  wide: {len(rows)} assets")
    return rows


def summarize_wide(rows):
    summary = {}
    for cls in ("crypto", "stock"):
        sub = [r for r in rows if r["class"] == cls]
        s = {"n_assets": len(sub)}
        for m in MODELS:
            for tau in WIDE_TAUS:
                key = f"{m}_tau{tau:g}"
                vals = [r[key] for r in sub if r.get(key)]
                if not vals:
                    continue
                edges = np.array([v["edge_bp"] for v in vals])
                accs = np.array([v["accuracy"] for v in vals])
                accs_nz = np.array([v["acc_nonzero"] for v in vals
                                    if v["acc_nonzero"] is not None])
                covs = np.array([v["coverage"] for v in vals])
                s[key] = {
                    "n": len(vals),
                    "median_edge_bp": float(np.median(edges)),
                    "median_accuracy": float(np.median(accs)),
                    "median_acc_nonzero": float(np.median(accs_nz)) if len(accs_nz) else None,
                    "median_coverage": float(np.median(covs)),
                    "frac_edge_gt": {f"{b:g}bp": float(np.mean(edges > b))
                                     for b in COST_BANDS_BP},
                }
        summary[cls] = s
    return summary


def main():
    print("focal cells:")
    cells = focal()
    print("wide universe:")
    rows = wide()
    out = {"config": {"tau_grid": TAU_GRID, "wide_taus": WIDE_TAUS,
                      "cost_bands_bp": COST_BANDS_BP, "n_boot": N_BOOT,
                      "block_days": BLOCK_DAYS},
           "focal": cells, "wide_assets": rows,
           "wide_summary": summarize_wide(rows)}
    (OUT / "selective.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT/'selective.json'}")
    for cls, s in out["wide_summary"].items():
        for m in ("ising",):
            k = f"{m}_tau0.02"
            if k in s:
                v = s[k]
                print(f"  {cls}: tau=0.02 median edge {v['median_edge_bp']:.2f}bp, "
                      f"median acc {v['median_accuracy']*100:.2f}%, "
                      f"frac>10bp {v['frac_edge_gt']['10bp']*100:.0f}%")


if __name__ == "__main__":
    main()
