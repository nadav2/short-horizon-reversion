"""The wrapper panel: same underlying, different market structure.

Every cross-market comparison in the main study varies the asset and the
venue together, leaving two accounts observationally equivalent: reversal as
a property of the VENUE (thin, round-the-clock, lightly-intermediated
trading generates and fails to compete away overreaction) versus of the
underlying PRICE PROCESS (any instrument priced off that process carries it,
however intermediated its own venue). The panel separates them: six
underlyings that trade in both worlds --

  gold / silver / platinum / palladium   OTC spot (Dukascopy) vs listed ETFs
                                          GLD, SLV, PPLT, PALL
  Bitcoin / Ether                         Binance spot vs the NAV-linked spot
                                          ETFs IBIT, FBTC / ETHA, FETH

-- plus four listed instruments CORRELATED with but not priced off an
underlying (GDX, NEM; COIN, MSTR) as controls.

Four readings per leg on the identical 15m walk-forward, models and
moving-block bootstrap used for the headline result:

  * 24h bars       -- the paper's baseline convention;
  * RTH-matched    -- every leg (including the 24/7 and OTC underlyings)
                      restricted to the same 09:30-16:00 New York slots: the
                      discriminating cells;
  * one-bar gap    -- each of the above with one candle skipped between
                      predictors and label (boundary-artifact control).

The output includes a panel summary: paired wrapper-underlying gaps, role
means, z-scores against the RTH stock universe (out/wide_rth.json).

    uv run python -m paper.natural
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

from .common import DATA_DIR
from .compare_markets import N_BOOT, SEED, SPAN, WINDOWS, autocorr1, block_boot_idx
from .models import ARLogit, IsingLogit
from .walkforward import walk_forward

OUT = Path(__file__).resolve().parent / "out"
NY = ZoneInfo("America/New_York")
INTERVAL = "15m"
BLOCK = 384
BARS_PER_DAY_RTH = 26
RTH_TRAIN, RTH_TEST, RTH_BLOCK = 60 * BARS_PER_DAY_RTH, 10 * BARS_PER_DAY_RTH, 4 * BARS_PER_DAY_RTH

# (label, underlying, [(asset, role, description)]) -- roles:
#   underlying  the round-the-clock / OTC price-discovery leg
#   wrapper     NAV-linked listed instrument (ETF priced off the underlying)
#   correlated  listed instrument exposed to but not priced off the underlying
EXPERIMENTS = [
    {"experiment": "gold", "underlying": "gold",
     "legs": [("xauusd", "underlying", "OTC spot 24/5"),
              ("gld", "wrapper", "listed ETF"),
              ("gdx", "correlated", "gold-miner ETF"),
              ("nem", "correlated", "miner single stock")]},
    {"experiment": "silver", "underlying": "silver",
     "legs": [("xagusd", "underlying", "OTC spot 24/5"),
              ("slv", "wrapper", "listed ETF")]},
    {"experiment": "platinum", "underlying": "platinum",
     "legs": [("xptusd", "underlying", "OTC spot 24/5"),
              ("pplt", "wrapper", "listed ETF")]},
    {"experiment": "palladium", "underlying": "palladium",
     "legs": [("xpdusd", "underlying", "OTC spot 24/5"),
              ("pall", "wrapper", "listed ETF")]},
    {"experiment": "bitcoin", "underlying": "BTC",
     "legs": [("btc", "underlying", "crypto spot 24/7"),
              ("ibit", "wrapper", "listed spot-BTC ETF"),
              ("fbtc", "wrapper", "listed spot-BTC ETF"),
              ("bitb", "wrapper", "listed spot-BTC ETF"),
              ("arkb", "wrapper", "listed spot-BTC ETF"),
              ("hodl", "wrapper", "listed spot-BTC ETF"),
              ("btco", "wrapper", "listed spot-BTC ETF"),
              ("ezbc", "wrapper", "listed spot-BTC ETF"),
              ("gbtc", "wrapper", "converted trust spot ETF"),
              ("bito", "wrapper_alt", "CME-futures NAV wrapper"),
              ("bitu", "wrapper_alt", "2x leveraged wrapper"),
              ("coin", "correlated", "listed equity proxy"),
              ("mstr", "correlated", "listed levered proxy")]},
    {"experiment": "ether", "underlying": "ETH",
     "legs": [("eth", "underlying", "crypto spot 24/7"),
              ("etha", "wrapper", "listed spot-ETH ETF"),
              ("feth", "wrapper", "listed spot-ETH ETF"),
              ("ethe", "wrapper", "converted trust spot ETF"),
              ("ethu", "wrapper_alt", "2x leveraged wrapper")]},
    # Dose test: spot FX carries a faint reversal copy (AUC ~0.515 at 15m),
    # so its wrappers test whether inheritance is dose-proportional.
    {"experiment": "euro", "underlying": "EURUSD",
     "legs": [("eurusd", "underlying", "OTC spot FX 24/5"),
              ("fxe", "wrapper", "listed currency ETF"),
              ("uup", "correlated", "USD-index basket ETF")]},
    {"experiment": "yen", "underlying": "USDJPY",
     "legs": [("usdjpy", "underlying", "OTC spot FX 24/5"),
              ("fxy", "wrapper", "listed currency ETF (inverse leg)")]},
]


def load(asset: str, rth: bool = False):
    raw = [d for d in json.loads((DATA_DIR / f"{asset}-{INTERVAL}.json").read_text())
           if SPAN[0] <= d["datetime"][:10] <= SPAN[1]]
    raw.sort(key=lambda d: d["timestamp"])
    if rth:
        keep = []
        for d in raw:
            t = datetime.fromtimestamp(d["timestamp"], timezone.utc).astimezone(NY)
            if t.weekday() < 5 and (9, 30) <= (t.hour, t.minute) < (16, 0):
                keep.append(d)
        raw = keep
    ch = np.array([d.get("change", 0.0) for d in raw], float)
    ups = np.array([bool(d["up"]) for d in raw], bool)
    return ch, ups


def models():
    return [IsingLogit(n_lags=12), ARLogit("free", n_lags=12)]


def evaluate(asset: str, rth: bool = False, gap: bool = False) -> dict | None:
    ch, ups = load(asset, rth=rth)
    tr, te = (RTH_TRAIN, RTH_TEST) if rth else WINDOWS[INTERVAL]
    block = RTH_BLOCK if rth else BLOCK
    if len(ch) < tr + te:
        return None
    if gap:
        # one-candle gap between the last predictor and the label: predict candle
        # t from t-2..t-13. Pure bid-ask bounce / staleness lives at the boundary
        # between adjacent bars and should not survive; genuine multi-lag reversion
        # should. Same control as Sec. "Skipping the most recent bar".
        ch = np.concatenate([[0.0], ch[:-1]])
    res, n_folds = walk_forward(ch, ups, tr, te, n_lags=12, models=models)
    y = res["ising"]["actuals"].astype(int)
    if len(y) == 0 or y.min() == y.max():
        return None
    p_is, p_fr = res["ising"]["probs"], res["free"]["probs"]
    idx = res["ising"]["idx"]

    auc_is, auc_fr = roc_auc_score(y, p_is), roc_auc_score(y, p_fr)
    rng = np.random.default_rng(SEED)
    b_is, b_fr = [], []
    for _ in range(N_BOOT):
        bi = block_boot_idx(len(y), block, rng)
        yb = y[bi]
        if yb.min() == yb.max():
            continue
        b_is.append(roc_auc_score(yb, p_is[bi]))
        b_fr.append(roc_auc_score(yb, p_fr[bi]))
    b_is, b_fr = np.array(b_is), np.array(b_fr)
    p_gt_is, p_gt_fr = float(np.mean(b_is <= 0.5)), float(np.mean(b_fr <= 0.5))

    # flat-bar-robust AUC: score only bars with r_t != 0
    nonflat = ch[idx] != 0
    auc_nf = (float(roc_auc_score(y[nonflat], p_is[nonflat]))
              if nonflat.sum() > 100 and len(np.unique(y[nonflat])) > 1 else float("nan"))

    # split-half stability: is the effect carried by the whole span or a few folds?
    h = len(y) // 2
    halves = []
    for sl in (slice(None, h), slice(h, None)):
        yy = y[sl]
        halves.append(float(roc_auc_score(yy, p_is[sl]))
                      if len(np.unique(yy)) > 1 else float("nan"))

    return {
        "asset": asset, "rth": rth, "gap": gap,
        "auc_half1": halves[0], "auc_half2": halves[1],
        "n_bars": int(len(ch)), "n_oos": int(len(y)),
        "n_folds": n_folds, "flat_frac": float(np.mean(ch == 0.0)),
        "ac1": autocorr1(ch), "base_rate_up": float(y.mean()),
        "auc_ising": float(auc_is),
        "auc_ci": [float(np.percentile(b_is, 2.5)), float(np.percentile(b_is, 97.5))],
        "auc_free": float(auc_fr),
        "p_ising": p_gt_is, "p_free": p_gt_fr,
        "conj_p": max(p_gt_is, p_gt_fr),
        "auc_nonflat": auc_nf,
        "A": float(np.mean([p["A"] for p in res["ising"]["fold_params"]])),
    }


def main():
    rows = []
    for exp in EXPERIMENTS:
        print(f"\n=== {exp['experiment']}: same underlying ({exp['underlying']}), "
              f"different market structure ===")
        for asset, role, structure in exp["legs"]:
            for rth, gap in ((False, False), (True, False), (False, True), (True, True)):
                r = evaluate(asset, rth=rth, gap=gap)
                tag = ("RTH" if rth else "24h") + ("+gap" if gap else "")
                if r is None:
                    print(f"  {asset:7s} {tag:8s} skipped (insufficient bars)")
                    continue
                r.update({"experiment": exp["experiment"], "role": role,
                          "structure": structure, "underlying": exp["underlying"]})
                rows.append(r)
                sig = "*" if r["conj_p"] < 0.05 else " "
                print(f"  {asset:7s} {tag:8s} {structure:22s} "
                      f"n={r['n_oos']:6d} AUC={r['auc_ising']:.4f} "
                      f"[{r['auc_ci'][0]:.3f},{r['auc_ci'][1]:.3f}] "
                      f"halves={r['auc_half1']:.3f}/{r['auc_half2']:.3f} "
                      f"nonflat={r['auc_nonflat']:.4f} p={r['conj_p']:.3f}{sig} "
                      f"A={r['A']:+.3f}", flush=True)

    panel = panel_summary(rows)
    (OUT / "natural.json").write_text(json.dumps(
        {"config": {"interval": INTERVAL, "span": SPAN, "n_boot": N_BOOT,
                    "block": BLOCK, "rth_block": RTH_BLOCK, "seed": SEED},
         "experiments": EXPERIMENTS, "rows": rows, "panel": panel}, indent=2))
    print(f"\nWrote {OUT/'natural.json'}")


def panel_summary(rows: list[dict]) -> dict:
    """Panel-level inheritance statistics on the RTH-matched cells.

    The RTH cell is the discriminating one: every leg (including the 24/7 and
    OTC underlyings) is scored on the SAME 09:30-16:00 New York slots, so the
    comparison is like-for-like in session terms. For each underlying with a
    NAV-linked wrapper we report the paired underlying-vs-wrapper AUC gap; the
    panel statistic is the mean wrapper excess over the RTH stock universe and
    the wrapper-vs-correlated contrast."""
    rth = {r["asset"]: r for r in rows if r["rth"] and not r["gap"]}
    by_role = {"underlying": [], "wrapper": [], "correlated": []}
    pairs = []
    for r in rows:
        if not (r["rth"] and not r["gap"]):
            continue
        by_role.setdefault(r["role"], []).append(r)
    # wrapper-underlying pairing
    for r in by_role["wrapper"]:
        u = next((x for x in by_role["underlying"]
                  if x["experiment"] == r["experiment"]), None)
        if u:
            pairs.append({"experiment": r["experiment"], "wrapper": r["asset"],
                          "underlying": u["asset"],
                          "auc_wrapper": r["auc_ising"], "auc_underlying": u["auc_ising"],
                          "auc_nonflat_wrapper": r["auc_nonflat"],
                          "gap": r["auc_ising"] - u["auc_ising"],
                          "A_wrapper": r["A"], "A_underlying": u["A"],
                          "sig_wrapper": r["conj_p"] < 0.05,
                          "sig_underlying": u["conj_p"] < 0.05})
    import numpy as _np
    # reference: the RTH-only wide stock universe
    try:
        wr = json.loads((OUT / "wide_rth.json").read_text())
        stock_rth = _np.array([x["auc_ising"] for x in (wr if isinstance(wr, list)
                               else wr.get("assets", []))], float)
        ref = {"mean": float(stock_rth.mean()), "sd": float(stock_rth.std()),
               "n": int(len(stock_rth))}
    except Exception:
        ref, stock_rth = None, None

    def stats(sel):
        a = _np.array([r["auc_ising"] for r in sel], float)
        out = {"n": len(sel), "mean_auc": float(a.mean()) if len(a) else None,
               "assets": {r["asset"]: round(r["auc_ising"], 4) for r in sel},
               "n_sig": sum(r["conj_p"] < 0.05 for r in sel),
               "mean_A": float(_np.mean([r["A"] for r in sel])) if sel else None}
        if ref and len(a):
            out["mean_z_vs_rth_stocks"] = float((a.mean() - ref["mean"]) / ref["sd"])
            out["frac_above_all_stocks"] = float(_np.mean(
                [(stock_rth < x).mean() == 1.0 for x in a]))
        return out

    return {"reference_rth_stocks": ref,
            "pairs": pairs,
            "underlying": stats(by_role["underlying"]),
            "wrapper": stats(by_role["wrapper"]),
            "correlated": stats(by_role["correlated"])}


if __name__ == "__main__":
    main()
