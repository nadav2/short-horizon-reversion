"""Realistic trading simulation: why an Ising-based bot beats other methods.

Each 15m/5m candle is a binary Up/Down contract. Crucially, the contract is NOT
priced at even money: a semi-efficient market prices it at
    m_t = 0.5 + rho * (ref_t - 0.5),
where ref_t is a simple, public reference model (an AR(1) logit pricing the obvious
lag-1 autocorrelation) and rho in [0,1] is market efficiency (rho=0: market has no
view, even money; rho=1: market fully prices the public model). A bot with belief
q_t buys YES when q_t > m_t (NO when q_t < m_t), sizes by fractional Kelly on the
edge versus the price, pays a spread, and compounds its bankroll.

This rewards a bot only for the skill it has BEYOND what the market already prices,
and rewards *calibration* (right-sized confidence) over raw accuracy. We sweep rho to
show how each method's edge survives as the market gets smarter, and report
risk-adjusted performance (CAGR, Sharpe, Calmar, max drawdown).

    uv run --active python -m paper.simulate
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import CANDLES_PER_DAY, classification_metrics, load_merged
from .models import ARLogit, build_models
from .walkforward import walk_forward, window_candles

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# Realistic bot mechanics. Bets are sized as a fraction of FIXED base capital (not a
# compounding bankroll) so P&L is additive and interpretable — the standard
# conservative convention that avoids the runaway compounding of a true persistent
# edge over tens of thousands of bets. Confidence still sets the size (rewarding
# calibration); it just isn't reinvested.
KELLY = 0.10          # fractional Kelly on the edge-vs-price
G_MAX = 0.05          # cap any single bet at 5% of base capital
SPREAD = 0.005        # half-spread paid on entry
EDGE_MIN = 0.01       # require >= 1 pp edge over the market price to trade
RHOS = [0.0, 0.5, 0.8, 1.0]
CELLS = [("btc", "15m"), ("eth", "15m"), ("btc", "5m"), ("eth", "5m")]
MODEL_ORDER = ["ising", "free", "l2", "l1", "markov1", "markov2"]
LABEL = {"ising": "Ising", "free": "Free AR-logit", "l2": "Ridge", "l1": "Lasso",
         "markov1": "Markov-1", "markov2": "Markov-2"}


def simulate(q, actual, m, idx_per_year, kelly=KELLY, g_max=G_MAX, spread=SPREAD,
             edge_min=EDGE_MIN):
    """Binary contract priced at m (per-candle). Bot belief q. Returns (metrics, equity)."""
    q = np.asarray(q, float)
    actual = np.asarray(actual, int)
    m = np.clip(np.asarray(m, float), 0.02, 0.98)
    up = actual > 0
    n = len(q)

    pnl = np.zeros(n)        # per-candle P&L as fraction of FIXED base capital
    stake = np.zeros(n)
    traded = np.zeros(n, bool)
    won = np.zeros(n, bool)
    for side_up in (True, False):
        if side_up:
            price = np.clip(m + spread, 0.02, 0.98)
            edge = q - price                       # buy YES when q exceeds ask
            take = edge > edge_min
            g = np.clip(kelly * edge / (1 - price), 0, g_max) * take
            r = np.where(up, g * (1 - price) / price, -g)   # YES payoff per unit base
        else:
            price = np.clip((1 - m) + spread, 0.02, 0.98)   # NO ask
            edge = (1 - q) - price
            take = edge > edge_min
            g = np.clip(kelly * edge / (1 - price), 0, g_max) * take
            r = np.where(~up, g * (1 - price) / price, -g)
        pnl = np.where(take, r, pnl)
        stake = np.where(take, g, stake)
        traded |= take
        won |= take & (up if side_up else ~up)

    equity = 1.0 + np.cumsum(pnl)             # additive (non-compounding)
    n_bets = int(traded.sum())
    if n_bets == 0:
        return {"total_return": 0.0, "ann_return": 0.0, "sharpe": float("nan"),
                "max_dd": 0.0, "calmar": float("nan"), "profit_factor": float("nan"),
                "expectancy_bp": 0.0, "win_rate": float("nan"), "n_bets": 0,
                "frac_traded": 0.0}, equity
    peak = np.maximum.accumulate(equity)
    max_dd = float((peak - equity).max())          # in units of base capital
    total_return = float(equity[-1] - 1.0)
    years = n / idx_per_year
    bet_pnl = pnl[traded]
    gains = bet_pnl[bet_pnl > 0].sum()
    losses = -bet_pnl[bet_pnl < 0].sum()
    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(idx_per_year)) if pnl.std() > 0 else float("nan")
    return {"total_return": total_return, "ann_return": float(total_return / years),
            "sharpe": sharpe, "max_dd": max_dd,
            "calmar": float((total_return / years) / max_dd) if max_dd > 0 else float("nan"),
            "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
            "expectancy_bp": float(bet_pnl.mean() * 1e4),   # mean P&L/bet in bp of base capital
            "win_rate": float(won[traded].sum() / n_bets), "n_bets": n_bets,
            "frac_traded": float(n_bets / n)}, equity


def run_asset(coin, interval):
    dts, ch, ups = load_merged(coin, interval)
    tr, te = window_candles(interval)
    res, n_folds = walk_forward(ch, ups, tr, te, n_lags=12)
    # market reference = AR(1) logit (public, obvious lag-1 signal), same walk-forward
    ref_res, _ = walk_forward(ch, ups, tr, te, n_lags=1,
                              models=lambda: [ARLogit("free", n_lags=1)])
    ref = ref_res["free"]["probs"]
    actual = res["ising"]["actuals"].astype(int)
    ipy = CANDLES_PER_DAY[interval] * 365

    cell = {"coin": coin, "interval": interval, "n_oos": int(len(actual)), "by_rho": {}}
    for rho in RHOS:
        m = 0.5 + rho * (ref - 0.5)
        rows = {}
        for short in MODEL_ORDER:
            q = res[short]["probs"]
            metrics, eq = simulate(q, actual, m, ipy)
            metrics["accuracy"] = classification_metrics(q, actual)["accuracy"]
            metrics["log_loss"] = classification_metrics(q, actual)["log_loss"]
            rows[short] = metrics
        cell["by_rho"][str(rho)] = rows
    return cell, res, ref, actual


def main():
    out = {"config": {"kelly": KELLY, "g_max": G_MAX, "spread": SPREAD,
                      "edge_min": EDGE_MIN, "rhos": RHOS}, "cells": {}}
    for coin, interval in CELLS:
        cell, *_ = run_asset(coin, interval)
        out["cells"][f"{coin}-{interval}"] = cell
        print(f"\n=== {coin.upper()} {interval}  (Kelly={KELLY}, gmax={G_MAX}, spread={SPREAD}, "
              f"n={cell['n_oos']}) ===")
        for rho in RHOS:
            print(f"  -- market efficiency rho={rho} --")
            print(f"     {'model':14s} {'acc%':>6} {'annRet%':>8} {'Sharpe':>7} {'maxDD%':>7} "
                  f"{'Calmar':>7} {'PF':>5} {'exp(bp)':>8} {'win%':>6}")
            for s in MODEL_ORDER:
                m = cell["by_rho"][str(rho)][s]
                print(f"     {LABEL[s]:14s} {m['accuracy']*100:6.2f} {m['ann_return']*100:8.1f} "
                      f"{m['sharpe']:7.2f} {m['max_dd']*100:7.1f} {m['calmar']:7.2f} "
                      f"{m['profit_factor']:5.2f} {m['expectancy_bp']:8.2f} "
                      f"{(m['win_rate'] or 0)*100:6.1f}")
    (OUT / "simulation.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT/'simulation.json'}")


if __name__ == "__main__":
    main()
