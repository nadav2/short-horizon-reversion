"""Stylized trading evaluation.

We translate each model's out-of-sample directional probabilities into a P&L on a
binary "up/down" contract, the structure traded on Polymarket's 15-minute BTC
market. To keep the comparison a property of the *signal* (and not of an assumed
mispricing edge), contracts are priced at fair even money: 1 share costs $0.50 and
pays $1 if the chosen side wins, $0 otherwise. A round-trip transaction cost
``fee`` (in $ per $1 staked) is deducted.

Net P&L on a $1 stake:
    win  -> +1 * (1 - fee) - stake_cost  (= +1 - fee - 0.5*... )  -> simplified below
We use the equivalent payoff-per-unit-stake formulation:
    correct  -> +1 - fee
    wrong    -> -1 - fee
so the expected return per bet is  (2*accuracy - 1) - fee.

A bet is placed only when the model's confidence exceeds a threshold
``|p - 0.5| >= tau``; ``tau`` is selected on TRAIN-equivalent data via the OOS-free
in-sample grid passed by the caller, or fixed. We report metrics at the
profit-maximizing in-sample tau and at tau = 0 (trade every candle).
"""

from __future__ import annotations

import numpy as np

TAU_GRID = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)


def _simulate(p_up, actual, tau, fee, stake=1.0):
    """Flat-stake even-money simulation at a fixed confidence threshold tau."""
    conf = np.abs(p_up - 0.5)
    trade = conf >= tau
    side_up = p_up > 0.5
    correct = (side_up == (actual > 0.5)) & trade
    wrong = trade & ~correct

    n_bets = int(trade.sum())
    if n_bets == 0:
        return {"n_bets": 0, "win_rate": float("nan"), "roi": 0.0, "profit_per_bet": 0.0,
                "total_pnl": 0.0, "max_drawdown": 0.0, "sharpe": float("nan"),
                "final_bankroll": 1.0, "frac_traded": 0.0}

    pnl = np.zeros(len(p_up))
    pnl[correct] = stake * (1.0 - fee)
    pnl[wrong] = stake * (-1.0 - fee)
    bet_pnl = pnl[trade]

    equity = np.cumsum(bet_pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))
    drawdown = peak[1:] - equity
    total_staked = n_bets * stake
    return {
        "n_bets": n_bets,
        "win_rate": float(correct.sum() / n_bets),
        "roi": float(bet_pnl.sum() / total_staked),          # return per $ staked
        "profit_per_bet": float(bet_pnl.mean()),
        "total_pnl": float(bet_pnl.sum()),
        "max_drawdown": float(drawdown.max()),
        "sharpe": float(bet_pnl.mean() / bet_pnl.std()) if bet_pnl.std() > 0 else float("nan"),
        "final_bankroll": float(1.0 + bet_pnl.sum() / total_staked),
        "frac_traded": float(n_bets / len(p_up)),
        "tau": float(tau),
    }


def trading_metrics(p_up, actual, fee=0.0, select_tau=True):
    """Return trading metrics at tau=0 (trade-all) and at the profit-maximizing tau.

    To avoid look-ahead in tau selection we choose tau on the first half of the OOS
    series and report the *second half* under that frozen tau (a held-out trading
    test); we also report the trade-all baseline on the full series."""
    p_up = np.asarray(p_up, float)
    actual = np.asarray(actual, float)
    all_metrics = _simulate(p_up, actual, 0.0, fee)

    if not select_tau or len(p_up) < 200:
        return {"trade_all": all_metrics, "thresholded": all_metrics}

    half = len(p_up) // 2
    best = (-np.inf, 0.0)
    for tau in TAU_GRID:
        m = _simulate(p_up[:half], actual[:half], tau, fee)
        if m["n_bets"] >= 20 and m["profit_per_bet"] > best[0]:
            best = (m["profit_per_bet"], tau)
    held = _simulate(p_up[half:], actual[half:], best[1], fee)
    return {"trade_all": all_metrics, "thresholded": held}
