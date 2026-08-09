"""Render markdown tables + summary statistics from out/results.json.

    uv run --active python -m paper.tables

Writes out/tables.md and prints headline aggregate statistics used in the paper text.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"

MODEL_ORDER = ["ising", "free", "l2", "l1", "markov1", "markov2", "base"]
MODEL_LABEL = {"ising": "Ising power-law", "free": "Free AR-logit", "l2": "Ridge AR-logit (L2)",
               "l1": "Lasso AR-logit (L1)", "markov1": "Markov-1", "markov2": "Markov-2",
               "base": "Base rate"}
N_PARAMS = {"ising": "3", "free": "13", "l2": "13", "l1": "13", "markov1": "2", "markov2": "4",
            "base": "1"}
CELLS = [f"{c}-{iv}" for c in ("btc", "eth", "sol", "xrp") for iv in ("5m", "15m", "1h", "4h")]


def fmt(x, p=2, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:.{p}f}%" if pct else f"{x:.{p}f}"


def coverage_table(R):
    L = ["| Cell | Span | Candles | Folds | OOS preds | P(up) |",
         "|------|------|--------:|------:|----------:|------:|"]
    for key in CELLS:
        c = R["cells"].get(key)
        if not c:
            continue
        i = c["info"]
        L.append(f"| {key} | {i['span'][0]}→{i['span'][1]} | {i['n_total']:,} | {i['n_folds']} | "
                 f"{i['n_oos']:,} | {i['base_rate_up']*100:.2f}% |")
    return "\n".join(L)


def pred_table(R, metric, label, p=4, pct=False, best="min"):
    """One row per cell, one column per model, for a prediction-quality metric."""
    head = "| Cell | " + " | ".join(MODEL_LABEL[m] for m in MODEL_ORDER) + " |"
    sep = "|------|" + "|".join(["------:"] * len(MODEL_ORDER)) + "|"
    L = [f"**{label}**\n", head, sep]
    for key in CELLS:
        c = R["cells"].get(key)
        if not c:
            continue
        vals = {m: c["models"][m]["pred"][metric] for m in MODEL_ORDER}
        comp = {m: v for m, v in vals.items() if m != "base" and not (isinstance(v, float) and np.isnan(v))}
        winner = (min if best == "min" else max)(comp, key=comp.get)
        cells = []
        for m in MODEL_ORDER:
            s = fmt(vals[m], p, pct)
            cells.append(f"**{s}**" if m == winner else s)
        L.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def trading_table(R, mode="trade_all"):
    head = "| Cell | Model | Bets | Win% | ROI/bet | Profit/bet | MaxDD |"
    sep = "|------|-------|-----:|-----:|--------:|-----------:|------:|"
    L = [head, sep]
    for key in CELLS:
        c = R["cells"].get(key)
        if not c:
            continue
        for m in ["ising", "free", "markov1"]:
            t = c["models"][m][mode]
            L.append(f"| {key} | {MODEL_LABEL[m]} | {t['n_bets']:,} | {fmt(t['win_rate'],1,True)} | "
                     f"{fmt(t['roi'],2,True)} | {t['profit_per_bet']:+.4f} | {fmt(t['max_drawdown'],1)} |")
    return "\n".join(L)


def lag_sweep_table(R):
    L = []
    for key, sweep in R["lag_sweep"].items():
        L.append(f"\n**Lag-order sweep — {key}** (OOS log-loss; train log-loss in parens)\n")
        Ns = sorted(int(n) for n in sweep)
        L.append("| N | Free AR-logit | Ridge (L2) | Ising (3 params) |")
        L.append("|--:|------|------|------|")
        for N in Ns:
            row = sweep[str(N)]
            def cell(m):
                return f"{row[m]['test_ll']:.4f} ({row[m]['train_ll']:.4f})"
            L.append(f"| {N} | {cell('free')} | {cell('l2')} | {cell('ising')} |")
    return "\n".join(L)


def headline_stats(R):
    """Aggregate win counts and average gaps used in the abstract / results prose."""
    cells = [k for k in CELLS if k in R["cells"]]
    n = len(cells)
    stats = {"n_cells": n}

    def wins(metric, best):
        w = 0
        for k in cells:
            mods = R["cells"][k]["models"]
            comp = {m: mods[m]["pred"][metric] for m in MODEL_ORDER if m != "base"}
            comp = {m: v for m, v in comp.items() if not (isinstance(v, float) and np.isnan(v))}
            best_m = (min if best == "min" else max)(comp, key=comp.get)
            if best_m == "ising":
                w += 1
        return w

    stats["ising_logloss_wins"] = wins("log_loss", "min")
    stats["ising_auc_wins"] = wins("auc", "max")
    stats["ising_acc_wins"] = wins("accuracy", "max")
    stats["ising_brier_wins"] = wins("brier", "min")

    # Ising vs free gaps
    dll = [(R["cells"][k]["models"]["free"]["pred"]["log_loss"]
            - R["cells"][k]["models"]["ising"]["pred"]["log_loss"]) for k in cells]
    dauc = [(R["cells"][k]["models"]["ising"]["pred"]["auc"]
             - R["cells"][k]["models"]["free"]["pred"]["auc"]) for k in cells]
    dacc = [(R["cells"][k]["models"]["ising"]["pred"]["accuracy"]
             - R["cells"][k]["models"]["free"]["pred"]["accuracy"]) * 100 for k in cells]
    stats["mean_logloss_gain_vs_free"] = float(np.mean(dll))
    stats["logloss_gain_pos_cells"] = int(np.sum(np.array(dll) > 0))
    stats["mean_auc_gain_vs_free"] = float(np.mean(dauc))
    stats["mean_acc_gain_vs_free_pp"] = float(np.mean(dacc))

    # overfitting gap (free) at largest N from sweep
    of = {}
    for key, sweep in R["lag_sweep"].items():
        Ns = sorted(int(x) for x in sweep)
        Nmax = max(Ns)
        of[key] = {
            "free_gap": sweep[str(Nmax)]["free"]["test_ll"] - sweep[str(Nmax)]["free"]["train_ll"],
            "ising_gap": sweep[str(Nmax)]["ising"]["test_ll"] - sweep[str(Nmax)]["ising"]["train_ll"],
            "Nmax": Nmax,
        }
    stats["overfit"] = of

    # alpha range
    alphas = [np.mean(R["cells"][k]["ising_params"]["alpha"]) for k in cells]
    stats["alpha_min"] = float(np.min(alphas))
    stats["alpha_max"] = float(np.max(alphas))
    stats["alpha_mean"] = float(np.mean(alphas))
    # amplitude sign (ferro vs antiferro): A>0 trend-following
    A_signs = {k: float(np.mean(R["cells"][k]["ising_params"]["A"])) for k in cells}
    stats["A_negative_cells"] = int(np.sum(np.array(list(A_signs.values())) < 0))
    stats["A_signs"] = A_signs
    return stats


def main():
    R = json.loads((OUT / "results.json").read_text())
    L = ["# Empirical results\n", "## Data coverage & walk-forward geometry\n", coverage_table(R), ""]
    L += ["\n## Prediction quality\n",
          pred_table(R, "accuracy", "Out-of-sample accuracy", 2, True, "max"), "",
          pred_table(R, "auc", "Out-of-sample AUC", 4, False, "max"), "",
          pred_table(R, "log_loss", "Out-of-sample log-loss (lower = better)", 4, False, "min"), "",
          pred_table(R, "brier", "Out-of-sample Brier score (lower = better)", 5, False, "min"), ""]
    L += ["\n## Trading performance (even-money binary, fee=2%)\n",
          "_Trade-every-candle:_\n", trading_table(R, "trade_all"), "",
          "\n_Confidence-thresholded (τ tuned on first OOS half, reported on held-out second half):_\n",
          trading_table(R, "thresholded"), ""]
    L += ["\n## Lag-order overfitting sweep\n", lag_sweep_table(R), ""]
    (OUT / "tables.md").write_text("\n".join(L))

    stats = headline_stats(R)
    (OUT / "headline_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nWrote {OUT/'tables.md'} and {OUT/'headline_stats.json'}")


if __name__ == "__main__":
    main()
