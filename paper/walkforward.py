"""Rolling walk-forward evaluation harness.

A fixed-length TRAIN window slides forward in steps equal to the TEST block size.
At every step each model is refit on the current TRAIN window and produces causal
predictions for the next (out-of-sample) TEST block. The OOS predictions from all
non-overlapping TEST blocks are concatenated into one series per model, which is
then scored. This mimics a deployment that periodically recalibrates on the most
recent window and trades the next block forward.
"""

from __future__ import annotations

import numpy as np

from .common import CANDLES_PER_DAY
from .models import build_models

# Walk-forward window geometry per interval, expressed in calendar days and
# converted to candle counts. Train windows shrink (in candles) for coarse
# intervals because far fewer candles exist per day.
WINDOW_DAYS = {
    "5m":  {"train": 90,  "test": 15},
    "15m": {"train": 150, "test": 21},
    "1h":  {"train": 240, "test": 30},
    "4h":  {"train": 365, "test": 45},
}


def window_candles(interval: str) -> tuple[int, int]:
    cpd = CANDLES_PER_DAY[interval]
    d = WINDOW_DAYS[interval]
    return d["train"] * cpd, d["test"] * cpd


def walk_forward(changes, ups, train_size, test_size, n_lags=12, models=None):
    """Run the rolling walk-forward. Returns ``{short: result_dict}`` where each
    result holds the concatenated OOS probabilities, aligned actuals, the absolute
    test indices, and the list of per-fold fitted-parameter dicts."""
    n = len(changes)
    factory = (lambda: build_models(n_lags=n_lags)) if models is None else models
    template = factory()
    results = {m.short: {"name": m.name, "probs": [], "actuals": [], "idx": [],
                         "fold_params": []} for m in template}

    n_folds = 0
    start = 0
    while start + train_size < n:
        train_lo, train_hi = start, start + train_size
        test_lo, test_hi = train_hi, min(train_hi + test_size, n)
        if test_hi <= test_lo:
            break
        n_folds += 1
        for m in factory():
            m.fit(changes, ups, train_lo, train_hi)
            p = np.asarray(m.predict_series(changes, ups, test_lo, test_hi), dtype=float)
            r = results[m.short]
            r["probs"].append(p)
            r["actuals"].append(ups[test_lo:test_hi].astype(int))
            r["idx"].append(np.arange(test_lo, test_hi))
            r["fold_params"].append(m.params())
        start += test_size

    for r in results.values():
        r["probs"] = np.concatenate(r["probs"]) if r["probs"] else np.array([])
        r["actuals"] = np.concatenate(r["actuals"]) if r["actuals"] else np.array([])
        r["idx"] = np.concatenate(r["idx"]) if r["idx"] else np.array([])
    return results, n_folds
