"""Data loading, feature construction, and classification metrics.

The directional target is ``up_t = 1[r_t > 0]`` where ``r_t`` is the intra-candle
open-to-close return ``(close_t - open_t) / open_t`` of candle ``t`` at the chosen
interval (note: flat candles, r_t == 0, are labeled "down"). The signed-return
feature used by all return-based models is ``r_t`` (field ``change`` in the
bundled JSON).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DATA_DIR = Path(os.environ.get("PAPER_DATA_DIR",
                               Path(__file__).resolve().parent.parent / "data"))

# Approximate number of candles per calendar day, used to translate the
# walk-forward window sizes (defined in days) into candle counts per interval.
CANDLES_PER_DAY = {"5m": 288, "15m": 96, "30m": 48, "1h": 24, "2h": 12, "4h": 6}


def load_merged(coin: str, interval: str):
    """Merge every available source for (coin, interval), dedup by datetime, sort by time.

    Returns ``(datetimes, changes, ups)`` where ``changes`` is a float array of signed
    returns and ``ups`` is a bool array of next-direction labels.
    """
    merged: dict[str, dict] = {}
    for fname in (f"{coin}-{interval}.json", f"{coin}_1m-{interval}.json"):
        path = DATA_DIR / fname
        if path.exists():
            for d in json.loads(path.read_text()):
                merged[d["datetime"]] = d
    if not merged:
        raise FileNotFoundError(f"no data for {coin}-{interval} in {DATA_DIR}")
    rows = sorted(merged.values(), key=lambda d: d["timestamp"])
    dts = [d["datetime"] for d in rows]
    changes = np.array([d.get("change", 0.0) for d in rows], dtype=float)
    ups = np.array([bool(d["up"]) for d in rows], dtype=bool)
    return dts, changes, ups


def spins(changes: np.ndarray, spin_scale: float) -> np.ndarray:
    """Soft-clipped spin transform ``s_t = tanh(scale * r_t) in (-1, +1)``.

    Bounds heavy-tailed crypto returns into a finite spin so that a few outlier
    candles cannot dominate the local field. ``spin_scale <= 0`` falls back to the
    hard sign (``+/-1``)."""
    if spin_scale > 0:
        return np.tanh(changes * spin_scale)
    return np.sign(changes)


def lag_matrix(x: np.ndarray, n_lags: int) -> np.ndarray:
    """Causal lag design matrix. Row ``t`` is ``[x_{t-1}, x_{t-2}, ..., x_{t-n_lags}]``
    (column 0 = most recent lag). Rows ``t < n_lags`` are NaN (insufficient history)."""
    n = len(x)
    out = np.full((n, n_lags), np.nan)
    if n > n_lags:
        # sliding_window_view(x, n_lags)[i] == x[i : i+n_lags]; row t needs x[t-n_lags:t] reversed.
        windows = sliding_window_view(x, n_lags)            # shape (n-n_lags+1, n_lags)
        out[n_lags:] = windows[:-1][:, ::-1]                # align to row t, reverse so col0 = lag1
    return out


def power_law_field(sp: np.ndarray, alpha: float, n_lags: int) -> np.ndarray:
    """Power-law-weighted local field ``field_t = sum_{k=1..N} s_{t-k} / k^alpha``.

    Distance ``k = 1`` is the most recent prior spin (matches the live Ising engine
    ``ising.predict.predict_from_spins``). Strictly causal. ``field_t`` for
    ``t < n_lags`` uses only the available history (shorter sum)."""
    n = len(sp)
    field = np.zeros(n, dtype=float)
    for k in range(1, n_lags + 1):
        field[k:] += sp[:n - k] / (k ** alpha)
    return field


# ── Metrics ──────────────────────────────────────────────────────────────────

def classification_metrics(p_up: np.ndarray, actual: np.ndarray) -> dict:
    """Standard binary-classification metrics for probabilistic predictions."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                  brier_score_loss, f1_score, log_loss,
                                  matthews_corrcoef, roc_auc_score)
    p_up = np.asarray(p_up, dtype=float)
    y = np.asarray(actual).astype(int)
    pred = (p_up > 0.5).astype(int)
    pc = np.clip(p_up, 1e-15, 1 - 1e-15)
    base = float(y.mean())
    majority_acc = max(base, 1 - base)
    two_class = len(np.unique(y)) > 1
    acc = accuracy_score(y, pred)
    return {
        "n": int(len(y)),
        "accuracy": float(acc),
        "edge": float(acc - majority_acc),
        "balanced_acc": float(balanced_accuracy_score(y, pred)) if two_class else 0.5,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if two_class and len(np.unique(pred)) > 1 else 0.0,
        "auc": float(roc_auc_score(y, p_up)) if two_class else float("nan"),
        "log_loss": float(log_loss(y, pc, labels=[0, 1])),
        "brier": float(brier_score_loss(y, pc)),
        "majority_acc": float(majority_acc),
        "base_rate_up": float(base),
    }
