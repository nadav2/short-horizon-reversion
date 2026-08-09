"""Does a flexible nonlinear ML model beat the 3-parameter Ising prior?

The paper's thesis is that a parsimonious physical prior beats complex, high-capacity
models in this ultra-low-SNR regime. We test that head-on by adding two strong,
flexible learners on the same 12 lagged returns — gradient-boosted trees and a small
MLP — fit under the identical walk-forward, and comparing out-of-sample log-loss/AUC
and the train→OOS overfitting gap against the Ising model and the free logit.

    uv run --active python -m paper.ml_baseline
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from .common import classification_metrics, lag_matrix, load_merged
from .models import ARLogit, IsingLogit, _nll
from .walkforward import walk_forward, window_candles

OUT = Path(__file__).resolve().parent / "out"
CELLS = [("btc", "15m"), ("eth", "15m"), ("xrp", "15m"), ("btc", "1h"), ("eth", "1h")]
N_LAGS = 12


VAL_FRAC = 0.20
# small per-fold hyperparameter grids, selected on the same chronological
# validation tail used by every other model (so the tuned learners get exactly
# the same information advantage as the Ising alpha grid)
GBM_GRID = [{"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": 0.7}
            for n in (50, 150) for d in (2, 3) for lr in (0.02, 0.05)]
MLP_GRID = [{"hidden_layer_sizes": h, "alpha": a, "max_iter": 300,
             "early_stopping": True, "n_iter_no_change": 10}
            for h in ((16,), (32, 16)) for a in (1e-4, 1e-2, 1.0)]


class _SklearnLagModel:
    """Wrap a sklearn classifier on the last N signed returns (standardized).

    ``tuned=True`` grid-searches the hyperparameters per fold by NLL on the
    chronological validation tail of the train window, then refits on the full
    window — the identical protocol used for the Ising alpha and the L1/L2 C."""
    def __init__(self, kind, tuned=False):
        self.kind = kind
        self.tuned = tuned
        self.n_lags = N_LAGS
        self.model = self.scaler = None
        base = {"gbm": "Gradient-boosted trees", "mlp": "Neural net (MLP)"}[kind]
        self.name = base + (" (tuned)" if tuned else "")
        self.short = kind + ("_t" if tuned else "")

    def _new(self, cfg=None):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.neural_network import MLPClassifier
        if self.kind == "gbm":
            cfg = cfg or {"n_estimators": 80, "max_depth": 3,
                          "learning_rate": 0.05, "subsample": 0.7}
            return GradientBoostingClassifier(**cfg)
        cfg = cfg or {"hidden_layer_sizes": (32, 16), "alpha": 1e-3, "max_iter": 300,
                      "early_stopping": True, "n_iter_no_change": 10}
        return MLPClassifier(**cfg)

    def fit(self, changes, ups, lo, hi):
        from sklearn.preprocessing import StandardScaler
        a = max(lo, self.n_lags)
        X = lag_matrix(changes, self.n_lags)[a:hi]
        y = ups.astype(int)[a:hi]
        ok = ~np.isnan(X).any(axis=1)
        X, y = X[ok], y[ok]
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = None
            if self.tuned:
                cut = int(len(Xs) * (1 - VAL_FRAC))
                grid = GBM_GRID if self.kind == "gbm" else MLP_GRID
                best = (np.inf, None)
                if cut > 100 and len(np.unique(y[:cut])) > 1:
                    for g in grid:
                        m = self._new(g).fit(Xs[:cut], y[:cut])
                        ll = _nll(m.predict_proba(Xs[cut:])[:, 1], y[cut:])
                        if ll < best[0]:
                            best = (ll, g)
                cfg = best[1]
            self.model = self._new(cfg).fit(Xs, y)

    def predict_series(self, changes, ups, lo, hi):
        X = lag_matrix(changes, self.n_lags)[lo:hi]
        bad = np.isnan(X).any(axis=1)
        p = self.model.predict_proba(self.scaler.transform(np.where(np.isnan(X), 0.0, X)))[:, 1]
        p[bad] = 0.5
        return p

    def params(self):
        return {"kind": self.kind}


def factory():
    return [IsingLogit(n_lags=N_LAGS), ARLogit("free", n_lags=N_LAGS),
            _SklearnLagModel("gbm"), _SklearnLagModel("mlp"),
            _SklearnLagModel("gbm", tuned=True), _SklearnLagModel("mlp", tuned=True)]


def train_oos_gap(coin, interval):
    """Per-fold train log-loss and pooled OOS log-loss/AUC for each model."""
    dts, ch, ups = load_merged(coin, interval)
    tr, te = window_candles(interval)
    n = len(ch)
    acc = {m.short: {"name": m.name, "tr": [], "te_p": [], "te_y": []} for m in factory()}
    start = 0
    while start + tr < n:
        tlo, thi = start, start + tr
        xlo, xhi = thi, min(thi + te, n)
        if xhi <= xlo:
            break
        for m in factory():
            m.fit(ch, ups, tlo, thi)
            ptr = m.predict_series(ch, ups, max(tlo, N_LAGS), thi)
            ytr = ups.astype(int)[max(tlo, N_LAGS):thi]
            acc[m.short]["tr"].append(_nll(ptr, ytr))
            acc[m.short]["te_p"].append(m.predict_series(ch, ups, xlo, xhi))
            acc[m.short]["te_y"].append(ups.astype(int)[xlo:xhi])
        start += te
    out = {}
    for s, d in acc.items():
        te_p = np.concatenate(d["te_p"]); te_y = np.concatenate(d["te_y"])
        cm = classification_metrics(te_p, te_y)
        train_ll = float(np.mean(d["tr"]))
        out[s] = {"name": d["name"], "train_ll": train_ll, "test_ll": cm["log_loss"],
                  "gap": cm["log_loss"] - train_ll, "auc": cm["auc"], "acc": cm["accuracy"]}
    return out


def main():
    results = {}
    for coin, interval in CELLS:
        r = train_oos_gap(coin, interval)
        results[f"{coin}-{interval}"] = r
        print(f"\n=== {coin.upper()} {interval} ===")
        print(f"  {'model':24s} {'trainLL':>8} {'testLL':>8} {'gap':>8} {'AUC':>7} {'acc%':>6}")
        for s in ["ising", "free", "gbm", "mlp", "gbm_t", "mlp_t"]:
            m = r[s]
            print(f"  {m['name']:24s} {m['train_ll']:8.4f} {m['test_ll']:8.4f} "
                  f"{m['gap']:8.4f} {m['auc']:7.4f} {m['acc']*100:6.2f}")
    (OUT / "ml_baseline.json").write_text(json.dumps(results, indent=2))
    # aggregate: how often does Ising beat GBM/MLP on OOS log-loss?
    n = len(results)
    bw = {s: sum(1 for r in results.values() if r["ising"]["test_ll"] < r[s]["test_ll"])
          for s in ["free", "gbm", "mlp", "gbm_t", "mlp_t"]}
    print(f"\nIsing OOS log-loss better than: free {bw['free']}/{n}, GBM {bw['gbm']}/{n}, "
          f"MLP {bw['mlp']}/{n}, GBM-tuned {bw['gbm_t']}/{n}, MLP-tuned {bw['mlp_t']}/{n}")
    print(f"Wrote {OUT/'ml_baseline.json'}")


if __name__ == "__main__":
    main()
