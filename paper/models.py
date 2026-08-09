"""Directional-prediction models, all fit by maximum likelihood (log-loss).

Every estimator exposes the same interface so the walk-forward harness can treat
them uniformly:

    fit(changes, ups, train_lo, train_hi)
        Estimate parameters on the TRAIN window [train_lo, train_hi). Any single
        hyperparameter (ridge/lasso strength C, or the Ising decay exponent alpha)
        is selected to MINIMIZE the negative log-likelihood on a chronological
        validation tail of TRAIN, then the model is refit on the full TRAIN window.

    predict_series(changes, ups, test_lo, test_hi) -> np.ndarray of P(up)
        One probability per test index. Strictly causal: index t uses only data
        before t (frozen TRAIN parameters + the actually-observed preceding returns).

The Ising model and the AR-logit baselines therefore differ ONLY in their
hypothesis space (3 power-law-tied parameters vs. N+1 free / penalized weights),
never in the fitting objective — this isolates the effect of the structural prior.
"""

from __future__ import annotations

import warnings

import numpy as np

from .common import lag_matrix, power_law_field, spins

VAL_FRAC = 0.20
N_LAGS = 12
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
ALPHA_GRID = tuple(np.round(np.linspace(0.0, 3.0, 31), 3))   # 0.0, 0.1, ..., 3.0
SPIN_SCALE = 150.0
# Volatility-standardized spin scale: lambda = VOL_SPIN_C / std(train returns).
# 0.45 reproduces lambda ~= 150 on BTC-15m (sigma ~= 30 bp), so the fixed and
# standardized variants coincide on the reference asset.
VOL_SPIN_C = 0.45


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _nll(p, y):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _val_cut(train_lo, train_hi):
    n = train_hi - train_lo
    return train_lo + int(round(n * (1 - VAL_FRAC)))


# ── Ising power-law logit (the proposed 3-parameter structural regularizer) ──

class IsingLogit:
    r"""P(up_t) = sigmoid( C + A * field_t ),  field_t = sum_{k=1}^{N} s_{t-k} / k^alpha,
    s = tanh(spin_scale * r).

    Identifiable parameters: intercept C = 2*beta*h0, amplitude A = 2*beta*J0, and the
    power-law decay exponent alpha. The inverse temperature beta is absorbed into (C, A)
    and is not separately identifiable from the likelihood. alpha is profiled over a grid
    by validation NLL; (C, A) are fit by unpenalized logistic MLE."""

    name = "Ising power-law logit"
    short = "ising"

    def __init__(self, n_lags: int = N_LAGS, spin_scale: float = SPIN_SCALE,
                 alpha_grid=ALPHA_GRID, spin_mode: str = "fixed"):
        assert spin_mode in ("fixed", "vol")
        self.n_lags = n_lags
        self.spin_scale = spin_scale
        self.alpha_grid = alpha_grid
        self.spin_mode = spin_mode
        self.alpha = self.C = self.A = None
        self._sp = None

    def _fit_CA(self, field, y):
        """Unpenalized logistic MLE of (C, A) on the single field feature."""
        from sklearn.linear_model import LogisticRegression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticRegression(C=1e8, solver="lbfgs", max_iter=2000)
            m.fit(field.reshape(-1, 1), y)
        return float(m.coef_[0, 0]), float(m.intercept_[0])

    def fit(self, changes, ups, train_lo, train_hi):
        if self.spin_mode == "vol":
            # causal: the scale is set on the TRAIN window only, then frozen
            sd = float(np.std(changes[train_lo:train_hi]))
            self.spin_scale = (VOL_SPIN_C / sd) if sd > 0 else SPIN_SCALE
        self._sp = spins(changes, self.spin_scale)
        y_all = ups.astype(int)
        cut = _val_cut(train_lo, train_hi)

        best = (np.inf, None)
        for a in self.alpha_grid:
            field = power_law_field(self._sp, a, self.n_lags)
            yf = y_all[train_lo:cut]
            if len(np.unique(yf)) < 2:
                continue
            A, C = self._fit_CA(field[train_lo:cut], yf)
            pv = _sigmoid(C + A * field[cut:train_hi])
            ll = _nll(pv, y_all[cut:train_hi])
            if ll < best[0]:
                best = (ll, a)
        self.alpha = best[1] if best[1] is not None else 1.0

        field = power_law_field(self._sp, self.alpha, self.n_lags)
        self.A, self.C = self._fit_CA(field[train_lo:train_hi], y_all[train_lo:train_hi])

    def predict_series(self, changes, ups, test_lo, test_hi):
        sp = self._sp if self._sp is not None else spins(changes, self.spin_scale)
        field = power_law_field(sp, self.alpha, self.n_lags)
        return _sigmoid(self.C + self.A * field)[test_lo:test_hi]

    def implied_weights(self) -> np.ndarray:
        """Per-lag effective weight A / k^alpha (k = 1..N), for plotting the kernel."""
        k = np.arange(1, self.n_lags + 1)
        return self.A / (k ** self.alpha)

    def params(self):
        return {"alpha": self.alpha, "A": self.A, "C": self.C,
                "n_lags": self.n_lags, "spin_scale": self.spin_scale,
                "spin_mode": self.spin_mode, "n_params": 3}


# ── Autoregressive logistic regression (free / L1 / L2) ──────────────────────

class ARLogit:
    """Logistic regression on the last ``n_lags`` signed returns.

    mode='free' -> unpenalized (N+1 free weights).
    mode='l1'/'l2' -> Lasso/Ridge penalty; strength C chosen by validation NLL."""

    def __init__(self, mode: str, n_lags: int = N_LAGS, c_grid=C_GRID):
        assert mode in ("free", "l1", "l2")
        self.mode = mode
        self.n_lags = n_lags
        self.c_grid = c_grid
        self.model = self.scaler = None
        self.C = None
        self.name = {"free": "Free AR-logit",
                     "l1": "Lasso AR-logit (L1)",
                     "l2": "Ridge AR-logit (L2)"}[mode]
        self.short = {"free": "free", "l1": "l1", "l2": "l2"}[mode]

    def _estimator(self, C):
        from sklearn.linear_model import LogisticRegression
        if self.mode == "free":
            return LogisticRegression(C=1e8, solver="lbfgs", max_iter=3000)
        if self.mode == "l2":
            return LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=3000)
        # L1: liblinear gives the exact Lasso-logit MLE far faster than saga at these sizes.
        return LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=3000)

    def _Xy(self, changes, ups, lo, hi):
        X = lag_matrix(changes, self.n_lags)[lo:hi]
        y = ups.astype(int)[lo:hi]
        ok = ~np.isnan(X).any(axis=1)
        return X[ok], y[ok], ok

    def fit(self, changes, ups, train_lo, train_hi):
        from sklearn.preprocessing import StandardScaler
        lo = max(train_lo, self.n_lags)
        X, y, _ = self._Xy(changes, ups, lo, train_hi)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.mode == "free":
                self.C = float("inf")
                self.model = self._estimator(None).fit(Xs, y)
                return
            # choose C by validation-tail NLL
            cut = int(len(Xs) * (1 - VAL_FRAC))
            best = (np.inf, self.c_grid[0])
            if len(np.unique(y[:cut])) > 1 and cut < len(Xs):
                scv = StandardScaler().fit(X[:cut])
                Xf, Xv = scv.transform(X[:cut]), scv.transform(X[cut:])
                for C in self.c_grid:
                    m = self._estimator(C).fit(Xf, y[:cut])
                    ll = _nll(m.predict_proba(Xv)[:, 1], y[cut:])
                    if ll < best[0]:
                        best = (ll, C)
            self.C = best[1]
            self.model = self._estimator(self.C).fit(Xs, y)

    def predict_series(self, changes, ups, test_lo, test_hi):
        X = lag_matrix(changes, self.n_lags)[test_lo:test_hi]
        bad = np.isnan(X).any(axis=1)
        Xf = np.where(np.isnan(X), 0.0, X)
        p = self.model.predict_proba(self.scaler.transform(Xf))[:, 1]
        p[bad] = 0.5
        return p

    def coef(self) -> np.ndarray:
        """Per-lag weights on the *standardized* features (col 0 = lag 1)."""
        return self.model.coef_.ravel()

    def params(self):
        nz = int(np.sum(np.abs(self.coef()) > 1e-6))
        return {"mode": self.mode, "C": self.C, "n_lags": self.n_lags,
                "n_params": self.n_lags + 1, "n_nonzero": nz}


# ── Markov chains on the direction sequence ──────────────────────────────────

class MarkovModel:
    def __init__(self, order: int = 1, laplace: float = 1.0):
        self.order = order
        self.laplace = laplace
        self.probs: dict[tuple, float] = {}
        self.default = 0.5
        self.name = f"Markov order-{order}"
        self.short = f"markov{order}"

    def fit(self, changes, ups, train_lo, train_hi):
        m = self.order
        counts: dict[tuple, list[float]] = {}
        u = ups.astype(bool)
        for t in range(max(train_lo, m), train_hi):
            state = tuple(u[t - m:t])
            c = counts.setdefault(state, [self.laplace, self.laplace])
            c[1 if u[t] else 0] += 1
        self.probs = {s: c[1] / (c[0] + c[1]) for s, c in counts.items()}
        self.default = float(u[train_lo:train_hi].mean())

    def predict_series(self, changes, ups, test_lo, test_hi):
        m = self.order
        u = ups.astype(bool)
        out = np.empty(test_hi - test_lo)
        for i, t in enumerate(range(test_lo, test_hi)):
            out[i] = self.probs.get(tuple(u[t - m:t]), self.default)
        return out

    def params(self):
        return {"order": self.order, "n_states": len(self.probs),
                "n_params": len(self.probs)}


class BaseRate:
    name = "Base rate"
    short = "base"

    def __init__(self):
        self.p = 0.5

    def fit(self, changes, ups, train_lo, train_hi):
        self.p = float(ups[train_lo:train_hi].astype(int).mean())

    def predict_series(self, changes, ups, test_lo, test_hi):
        return np.full(test_hi - test_lo, self.p)

    def params(self):
        return {"p_up": self.p, "n_params": 1}


def build_models(n_lags: int = N_LAGS):
    return [
        IsingLogit(n_lags=n_lags),
        ARLogit("free", n_lags=n_lags),
        ARLogit("l1", n_lags=n_lags),
        ARLogit("l2", n_lags=n_lags),
        MarkovModel(order=1),
        MarkovModel(order=2),
        BaseRate(),
    ]
