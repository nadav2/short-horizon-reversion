"""Cross-sectional microstructure regression of per-pair predictability
(referee point 5).

The liquidity script reports univariate rank correlations of AUC with volume
and thinness. A referee for a finance journal wants the microstructure
variables entered jointly, so that the marginal contribution of each is
visible and the flat-bar (staleness) channel is separated from a genuine
volume/volatility gradient. We regress each crypto pair's flat-bar-robust
out-of-sample AUC (auc_nz) on standardized log dollar volume, flat-bar
fraction, and median |return| (a per-pair realized-volatility proxy), with
HC1 heteroskedasticity-robust standard errors.

    uv run --active python -m paper.micro_regression
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"


def ols_hc1(X, y):
    """OLS with HC1 robust SEs. X includes intercept column. Returns beta, se, t."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    # HC1
    S = (X * resid[:, None])
    meat = S.T @ S
    cov = XtX_inv @ meat @ XtX_inv * (n / (n - k))
    se = np.sqrt(np.diag(cov))
    t = beta / se
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot
    return beta, se, t, r2


def main():
    liq = json.loads((OUT / "liquidity.json"))["assets"] if False else \
        json.loads((OUT / "liquidity.json").read_text())["assets"]
    crypto = [a for a in liq if a["class"] == "crypto"
              and a.get("volume") and a["volume"] > 0
              and np.isfinite(a.get("auc_nz_ising", np.nan))]
    print(f"{len(crypto)} crypto pairs with complete microstructure covariates")

    auc = np.array([a["auc_nz_ising"] for a in crypto])           # flat-bar-robust AUC
    logvol = np.log(np.array([a["volume"] for a in crypto]))
    zerofrac = np.array([a["zero_frac"] for a in crypto])
    medabs = np.array([a["med_abs_change"] for a in crypto])

    def z(v):
        return (v - v.mean()) / v.std(ddof=0)

    names = ["intercept", "log_volume", "flat_frac", "med_abs_ret"]
    X = np.column_stack([np.ones(len(auc)), z(logvol), z(zerofrac), z(medabs)])
    beta, se, t, r2 = ols_hc1(X, auc)

    res = {"n": len(crypto), "dep_var": "auc_nz_ising (flat-bar-robust)",
           "standardized": True, "r2": float(r2), "coef": {}}
    print(f"\nDependent: flat-bar-robust OOS Ising AUC (crypto pairs)")
    print(f"Standardized predictors, HC1 robust SEs.  R^2 = {r2:.3f}\n")
    print(f"{'term':<14}{'beta':>10}{'se':>10}{'t':>8}")
    for nm, b, s, tt in zip(names, beta, se, t):
        res["coef"][nm] = {"beta": float(b), "se": float(s), "t": float(tt)}
        print(f"{nm:<14}{b:>10.4f}{s:>10.4f}{tt:>8.2f}")

    # also the mean AUC the intercept corresponds to, and simple correlations
    res["mean_auc_nz"] = float(auc.mean())
    (OUT / "micro_regression.json").write_text(json.dumps(res, indent=2))
    print(f"\nMean flat-bar-robust AUC = {auc.mean():.4f}")
    print(f"Wrote {OUT/'micro_regression.json'}")


if __name__ == "__main__":
    main()
