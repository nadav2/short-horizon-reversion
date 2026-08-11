"""Within-panel multiplicity control and the family-level inheritance slope
(referee response, Tier 1 #2 and #3).

Two gaps in the wrapper panel's inference, both closed here without any refit.

(a) MULTIPLICITY. The wide study applies Benjamini-Hochberg jointly across all 370
    assets, but the 33 panel legs of Table `tab:panel` are reported at raw 5%. The
    panel's headline counts ("all eight spot-Bitcoin funds individually two-model
    significant") are therefore uncorrected. We apply BH at q=0.05 within the panel
    family and report which cells survive.

(b) INHERITANCE IS A SLOPE, NOT A CORRELATION. `family_inference.py` reports
    corr(wrapper mean, underlying) = 0.658 across the eight families. But inheritance
    is the claim wrapper = underlying, i.e. slope 1 and intercept 0 in
      wrapper_mean = a + b * underlying.
    A correlation cannot distinguish b=1 from b=0.3, and it is maximised by any
    monotone relation -- including one driven by a family (FXE) that visibly fails to
    inherit. We fit the regression, and test b against BOTH hypotheses the design was
    built to separate: full inheritance (b=1) and no inheritance (b=0).

    uv run --active python -m paper.panel_fdr
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"

# Panel legs as printed in Table `tab:panel` (conjunction p-values, all-bars fit).
# role: wrapper | underlying | correlated
PANEL: list[tuple[str, str, str, float]] = [
    ("gold", "XAUUSD", "underlying", 0.282),
    ("gold", "GLD", "wrapper", 0.233),
    ("gold", "GDX", "correlated", 0.605),
    ("gold", "NEM", "correlated", 0.704),
    ("silver", "XAGUSD", "underlying", 0.546),
    ("silver", "SLV", "wrapper", 0.276),
    ("platinum", "XPTUSD", "underlying", 0.001),
    ("platinum", "PPLT", "wrapper", 0.129),
    ("palladium", "XPDUSD", "underlying", 0.006),
    ("palladium", "PALL", "wrapper", 0.157),
    ("bitcoin", "BTC", "underlying", 0.025),
    ("bitcoin", "IBIT", "wrapper", 0.004),
    ("bitcoin", "FBTC", "wrapper", 0.008),
    ("bitcoin", "BITB", "wrapper", 0.002),
    ("bitcoin", "ARKB", "wrapper", 0.009),
    ("bitcoin", "HODL", "wrapper", 0.000),
    ("bitcoin", "BTCO", "wrapper", 0.017),
    ("bitcoin", "EZBC", "wrapper", 0.026),
    ("bitcoin", "GBTC", "wrapper", 0.005),
    ("bitcoin", "BITO", "wrapper", 0.030),
    ("bitcoin", "BITU", "wrapper", 0.004),
    ("bitcoin", "COIN", "correlated", 0.364),
    ("bitcoin", "MSTR", "correlated", 0.248),
    ("ether", "ETH", "underlying", 0.150),
    ("ether", "ETHA", "wrapper", 0.407),
    ("ether", "FETH", "wrapper", 0.398),
    ("ether", "ETHE", "wrapper", 0.345),
    ("ether", "ETHU", "wrapper", 0.378),
    ("euro", "EURUSD", "underlying", 0.263),
    ("euro", "FXE", "wrapper", 0.967),
    ("euro", "UUP", "correlated", 0.472),
    ("yen", "USDJPY", "underlying", 0.148),
    ("yen", "FXY", "wrapper", 0.638),
]


def bh(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg mask at level q."""
    n = len(pvals)
    order = np.argsort(pvals, kind="stable")
    thresh = (np.arange(1, n + 1) / n) * q
    passed = pvals[order] <= thresh
    mask = np.zeros(n, bool)
    if passed.any():
        kmax = np.max(np.flatnonzero(passed))
        mask[order[: kmax + 1]] = True
    return mask


def panel_multiplicity(q: float = 0.05) -> dict:
    names = [n for _, n, _, _ in PANEL]
    roles = [r for _, _, r, _ in PANEL]
    fams = [f for f, _, _, _ in PANEL]
    p = np.array([x for _, _, _, x in PANEL])
    raw = p < q
    keep = bh(p, q)

    lost = [names[i] for i in range(len(p)) if raw[i] and not keep[i]]
    btc_w = [i for i in range(len(p)) if fams[i] == "bitcoin" and roles[i] == "wrapper"
             and names[i] not in ("BITO", "BITU")]
    return {
        "n_legs": len(p),
        "q": q,
        "n_sig_raw": int(raw.sum()),
        "n_sig_bh": int(keep.sum()),
        "lost_under_bh": lost,
        "bh_threshold_at_last_pass": float(np.sort(p)[keep.sum() - 1]) if keep.any() else None,
        "spot_btc_funds_raw": f"{sum(raw[i] for i in btc_w)}/{len(btc_w)}",
        "spot_btc_funds_bh": f"{sum(keep[i] for i in btc_w)}/{len(btc_w)}",
        "surviving": [names[i] for i in range(len(p)) if keep[i]],
    }


def inheritance_slope(n_boot: int = 20000, seed: int = 20260811) -> dict:
    fam = json.loads((OUT / "family_inference.json").read_text())["families"]
    keys = sorted(fam)
    x = np.array([fam[k]["underlying_auc"] for k in keys])
    y = np.array([fam[k]["wrapper_mean_auc"] for k in keys])
    n = len(x)

    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 2
    s2 = float(resid @ resid / dof)
    cov = s2 * np.linalg.inv(X.T @ X)
    se_b = float(np.sqrt(cov[1, 1]))
    se_a = float(np.sqrt(cov[0, 0]))
    b, a = float(beta[1]), float(beta[0])

    # two-sided t tests against full inheritance (b=1) and no inheritance (b=0)
    from math import erf, sqrt

    def t_p(t: float, df: int) -> float:
        """Two-sided p from a t statistic, normal approximation with small-df widening."""
        # exact-enough for reporting: use the survival function of |t| under t(df)
        # via the incomplete beta relation implemented with a numeric integral.
        import math
        xs = np.linspace(-50, 50, 400001)
        pdf = (math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))) * \
              (1 + xs**2 / df) ** (-(df + 1) / 2)
        cdf = np.trapezoid(pdf[xs <= abs(t)], xs[xs <= abs(t)])
        return float(2 * (1 - cdf))

    # leave-one-family-out sensitivity on the reported correlation
    loo = {}
    for i, k in enumerate(keys):
        m = np.ones(n, bool)
        m[i] = False
        loo[k] = float(np.corrcoef(x[m], y[m])[0, 1])
    # leave-one-CLASS-out: drop the two crypto families
    crypto = [i for i, k in enumerate(keys) if k in ("bitcoin", "ether")]
    m = np.ones(n, bool)
    m[crypto] = False
    corr_nocrypto = float(np.corrcoef(x[m], y[m])[0, 1])

    # n=8 families -> df=6; use the t critical value, not the normal one
    tcrit = {4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}.get(dof, 1.96)
    return {
        "n_families": n,
        "dof": dof, "t_crit_975": tcrit,
        "slope": b, "slope_se": se_b,
        "slope_ci95": [b - tcrit * se_b, b + tcrit * se_b],
        "intercept": a, "intercept_se": se_a,
        "t_vs_full_inheritance_b1": (b - 1) / se_b,
        "p_vs_full_inheritance_b1": t_p((b - 1) / se_b, dof),
        "t_vs_no_inheritance_b0": b / se_b,
        "p_vs_no_inheritance_b0": t_p(b / se_b, dof),
        "corr_all": float(np.corrcoef(x, y)[0, 1]),
        "corr_leave_one_family_out": loo,
        "corr_excluding_crypto_families": corr_nocrypto,
        "n_excluding_crypto": int(m.sum()),
    }


if __name__ == "__main__":
    res = {"multiplicity": panel_multiplicity(), "inheritance": inheritance_slope()}
    (OUT / "panel_fdr.json").write_text(json.dumps(res, indent=2))

    m = res["multiplicity"]
    print("── within-panel BH ──")
    print(f"  legs {m['n_legs']}, significant raw {m['n_sig_raw']} -> BH {m['n_sig_bh']}")
    print(f"  lost under BH: {', '.join(m['lost_under_bh'])}")
    print(f"  spot-Bitcoin funds: {m['spot_btc_funds_raw']} raw -> {m['spot_btc_funds_bh']} BH")

    i = res["inheritance"]
    print("\n── family inheritance slope ──")
    print(f"  slope {i['slope']:.3f} (SE {i['slope_se']:.3f}) CI "
          f"[{i['slope_ci95'][0]:.2f},{i['slope_ci95'][1]:.2f}]  intercept {i['intercept']:.3f}")
    print(f"  vs full inheritance b=1: t={i['t_vs_full_inheritance_b1']:+.2f} p={i['p_vs_full_inheritance_b1']:.3f}")
    print(f"  vs no   inheritance b=0: t={i['t_vs_no_inheritance_b0']:+.2f} p={i['p_vs_no_inheritance_b0']:.3f}")
    print(f"  corr all {i['corr_all']:.3f}; excluding crypto families {i['corr_excluding_crypto_families']:.3f}")
