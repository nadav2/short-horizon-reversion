"""Two objections that could move the headline (referee response, Tier 2 #6 and #7).

Both are run on the model-free kernel sign correlation R of Eq. (2), which needs
nothing fitted, so neither result is contingent on the estimator under review. R
reproduces the class gap in the paper (-0.035), so it is the right instrument for
asking whether that gap survives a transformation of the data.

(6) COMMON FACTOR. With mean pairwise return correlation 0.40 the crypto
    cross-section carries N_eff ~ 2.5 independent observations, so "90% of 183 pairs"
    may be one market factor with 183 loadings. We build an equal-weight crypto market
    factor on the shared slot grid, project it out of every crypto pair (beta from the
    TRAINING span only, so nothing is fitted on the scored data), and recompute R on
    the residuals. If reversal is a property of the factor, residual R collapses to
    zero; if it is a property of individual pairs, residual R survives.

(7) INTRADAY VARIANCE. AUC ranks bars by a score that is a fixed function of raw
    returns, so when the conditional variance moves over the day the ranking pools
    regimes and real dependence can fail to rank (the paper's own SPX example:
    rho_1 = -0.030 yet AUC 0.492, lifting to 0.515 on variance homogenisation).
    Listed instruments carry 17.6x intraday heteroskedasticity against 5.0x for
    crypto, so the class gap may be a detectability gap. We standardise each series by
    a CAUSAL slot-of-day volatility profile (estimated on the training span only) and
    recompute R for both classes.

    uv run --active python -m paper.factor_variance
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dependence import BLOCK, OUT, SLOT, fast_auc  # noqa: F401
from .signlag import corr, kernel_field, sign_series

BULK = Path(__file__).resolve().parent / "bulk_data"
INTERVAL = "15m"
N_LAGS = 12
DAY = 96                       # 15m slots per 24h
TRAIN_FRAC = 0.5               # span used to estimate betas / vol profiles
N_BOOT, SEED = 1000, 23
MIN_BARS = 8720


def _load(asset: str):
    p = BULK / f"{asset}-{INTERVAL}.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    raw.sort(key=lambda d: d["timestamp"])
    ch = np.array([d.get("change", 0.0) for d in raw], float)
    ts = np.array([d["timestamp"] for d in raw], np.int64)
    if len(ch) < MIN_BARS:
        return None
    return ch, ts // SLOT


def _grid(assets: list[str]):
    series = {}
    for a in assets:
        r = _load(a)
        if r is not None:
            series[a] = r
    lo = min(int(s.min()) for _, s in series.values())
    hi = max(int(s.max()) for _, s in series.values())
    return series, lo, hi - lo + 1


def residualise_factor() -> dict:
    rows = json.loads((OUT / "wide.json").read_text())
    crypto = [r["asset"] for r in rows if r["class"] == "crypto"]
    series, lo, T = _grid(crypto)

    # dense matrix on the shared grid (nan = no bar)
    M = np.full((len(series), T), np.nan)
    names = sorted(series)
    for i, a in enumerate(names):
        ch, sl = series[a]
        M[i, sl - lo] = ch

    # equal-weight market factor: cross-sectional mean of available pairs per slot
    with np.errstate(invalid="ignore"):
        factor = np.nanmean(M, axis=0)
    cut = int(T * TRAIN_FRAC)

    out_raw, out_res, betas = [], [], []
    for i, a in enumerate(names):
        x = M[i]
        ok = np.isfinite(x) & np.isfinite(factor)
        tr = ok.copy()
        tr[cut:] = False
        if tr.sum() < 2000:
            continue
        # beta on the TRAINING span only
        f_tr, x_tr = factor[tr], x[tr]
        b = float(np.cov(x_tr, f_tr)[0, 1] / np.var(f_tr))
        betas.append(b)
        resid = np.where(ok, x - b * factor, np.nan)
        r_raw = corr(sign_series(x[ok])[N_LAGS:], kernel_field(sign_series(x[ok]))[N_LAGS:])
        rr = resid[ok]
        r_res = corr(sign_series(rr)[N_LAGS:], kernel_field(sign_series(rr))[N_LAGS:])
        if np.isfinite(r_raw) and np.isfinite(r_res):
            out_raw.append(r_raw)
            out_res.append(r_res)

    raw, res = np.array(out_raw), np.array(out_res)
    return {
        "n_pairs": len(raw),
        "beta_median": float(np.median(betas)),
        "R_raw_mean": float(raw.mean()),
        "R_residual_mean": float(res.mean()),
        "R_raw_frac_negative": float((raw < 0).mean()),
        "R_residual_frac_negative": float((res < 0).mean()),
        "retained_fraction": float(res.mean() / raw.mean()),
        "factor_R": None,
    }


def variance_standardise() -> dict:
    """Slot-of-day STRATIFIED AUC on the frozen scores.

    A sign statistic like R cannot answer this objection: R is invariant to dividing
    returns by any positive volatility profile, because that leaves every sign
    unchanged. The channel operates on AUC, which ranks bars by a score that is a
    fixed function of raw returns, so pooling bars from high- and low-variance hours
    into one ROC mixes regimes whose scores are not comparable.

    The direct test needs no refit: recompute each asset's AUC WITHIN each slot-of-day
    bucket and average over buckets (weighted by bucket size). Stratifying removes the
    cross-regime pooling exactly. If the class gap is a detectability artefact of
    pooling, the stratified gap should be materially larger than the pooled one --
    most of all for listed instruments, whose intraday variance profile is far steeper.
    """
    rows = json.loads((OUT / "wide.json").read_text())
    OOS = OUT / "wide_oos"
    res: dict[str, dict] = {}
    for cls in ("crypto", "stock"):
        pooled, strat, het = [], [], []
        for r in rows:
            if r["class"] != cls:
                continue
            f = OOS / f"{r['asset']}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            y, p, sl = z["actual"].astype(np.int8), z["p_ising"], z["ts"] // SLOT
            if len(np.unique(y)) < 2:
                continue
            a_pool = fast_auc(y, p)
            # hour-of-day buckets (4 slots each): listed instruments carry only ~6k
            # out-of-sample bars over 68 slots-of-day, too few per 15m slot to rank.
            tod = ((sl % DAY) // 4).astype(int)
            num = den = 0.0
            vols = []
            for s in np.unique(tod):
                m = tod == s
                if m.sum() < 100 or len(np.unique(y[m])) < 2:
                    continue
                a = fast_auc(y[m], p[m])
                if np.isfinite(a):
                    num += a * m.sum()
                    den += m.sum()
                vols.append(float(np.std(p[m])))
            if den == 0 or not np.isfinite(a_pool):
                continue
            pooled.append(a_pool)
            strat.append(num / den)
            if len(vols) >= 4:
                q = np.percentile(vols, [10, 90])
                if q[0] > 0:
                    het.append(float(q[1] / q[0]))
        res[cls] = {
            "n": len(pooled),
            "auc_pooled_mean": float(np.mean(pooled)),
            "auc_stratified_mean": float(np.mean(strat)),
            "median_score_dispersion_ratio_p90_p10": float(np.median(het)) if het else None,
        }
    res["gap_pooled"] = res["crypto"]["auc_pooled_mean"] - res["stock"]["auc_pooled_mean"]
    res["gap_stratified"] = res["crypto"]["auc_stratified_mean"] - res["stock"]["auc_stratified_mean"]
    res["gap_change"] = res["gap_stratified"] - res["gap_pooled"]
    return res


if __name__ == "__main__":
    f = residualise_factor()
    v = variance_standardise()
    (OUT / "factor_variance.json").write_text(json.dumps({"factor": f, "variance": v}, indent=2))

    print("── (6) common-factor decomposition, crypto ──")
    print(f"  pairs {f['n_pairs']}, median beta to market factor {f['beta_median']:.2f}")
    print(f"  mean R  raw {f['R_raw_mean']:+.4f}  -> residual {f['R_residual_mean']:+.4f}"
          f"   ({f['retained_fraction']*100:.0f}% retained)")
    print(f"  fraction negative  raw {f['R_raw_frac_negative']*100:.0f}%"
          f"  -> residual {f['R_residual_frac_negative']*100:.0f}%")

    print("\n── (7) slot-of-day stratified AUC (frozen scores, no refit) ──")
    for c in ("crypto", "stock"):
        d = v[c]
        print(f"  {c:7s} n={d['n']:3d}   AUC pooled {d['auc_pooled_mean']:.4f}"
              f" -> stratified {d['auc_stratified_mean']:.4f}")
    print(f"  class gap: pooled {v['gap_pooled']:+.5f} -> stratified {v['gap_stratified']:+.5f}"
          f"   (change {v['gap_change']:+.5f})")
