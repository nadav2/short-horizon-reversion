"""Multi-lag SIGN autocorrelation: the model-free statistic that matches the model.

The paper's model-free reference has been the lag-1 return autocorrelation rho_1,
which is ~0 everywhere including crypto. But the constrained logit reads TWELVE
lags of the *soft spin*, and the lambda-sweep shows the effect survives from a
near-linear encoding (lambda=50) to a near-sign encoding (lambda=1000). So the
estimator is, to a good approximation, reading the SIGN sequence over many lags,
and rho_1 on returns is not its model-free counterpart -- it is a one-lag,
magnitude-weighted statistic compared against a twelve-lag, sign-weighted model.
That comparison is underpowered by construction, not by nature, and it makes the
finding look more estimator-dependent than it is.

This module computes the matched statistic on the sign series
sigma_t = sign(r_t) in {-1, 0, +1} (flat bars enter as 0, so the r_t == 0 label
convention cannot contribute -- the statistic is flat-bar-artifact-free by
construction):

  * rho_k^sign for k = 1..12, per asset;
  * a joint Ljung-Box Q(12) on the sign series with its chi^2_12 p-value;
  * the kernel-weighted sign correlation
        R_kernel = corr( sigma_t , sum_{k=1..12} sigma_{t-k} / k^alpha )
    with alpha FIXED at 1.0 a priori (no fitted parameter enters, so this is a
    fully model-free statistic, not a re-read of the fitted model);
  * the class contrast crypto vs stocks, with a JOINT moving-block bootstrap on
    the shared 15m grid -- the same dependence-aware inference used for the AUC
    gap in dependence.py;
  * the class-pooled sign-flip rate by decile of |r_{t-1}| (flat bars excluded
    on both sides), i.e. per-asset decile curves averaged within class -- the
    mechanism panel: reversal strengthens with the size of the preceding move.

    uv run python -m paper.signlag
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BULK = Path(__file__).resolve().parent / "bulk_data"
OUT = Path(__file__).resolve().parent / "out"
SLOT = 900                 # 15m in seconds
BLOCK = 384                # ~4 days, matching the AUC bootstrap
N_BOOT, SEED = 1000, 7
N_LAGS = 12
ALPHA_FIXED = 1.0          # declared a priori; NOT the fitted alpha
MIN_CANDLES = 8000
MIN_OBS = 1000


def sign_series(ch: np.ndarray) -> np.ndarray:
    """sigma_t = sign(r_t) in {-1, 0, +1}. Flat bars contribute zero weight, so no
    part of this statistic depends on the up = 1[r>0] flat-bar convention."""
    return np.sign(np.asarray(ch, float))


def autocorr_k(x: np.ndarray, k: int) -> float:
    x = np.asarray(x, float) - np.mean(x)
    den = float(np.sum(x * x))
    return float(np.sum(x[k:] * x[:-k]) / den) if den > 0 else 0.0


def ljung_box(x: np.ndarray, m: int = N_LAGS) -> tuple[float, float, list[float]]:
    """Ljung-Box Q(m) and its chi^2_m p-value for the series x."""
    from scipy.stats import chi2
    n = len(x)
    rhos = [autocorr_k(x, k) for k in range(1, m + 1)]
    Q = n * (n + 2) * sum(r * r / (n - k) for k, r in enumerate(rhos, start=1))
    return float(Q), float(chi2.sf(Q, m)), [float(r) for r in rhos]


def kernel_field(sig: np.ndarray, alpha: float = ALPHA_FIXED, n_lags: int = N_LAGS) -> np.ndarray:
    """field_t = sum_{k=1..N} sigma_{t-k} / k^alpha (strictly causal)."""
    n = len(sig)
    field = np.zeros(n, float)
    for k in range(1, n_lags + 1):
        field[k:] += sig[:n - k] / (k ** alpha)
    return field


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / den) if den > 0 else 0.0


def flip_by_decile(ch: np.ndarray, n_bins: int = 10) -> list[float] | None:
    """Strict sign-flip rate P(sign(r_t) = -sign(r_{t-1})) by decile of |r_{t-1}|,
    with flat bars excluded on BOTH sides. Without the exclusion a flat next bar
    counts as a "flip" (sign 0 != sign r_{t-1}), which inflates the rate on
    illiquid assets exactly where |r_{t-1}| is small -- the pooled stock curve
    then bends UP at the small-move end for reasons that have nothing to do
    with reversal. Returns None when an asset's deciles are degenerate (e.g.
    stablecoins whose |r| quantiles collapse onto identical values)."""
    r_prev, r_now = ch[:-1], ch[1:]
    keep = (r_prev != 0) & (r_now != 0)
    r_prev, r_now = r_prev[keep], r_now[keep]
    if len(r_prev) < MIN_OBS:
        return None
    edges = np.quantile(np.abs(r_prev), np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.inf
    rates = []
    for i in range(n_bins):
        m = (np.abs(r_prev) >= edges[i]) & (np.abs(r_prev) < edges[i + 1])
        if m.sum() < 50:
            return None
        rates.append(float(np.mean(np.sign(r_now[m]) == -np.sign(r_prev[m]))))
    return rates


def reversal_by_magnitude(ch: np.ndarray, n_bins: int = 10) -> dict:
    """Why rho_1(return) ~ 0 while rho_1(sign) << 0: the sign-flip rate and the
    magnitude-weighted covariance contribution, by decile of |r_{t-1}|.

    rho_1 on returns weights each (t-1, t) pair by r_{t-1} * r_t, so the top
    decile of |r_{t-1}| dominates it. If the flip rate is elevated at typical
    move sizes but the covariance in the largest bin is offsetting, the
    magnitude-weighted statistic cancels while the sign-weighted one does not."""
    r_prev, r_now = ch[:-1], ch[1:]
    keep = r_prev != 0
    r_prev, r_now = r_prev[keep], r_now[keep]
    edges = np.quantile(np.abs(r_prev), np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.inf
    x = r_prev - ch.mean(); y = r_now - ch.mean()
    # normalise by the VARIANCE, so the per-decile terms sum exactly to rho_1
    denom = float(np.sum((ch - ch.mean()) ** 2))
    out = []
    for i in range(n_bins):
        m = (np.abs(r_prev) >= edges[i]) & (np.abs(r_prev) < edges[i + 1])
        if m.sum() < 50:
            continue
        flip = float(np.mean(np.sign(r_now[m]) != np.sign(r_prev[m])))
        out.append({
            "decile": i + 1,
            "n": int(m.sum()),
            "median_abs_r_prev": float(np.median(np.abs(r_prev[m]))),
            "flip_rate": flip,
            # contribution of this decile to rho_1 (the terms sum to rho_1)
            "rho1_contrib": float(np.sum(x[m] * y[m]) / denom) if denom > 0 else 0.0,
        })
    return {"bins": out, "total_rho1": autocorr_k(ch, 1),
            "sum_contrib": float(sum(b["rho1_contrib"] for b in out))}


# ── per-asset statistics ─────────────────────────────────────────────────────

def asset_stats(ch: np.ndarray) -> dict:
    sig = sign_series(ch)
    Q, p_lb, rhos = ljung_box(sig, N_LAGS)
    field = kernel_field(sig)
    valid = slice(N_LAGS, None)
    return {
        "rho1_return": autocorr_k(ch, 1),
        "rho1_sign": rhos[0],
        "rho_sign_lags": rhos,
        "sum_abs_rho_sign": float(np.sum(np.abs(rhos))),
        "ljung_box_Q": Q,
        "ljung_box_p": p_lb,
        "R_kernel": corr(sig[valid], field[valid]),
        "n": int(len(ch)),
        "flat_frac": float(np.mean(ch == 0.0)),
    }


# ── joint, dependence-preserving bootstrap of the class gap in R_kernel ──────

def joint_kernel_bootstrap(assets: dict[str, str]) -> dict:
    """assets: {asset -> class}. Resample 15m time blocks ONCE on the shared grid
    and apply to every asset simultaneously, preserving cross-sectional dependence."""
    data = {}
    for a, cls in assets.items():
        raw = json.loads((BULK / f"{a}-15m.json").read_text())
        raw.sort(key=lambda d: d["timestamp"])
        ch = np.array([d.get("change", 0.0) for d in raw], float)
        if len(ch) < MIN_CANDLES:
            continue
        slot = np.array([d["timestamp"] // SLOT for d in raw], np.int64)
        sig = sign_series(ch)
        field = kernel_field(sig)
        data[a] = {"cls": cls, "slot": slot[N_LAGS:],
                   "sig": sig[N_LAGS:], "field": field[N_LAGS:]}

    lo = min(int(d["slot"].min()) for d in data.values())
    hi = max(int(d["slot"].max()) for d in data.values())
    T = hi - lo + 1
    for d in data.values():
        lut = np.full(T, -1, np.int32)
        lut[d["slot"] - lo] = np.arange(len(d["slot"]), dtype=np.int32)
        d["lut"] = lut

    def class_means(take_slots):
        per = {"crypto": [], "stock": []}
        for d in data.values():
            if take_slots is None:
                s, f = d["sig"], d["field"]
            else:
                obs = d["lut"][take_slots]
                obs = obs[obs >= 0]
                if len(obs) < MIN_OBS:
                    continue
                s, f = d["sig"][obs], d["field"][obs]
            per[d["cls"]].append(corr(s, f))
        return {c: float(np.mean(v)) for c, v in per.items() if v}

    obs = class_means(None)
    obs_gap = obs["crypto"] - obs["stock"]

    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(T / BLOCK))
    gaps = []
    for b in range(N_BOOT):
        starts = rng.integers(0, max(1, T - BLOCK + 1), size=nb)
        slots = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:T]
        m = class_means(slots)
        if "crypto" in m and "stock" in m:
            gaps.append(m["crypto"] - m["stock"])
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b+1}/{N_BOOT}", flush=True)
    g = np.array(gaps)
    return {
        "n_assets_used": len(data), "n_boot": N_BOOT, "block_slots": BLOCK,
        "alpha_fixed": ALPHA_FIXED, "n_lags": N_LAGS,
        "crypto_mean_R": obs["crypto"], "stock_mean_R": obs["stock"],
        "obs_gap": obs_gap,
        "gap_ci": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
        "p_gap_ge0": float(np.mean(g >= 0)),
    }


def main():
    wide = {r["asset"]: r for r in json.loads((OUT / "wide.json").read_text())}
    rows = []
    flips: dict[str, list[list[float]]] = {"crypto": [], "stock": []}
    files = sorted(BULK.glob("*-15m.json"))
    for i, f in enumerate(files):
        asset = f.name[: -len("-15m.json")]
        w = wide.get(asset)
        if w is None:
            continue
        raw = json.loads(f.read_text())
        raw.sort(key=lambda d: d["timestamp"])
        ch = np.array([d.get("change", 0.0) for d in raw], float)
        if len(ch) < MIN_CANDLES:
            continue
        st = asset_stats(ch)
        st.update({"asset": asset, "class": w["class"],
                   "auc_ising": w["auc_ising"], "A": w["A"]})
        rows.append(st)
        if w["class"] in flips:
            fr = flip_by_decile(ch)
            if fr is not None:
                flips[w["class"]].append(fr)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(files)}]", flush=True)

    summary = {}
    for c in ("crypto", "stock"):
        sel = [r for r in rows if r["class"] == c]
        arr = lambda k: np.array([r[k] for r in sel], float)
        summary[c] = {
            "n": len(sel),
            "mean_rho1_return": float(arr("rho1_return").mean()),
            "mean_rho1_sign": float(arr("rho1_sign").mean()),
            "mean_R_kernel": float(arr("R_kernel").mean()),
            "median_R_kernel": float(np.median(arr("R_kernel"))),
            "frac_R_negative": float(np.mean(arr("R_kernel") < 0)),
            "frac_lb_sig": float(np.mean(arr("ljung_box_p") < 0.05)),
            "median_lb_p": float(np.median(arr("ljung_box_p"))),
            "mean_rho_by_lag": [float(np.mean([r["rho_sign_lags"][k] for r in sel]))
                                for k in range(N_LAGS)],
        }
    # class-pooled sign-flip rate by decile of |r_{t-1}| (flat bars excluded):
    # per-asset decile curves averaged within class, with the cross-asset s.e.
    summary["flip_by_class"] = {}
    for c, curves in flips.items():
        arr = np.array(curves)
        summary["flip_by_class"][c] = {
            "n_assets": int(arr.shape[0]),
            "mean": [float(x) for x in arr.mean(axis=0)],
            "se": [float(x) for x in arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])],
            # fraction of assets whose top-3-decile mean exceeds the bottom-3
            "frac_rising": float(np.mean(arr[:, -3:].mean(axis=1) > arr[:, :3].mean(axis=1))),
        }

    # magnitude decomposition on the focal coins + a stock reference
    summary["magnitude_decomposition"] = {}
    for a in ("btc", "eth", "xrp", "sol", "aapl"):
        f = BULK / f"{a}-15m.json"
        if not f.exists():
            continue
        raw = sorted(json.loads(f.read_text()), key=lambda d: d["timestamp"])
        ch = np.array([d.get("change", 0.0) for d in raw], float)
        summary["magnitude_decomposition"][a] = reversal_by_magnitude(ch)

    R = np.array([r["R_kernel"] for r in rows]); AU = np.array([r["auc_ising"] for r in rows])
    A = np.array([r["A"] for r in rows]); isc = np.array([r["class"] == "crypto" for r in rows])
    summary["corr_Rkernel_auc"] = float(np.corrcoef(R, AU)[0, 1])
    summary["corr_Rkernel_A"] = float(np.corrcoef(R, A)[0, 1])
    summary["within_crypto_corr_Rkernel_auc"] = float(np.corrcoef(R[isc], AU[isc])[0, 1])

    print("\n=== joint block bootstrap of the class gap in R_kernel ===")
    summary["joint_gap"] = joint_kernel_bootstrap({r["asset"]: r["class"] for r in rows})

    (OUT / "signlag.json").write_text(json.dumps({"summary": summary, "assets": rows}, indent=2))

    print("\n=== Multi-lag SIGN autocorrelation (15m, sigma_t = sign(r_t)) ===")
    for c in ("crypto", "stock"):
        s = summary[c]
        print(f"  {c:6s} n={s['n']:4d}  rho1(return)={s['mean_rho1_return']:+.4f}  "
              f"rho1(sign)={s['mean_rho1_sign']:+.4f}  R_kernel={s['mean_R_kernel']:+.4f}  "
              f"frac R<0 {s['frac_R_negative']*100:.0f}%  Ljung-Box p<.05 in {s['frac_lb_sig']*100:.0f}%")
    jg = summary["joint_gap"]
    print(f"\n  gap in R_kernel = {jg['obs_gap']:+.4f}  CI=[{jg['gap_ci'][0]:+.4f},{jg['gap_ci'][1]:+.4f}]  "
          f"p(gap>=0)={jg['p_gap_ge0']:.4f}")
    print(f"  corr(R_kernel, OOS AUC) = {summary['corr_Rkernel_auc']:+.3f}   "
          f"corr(R_kernel, A) = {summary['corr_Rkernel_A']:+.3f}")
    print(f"\nWrote {OUT/'signlag.json'}")


if __name__ == "__main__":
    main()
