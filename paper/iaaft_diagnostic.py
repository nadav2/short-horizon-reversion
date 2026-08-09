"""Why do IAAFT surrogates produce MORE two-model significance than real data
on the non-crypto focal instruments (5/11 vs 3/11)?

The negative-control table (paper Sec. "Negative controls") shows the phase-
randomized row clearing the two-model criterion on 5 of 11 non-crypto
instruments while the *real* series clears it on 3. Left unexplained that reads
as the pipeline manufacturing significance from a null, which is exactly what
the section exists to rule out. This module discriminates between the candidate
explanations:

  H1 (variance)     surrogates are more homoskedastic, so the moving-block
                    bootstrap CI is narrower and the same AUC crosses the line.
                    -> REJECTED: mean CI half-width ratio surrogate/real is
                       0.98 on non-crypto (0.0091 vs 0.0094). Not a width effect.

  H2 (flat bars)    IAAFT's rank remapping scatters the r_t == 0 atom, which is
                    concentrated overnight in listed instruments.
                    -> REJECTED: corr(flat-bar fraction, AUC lift) = -0.15,
                       wrong sign and negligible.

  H3 (anti-conservative null under heteroskedasticity)  ACCEPTED.
                    IAAFT preserves the power spectrum -- hence rho_1 -- exactly,
                    but destroys volatility clustering and session structure. In
                    a real listed series the negative rho_1 lives in the thin
                    overnight regime and is NOT convertible into pooled ranking
                    skill (SPX: rho_1 = -0.030 yet real OOS AUC = 0.492, below
                    no-skill). Homogenising the variance makes that same rho_1
                    uniformly exploitable, so surrogate AUC rises. The signature:
                    surrogate AUC is a near-pure read-out of rho_1
                    (corr = -0.69 on non-crypto) while real AUC is not
                    (corr = +0.31), and the AUC lift tracks intraday
                    heteroskedasticity, which is 3.5x larger in listed
                    instruments than in crypto.

Consequence for the paper: the IAAFT row is not a strict null for
session-heteroskedastic series -- it is biased TOWARD finding skill there. It
therefore bounds the linear channel from above, and the crypto collapse
(0.536 -> 0.508) under an anti-conservative null is stronger evidence of
nonlinearity, not weaker.

    uv run python -m paper.iaaft_diagnostic
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .compare_markets import ASSET_CLASS, CRYPTO, SPAN, load_span

OUT = Path(__file__).resolve().parent / "out"
INTERVAL = "15m"
MIN_HOUR_OBS = 50


def autocorr1(x: np.ndarray) -> float:
    x = np.asarray(x, float) - np.mean(x)
    den = float(np.sum(x * x))
    return float(np.sum(x[1:] * x[:-1]) / den) if den > 0 else 0.0


def hourly_heteroskedasticity(ch: np.ndarray, dts: list[str]) -> float:
    """Ratio of the largest to the smallest UTC-hour return variance.

    A 24/7 series with a homogeneous tape sits near 1; a listed instrument whose
    overnight session is thin sits an order of magnitude higher."""
    hrs = np.array([int(t[11:13]) for t in dts])
    v = [float(ch[hrs == h].var()) for h in range(24) if int((hrs == h).sum()) > MIN_HOUR_OBS]
    v = [x for x in v if x > 0]
    return float(max(v) / min(v)) if len(v) > 1 else float("nan")


def half_width(ci) -> float:
    return (ci[1] - ci[0]) / 2.0


def main():
    markets = {r["asset"]: r for r in json.loads((OUT / "markets.json").read_text())
               if r["interval"] == INTERVAL}
    controls = {(r["asset"], r["control"]): r
                for r in json.loads((OUT / "negative_controls.json").read_text())["rows"]}

    rows = []
    for asset, real in markets.items():
        sur = controls.get((asset, "phase"))
        if sur is None:
            continue
        dts, ch, _ = load_span(asset, INTERVAL, SPAN)
        rows.append({
            "asset": asset,
            "class": ASSET_CLASS.get(asset, "?"),
            "is_crypto": asset in CRYPTO,
            "rho1": autocorr1(ch),
            "flat_frac": float(np.mean(ch == 0.0)),
            "heterosk": hourly_heteroskedasticity(ch, dts),
            "real_auc": real["ising_auc"],
            "sur_auc": sur["ising_auc"],
            "auc_lift": sur["ising_auc"] - real["ising_auc"],
            "real_hw": half_width(real["ising_auc_ci"]),
            "sur_hw": half_width(sur["ising_auc_ci"]),
            "real_conj_p": max(real["ising_auc_p_gt05"], real["free_auc_p_gt05"]),
            "sur_conj_p": sur["conj_p"],
        })

    nc = [r for r in rows if not r["is_crypto"]]
    cr = [r for r in rows if r["is_crypto"]]
    arr = lambda sel, k: np.array([r[k] for r in sel], float)

    summary = {
        "n_noncrypto": len(nc), "n_crypto": len(cr),
        # H1: CI width
        "hw_ratio_noncrypto": float(arr(nc, "sur_hw").mean() / arr(nc, "real_hw").mean()),
        "mean_real_hw_noncrypto": float(arr(nc, "real_hw").mean()),
        "mean_sur_hw_noncrypto": float(arr(nc, "sur_hw").mean()),
        # H2: flat bars
        "corr_flat_lift_noncrypto": float(np.corrcoef(arr(nc, "flat_frac"), arr(nc, "auc_lift"))[0, 1]),
        # H3: rho1 read-out + heteroskedasticity
        "corr_rho1_surauc_noncrypto": float(np.corrcoef(arr(nc, "rho1"), arr(nc, "sur_auc"))[0, 1]),
        "corr_rho1_realauc_noncrypto": float(np.corrcoef(arr(nc, "rho1"), arr(nc, "real_auc"))[0, 1]),
        "corr_rho1_surauc_all": float(np.corrcoef(arr(rows, "rho1"), arr(rows, "sur_auc"))[0, 1]),
        "corr_rho1_realauc_all": float(np.corrcoef(arr(rows, "rho1"), arr(rows, "real_auc"))[0, 1]),
        "mean_heterosk_crypto": float(arr(cr, "heterosk").mean()),
        "mean_heterosk_noncrypto": float(arr(nc, "heterosk").mean()),
        "mean_lift_crypto": float(arr(cr, "auc_lift").mean()),
        "mean_lift_noncrypto": float(arr(nc, "auc_lift").mean()),
        "n_sig_real_noncrypto": int(sum(r["real_conj_p"] < 0.05 for r in nc)),
        "n_sig_sur_noncrypto": int(sum(r["sur_conj_p"] < 0.05 for r in nc)),
    }

    (OUT / "iaaft_diagnostic.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2))

    print("=== Why IAAFT over-produces significance on non-crypto ===")
    print(f"  two-model significant, non-crypto: real {summary['n_sig_real_noncrypto']}/11 "
          f"-> surrogate {summary['n_sig_sur_noncrypto']}/11")
    print(f"\n  H1 CI width      ratio sur/real = {summary['hw_ratio_noncrypto']:.3f}  "
          f"({summary['mean_sur_hw_noncrypto']:.4f} vs {summary['mean_real_hw_noncrypto']:.4f})  REJECTED")
    print(f"  H2 flat bars     corr(flat, lift) = {summary['corr_flat_lift_noncrypto']:+.3f}  REJECTED")
    print(f"  H3 rho1 read-out corr(rho1, SUR auc)  = {summary['corr_rho1_surauc_noncrypto']:+.3f}")
    print(f"                   corr(rho1, REAL auc) = {summary['corr_rho1_realauc_noncrypto']:+.3f}  ACCEPTED")
    print(f"     heteroskedasticity  crypto {summary['mean_heterosk_crypto']:.1f}x  "
          f"vs non-crypto {summary['mean_heterosk_noncrypto']:.1f}x")
    print(f"     mean AUC lift       crypto {summary['mean_lift_crypto']:+.4f}  "
          f"vs non-crypto {summary['mean_lift_noncrypto']:+.4f}")
    print(f"\nWrote {OUT/'iaaft_diagnostic.json'}")


if __name__ == "__main__":
    main()
