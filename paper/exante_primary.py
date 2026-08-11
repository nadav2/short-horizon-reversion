"""Full primary-test statistics on the ex-ante-selected crypto universe.

exante.py established that re-selecting the crypto universe on
first-sample-month (January 2025) volume leaves the class-mean gap
unchanged or larger, but it froze only the Ising-model gap. To let the
ex-ante universe stand as a headline specification alongside the as-fixed
183-pair test, this module recomputes on the 133 ex-ante pairs (plus the
unchanged 187 stocks) everything the primary test reports:

  * two-model joint dependence-preserving block-bootstrap class gap
    (reusing dependence.joint_gap_bootstrap unchanged);
  * BH-FDR at q=0.05 on the two-model conjunction, jointly across the
    320-asset ex-ante universe (reusing fdr.bh_mask);
  * class-mean AUC levels, coupling summaries, and within-class
    dependence (mean pairwise correlation, N_eff) for the 133 pairs.

Everything is aggregation over frozen per-asset outputs (out/wide.json,
out/wide_oos/*.npz); no model is refit. Writes out/exante_primary.json.

    uv run --active python -m paper.exante_primary
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dependence import BULK, joint_gap_bootstrap, pairwise_mean_corr
from .fdr import bh_mask

OUT = Path(__file__).resolve().parent / "out"
Q = 0.05
P_FLOOR = 1.0 / 301                      # wide.py used 300 bootstrap resamples


def main() -> None:
    wide = json.loads((OUT / "wide.json").read_text())
    exante = set(json.loads((OUT / "exante_volume.json").read_text()))

    rows = [r for r in wide
            if r["class"] == "stock" or r["asset"] in exante]
    crypto = [r for r in rows if r["class"] == "crypto"]
    stocks = [r for r in rows if r["class"] == "stock"]
    print(f"ex-ante universe: {len(crypto)} crypto + {len(stocks)} stocks")

    # BH-FDR at q=0.05 on the two-model conjunction, jointly across the
    # ex-ante universe (same construction as fdr.py, restricted universe).
    p_is = np.maximum([r["p_ising"] for r in rows], P_FLOOR)
    p_fr = np.maximum([r["p_free"] for r in rows], P_FLOOR)
    p_conj = np.maximum(p_is, p_fr)
    m_conj = bh_mask(p_conj, Q)
    cls = np.array([r["class"] for r in rows])
    fdr = {}
    for c in ("crypto", "stock"):
        sel = cls == c
        fdr[c] = {"n": int(sel.sum()),
                  "bh_conj_sig": int(np.sum(m_conj & sel)),
                  "bh_conj_frac": float(np.sum(m_conj & sel) / sel.sum())}

    # Two-model joint dependence-preserving bootstrap on the subset.
    joint = joint_gap_bootstrap(rows)
    for label in ("ising", "free"):
        joint[label].pop("gap_samples", None)

    # Within-class dependence for the ex-ante crypto set.
    corr = pairwise_mean_corr(
        [r["asset"] for r in crypto if (BULK / f"{r['asset']}-15m.json").exists()])

    A = np.array([r["A"] for r in crypto])
    res = {
        "selection": "pairs with Binance quote volume over 2025-01-01.."
                     "2025-01-31 (the first sample month); see exante.py",
        "n_crypto": len(crypto), "n_stocks": len(stocks),
        "crypto_mean_auc_ising": float(np.mean([r["auc_ising"] for r in crypto])),
        "crypto_mean_auc_free": float(np.mean([r["auc_free"] for r in crypto])),
        "stock_mean_auc_ising": float(np.mean([r["auc_ising"] for r in stocks])),
        "coupling": {"mean_A": float(A.mean()), "frac_neg": float(np.mean(A < 0))},
        "fdr_q05_conjunction": fdr,
        "joint_gap": joint,
        "crypto_dependence": corr,
    }
    (OUT / "exante_primary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "joint_gap"}, indent=1))
    for label in ("ising", "free"):
        g = joint[label]
        print(f"{label:5s}: gap={g['obs_gap']:+.4f} CI=[{g['gap_ci'][0]:+.4f},"
              f"{g['gap_ci'][1]:+.4f}] p={g['p_gap_le0']:.4f}")


if __name__ == "__main__":
    main()
