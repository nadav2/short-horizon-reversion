"""Sign-randomization surrogate: the clean null the negative-control battery lacked.

IAAFT preserves linear autocovariance but destroys volatility clustering, which
makes it anti-conservative on session-heteroskedastic series (7/14 instruments
significant on surrogates). Holding the |r| path fixed and randomizing only the
signs inverts the trade: volatility clustering and the diurnal profile are
preserved EXACTLY, and precisely the sign dependence under test is destroyed.
Runs the identical evaluate_control machinery (walk-forward, both models,
per-asset block bootstrap) over the 14 focal instruments.

    uv run --active python -m paper.signrand_control
"""

from __future__ import annotations

import json
import time

import numpy as np

from .compare_markets import CRYPTO, TRADITIONAL
from .negative_controls import OUT, SEED, evaluate_control


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for asset in CRYPTO + TRADITIONAL:
        t0 = time.time()
        r = evaluate_control(asset, "signrand", rng)
        if r is None:
            print(f"  {asset:7s} skipped", flush=True)
            continue
        rows.append(r)
        sig = "*" if r["conj_p"] < 0.05 else " "
        print(f"  {asset:7s} AUC_ising={r['ising_auc']:.4f} p={r['ising_auc_p_gt05']:.3f}"
              f"  AUC_free={r['free_auc']:.4f} p={r['free_auc_p_gt05']:.3f}"
              f"  conj={r['conj_p']:.3f}{sig}  ({time.time()-t0:.0f}s)", flush=True)

    crypto = [r for r in rows if r["asset"] in CRYPTO]
    noncr = [r for r in rows if r["asset"] not in CRYPTO]
    summary = {
        "n_assets": len(rows),
        "n_two_model_sig": sum(1 for r in rows if r["conj_p"] < 0.05),
        "mean_ising_auc": float(np.mean([r["ising_auc"] for r in rows])),
        "mean_free_auc": float(np.mean([r["free_auc"] for r in rows])),
        "crypto_mean_ising_auc": float(np.mean([r["ising_auc"] for r in crypto])),
        "crypto_n_sig": sum(1 for r in crypto if r["conj_p"] < 0.05),
        "noncrypto_mean_ising_auc": float(np.mean([r["ising_auc"] for r in noncr])),
        "noncrypto_n_sig": sum(1 for r in noncr if r["conj_p"] < 0.05),
    }
    (OUT / "signrand.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nsignrand: {summary['n_two_model_sig']}/{summary['n_assets']} two-model sig; "
          f"crypto mean {summary['crypto_mean_ising_auc']:.4f} ({summary['crypto_n_sig']}/{len(crypto)} sig); "
          f"non-crypto mean {summary['noncrypto_mean_ising_auc']:.4f} ({summary['noncrypto_n_sig']}/{len(noncr)})")


if __name__ == "__main__":
    main()
