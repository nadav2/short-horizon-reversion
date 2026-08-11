"""Per-asset linear-channel accounting (resolves the IAAFT class-level discrepancy).

For an elliptical process a lag-1 return correlation of rho implies an AUC of
Phi(sqrt(4/pi)|rho|) for a sign predictor reading it -- the "linear channel."
The class-mean crypto rho_1 of -0.046 implies 0.521 (two thirds of the observed
excess) yet IAAFT surrogates retain only a fifth, a discrepancy the paper
flagged as likely compositional: the class-mean rho_1 may be carried by
bounce-prone pairs whose AUC excess is no larger. This module computes the
per-asset comparison directly: each asset's own rho_1 (full span, bulk_data),
its implied linear AUC excess, and its observed out-of-sample excess (wide.json).

    uv run --active python -m paper.linear_channel
"""

from __future__ import annotations

import json
from math import erf, pi, sqrt
from pathlib import Path

import numpy as np

BULK = Path(__file__).resolve().parent / "bulk_data"
OUT = Path(__file__).resolve().parent / "out"
C = 2 * sqrt(2 / pi)  # sqrt(4/pi)


def phi(z: float) -> float:
    return 0.5 * (1 + erf(z / sqrt(2)))


def rho1(x: np.ndarray) -> float:
    x = x - x.mean()
    d = float(x @ x)
    return float((x[1:] @ x[:-1]) / d) if d > 0 else 0.0


def main():
    wide = json.loads((OUT / "wide.json").read_text())
    rows = []
    for r in wide:
        f = BULK / f"{r['asset']}-15m.json"
        if not f.exists():
            continue
        raw = json.loads(f.read_text())
        raw.sort(key=lambda d: d["timestamp"])
        ch = np.array([d.get("change", 0.0) for d in raw], float)
        rho = rho1(ch)
        rows.append({
            "asset": r["asset"], "class": r["class"], "rho1": rho,
            "implied_excess": phi(C * abs(rho)) - 0.5,
            "observed_excess": r["auc_ising"] - 0.5,
        })

    res = {"n_assets": len(rows)}
    for cls in ("crypto", "stock"):
        sel = [x for x in rows if x["class"] == cls]
        rho_arr = np.array([x["rho1"] for x in sel])
        imp = np.array([x["implied_excess"] for x in sel])
        obs = np.array([x["observed_excess"] for x in sel])
        res[cls] = {
            "n": len(sel),
            "mean_rho1": float(rho_arr.mean()),
            "median_rho1": float(np.median(rho_arr)),
            "mean_implied_excess": float(imp.mean()),
            "mean_observed_excess": float(obs.mean()),
            "corr_implied_observed": float(np.corrcoef(imp, obs)[0, 1]),
            "corr_absrho_observed": float(np.corrcoef(np.abs(rho_arr), obs)[0, 1]),
            # the compositional question: does the class-mean rho1 come from the
            # same pairs that carry the AUC excess?
            "mean_rho1_top_obs_quartile": float(
                rho_arr[obs >= np.percentile(obs, 75)].mean()),
            "mean_rho1_bottom_obs_quartile": float(
                rho_arr[obs <= np.percentile(obs, 25)].mean()),
        }
        print(f"{cls:7s} mean rho1 {res[cls]['mean_rho1']:+.4f} "
              f"(median {res[cls]['median_rho1']:+.4f}); implied excess "
              f"{res[cls]['mean_implied_excess']:+.4f} vs observed "
              f"{res[cls]['mean_observed_excess']:+.4f}; "
              f"corr(implied,observed)={res[cls]['corr_implied_observed']:+.2f}")
    (OUT / "linear_channel.json").write_text(json.dumps({**res, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
