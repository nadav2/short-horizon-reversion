"""Generative self-consistency check for the fitted kinetic Ising model.

The fitted model is not just a classifier: with (C, A, alpha) it defines Glauber
dynamics that can be *simulated*. If the model is an adequate description of the
sign dynamics, synthetic chains generated from the fitted parameters should
reproduce the stylized facts of the data it was fit on:

  (i)   near-zero linear autocorrelation of returns (rho_1 ~ 0),
  (ii)  variance ratios mildly below one (diffuse mean-reversion), and
  (iii) the same out-of-sample AUC level when the model scores its own chain.

That combination is the resolution of the apparent paradox in the data (rho_1 ~ 0
yet AUC > 0.5): a weak multi-lag magnitude-aware reversal produces almost no
first-order linear signature. Magnitudes |r_t| are resampled from the empirical
series in day-long blocks (preserving volatility clustering, which the spin model
does not attempt to describe); signs follow the fitted Glauber law.

    uv run --active python -m paper.generative
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .common import CANDLES_PER_DAY, load_merged
from .compare_markets import autocorr1
from .models import N_LAGS, SPIN_SCALE
from .robustness import variance_ratio

OUT = Path(__file__).resolve().parent / "out"
N_SIMS = 20
BURN = 200
SEED = 23


def simulate_chain(abs_r: np.ndarray, C: float, A: float, alpha: float,
                   n: int, block: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Simulate returns r_t = sigma_t * |r_t| with Glauber signs and block-resampled
    empirical magnitudes. Returns (returns, model_probs) after burn-in."""
    w = A / (np.arange(1, N_LAGS + 1) ** alpha)
    total = n + BURN
    # magnitudes: day-blocks from the empirical |r| series (vol clustering kept)
    nblocks = int(np.ceil(total / block))
    starts = rng.integers(0, max(1, len(abs_r) - block + 1), size=nblocks)
    mag = np.concatenate([abs_r[s:s + block] for s in starts])[:total]

    s_hist = np.zeros(N_LAGS)            # s_{t-1}, s_{t-2}, ...
    r = np.empty(total)
    probs = np.empty(total)
    u = rng.random(total)
    for t in range(total):
        p_up = 1.0 / (1.0 + np.exp(-(C + w @ s_hist)))
        sigma = 1.0 if u[t] < p_up else -1.0
        r[t] = sigma * mag[t]
        probs[t] = p_up
        s_hist[1:] = s_hist[:-1]
        s_hist[0] = np.tanh(SPIN_SCALE * r[t])
    return r[BURN:], probs[BURN:]


def main():
    results = json.loads((OUT / "results.json").read_text())
    rng = np.random.default_rng(SEED)
    rows = []
    for cell, cdata in results["cells"].items():
        coin, interval = cell.split("-")
        _, ch, _ = load_merged(coin, interval)
        block = CANDLES_PER_DAY[interval]
        C = float(np.median(cdata["ising_params"]["C"]))
        A = float(np.median(cdata["ising_params"]["A"]))
        alpha = float(np.median(cdata["ising_params"]["alpha"]))
        emp_auc = cdata["models"]["ising"]["pred"]["auc"]

        emp_vr16, _, _ = variance_ratio(ch, 16)
        n = min(len(ch), 80_000)
        sims = {"rho1": [], "vr16": [], "auc": []}
        abs_r = np.abs(ch)
        for _ in range(N_SIMS):
            r, p = simulate_chain(abs_r, C, A, alpha, n, block, rng)
            sims["rho1"].append(autocorr1(r))
            sims["vr16"].append(variance_ratio(r, 16)[0])
            sims["auc"].append(roc_auc_score((r > 0).astype(int), p))
        row = {
            "cell": cell, "C": C, "A": A, "alpha": alpha, "n_sim": n,
            "emp_rho1": autocorr1(ch), "emp_vr16": float(emp_vr16), "emp_auc_oos": emp_auc,
            "sim_rho1": float(np.mean(sims["rho1"])), "sim_rho1_sd": float(np.std(sims["rho1"])),
            "sim_vr16": float(np.mean(sims["vr16"])), "sim_vr16_sd": float(np.std(sims["vr16"])),
            "sim_auc": float(np.mean(sims["auc"])), "sim_auc_sd": float(np.std(sims["auc"])),
        }
        rows.append(row)
        print(f"  {cell:8s} A={A:+.2f} α={alpha:.2f} | rho1 emp={row['emp_rho1']:+.4f} "
              f"sim={row['sim_rho1']:+.4f}±{row['sim_rho1_sd']:.4f} | "
              f"VR16 emp={row['emp_vr16']:.3f} sim={row['sim_vr16']:.3f}±{row['sim_vr16_sd']:.3f} | "
              f"AUC emp={row['emp_auc_oos']:.4f} sim={row['sim_auc']:.4f}±{row['sim_auc_sd']:.4f}")

    (OUT / "selfcons.json").write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {OUT/'selfcons.json'}")


if __name__ == "__main__":
    main()
