"""Family-level inference for the wrapper panel.

The wrapper panel of panel.py scores 17 NAV-linked wrappers on 8
underlyings.  Multiple wrappers on the same underlying (e.g. the eight US
spot-Bitcoin funds) share one price process and one sample window, so they
are within-family replications, not independent observations.  This module
re-runs the panel's concordance inference with the *underlying family* as
the unit:

  * family gap      -- mean wrapper AUC within a family minus the
                       underlying's session-matched AUC;
  * exact permutation test over the 8! reassignments of underlying AUCs to
    family wrapper means (statistic: mean |family gap|);
  * exact permutation p for the family-level wrapper-underlying AUC
    correlation;
  * within-family dispersion of the Bitcoin wrappers, reported as the
    replication scale against which the family gaps should be read.

Reads out/natural.json (written by panel.py); writes
out/family_inference.json.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "out"


def main() -> None:
    data = json.loads((OUT / "natural.json").read_text())
    pairs = data["panel"]["pairs"]
    ref = data["panel"]["reference_rth_stocks"]

    fams: dict[str, dict] = {}
    for p in pairs:
        f = fams.setdefault(
            p["experiment"], {"underlying_auc": p["auc_underlying"], "wrappers": {}}
        )
        f["wrappers"][p["wrapper"]] = p["auc_wrapper"]

    names = sorted(fams)
    u = np.array([fams[f]["underlying_auc"] for f in names])
    w = np.array([np.mean(list(fams[f]["wrappers"].values())) for f in names])
    gaps = w - u

    obs_mean_abs_gap = float(np.mean(np.abs(gaps)))
    obs_corr = float(np.corrcoef(w, u)[0, 1])

    # Exact permutation distribution over all 8! pairings.
    n_le, n_corr_ge, n_tot = 0, 0, 0
    for perm in itertools.permutations(range(len(names))):
        up = u[list(perm)]
        n_tot += 1
        if np.mean(np.abs(w - up)) <= obs_mean_abs_gap + 1e-15:
            n_le += 1
        if np.corrcoef(w, up)[0, 1] >= obs_corr - 1e-15:
            n_corr_ge += 1

    btc = np.array(list(fams["bitcoin"]["wrappers"].values()))

    res = {
        "n_families": len(names),
        "families": {
            f: {
                "underlying_auc": round(fams[f]["underlying_auc"], 4),
                "n_wrappers": len(fams[f]["wrappers"]),
                "wrapper_mean_auc": round(float(np.mean(list(fams[f]["wrappers"].values()))), 4),
                "family_gap": round(float(np.mean(list(fams[f]["wrappers"].values())) - fams[f]["underlying_auc"]), 4),
                "wrapper_z_vs_rth_stocks": round(
                    (float(np.mean(list(fams[f]["wrappers"].values()))) - ref["mean"]) / ref["sd"], 2
                ),
            }
            for f in names
        },
        "mean_abs_family_gap": round(obs_mean_abs_gap, 4),
        "perm_p_mean_abs_gap": n_le / n_tot,
        "corr_wrapper_underlying": round(obs_corr, 3),
        "perm_p_corr": n_corr_ge / n_tot,
        "n_permutations": n_tot,
        "bitcoin_within_family": {
            "n": int(btc.size),
            "sd": round(float(btc.std(ddof=1)), 4),
            "range": [round(float(btc.min()), 4), round(float(btc.max()), 4)],
        },
    }
    (OUT / "family_inference.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
