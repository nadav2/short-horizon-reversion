"""Reliability diagram for the calibration claim.

The paper's stated reason for the kernel constraint is that it buys
*calibration* rather than detection (app:why): constrained and free logit
agree on AUC while only the constrained model clears out-of-sample log-loss
significance. This module draws the standard exhibit for that claim -- a
reliability diagram (predicted probability against realised frequency) on the
pooled crypto out-of-sample predictions of the wide study -- which the
manuscript otherwise evidences only through aggregate log-loss/Brier deltas.

Inputs are the per-asset OOS dumps of `wide.py --dump-oos`
(out/wide_oos/{asset}.npz: ts / actual / p_ising / p_free). All 183 crypto
pairs are pooled; bins are equal-count deciles of each model's own pooled
predictions. Uncertainty respects cross-sectional and serial dependence: a
JOINT moving-block bootstrap on the shared 15m grid (one set of time blocks
per replicate, applied to every asset at once, as in paper.dependence),
implemented on per-slot sufficient statistics so each replicate is a row-sum.

Also reported with joint CIs: the linear calibration slope cov(y,p)/var(p)
per model (1 = perfectly calibrated on this range, <1 = over-dispersed
predictions) and the pooled log-loss / Brier differences free - constrained.

    uv run --active python -m paper.reliability

Writes docs/figures/reliability.{pdf,png} + out/reliability.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fetch_bulk_stocks import UNIVERSE

OUT = Path(__file__).resolve().parent / "out"
OOS = OUT / "wide_oos"
SLOT = 900
BLOCK, N_BOOT, SEED = 384, 1000, 13
N_BINS = 10
STOCK_SET = {s.lower().replace(".", "-") for s in UNIVERSE}
MODELS = ("ising", "free")


def load_pooled():
    """Pooled crypto OOS rows: slot index, label, and both models' p."""
    slots, ys, ps = [], [], {m: [] for m in MODELS}
    for f in sorted(OOS.glob("*.npz")):
        if f.stem in STOCK_SET:
            continue
        d = np.load(f)
        slots.append(d["ts"] // SLOT)
        ys.append(d["actual"].astype(np.int8))
        for m in MODELS:
            ps[m].append(d[f"p_{m}"].astype(np.float64))
    slots = np.concatenate(slots)
    lo = int(slots.min())
    return (slots - lo).astype(np.int64), np.concatenate(ys), \
        {m: np.concatenate(ps[m]) for m in MODELS}


def suff_stats(slot, y, p, edges):
    """Per-slot sufficient statistics; every reported quantity is a linear
    functional of these, so a joint block bootstrap is a row-sum."""
    T = int(slot.max()) + 1
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, N_BINS - 1)
    cnt = np.zeros((T, N_BINS))
    up = np.zeros((T, N_BINS))
    psum = np.zeros((T, N_BINS))
    np.add.at(cnt, (slot, b), 1.0)
    np.add.at(up, (slot, b), y.astype(float))
    np.add.at(psum, (slot, b), p)
    pc = np.clip(p, 1e-15, 1 - 1e-15)
    ll = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    br = (p - y) ** 2
    mom = np.zeros((T, 6))          # n, sum_y, sum_p, sum_p2, sum_yp, + spare
    scal = np.zeros((T, 2))         # sum log-loss, sum brier
    np.add.at(mom[:, 0], slot, 1.0)
    np.add.at(mom[:, 1], slot, y.astype(float))
    np.add.at(mom[:, 2], slot, p)
    np.add.at(mom[:, 3], slot, p * p)
    np.add.at(mom[:, 4], slot, y * p)
    np.add.at(scal[:, 0], slot, ll)
    np.add.at(scal[:, 1], slot, br)
    return {"cnt": cnt, "up": up, "psum": psum, "mom": mom, "scal": scal}


def stats_from_rows(S, rows):
    cnt = S["cnt"][rows].sum(0)
    up = S["up"][rows].sum(0)
    psum = S["psum"][rows].sum(0)
    n, sy, sp, sp2, syp, _ = S["mom"][rows].sum(0)
    sll, sbr = S["scal"][rows].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        obs = up / cnt
        pred = psum / cnt
    var_p = sp2 / n - (sp / n) ** 2
    cov_yp = syp / n - (sy / n) * (sp / n)
    slope = cov_yp / var_p if var_p > 0 else float("nan")
    ece = float(np.nansum(cnt * np.abs(obs - pred)) / cnt.sum())
    return {"obs": obs, "pred": pred, "slope": float(slope), "ece": ece,
            "log_loss": float(sll / n), "brier": float(sbr / n)}


def main():
    slot, y, ps = load_pooled()
    T = int(slot.max()) + 1
    print(f"pooled crypto OOS: {len(y):,} predictions on {T:,} slots")

    S, edges, full = {}, {}, {}
    for m in MODELS:
        e = np.unique(np.quantile(ps[m], np.linspace(0, 1, N_BINS + 1)))
        e[0], e[-1] = -np.inf, np.inf
        edges[m] = e
        S[m] = suff_stats(slot, y, ps[m], e)
        full[m] = stats_from_rows(S[m], np.arange(T))

    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(T / BLOCK))
    boots = {m: {"obs": [], "slope": [], "ece": []} for m in MODELS}
    dll, dbr = [], []
    for _ in range(N_BOOT):
        starts = rng.integers(0, max(1, T - BLOCK + 1), size=nb)
        rows = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:T]
        rows = rows[rows < T]
        st = {m: stats_from_rows(S[m], rows) for m in MODELS}
        for m in MODELS:
            boots[m]["obs"].append(st[m]["obs"])
            boots[m]["slope"].append(st[m]["slope"])
            boots[m]["ece"].append(st[m]["ece"])
        dll.append(st["free"]["log_loss"] - st["ising"]["log_loss"])
        dbr.append(st["free"]["brier"] - st["ising"]["brier"])

    res = {"n_pooled": int(len(y)), "n_slots": T, "n_bins": N_BINS,
           "block": BLOCK, "n_boot": N_BOOT, "seed": SEED}
    for m in MODELS:
        ob = np.array(boots[m]["obs"])
        sl = np.array(boots[m]["slope"])
        ec = np.array(boots[m]["ece"])
        res[m] = {
            "pred": full[m]["pred"].tolist(),
            "obs": full[m]["obs"].tolist(),
            "obs_ci_lo": np.nanpercentile(ob, 2.5, axis=0).tolist(),
            "obs_ci_hi": np.nanpercentile(ob, 97.5, axis=0).tolist(),
            "slope": full[m]["slope"],
            "slope_ci": [float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))],
            "ece": full[m]["ece"],
            "ece_ci": [float(np.percentile(ec, 2.5)), float(np.percentile(ec, 97.5))],
            "log_loss": full[m]["log_loss"], "brier": full[m]["brier"],
        }
    res["delta_free_minus_constr"] = {
        "log_loss": float(full["free"]["log_loss"] - full["ising"]["log_loss"]),
        "log_loss_ci": [float(np.percentile(dll, 2.5)), float(np.percentile(dll, 97.5))],
        "brier": float(full["free"]["brier"] - full["ising"]["brier"]),
        "brier_ci": [float(np.percentile(dbr, 2.5)), float(np.percentile(dbr, 97.5))],
    }
    (OUT / "reliability.json").write_text(json.dumps(res, indent=2))

    # ── figure ───────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    from .style import MODEL_COLOR, SINGLE, save
    C_ISING, C_FREE = MODEL_COLOR["ising"], MODEL_COLOR["free"]

    fig, ax = plt.subplots(figsize=(SINGLE, 2.9))
    lim_lo = min(min(res[m]["pred"]) for m in MODELS) - 0.005
    lim_hi = max(max(res[m]["pred"]) for m in MODELS) + 0.005
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k:", lw=0.8,
            label="perfect calibration")
    for m, color, lab in ((MODELS[0], C_ISING, "constrained logit"),
                          (MODELS[1], C_FREE, "free AR(12) logit")):
        pred = np.array(res[m]["pred"])
        obs = np.array(res[m]["obs"])
        err = np.vstack([obs - np.array(res[m]["obs_ci_lo"]),
                         np.array(res[m]["obs_ci_hi"]) - obs])
        ax.errorbar(pred, obs, yerr=err, fmt="o-", color=color, markersize=2.6,
                    lw=1.0, capsize=1.5,
                    label=f"{lab} (slope {res[m]['slope']:.2f})")
    ax.set_xlabel(r"predicted $P(\mathrm{up})$ (decile mean)")
    ax.set_ylabel("observed up-frequency")
    ax.legend(loc="upper left")
    save(fig, "reliability")

    for m in MODELS:
        r = res[m]
        print(f"{m:>6}: slope {r['slope']:.3f} CI [{r['slope_ci'][0]:.3f}, "
              f"{r['slope_ci'][1]:.3f}]  ECE {r['ece']:.5f}  "
              f"logloss {r['log_loss']:.5f}  brier {r['brier']:.5f}")
    d = res["delta_free_minus_constr"]
    print(f"free - constrained: dlogloss {d['log_loss']:+.5f} CI "
          f"[{d['log_loss_ci'][0]:+.5f}, {d['log_loss_ci'][1]:+.5f}]  "
          f"dbrier {d['brier']:+.6f} CI [{d['brier_ci'][0]:+.6f}, {d['brier_ci'][1]:+.6f}]")
    print(f"Wrote {OUT/'reliability.json'}")


if __name__ == "__main__":
    main()
