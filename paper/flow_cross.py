"""Cross-sectional flow mechanism test on the wide crypto universe.

Joins per-pair flow statistics (from bulk_flow/, taker-buy klines) with the
wide study's fitted coupling A and OOS AUC (out/wide.json). Under the
overreaction-correction account, pairs where flow-driven moves revert harder
(delta_flip = flip rate after flow-driven bars minus after flow-opposed bars)
should carry a more negative coupling and a higher AUC; and if the corrected
flow is retail, reversal should strengthen as the typical trade gets smaller,
holding dollar volume fixed.

    uv run --active python -m paper.flow_cross
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
FLOW = HERE / "bulk_flow"
OUT = HERE / "out"


def pair_metrics(path):
    raw = json.loads(path.read_text())
    ch = np.array([d["change"] for d in raw])
    vol = np.array([d["vol"] for d in raw])
    qvol = np.array([d["qvol"] for d in raw])
    ntr = np.array([d["ntr"] for d in raw])
    tbv = np.array([d["tbv"] for d in raw])
    with np.errstate(divide="ignore", invalid="ignore"):
        imb = np.where(vol > 0, 2 * tbv / vol - 1, np.nan)
        tsize = np.where(ntr > 0, qvol / ntr, np.nan)

    s = np.sign(ch)
    t = np.arange(len(ch) - 1)
    ok = (s[t] != 0) & (s[t + 1] != 0) & np.isfinite(imb[t]) & (imb[t] != 0)
    t = t[ok]
    if len(t) < 2000:
        return None
    driven = s[t] * imb[t] > 0
    flip = s[t + 1] == -s[t]
    return {
        "delta_flip": float(flip[driven].mean() - flip[~driven].mean()),
        "flip_driven": float(flip[driven].mean()),
        "flip_opposed": float(flip[~driven].mean()),
        "driven_share": float(driven.mean()),
        "mean_abs_imb": float(np.nanmean(np.abs(imb))),
        "med_trade_usd": float(np.nanmedian(tsize)),
        "dollar_vol": float(np.nanmean(qvol)),
        "n": int(len(t)),
    }


def hc1_ols(y, X):
    """OLS with intercept and HC1 robust t-stats."""
    X = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    S = X.T @ np.diag(e ** 2) @ X * (n / (n - k))
    se = np.sqrt(np.diag(XtXi @ S @ XtXi))
    return beta, beta / se


def main():
    wide = {r["asset"]: r for r in json.loads((OUT / "wide.json").read_text())
            if r["class"] == "crypto"}
    rows = []
    for f in sorted(FLOW.glob("*-15m-flow.json")):
        sid = f.name[:-len("-15m-flow.json")]
        if sid not in wide:
            continue
        m = pair_metrics(f)
        if m is None:
            continue
        m.update({"asset": sid, "A": wide[sid]["A"],
                  "auc": wide[sid]["auc_ising"]})
        rows.append(m)
    print(f"{len(rows)} pairs joined")

    A = np.array([r["A"] for r in rows])
    auc = np.array([r["auc"] for r in rows])
    dflip = np.array([r["delta_flip"] for r in rows])
    tsz = np.log10([r["med_trade_usd"] for r in rows])
    dvol = np.log10([r["dollar_vol"] for r in rows])
    absimb = np.array([r["mean_abs_imb"] for r in rows])

    def spear(name, x, y):
        rho, p = stats.spearmanr(x, y)
        print(f"  spearman {name:28s} rho={rho:+.3f}  p={p:.2g}")
        return {"rho": float(rho), "p": float(p)}

    print("\ncross-sectional correlations:")
    cors = {
        "auc_vs_delta_flip": spear("AUC ~ delta_flip", dflip, auc),
        "A_vs_delta_flip": spear("A ~ delta_flip", dflip, A),
        "auc_vs_trade_size": spear("AUC ~ log med trade $", tsz, auc),
        "A_vs_trade_size": spear("A ~ log med trade $", tsz, A),
        "auc_vs_abs_imb": spear("AUC ~ mean |imb|", absimb, auc),
    }

    # joint: AUC on delta_flip + retail proxy + volume (standardized, HC1)
    z = lambda x: (x - x.mean()) / x.std()
    beta, tstat = hc1_ols(auc, [z(dflip), z(tsz), z(dvol)])
    names = ["const", "delta_flip", "log_trade_usd", "log_dollar_vol"]
    print("\nOLS AUC ~ z(delta_flip) + z(log trade size) + z(log $vol), HC1:")
    for nm, b, tt in zip(names, beta, tstat):
        print(f"  {nm:15s} {b:+.5f}  t={tt:+.2f}")

    OUT.mkdir(exist_ok=True)
    (OUT / "flow_cross.json").write_text(json.dumps(
        {"n_pairs": len(rows), "correlations": cors,
         "ols_auc": {"names": names, "beta": beta.tolist(), "t": tstat.tolist()},
         "rows": rows}, indent=2))
    print(f"\nwrote {OUT / 'flow_cross.json'}")


if __name__ == "__main__":
    main()
