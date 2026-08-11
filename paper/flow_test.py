"""Order-flow mechanism tests on the focal coins (15m, paper span).

Uses the taker-buy volume field of Binance klines (fetched by paper.fetch_flow)
to sign each bar's aggressor flow: imbalance i_t = 2*tbv/vol - 1 in [-1, 1].
Three tests discriminate an overreaction-correction / liquidity-provision
mechanism from alternatives:

1. MODEL-FREE FLIP RATES. A bar is "flow-driven" when its price move agrees in
   sign with its taker imbalance (aggressors pushed price the way it went) and
   "flow-opposed" otherwise. Liquidity-provision reversal predicts the next-bar
   sign flip concentrates after flow-driven bars and grows with |imbalance|.

2. CONDITIONAL AUC. The paper's walk-forward Ising OOS predictions, bucketed by
   whether the immediately preceding bar was flow-driven or flow-opposed.

3. SUBSUMPTION. Walk-forward logistic models on (a) 12 sign lags, (b) 12
   imbalance lags, (c) both. If imbalance subsumes the sign kernel, the
   mechanism is inventory/flow pressure; if signs survive, it is not reducible
   to bar-level flow.

    uv run --active python -m paper.flow_test
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .compare_markets import SPAN, WINDOWS, block_boot_idx, BLOCK
from .models import IsingLogit
from .walkforward import walk_forward

OUT = Path(__file__).resolve().parent / "out"
FLOW_DIR = Path(__file__).resolve().parent / "flow_data"
COINS = ["btc", "eth", "sol", "xrp"]
N_LAGS = 12
N_BOOT = 1000
SEED = 7


def load(coin):
    raw = json.loads((FLOW_DIR / f"{coin}-15m-flow.json").read_text())
    raw = [d for d in raw if SPAN[0] <= d["datetime"][:10] <= SPAN[1]]
    raw.sort(key=lambda d: d["timestamp"])
    ch = np.array([d["change"] for d in raw])
    ups = np.array([bool(d["up"]) for d in raw])
    vol = np.array([d["vol"] for d in raw])
    tbv = np.array([d["tbv"] for d in raw])
    with np.errstate(divide="ignore", invalid="ignore"):
        imb = np.where(vol > 0, 2 * tbv / vol - 1, np.nan)
    return ch, ups, imb


# ---------------------------------------------------------------- test 1
def flip_rates(ch, imb):
    s = np.sign(ch)
    t = np.arange(len(ch) - 1)
    ok = (s[t] != 0) & (s[t + 1] != 0) & np.isfinite(imb[t]) & (imb[t] != 0)
    t = t[ok]
    driven = s[t] * imb[t] > 0
    flip = s[t + 1] == -s[t]

    def rate(mask):
        n = int(mask.sum())
        p = float(flip[mask].mean()) if n else float("nan")
        return {"rate": p, "n": n, "se": float(np.sqrt(p * (1 - p) / n)) if n else float("nan")}

    res = {"flow_driven": rate(driven), "flow_opposed": rate(~driven)}
    # dose-response: flip rate by |imbalance| quintile, within each condition
    q = np.nanquantile(np.abs(imb[t]), [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(np.abs(imb[t]), q)
    for name, mask in (("driven_by_absimb", driven), ("opposed_by_absimb", ~driven)):
        res[name] = [rate(mask & (bins == b)) for b in range(5)]
    return res


# ---------------------------------------------------------------- test 2
def conditional_auc(ch, ups, imb):
    tr, te = WINDOWS["15m"]
    res, _ = walk_forward(ch, ups, tr, te, n_lags=N_LAGS,
                          models=lambda: [IsingLogit(n_lags=N_LAGS)])
    r = res["ising"]
    idx = r["idx"].astype(int)
    prev = idx - 1
    s_prev = np.sign(ch[prev])
    ok = (s_prev != 0) & np.isfinite(imb[prev]) & (imb[prev] != 0)
    driven = ok & (s_prev * imb[prev] > 0)
    opposed = ok & (s_prev * imb[prev] < 0)
    return {"probs": r["probs"], "actuals": r["actuals"].astype(int),
            "driven": driven, "opposed": opposed}


# ---------------------------------------------------------------- test 3
def lagmat(x, n_lags):
    """Row t holds x[t-1], ..., x[t-n_lags]; rows < n_lags are invalid."""
    n = len(x)
    m = np.full((n, n_lags), np.nan)
    for k in range(1, n_lags + 1):
        m[k:, k - 1] = x[:n - k]
    return m


def subsumption(ch, ups, imb):
    tr, te = WINDOWS["15m"]
    s = np.sign(ch)
    imb0 = np.nan_to_num(imb, nan=0.0)
    S = lagmat(s, N_LAGS)
    I = lagmat(imb0, N_LAGS)
    feats = {"signs": S, "imb": I, "both": np.hstack([S, I])}
    y = ups.astype(int)
    n = len(ch)

    out = {k: {"probs": [], "y": []} for k in feats}
    coef_sign1 = {"signs": [], "both": []}
    start = N_LAGS  # first row with a full lag window
    while start + tr + te <= n:
        lo, hi = start, start + tr
        t_lo, t_hi = hi, min(hi + te, n)
        for k, X in feats.items():
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit(X[lo:hi], y[lo:hi])
            out[k]["probs"].append(clf.predict_proba(X[t_lo:t_hi])[:, 1])
            out[k]["y"].append(y[t_lo:t_hi])
            if k in coef_sign1:
                coef_sign1[k].append(float(clf.coef_[0][0]))
        start += te

    res = {}
    for k in feats:
        p = np.concatenate(out[k]["probs"])
        yy = np.concatenate(out[k]["y"])
        res[k] = {"auc": float(roc_auc_score(yy, p)), "probs": p, "y": yy}
    res["coef_sign_lag1"] = {k: float(np.mean(v)) for k, v in coef_sign1.items()}
    return res


def boot_auc_diff(y, p_a, p_b, rng):
    """Moving-block bootstrap CI for AUC(a) - AUC(b) on shared OOS points."""
    diffs = []
    for _ in range(N_BOOT):
        bi = block_boot_idx(len(y), BLOCK["15m"], rng)
        yb = y[bi]
        if len(np.unique(yb)) < 2:
            continue
        diffs.append(roc_auc_score(yb, p_a[bi]) - roc_auc_score(yb, p_b[bi]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean": float(np.mean(diffs)), "ci": [float(lo), float(hi)]}


def main():
    rng = np.random.default_rng(SEED)
    summary = {}
    pool2 = {"probs": [], "actuals": [], "driven": [], "opposed": []}
    pool3 = {k: {"p": [], "y": []} for k in ("signs", "imb", "both")}

    for coin in COINS:
        ch, ups, imb = load(coin)
        print(f"\n=== {coin} (n={len(ch)}) ===")
        fr = flip_rates(ch, imb)
        d, o = fr["flow_driven"], fr["flow_opposed"]
        z = (d["rate"] - o["rate"]) / np.hypot(d["se"], o["se"])
        print(f"  flip| flow-driven {d['rate']:.4f} (n={d['n']})   "
              f"flow-opposed {o['rate']:.4f} (n={o['n']})   z={z:+.1f}")
        print("  driven flip by |imb| quintile: "
              + "  ".join(f"{r['rate']:.4f}" for r in fr["driven_by_absimb"]))
        print("  opposed flip by |imb| quintile: "
              + "  ".join(f"{r['rate']:.4f}" for r in fr["opposed_by_absimb"]))

        ca = conditional_auc(ch, ups, imb)
        for k in pool2:
            pool2[k].append(ca[k])
        auc_d = roc_auc_score(ca["actuals"][ca["driven"]], ca["probs"][ca["driven"]])
        auc_o = roc_auc_score(ca["actuals"][ca["opposed"]], ca["probs"][ca["opposed"]])
        print(f"  ising OOS AUC | prior bar flow-driven {auc_d:.4f}  flow-opposed {auc_o:.4f}")

        sub = subsumption(ch, ups, imb)
        for k in pool3:
            pool3[k]["p"].append(sub[k]["probs"])
            pool3[k]["y"].append(sub[k]["y"])
        print(f"  logit AUC | signs {sub['signs']['auc']:.4f}  "
              f"imb {sub['imb']['auc']:.4f}  both {sub['both']['auc']:.4f}   "
              f"sign-lag1 coef: alone {sub['coef_sign_lag1']['signs']:+.3f} "
              f"with-imb {sub['coef_sign_lag1']['both']:+.3f}")
        summary[coin] = {
            "flip": {k: fr[k] for k in ("flow_driven", "flow_opposed")},
            "flip_dose": {k: fr[k] for k in ("driven_by_absimb", "opposed_by_absimb")},
            "auc_driven": float(auc_d), "auc_opposed": float(auc_o),
            "sub_auc": {k: sub[k]["auc"] for k in ("signs", "imb", "both")},
            "coef_sign_lag1": sub["coef_sign_lag1"],
        }

    # pooled
    print("\n=== pooled (4 coins) ===")
    P = np.concatenate(pool2["probs"]); Y = np.concatenate(pool2["actuals"])
    D = np.concatenate(pool2["driven"]); O = np.concatenate(pool2["opposed"])
    auc_d, auc_o = roc_auc_score(Y[D], P[D]), roc_auc_score(Y[O], P[O])
    print(f"  ising OOS AUC | flow-driven {auc_d:.4f} (n={D.sum()})  "
          f"flow-opposed {auc_o:.4f} (n={O.sum()})")

    p3 = {k: np.concatenate(v["p"]) for k, v in pool3.items()}
    y3 = np.concatenate(pool3["signs"]["y"])
    aucs = {k: float(roc_auc_score(y3, p)) for k, p in p3.items()}
    print(f"  logit AUC | signs {aucs['signs']:.4f}  imb {aucs['imb']:.4f}  "
          f"both {aucs['both']:.4f}")
    d_both_signs = boot_auc_diff(y3, p3["both"], p3["signs"], rng)
    d_signs_imb = boot_auc_diff(y3, p3["signs"], p3["imb"], rng)
    print(f"  AUC(both)-AUC(signs) = {d_both_signs['mean']:+.4f} "
          f"CI {d_both_signs['ci']}")
    print(f"  AUC(signs)-AUC(imb)  = {d_signs_imb['mean']:+.4f} "
          f"CI {d_signs_imb['ci']}")

    summary["pooled"] = {
        "auc_driven": float(auc_d), "n_driven": int(D.sum()),
        "auc_opposed": float(auc_o), "n_opposed": int(O.sum()),
        "sub_auc": aucs,
        "boot_both_minus_signs": d_both_signs,
        "boot_signs_minus_imb": d_signs_imb,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "flow_test.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'flow_test.json'}")


if __name__ == "__main__":
    main()
