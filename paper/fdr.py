"""Multiple-testing control for the wide-universe study (Benjamini-Hochberg FDR).

The wide study tests hundreds of assets simultaneously; raw per-asset 5% tests
overstate the population claim. This script applies BH-FDR jointly across the
full universe (crypto + stocks together, the conservative choice) at q=0.05 to

  (a) each model's one-sided bootstrap p(AUC <= 0.5) separately, and
  (b) the two-model conjunction via the intersection-union test
      p_conj = max(p_ising, p_free)  (valid p-value for "both models have skill").

Bootstrap p-values of exactly 0 mean "below the bootstrap resolution"; they are
floored at 1/(N_BOOT+1) before BH so the procedure stays valid.

    uv run --active python -m paper.fdr
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
N_BOOT = {"wide.json": 300, "wide_1h.json": 300}   # resamples used by wide / wide_horizon
Q = 0.05


def bh_mask(pvals: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg step-up: True where the hypothesis is rejected at FDR q."""
    n = len(pvals)
    order = np.argsort(pvals)
    thresh = q * (np.arange(1, n + 1)) / n
    passed = pvals[order] <= thresh
    kmax = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    mask = np.zeros(n, dtype=bool)
    mask[order[:kmax]] = True
    return mask


def analyze(fname: str) -> dict:
    rows = json.loads((OUT / fname).read_text())
    floor = 1.0 / (N_BOOT[fname] + 1)
    p_is = np.maximum([r["p_ising"] for r in rows], floor)
    p_fr = np.maximum([r["p_free"] for r in rows], floor)
    p_conj = np.maximum(p_is, p_fr)

    m_is, m_fr, m_conj = bh_mask(p_is, Q), bh_mask(p_fr, Q), bh_mask(p_conj, Q)
    cls = np.array([r["class"] for r in rows])

    out = {"file": fname, "q": Q, "p_floor": floor, "n_total": len(rows), "classes": {}}
    for c in sorted(set(cls)):
        sel = cls == c
        n = int(sel.sum())
        raw_both = int(np.sum((p_is[sel] < 0.05) & (p_fr[sel] < 0.05)))
        out["classes"][c] = {
            "n": n,
            "raw_both_sig": raw_both,
            "raw_both_frac": raw_both / n,
            "bh_ising_sig": int(np.sum(m_is & sel)),
            "bh_ising_frac": float(np.mean(m_is[sel])),
            "bh_conj_sig": int(np.sum(m_conj & sel)),
            "bh_conj_frac": float(np.mean(m_conj[sel])),
        }
        out["classes"][c]["bh_conj_assets_not_sig"] = sorted(
            r["asset"] for r, mc in zip(rows, m_conj) if r["class"] == c and not mc
        ) if c == "crypto" else None
    return out


def main():
    res = {}
    for fname in N_BOOT:
        if not (OUT / fname).exists():
            print(f"{fname}: missing, skipped")
            continue
        r = analyze(fname)
        res[fname.replace(".json", "")] = r
        print(f"\n=== {fname} (BH-FDR q={Q}, joint over {r['n_total']} assets) ===")
        for c, s in r["classes"].items():
            print(f"  {c:6s} n={s['n']:4d}  raw both-model {s['raw_both_frac']*100:4.0f}%  "
                  f"BH ising {s['bh_ising_frac']*100:4.0f}%  BH conjunction {s['bh_conj_frac']*100:4.0f}%")
    (OUT / "fdr.json").write_text(json.dumps(res, indent=2))
    print(f"\nWrote {OUT/'fdr.json'}")


if __name__ == "__main__":
    main()
