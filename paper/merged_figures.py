"""Per-section figures for the merged manuscript (docs/paper.tex).

One main visual per section of the merged paper:

  m_location.pdf      the wrapper panel: 17 NAV-linked wrappers (plus
                      futures/leveraged and correlated controls) against
                      their underlying's session-matched AUC.
  m_transmission.pdf  wrapper vs underlying AUC across the 14 event-study
                      era cells (births, BITO eras, leveraged, full eras).
  m_flow.pdf          (a) pooled next-bar sign-flip rate by prior-bar
                      |taker imbalance| quintile, flow-driven vs opposed;
                      (b) per-coin flow-driven minus flow-opposed flip-rate
                      difference with bootstrap CIs; (c) delta-flip vs
                      fitted coupling across the 183-pair cross-section.

    uv run --active python -m paper.merged_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .style import OI, ONEHALF, TEXT, letter, plt, save

OUT = Path(__file__).resolve().parent / "out"

FAMILY = {"gold": OI["orange"], "silver": OI["orange"], "platinum": OI["orange"],
          "palladium": OI["orange"], "bitcoin": OI["vermillion"],
          "ether": OI["vermillion"], "euro": OI["purple"], "yen": OI["purple"]}


def _swarm_levels(xs, thresh=0.0011):
    """Greedy beeswarm: for sorted xs, the vertical level that keeps markers
    at least `thresh` apart in x on each level."""
    last = {}
    levels = []
    for x in xs:
        lvl = 0
        while lvl in last and x - last[lvl] < thresh:
            lvl += 1
        last[lvl] = x
        levels.append(lvl)
    return levels


def fig_location():
    """One row per underlying family (sorted by underlying AUC): diamond =
    underlying with 95% CI, dots = its NAV-linked funds, connector = family
    gap; correlated controls in a bottom strip; stock band as backdrop."""
    d = json.loads((OUT / "natural.json").read_text())
    panel = d["panel"]
    ref = panel["reference_rth_stocks"]
    rth = {r["asset"]: r for r in d["rows"] if r["rth"] and not r["gap"]}
    und = {r["experiment"]: r for r in d["rows"]
           if r["rth"] and not r["gap"] and r["role"] == "underlying"}
    by_fam: dict[str, list] = {}
    for p in panel["pairs"]:
        by_fam.setdefault(p["experiment"], []).append(p)

    ALT = {"bitcoin": ["bito", "bitu"], "ether": ["ethu"]}
    CTRL = [("gdx", "gold"), ("nem", "gold"), ("uup", "euro"),
            ("mstr", "bitcoin"), ("coin", "bitcoin")]
    LABEL = {"bitcoin": "Bitcoin (8 spot funds)", "ether": "Ether (3 funds)",
             "platinum": "platinum (PPLT)", "palladium": "palladium (PALL)",
             "gold": "gold (GLD)", "silver": "silver (SLV)",
             "euro": "euro (FXE)", "yen": "yen (FXY)"}
    OFF_U, OFF_W = 0.18, -0.14              # underlying / wrapper sub-lines
    LVL_DY = (0.0, -0.15, 0.12, -0.30)      # beeswarm offsets off the wrapper line

    fams = sorted(by_fam, key=lambda e: -und[e]["auc_ising"])
    fig, ax = plt.subplots(figsize=(ONEHALF, ONEHALF * 0.80))
    m, s = ref["mean"], ref["sd"]
    ax.axvspan(m - 2 * s, m + 2 * s, color="0.88", alpha=0.6, zorder=0)
    ax.axvline(m, color="0.55", lw=0.7, ls="--", zorder=1, ymax=0.92)
    ax.text(m, len(fams) + 0.52, "187 listed stocks/ETFs (RTH)\nmean ± 2 SD",
            fontsize=6.2, color="0.35", ha="center", va="center")

    ticks, labels = [], []
    for i, fam in enumerate(fams):
        y = len(fams) - i
        c = FAMILY[fam]
        u = und[fam]
        ux, uci = u["auc_ising"], u["auc_ci"]
        spot = sorted(p["auc_wrapper"] for p in by_fam[fam])
        wmean = float(np.mean(spot))

        ax.plot([ux, wmean], [y + OFF_U, y + OFF_W], color="0.6", lw=0.8,
                zorder=2)
        ax.errorbar(ux, y + OFF_U, xerr=[[ux - uci[0]], [uci[1] - ux]],
                    fmt="none", ecolor="0.72", elinewidth=1.0, capsize=1.6,
                    zorder=2)
        ax.scatter(ux, y + OFF_U, s=30, marker="D", color=c, edgecolor="k",
                   linewidths=0.5, zorder=4)

        marks = [(x, "o") for x in spot]
        marks += [(rth[sid]["auc_ising"], "^") for sid in ALT.get(fam, ())
                  if sid in rth]
        marks.sort()
        for (x, mk), lvl in zip(marks, _swarm_levels([x for x, _ in marks])):
            ax.scatter(x, y + OFF_W + LVL_DY[lvl], s=23 if mk == "o" else 27,
                       marker=mk, color=c, edgecolor="k", linewidths=0.45,
                       zorder=3)
        ticks.append(y)
        labels.append(LABEL[fam])

    ax.axhline(0.45, color="0.82", lw=0.6, zorder=1)
    yc = -0.25
    for k, (sid, exp) in enumerate(sorted(CTRL, key=lambda t: rth[t[0]]["auc_ising"])):
        x = rth[sid]["auc_ising"]
        ax.scatter(x, yc, s=26, facecolor="none", edgecolor=FAMILY[exp],
                   linewidths=1.1, zorder=3)
        ax.text(x, yc + 0.30 if k % 2 else yc - 0.33, sid.upper(),
                fontsize=5.8, color="0.3", ha="center",
                va="bottom" if k % 2 else "top")
    ticks.append(yc)
    labels.append("correlated controls")

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-0.85, len(fams) + 0.85)
    ax.set_xlim(0.4755, 0.5495)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("AUC (identical session-matched RTH 15-min slots)")

    hnd = [plt.Line2D([], [], marker="D", ls="", color="0.45",
                      markeredgecolor="k", markersize=4.5, label="underlying"),
           plt.Line2D([], [], marker="o", ls="", color="0.45",
                      markeredgecolor="k", markersize=5, label="NAV-linked wrapper"),
           plt.Line2D([], [], marker="^", ls="", color="0.45",
                      markeredgecolor="k", markersize=5,
                      label="futures-NAV / 2× leveraged"),
           plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
                      markeredgecolor="0.45", markersize=5,
                      label="correlated, no NAV link")]
    ax.legend(handles=hnd, loc="upper center", bbox_to_anchor=(0.5, -0.115),
              ncol=4, fontsize=6.2, handletextpad=0.1, columnspacing=0.7)
    save(fig, "m_location")


def fig_transmission():
    """Wrapper-minus-underlying AUC gap per era cell, grouped by design."""
    gb = {r["cell"]: r for r in
          json.loads((OUT / "wrapper_gap_boot.json").read_text())["rows"]}

    def row(cell, label):
        r = gb[cell]
        return (label, r["gap"], r["gap_ci"], r["auc_underlying"])

    groups = [
        ("first 6 months of trading", OI["blue"], [
            row("ibit_birth", "IBIT"), row("fbtc_birth", "FBTC"),
            row("gbtc_birth", "GBTC"), row("etha_birth", "ETHA"),
            row("feth_birth", "FETH"), row("ethe_birth", "ETHE")]),
        ("futures-based NAV", OI["green"], [
            row("bito_early", "BITO 2021–23"), row("bito_late", "BITO 2024–26")]),
        ("2× leveraged", OI["purple"], [
            row("bitu_panel", "BITU"), row("ethu_panel", "ETHU")]),
        ("full post-launch era", OI["vermillion"], [
            row("ibit_full", "IBIT"), row("fbtc_full", "FBTC"),
            row("gbtc_etf", "GBTC"), row("ethe_etf", "ETHE")]),
    ]

    fig, ax = plt.subplots(figsize=(ONEHALF, ONEHALF * 0.70))

    y, ticks, labels = 0.0, [], []
    ax.axvline(0, color="0.3", lw=0.9, zorder=1)
    for gname, color, rows in groups:
        y -= 0.9
        ax.text(-0.0475, y, gname, fontsize=6.3, style="italic", color="0.35",
                va="center")
        for lab, gap, ci, uauc in rows:
            y -= 1.0
            ax.errorbar(gap, y, xerr=[[gap - ci[0]], [ci[1] - gap]], fmt="none",
                        ecolor="0.62", elinewidth=0.9, capsize=1.8, zorder=2)
            ax.scatter(gap, y, s=30, color=color, edgecolor="k", linewidths=0.45,
                       zorder=3)
            ax.text(0.0475, y, f"{uauc:.3f}", fontsize=6.0, color="0.45",
                    va="center", ha="right")
            ticks.append(y); labels.append(lab)

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_ylim(y - 0.9, 1.6)
    ax.set_xlim(-0.05, 0.05)
    ax.set_xlabel("wrapper AUC $-$ underlying AUC (identical dates and slots)")
    ax.text(0.0475, 0.35, "underlying AUC", fontsize=6.0, color="0.45",
            ha="right", va="center")
    ax.text(0.0, 0.35, "perfect inheritance", fontsize=6.2, color="0.3",
            ha="center", va="center")
    ax.grid(axis="y", visible=False)
    save(fig, "m_transmission")


def fig_flow():
    ft = json.loads((OUT / "flow_test.json").read_text())
    fb = json.loads((OUT / "flow_boot.json").read_text())
    fc = json.loads((OUT / "flow_cross.json").read_text())
    coins = ["btc", "eth", "sol", "xrp"]

    canvas = TEXT + 1.26            # tight-bbox label margin; delivers 6.50 in
    fig, (ax, ax2, ax3) = plt.subplots(
        1, 3, figsize=(canvas, canvas * 0.30), gridspec_kw={"wspace": 0.42})

    for cond, color, lab in (("driven_by_absimb", OI["vermillion"], "flow-driven prior bar"),
                             ("opposed_by_absimb", OI["blue"], "flow-opposed prior bar")):
        Q = np.zeros(5)
        N = np.zeros(5)
        for c in coins:
            for q, r in enumerate(ft[c]["flip_dose"][cond]):
                Q[q] += r["rate"] * r["n"]
                N[q] += r["n"]
        ax.plot(np.arange(1, 6), Q / N, "-o", color=color, markersize=4,
                markeredgecolor="k", markeredgewidth=0.4, label=lab)
    ax.axhline(0.5, color="0.55", lw=0.7, ls="--")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel(r"prior-bar $|\mathrm{taker\ imbalance}|$ quintile")
    ax.set_ylabel("next-bar sign-flip rate")
    ax.set_ylim(0.49, 0.545)
    ax.legend(loc="lower right", fontsize=6.5)
    letter(ax, "(a)")

    xs = np.arange(4)
    pts = [fb[c]["delta_flip"] for c in coins]
    los = [fb[c]["delta_flip"] - fb[c]["ci"][0] for c in coins]
    his = [fb[c]["ci"][1] - fb[c]["delta_flip"] for c in coins]
    ax2.errorbar(xs, pts, yerr=[los, his], fmt="o", color=OI["vermillion"],
                 markersize=4.5, markeredgecolor="k", markeredgewidth=0.4,
                 ecolor="0.35", elinewidth=0.9, capsize=2.5)
    ax2.axhline(0, color="0.3", lw=0.7)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([c.upper() for c in coins], fontsize=6.5)
    ax2.set_ylabel(r"$\Delta$ flip rate (driven $-$ opposed)")
    ax2.set_ylim(-0.005, 0.040)
    letter(ax2, "(b)")

    dx = [r["delta_flip"] for r in fc["rows"]]
    dy = [r["A"] for r in fc["rows"]]
    ax3.scatter(dx, dy, s=9, color=OI["vermillion"], alpha=0.55, linewidths=0,
                zorder=3)
    ax3.axhline(0, color="0.55", lw=0.6, ls="--")
    ax3.axvline(0, color="0.55", lw=0.6, ls="--")
    rho = fc["correlations"]["A_vs_delta_flip"]["rho"]
    ax3.annotate(f"Spearman $\\rho$ = ${rho:+.2f}$\n$p = 4\\times10^{{-7}}$",
                 xy=(-0.033, -0.50), fontsize=6.5)
    ax3.set_xlim(-0.037, 0.037)
    ax3.set_ylim(-0.58, 0.18)   # 3 extreme-|A| pegged/microcap pairs clipped
    ax3.set_xlabel(r"$\Delta$ flip rate (183 pairs)")
    ax3.set_ylabel("fitted coupling $A$")
    letter(ax3, "(c)")
    save(fig, "m_flow")


if __name__ == "__main__":
    fig_location()
    fig_transmission()
    fig_flow()
