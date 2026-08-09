"""Per-calendar-year walk-forward for the focal US-listed instruments, 2021-2026.

Companion to paper.multiyear (crypto): the same matched 15m geometry and
two-model block-bootstrap scoring, applied to the equity/commodity/bond side
so the "absent in listed instruments" claim is tested across five years and
several regimes (incl. the 2022 bear market), not just the 14-month matched
window. Requires paper.fetch_multiyear_stocks to have been run first.

    uv run --active python -m paper.multiyear_stocks
"""

from __future__ import annotations

import json

from .multiyear import DATA, OUT, score_asset

STOCKS = ["spx", "qqq", "iwm", "aapl", "nvda", "tsla", "gld", "tlt"]


def main():
    rows = []
    for sid in STOCKS:
        if not (DATA / f"{sid}-15m.json").exists():
            print(f"  {sid}: no data, skipping (run fetch_multiyear_stocks)")
            continue
        rows.extend(score_asset(sid))
    (OUT / "multiyear_stocks.json").write_text(json.dumps(rows, indent=2))
    n_sig = sum(r["p_ising"] < 0.05 and r["p_free"] < 0.05 for r in rows)
    print(f"\n{n_sig}/{len(rows)} instrument-years two-model significant")
    print(f"Wrote {OUT/'multiyear_stocks.json'}")


if __name__ == "__main__":
    main()
