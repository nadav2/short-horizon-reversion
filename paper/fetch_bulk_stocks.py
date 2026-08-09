"""Bulk-fetch a wide universe of liquid US large-cap stocks and broad ETFs from
Alpaca at 15m (matched span), for the hundreds-of-assets cross-market study.

We deliberately use mature, liquid, *unlevered* instruments (S&P-500-style large
caps across all sectors + broad/sector ETFs) — the clean "efficiently intermediated
market" comparison set — not leveraged/inverse ETFs.

Writes paper/bulk_data/{sym}-15m.json with {timestamp, datetime, change, up}.

    cd scripts/data/py && dotenvx run -f ../../../.env -- \
        uv run --active python /abs/.../paper/fetch_bulk_stocks.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent / "bulk_data"
START, END = "2025-01-01", "2026-02-11"

# Curated universe: liquid large caps across sectors + broad/sector ETFs (~230 names).
UNIVERSE = """
AAPL MSFT NVDA GOOGL GOOG AMZN META AVGO TSLA ORCL ADBE CRM AMD INTC CSCO QCOM TXN
IBM NOW INTU AMAT MU LRCX KLAC SNPS CDNS ADI MRVL PANW ANET FTNT ABNB UBER PYPL SHOP
JPM BAC WFC C GS MS BLK SCHW AXP USB PNC TFC COF BK SPGI CME ICE MCO V MA FIS FISV
BRK.B AIG MET PRU ALL TRV PGR CB AFL
UNH JNJ LLY ABBV MRK PFE TMO ABT DHR BMY AMGN GILD CVS CI ELV HUM ISRG MDT SYK BSX
VRTX REGN ZTS BDX
WMT PG KO PEP COST MCD NKE SBUX TGT LOW HD DIS CMCSA NFLX TMUS VZ T CL KMB MDLZ MO PM
KHC GIS HSY KR DG DLTR ROST TJX YUM CMG MAR HLT BKNG
XOM CVX COP SLB EOG MPC PSX VLO OXY WMB KMI OKE HAL DVN HES
BA CAT DE HON GE MMM UPS FDX LMT RTX NOC GD UNP CSX NSC EMR ETN ITW PH ROK CMI WM
CARR OTIS PCAR
LIN APD SHW ECL FCX NEM DOW DD NUE
NEE DUK SO D AEP EXC SRE XEL PEG ED WEC
AMT PLD CCI EQIX PSA O SPG WELL DLR VICI
ADP PYPL SQ COIN PLTR SNOW DDOG NET CRWD ZS MDB
SPY QQQ IWM DIA VTI VOO IVV MDY RSP
XLK XLF XLE XLV XLY XLP XLI XLU XLB XLRE XLC
GLD SLV GDX USO UNG DBC
TLT IEF SHY HYG LQD AGG TIP BND
EEM EFA VEA VWO FXI EWZ EWJ INDA
ARKK SMH SOXX XBI IBB KRE ITB XHB VNQ
""".split()


def fetch_batch(client, symbols):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    req = StockBarsRequest(symbol_or_symbols=symbols,
                           timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                           start=datetime.fromisoformat(f"{START}T00:00:00"),
                           end=datetime.fromisoformat(f"{END}T23:59:59"))
    return client.get_stock_bars(req)


def main():
    from alpaca.data.historical import StockHistoricalDataClient
    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    OUTDIR.mkdir(exist_ok=True)
    syms = sorted(set(UNIVERSE))
    print(f"fetching {len(syms)} stock/ETF symbols → {OUTDIR}")
    done = 0
    BATCH = 40
    for b in range(0, len(syms), BATCH):
        batch = [s for s in syms[b:b + BATCH] if not (OUTDIR / f"{s.lower().replace('.', '-')}-15m.json").exists()]
        if not batch:
            continue
        try:
            bars = fetch_batch(client, batch)
        except Exception as e:
            print(f"  batch {b} FAILED: {e}")
            continue
        for sym in batch:
            data = bars.data.get(sym, [])
            if len(data) < 5000:
                continue
            rows = []
            for bar in sorted(data, key=lambda x: x.timestamp):
                o, c = float(bar.open), float(bar.close)
                ts = int(bar.timestamp.timestamp())
                rows.append({"timestamp": ts,
                             "datetime": bar.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                             "up": c > o, "change": (c - o) / o if o else 0.0})
            sid = sym.lower().replace(".", "-")
            (OUTDIR / f"{sid}-15m.json").write_text(json.dumps(rows))
            done += 1
        print(f"  batch {b}-{b+len(batch)}: total written {done}")
    print(f"done: {done} stock/ETF symbols written")


if __name__ == "__main__":
    main()
