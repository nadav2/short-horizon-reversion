# Short-horizon mean reversion in cryptocurrency markets — replication package

Replication code, frozen symbol lists, and frozen result files for the paper

> **Short-horizon mean reversion in cryptocurrency markets: a
> kernel-constrained logit measurement of directional predictability**
> (N. Kitron)

**The manuscript is included here as [`paper.pdf`](paper.pdf) until the
arXiv version is live; it will then be replaced by the arXiv link.**

Every number, table, and figure in the manuscript is produced by a module in
`paper/`. The result files the manuscript was typeset from are frozen in
`paper/out/` with SHA-256 hashes in `HASHES.sha256`.

## Layout

| Path | Contents |
|------|----------|
| `paper/` | the analysis pipeline (plain Python modules, run as `python -m paper.<module>`) |
| `symbols/` | **frozen universes**: `crypto_universe.txt` (183 Binance USDT pairs), `stock_universe.txt` (187 US stocks/ETFs), `focal_instruments.txt` |
| `paper/out/` | frozen result JSONs the manuscript numbers come from |
| `HASHES.sha256` | SHA-256 of every frozen file (`shasum -a 256 -c HASHES.sha256`) |
| `data/` | input candles (gitignored; rebuilt by the fetch scripts below) |
| `paper/bulk_data/`, `paper/multiyear_data/`, `paper/fx_raw/` | wide-universe / multi-year / FX raw data (gitignored; rebuilt) |
| `figures/` | regenerated manuscript figures (gitignored) |

## Environment

Python ≥ 3.11 via [uv](https://docs.astral.sh/uv/):

```bash
uv sync                 # numpy, scikit-learn, scipy, matplotlib
uv sync --extra fetch   # adds alpaca-py (only needed to re-download stock data)
```

All bootstrap and permutation procedures are seeded; given identical inputs,
reruns are deterministic.

## Rebuilding the input data

All sources are public. Crypto and FX need no credentials; US-listed bars
need free Alpaca market-data keys (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`).

```bash
# 1. Focal coins (BTC/ETH/SOL/XRP at 5m/15m/1h/4h, public Binance klines)
uv run python -m paper.fetch_focal_crypto

# 2. Wide crypto universe (15m, matched span). The published universe is the
#    FROZEN list, not the volume ranking (which is time-dependent):
uv run python -m paper.fetch_bulk_crypto --symbols \
    $(tr 'a-z' 'A-Z' < symbols/crypto_universe.txt | sed 's/$/USDT/')

# 3. Focal US-listed instruments (Alpaca; SPY/QQQ/IWM/AAPL/NVDA/TSLA/GLD/TLT,
#    15m + resampled 1h/4h)
uv run python -m paper.fetch_markets

# 4. Wide stock/ETF universe (Alpaca, 15m)
uv run python -m paper.fetch_bulk_stocks

# 5. Spot FX (Dukascopy via dukascopy-node, then convert to the project schema)
cd paper && npx dukascopy-node -i eurusd -from 2025-01-01 -to 2026-02-12 -t m15 -f json -dir fx_raw
#   ... same for usdjpy, gbpusd ... then:
cd .. && uv run python -m paper.fetch_fx

# 6. Multi-year histories (2021-01-01 onward)
uv run python -m paper.fetch_multiyear          # crypto, public Binance
uv run python -m paper.fetch_multiyear_stocks   # stocks, Alpaca

# 7. External-validity data: independent venues + quote-mid bars
uv run python -m paper.fetch_exchanges          # Coinbase/OKX/Bybit/Binance 15m OHLCV (public)
#    Kraken is omitted: its public OHLC endpoint caps history at 720 bars.
#    Dukascopy bid+ask candles (quote-midprice bars; run from paper/):
#    for i in btcusd ethusd; do for p in bid ask; do
#      npx dukascopy-node -i $i -from 2025-01-01 -to 2026-02-12 -t m15 -f json -p $p -dir mid_raw
#    done; done
```

```bash
# 8. Natural-experiment instruments: spot gold (Dukascopy) + listed BTC wrappers (Alpaca)
cd paper && npx dukascopy-node -i xauusd -from 2025-01-01 -to 2026-02-12 -t m15 -f json -dir natural_raw
cd .. && uv run python -m paper.fetch_natural --fx
uv run python -m paper.fetch_natural --alpaca     # IBIT, FBTC, COIN, MSTR
```

Notes on data quirks documented in the paper: US-listed bars are 24-hour
bars (a regular-trading-hours-only robustness check is part of the
pipeline); FX closed-market bars are removed by `fetch_fx`; flat candles
(`change == 0`) are labeled "down" and their effect is quantified by
`paper.liquidity`; the crypto universe excludes stablecoin-quote pairs and
leveraged tokens by construction.

## The main test

The paper's headline claim is the boxed pre-specified test in the
manuscript's protocol section. It is exactly:

```bash
uv run python -m paper.wide --dump-oos   # 370-asset walk-forward -> paper/out/wide.json, paper/out/wide_oos/
uv run python -m paper.fdr               # conjunction p-values + joint BH FDR -> paper/out/fdr.json
uv run python -m paper.dependence        # joint time-block bootstrap of the class-mean AUC gap
uv run python -m paper.dependence_sensitivity  # the gap bootstrap at B=5,000 across block lengths {192,384,768}
uv run python -m paper.exante            # ex-ante (Jan-2025 volume) universe-selection robustness -> paper/out/exante.json
```

## Everything else

```bash
uv run python -m paper.compare_markets     # focal cross-market study -> paper/out/markets.json
uv run python -m paper.exchanges           # cross-exchange replication + composite -> paper/out/exchanges.json
uv run python -m paper.midprice            # quote-midprice bars (Dukascopy bid/ask) -> paper/out/midprice.json
uv run python -m paper.return_defs         # open-close / close-close / VWAP-VWAP labels -> paper/out/return_defs.json
uv run python -m paper.robustness          # variance ratios, permutation test, sub-periods
uv run python -m paper.robust_spin         # volatility-standardized spins
uv run python -m paper.rth                 # regular-trading-hours-only rerun
uv run python -m paper.multiyear           # per-year crypto stability 2021-2026
uv run python -m paper.multiyear_stocks    # per-year stocks 2021-2026
uv run python -m paper.run                 # within-crypto walk-forward + lag sweep
uv run python -m paper.ml_baseline         # GBM + MLP baselines (untuned and tuned)
uv run python -m paper.kernel_shapes       # power-law vs exponential vs flat kernels
uv run python -m paper.gap_test --all      # skip-one-bar microstructure test
uv run python -m paper.liquidity           # volume stratification + flat-bar AUC
uv run python -m paper.negative_controls   # shuffled labels, IAAFT surrogates, time reversal
uv run python -m paper.wide_null           # shuffled-label null over the FULL 370-asset universe,
                                           #   identical pipeline incl. joint BH-FDR -> paper/out/wide_null.json
uv run python -m paper.full_boot           # full-procedure bootstrap: complete walk-forward refits inside
                                           #   each of B=200 raw-series block resamples, focal coins +
                                           #   50 representative assets -> paper/out/full_boot.json
uv run python -m paper.generative          # Glauber self-consistency
uv run python -m paper.dfa                 # DFA-1 Hurst exponents + joint block bootstrap of the class-mean H gap
uv run python -m paper.signlag             # model-free multi-lag SIGN autocorrelation, Ljung-Box Q(12),
                                           #   kernel-weighted R (alpha=1 fixed a priori), joint gap bootstrap,
                                           #   and the |r|-decile decomposition of rho_1
uv run python -m paper.iaaft_diagnostic    # why IAAFT surrogates over-produce significance on non-crypto
                                           #   (no refits: frozen results + focal candles only)
uv run python -m paper.natural             # same-asset natural experiments: XAUUSD vs GLD,
                                           #   BTC vs IBIT/FBTC/COIN/MSTR, on 24h / RTH / one-bar-gap bars
uv run python -m paper.wide_rth            # RTH-only rerun of the full 187-stock wide universe
uv run python -m paper.micro_regression    # cross-sectional HC1 regression of flat-bar-robust AUC
uv run python -m paper.selective           # coverage-accuracy-edge, cost of capture
uv run python -m paper.simulate            # stylized contract-level simulation
uv run python -m paper.mechanism           # volatility-regime / time-of-day checks
uv run python -m paper.wide_horizon        # wide universe at 1h
uv run python -m paper.tables              # within-crypto tables -> paper/out/tables.md

# figures (written to figures/)
uv run python -m paper.market_figures
uv run python -m paper.figures
uv run python -m paper.wide_figures
uv run python -m paper.dfa_figures
uv run python -m paper.kernel_figures
uv run python -m paper.micro_figures
uv run python -m paper.sim_figures
uv run python -m paper.selective_figures
uv run python -m paper.signlag_figures
```

Dependencies between steps: `paper.run` and `paper.wide --dump-oos` write
the out-of-sample dumps (`paper/out/oos_*.npz`, `paper/out/wide_oos/`) that
`paper.selective`, `paper.liquidity`, `paper.dependence`,
`paper.dependence_sensitivity`, and parts of `paper.figures` consume.
`paper.signlag` and `paper.dfa` read the wide-universe candles
(`paper/bulk_data/`); `paper.iaaft_diagnostic` reads frozen results plus
the focal candles; `paper.micro_regression` reads only `liquidity.json`. These intermediates are large and therefore not
frozen in the repository; rerun the two producer commands to rebuild them.

## Verifying the frozen results

```bash
shasum -a 256 -c HASHES.sha256
```

Re-running the pipeline on freshly fetched data reproduces the frozen
numbers exactly when the data snapshot matches, and to within ordinary
data-revision noise otherwise (exchanges occasionally restate candles).

## Walk-forward fold definition

With train size `s_tr` and test size `s_te` (in candles; 15m: 5760/960,
1h: 2160/360, 4h: 720/120), fold `j = 0, 1, ...` trains on
`[j*s_te, j*s_te + s_tr)` and tests on
`[j*s_te + s_tr, min(j*s_te + s_tr + s_te, n))`. Out-of-sample blocks are
concatenated and scored once per (asset, interval, model).

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `PAPER_DATA_DIR` | `./data` | input candle directory |
| `PAPER_FIG_DIR` | `./figures` | figure output directory |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | — | only for the stock fetchers |
