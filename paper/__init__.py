"""Empirical pipeline for the paper:

    "Power-Law Regularized Autoregressive Logit Models via an Ising Prior"

Everything in this package fits *every* model by the same maximum-likelihood
(log-loss) objective so the comparison is apples-to-apples, then evaluates
out-of-sample under a rolling walk-forward protocol across multiple sampling
intervals (5m / 15m / 1h / 4h) and assets (BTC, ETH, SOL, XRP).
"""
