"""Portfolio weight rules.

Each rule takes a trailing window of monthly returns (T x N, T months of
history strictly before the rebalance date, N assets) and returns a weight
vector of length N that sums to 1. Keeping a common signature lets the
backtest harness in `backtest.py` treat every strategy interchangeably.
"""

import numpy as np
from sklearn.covariance import LedoitWolf


def equal_weight(window_returns):
    """1/N weights. Ignores the window's contents; only its asset count."""
    n = window_returns.shape[1]
    return np.ones(n) / n


def _min_variance_weights(cov):
    """Unconstrained minimum-variance weights for a given covariance matrix."""
    n = cov.shape[0]
    ones = np.ones(n)
    inv_cov_ones = np.linalg.solve(cov, ones)
    return inv_cov_ones / (ones @ inv_cov_ones)


def min_variance(window_returns):
    """Sample-covariance minimum-variance weights, no short-sale constraint."""
    cov = np.cov(window_returns, rowvar=False)
    return _min_variance_weights(cov)


def tangency(window_returns):
    """Sample mean-variance (tangency) weights: Sigma^-1 mu, normalized to sum to 1."""
    cov = np.cov(window_returns, rowvar=False)
    mu = np.mean(window_returns, axis=0)
    raw = np.linalg.solve(cov, mu)
    return raw / np.sum(raw)


def shrinkage_min_variance(window_returns):
    """Minimum-variance weights using a Ledoit-Wolf shrinkage covariance estimate."""
    cov = LedoitWolf().fit(window_returns).covariance_
    return _min_variance_weights(cov)
