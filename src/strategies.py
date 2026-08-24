"""Portfolio weight rules.

Each rule takes a trailing window of monthly returns (T x N, T months of
history strictly before the rebalance date, N assets) and returns a weight
vector of length N that sums to 1. Keeping a common signature lets the
backtest harness in `backtest.py` treat every strategy interchangeably.
"""

import numpy as np


def equal_weight(window_returns):
    """1/N weights. Ignores the window's contents; only its asset count."""
    n = window_returns.shape[1]
    return np.ones(n) / n
