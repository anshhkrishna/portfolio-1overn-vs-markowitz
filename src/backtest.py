"""Rolling out-of-sample backtest harness.

At each rebalance month t, a strategy's weights are computed from the
`window` months of returns strictly before t, then held fixed while month
t's already-realized return is applied to them. No return used to compute
weights for month t is ever the return of month t or later.
"""

import numpy as np


def rolling_backtest(dates, returns, window, weight_fn):
    """Run one strategy's rolling out-of-sample backtest.

    Returns (realized_dates, realized_returns): realized_returns[i] is the
    portfolio return earned in month realized_dates[i], using weights
    computed from the `window` months strictly before it.
    """
    n_months = returns.shape[0]
    realized_dates = []
    realized_returns = []
    for t in range(window, n_months):
        train_window = returns[t - window:t]
        weights = weight_fn(train_window)
        realized_returns.append(returns[t] @ weights)
        realized_dates.append(dates[t])
    return np.array(realized_dates), np.array(realized_returns)


def annualized_stats(monthly_returns):
    """(mean, volatility, Sharpe ratio), all annualized from monthly returns."""
    mean = np.mean(monthly_returns) * 12
    vol = np.std(monthly_returns, ddof=1) * np.sqrt(12)
    sharpe = mean / vol
    return mean, vol, sharpe
