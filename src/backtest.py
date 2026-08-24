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
    computed from the `window` months strictly before it. Asserts on every
    step that the training window's latest date is strictly before the
    realized month, so a slicing bug that leaks the realized return into its
    own weight computation fails loudly instead of silently.
    """
    n_months = returns.shape[0]
    realized_dates = []
    realized_returns = []
    for t in range(window, n_months):
        train_window = returns[t - window:t]
        train_dates = dates[t - window:t]
        assert train_dates.max() < dates[t], (
            f"leakage: training window for {dates[t]} includes a date "
            f">= the realized month ({train_dates.max()})"
        )
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


def block_bootstrap_sharpe_ci(monthly_returns, block_length=12, n_boot=2000, seed=0, alpha=0.05):
    """Percentile confidence interval on the annualized Sharpe ratio.

    Resamples fixed-length, overlapping blocks of the realized monthly
    return series with replacement (the moving block bootstrap), so
    month-to-month autocorrelation within a block is preserved rather than
    destroyed by resampling individual months independently.
    """
    rng = np.random.default_rng(seed)
    n = len(monthly_returns)
    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length
    boot_sharpes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        resampled = np.concatenate(
            [monthly_returns[s:s + block_length] for s in starts]
        )[:n]
        boot_sharpes[b] = annualized_stats(resampled)[2]
    lo, hi = np.percentile(boot_sharpes, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi
