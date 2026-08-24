"""Run the full rolling-window out-of-sample comparison.

For each universe (25 size/book-to-market portfolios, 49 industry
portfolios) and each estimation window length, runs all four strategies
(1/N, sample minimum-variance, sample tangency, shrinkage minimum-variance)
through the rolling backtest and prints annualized out-of-sample mean,
volatility, and Sharpe ratio for every (strategy, universe, window)
combination.
"""

from backtest import annualized_stats, rolling_backtest
from data import load_25_portfolios, load_49_industries
from strategies import equal_weight, min_variance, shrinkage_min_variance, tangency

WINDOWS = (60, 120, 240)

STRATEGIES = (
    ("1/N", equal_weight),
    ("sample min-variance", min_variance),
    ("tangency", tangency),
    ("shrinkage min-variance", shrinkage_min_variance),
)

UNIVERSES = (
    ("25 portfolios", load_25_portfolios),
    ("49 industries", load_49_industries),
)


def run(universe_name, loader):
    dates, returns = loader()
    for window in WINDOWS:
        if window >= len(dates):
            print(f"{universe_name}, window={window}: skipped, "
                  f"only {len(dates)} months of usable history")
            continue
        for strategy_name, weight_fn in STRATEGIES:
            realized_dates, realized_returns = rolling_backtest(
                dates, returns, window, weight_fn
            )
            mean, vol, sharpe = annualized_stats(realized_returns)
            print(f"{universe_name}, window={window}, {strategy_name}: "
                  f"{len(realized_dates)} out-of-sample months, "
                  f"{realized_dates[0]}-{realized_dates[-1]}, "
                  f"mean={mean:.4f} vol={vol:.4f} sharpe={sharpe:.4f}")


if __name__ == "__main__":
    for universe_name, loader in UNIVERSES:
        run(universe_name, loader)
