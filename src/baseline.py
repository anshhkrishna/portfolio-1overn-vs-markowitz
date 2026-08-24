"""Run the 1/N baseline on both universes at the shortest sweep window.

Prints annualized mean, volatility, and Sharpe ratio for the equal-weight
strategy on the 25-portfolio and 49-industry universes, out of sample, at a
60-month rolling estimation window.
"""

from backtest import annualized_stats, rolling_backtest
from data import load_25_portfolios, load_49_industries
from strategies import equal_weight

WINDOW = 60


def run(name, loader):
    dates, returns = loader()
    realized_dates, realized_returns = rolling_backtest(
        dates, returns, WINDOW, equal_weight
    )
    mean, vol, sharpe = annualized_stats(realized_returns)
    print(f"{name}: {len(realized_dates)} out-of-sample months, "
          f"{realized_dates[0]}-{realized_dates[-1]}, window={WINDOW}")
    print(f"{name}: annualized mean={mean:.4f} vol={vol:.4f} sharpe={sharpe:.4f}")


if __name__ == "__main__":
    run("25 portfolios, 1/N", load_25_portfolios)
    run("49 industries, 1/N", load_49_industries)
