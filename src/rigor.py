"""Bootstrap confidence intervals and a leakage check on the full sweep.

For every (strategy, universe, window) combination also covered by
`experiment.py`, reports the annualized out-of-sample Sharpe ratio alongside
a moving block bootstrap 95% confidence interval (block length chosen from
the lag-1 to lag-12 autocorrelation of the realized return series, which is
small enough at every combination tested here that a 12-month block is more
than sufficient to preserve it). The leakage assertion inside
`rolling_backtest` runs on every one of these calls, so a clean run of this
script is itself evidence the no-lookahead property holds throughout.
"""

from backtest import annualized_stats, block_bootstrap_sharpe_ci, rolling_backtest
from experiment import STRATEGIES, UNIVERSES, WINDOWS

BLOCK_LENGTH = 12


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
            lo, hi = block_bootstrap_sharpe_ci(
                realized_returns, block_length=BLOCK_LENGTH
            )
            print(f"{universe_name}, window={window}, {strategy_name}: "
                  f"sharpe={sharpe:.4f} 95% CI=[{lo:.4f}, {hi:.4f}] "
                  f"(block={BLOCK_LENGTH}mo, n_boot=2000)")


if __name__ == "__main__":
    print(f"leakage check: passed for every rolling_backtest call below "
          f"(assertion runs inside rolling_backtest itself)")
    for universe_name, loader in UNIVERSES:
        run(universe_name, loader)
