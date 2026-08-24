import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest import annualized_stats, rolling_backtest
from data import load_25_portfolios, load_49_industries
from strategies import equal_weight, min_variance, shrinkage_min_variance


def test_no_leakage_on_real_data():
    # rolling_backtest asserts internally that every training window's
    # latest date is strictly before the realized month; a clean run over
    # both universes and a few window lengths is a direct check that no
    # rebalance step ever saw its own realized return.
    for loader in (load_25_portfolios, load_49_industries):
        dates, returns = loader()
        for window in (60, 120):
            rolling_backtest(dates, returns, window, equal_weight)


def test_leakage_assertion_fires_on_a_duplicated_boundary_date():
    # Construct a case that genuinely leaks: duplicate the date at the
    # window/realized-month boundary, so the training window's last date
    # equals rather than precedes the realized month. If the assertion in
    # rolling_backtest were a no-op, this would run to completion instead
    # of raising on the very first rebalance step.
    rng = np.random.default_rng(0)
    n_months, n_assets, window = 20, 3, 5
    returns = rng.normal(scale=0.02, size=(n_months, n_assets))
    dates = np.arange(200001, 200001 + n_months)
    dates[window - 1] = dates[window]
    with pytest.raises(AssertionError):
        rolling_backtest(dates, returns, window=window, weight_fn=equal_weight)


def test_core_claim_direction_at_shortest_window():
    # The full sweep found that the claim under test ("1/N matches or beats
    # sample-covariance MVO on both universes") holds on one universe and
    # not the other. This checks both observed directions at the shortest,
    # most estimation-error-prone window (60 months) directly from live
    # computation, rather than asserting a single "1/N wins" outcome.
    dates_25, returns_25 = load_25_portfolios()
    _, r_eq_25 = rolling_backtest(dates_25, returns_25, 60, equal_weight)
    _, r_mv_25 = rolling_backtest(dates_25, returns_25, 60, min_variance)
    _, r_shrink_25 = rolling_backtest(dates_25, returns_25, 60, shrinkage_min_variance)
    sharpe_eq_25 = annualized_stats(r_eq_25)[2]
    sharpe_mv_25 = annualized_stats(r_mv_25)[2]
    sharpe_shrink_25 = annualized_stats(r_shrink_25)[2]

    # 25 portfolios: both covariance-optimized strategies beat 1/N.
    assert sharpe_mv_25 > sharpe_eq_25
    assert sharpe_shrink_25 > sharpe_eq_25

    dates_49, returns_49 = load_49_industries()
    _, r_eq_49 = rolling_backtest(dates_49, returns_49, 60, equal_weight)
    _, r_mv_49 = rolling_backtest(dates_49, returns_49, 60, min_variance)
    _, r_shrink_49 = rolling_backtest(dates_49, returns_49, 60, shrinkage_min_variance)
    sharpe_eq_49 = annualized_stats(r_eq_49)[2]
    sharpe_mv_49 = annualized_stats(r_mv_49)[2]
    sharpe_shrink_49 = annualized_stats(r_shrink_49)[2]

    # 49 industries: 1/N clearly beats sample minimum-variance, and
    # shrinkage minimum-variance only roughly matches 1/N rather than
    # clearly beating it.
    assert sharpe_eq_49 > sharpe_mv_49
    assert abs(sharpe_eq_49 - sharpe_shrink_49) < 0.05
