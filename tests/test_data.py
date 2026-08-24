import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import load_25_portfolios, load_49_industries, _next_yyyymm


@pytest.mark.parametrize(
    "loader,n_columns",
    [(load_25_portfolios, 25), (load_49_industries, 49)],
)
def test_shape(loader, n_columns):
    dates, returns = loader()
    assert dates.ndim == 1
    assert returns.shape == (dates.shape[0], n_columns)
    assert dates.shape[0] > 0


@pytest.mark.parametrize("loader", [load_25_portfolios, load_49_industries])
def test_no_missing_value_codes(loader):
    _, returns = loader()
    percent = returns * 100.0
    assert not np.any(np.isclose(percent, -99.99))
    assert not np.any(np.isclose(percent, -999.0))


@pytest.mark.parametrize("loader", [load_25_portfolios, load_49_industries])
def test_dates_strictly_increasing_no_gaps(loader):
    dates, _ = loader()
    for i in range(len(dates) - 1):
        assert dates[i] < dates[i + 1]
        assert _next_yyyymm(dates[i]) == dates[i + 1]


def test_next_yyyymm_year_rollover():
    assert _next_yyyymm(202612) == 202701
    assert _next_yyyymm(192607) == 192608


def test_25_portfolios_spot_check_matches_raw_file():
    # 192607, "SMALL LoBM" column, first row of the monthly value-weighted
    # section: raw file reads "5.8276" (percent).
    dates, returns = load_25_portfolios()
    assert dates[0] == 192607
    assert returns[0, 0] == pytest.approx(0.058276, abs=1e-4)


def test_49_industries_spot_check_matches_raw_file():
    # Earliest fully-populated month is 196907; raw file reads "-8.33"
    # (percent) for the "Agric" column on that date.
    dates, returns = load_49_industries()
    assert dates[0] == 196907
    assert returns[0, 0] == pytest.approx(-0.0833, abs=1e-4)


def test_49_industries_earliest_run_excludes_earlier_missing_months():
    dates, _ = load_49_industries()
    assert dates[0] > 192607
