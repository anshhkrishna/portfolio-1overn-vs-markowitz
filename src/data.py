"""Loaders for the Ken French portfolio return CSVs used by this project.

Each source file opens with a prose header, then several sections (monthly and
annual, value- and equal-weighted) separated by blank lines. Only the first
section, "Average Value Weighted Returns -- Monthly", is loaded. Values in that
section are percent returns with two sentinel codes, -99.99 and -999, marking a
portfolio that had no firms in it yet that month.
"""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ken-french"
PATH_25_PORTFOLIOS = DATA_DIR / "25_Portfolios_5x5.csv"
PATH_49_INDUSTRIES = DATA_DIR / "49_Industry_Portfolios.csv"

SECTION_HEADER = "Average Value Weighted Returns -- Monthly"
MISSING_CODES = (-99.99, -999.0)


def _next_yyyymm(date):
    year, month = divmod(date, 100)
    if month == 12:
        return (year + 1) * 100 + 1
    return date + 1


def _read_section(path, section_header=SECTION_HEADER):
    with open(path) as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if section_header in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"section {section_header!r} not found in {path}")

    columns = [c.strip() for c in lines[header_idx + 1].strip().split(",")[1:]]

    dates = []
    rows = []
    for line in lines[header_idx + 2:]:
        if not line.strip():
            break
        parts = line.strip().split(",")
        dates.append(int(parts[0]))
        rows.append([float(x) for x in parts[1:]])

    dates = np.array(dates, dtype=np.int64)
    values = np.array(rows, dtype=np.float64)
    return dates, values, columns


def _restrict_to_earliest_valid_run(dates, values):
    missing = np.zeros(values.shape[0], dtype=bool)
    for code in MISSING_CODES:
        missing |= np.any(np.isclose(values, code), axis=1)
    valid = ~missing

    start = np.argmax(valid)
    if not valid[start]:
        raise ValueError("no fully populated month found")

    end = start
    while end + 1 < len(valid) and valid[end + 1]:
        end += 1

    return dates[start:end + 1], values[start:end + 1]


def load_portfolio_returns(path):
    """Load one Ken French monthly value-weighted returns file.

    Returns (dates, returns): dates is a strictly increasing array of YYYYMM
    integers with no gaps, returns is a same-length array of decimal (not
    percent) returns with no missing-value codes remaining, restricted to the
    earliest contiguous run of months where every column has data.
    """
    dates, values, _ = _read_section(path)
    dates, values = _restrict_to_earliest_valid_run(dates, values)
    return dates, values / 100.0


def load_25_portfolios(path=PATH_25_PORTFOLIOS):
    return load_portfolio_returns(path)


def load_49_industries(path=PATH_49_INDUSTRIES):
    return load_portfolio_returns(path)
