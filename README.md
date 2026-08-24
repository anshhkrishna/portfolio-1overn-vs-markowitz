# portfolio-1overn-vs-markowitz

Status: scaffolded, not yet built.

## Claim under test

Out of sample, the naive 1/N (equal-weight) portfolio matches or beats sample-covariance
mean-variance optimization on Sharpe ratio, across realistic rolling estimation window
lengths, on both a size/book-to-market-sorted universe and an industry universe. The
estimation window a sample-covariance strategy needs before it reliably beats 1/N is
longer than the return history most of these universes actually have.

## Baseline

1/N equal weight, rebalanced every period to the same fixed weights.

## Compared against

- Sample-covariance minimum-variance portfolio.
- Sample mean-variance (tangency) portfolio.
- Shrinkage-covariance minimum-variance portfolio, so a loss for the sample-covariance
  strategy isn't just "estimated badly" — shrinkage is the standard fix.

## Data

`data/ken-french/25_Portfolios_5x5.csv` (25 portfolios formed on size and
book-to-market) and `data/ken-french/49_Industry_Portfolios.csv` (49 industry
portfolios), monthly value-weighted returns. See `data/README.md`.
