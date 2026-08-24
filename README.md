# portfolio-1overn-vs-markowitz

> out-of-sample 1/n vs sample and shrinkage-covariance mean-variance portfolios on ken french equity returns, numpy and scikit-learn

## what this tests

a rolling out-of-sample backtest compares four portfolio weight rules on monthly returns
from two ken french universes: 25 portfolios sorted on size and book-to-market, and 49
industry portfolios. `src/data.py` parses both csvs' "average value weighted returns,
monthly" section, converts percent to decimal returns, and restricts each file to its
earliest contiguous run of fully populated months, since several industries have no
firms classified into them in the file's earliest decades. at each rebalance month, a
strategy's weights are computed from a trailing window of returns strictly before that
month (`src/strategies.py`, `src/backtest.py`), then held fixed while that month's
already-realized return is applied and recorded, so no strategy ever sees the return it
is scored against.

**claim:** the naive `1/N` (equal-weight) rule matches or beats sample-covariance
mean-variance optimization on out-of-sample sharpe ratio, across window lengths
realistic for a monthly return history, on both universes. **baseline:** `1/N` itself is
what the other three rules are measured against: sample minimum-variance, sample
tangency (mean-variance), and shrinkage minimum-variance, the last using
`sklearn.covariance.LedoitWolf` so a loss for the sample-covariance rule cannot be
waved away as bad estimation alone.

the sweep covers three window lengths, 60, 120, and 240 months (`src/experiment.py`),
and every out-of-sample sharpe ratio carries a block-bootstrap 95% confidence interval
(12-month blocks, 2000 resamples, `src/backtest.py`) alongside a leakage assertion that
fires if any window's weights are ever computed on or after the month they are scored
against.

## result: `1/N` holds on the industry universe, loses to min-variance on size and book-to-market

- **sample min-variance** beats `1/N` on the 25-portfolio universe at every window
  tested (sharpe 0.7631 vs 0.6270 at window=60, growing to 1.1093 vs 0.7690 at
  window=240), but loses to it on the 49-industry universe at the shortest window
  (0.2473 vs 0.8029 at window=60) (`results/rigor.log` lines 2, 3, 10, 11, 14 and 15).
- **shrinkage min-variance** also beats `1/N` on the 25-portfolio universe at every
  window (0.8205 vs 0.6270 at window=60), but on the 49-industry universe it roughly
  matches `1/N` instead of losing to it (0.8051 vs 0.8029 at window=60, 0.9249 vs 0.8092
  at window=120) (`results/rigor.log` lines 2, 5, 14, 17, 18 and 21).
- **tangency** is the clear loser everywhere: it never beats `1/N` on either universe,
  and its sharpe ratio goes negative on the 49-industry universe at the shortest window
  (-0.0391) (`results/rigor.log` line 16).
- the reversal on the 25-portfolio universe already shows up at the shortest, most
  estimation-error-prone window, not only at the longer windows where more data should
  help the optimized rules.
- every one of these gaps sits inside overlapping 95% bootstrap confidence intervals: at
  window=60 on the 25-portfolio universe, `1/N`'s ci is [0.4471, 0.8966] and sample
  min-variance's is [0.5228, 1.0906] (`results/rigor.log` lines 2 and 3), so the
  point-estimate differences above are not sharply distinguishable from noise.
- a leakage assertion inside `rolling_backtest` confirmed no strategy's weights were
  ever computed from a window overlapping the month they were scored against, across
  every universe, window, and strategy in the sweep (`results/rigor.log` line 1).

full reasoning, including a candidate explanation for why the two universes disagree, is
in `results/FINDING.md`.

## reproducing

first, the pins are exact, so start clean.

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

second, run `1/N` alone at the shortest window on both universes. its output is what
`results/baseline.log` holds, verbatim.

```
python src/baseline.py
```

third, run the full sweep, every strategy across both universes and every window
length. its output is `results/run.log`.

```
python src/experiment.py
```

fourth, add the bootstrap confidence intervals and the leakage check. its output is
`results/rigor.log`, where every number quoted above comes from.

```
python src/rigor.py
```

fifth, redraw the headline chart from that log, and confirm the tests still pass.

```
python src/plot.py
pytest tests/
```

the whole sweep finishes in well under a minute on a laptop cpu. `results/FINDING.md` is
the short prose version of the same result.
