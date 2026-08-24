Tested whether the naive 1/N portfolio matches or beats sample-covariance mean-variance
optimization out of sample, across three estimation window lengths (60, 120, 240 months)
on two universes: 25 size and book-to-market portfolios and 49 industry portfolios, both
from Ken French's data library. On the 49-industry universe 1/N's Sharpe ratio (0.8029 at
a 60-month window) held up against the optimized strategies, roughly matched by shrinkage
minimum-variance (0.8051) and clearly ahead of sample minimum-variance (0.2473) and
tangency (-0.0391). On the 25-portfolio universe the result reversed: sample
minimum-variance (0.7631) and shrinkage minimum-variance (0.8205) both beat 1/N (0.6270)
at every window length tested, the opposite direction from the industry universe. The
surprising part is that this reversal shows up at the shortest, most
estimation-error-prone window, exactly where the naive prior said 1/N should have its
biggest edge. A block-bootstrap 95% confidence interval on every Sharpe ratio overlaps
substantially between 1/N and every optimized strategy at every window and universe
tested, so none of these point-estimate differences is sharply distinguishable from noise.
