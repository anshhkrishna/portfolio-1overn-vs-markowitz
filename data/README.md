# data

Two files from Ken French's data library, used as the two test universes in DeMiguel,
Garlappi & Uppal's original comparison of naive and optimized diversification.

| file | contents |
|---|---|
| `ken-french/25_Portfolios_5x5.csv` | monthly returns, 25 portfolios formed on size and book-to-market |
| `ken-french/49_Industry_Portfolios.csv` | monthly returns, 49 industry portfolios |

Both are published free for research use by Kenneth R. French, downloaded unmodified
from [mba.tuck.dartmouth.edu/pages/faculty/ken.french](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
`MANIFEST.tsv` records each file's size, sha-256 checksum, source URL, and retrieval
date.

Each file opens with a prose header block, then one or more sections ("Average Value
Weighted Returns -- Monthly", "Average Equal Weighted Returns -- Monthly", annual
versions, firm counts, and so on) separated by blank lines. Only the first, monthly,
value-weighted section is used here. Returns are in percent, dates are `YYYYMM`, and
`-99.99` / `-999` mark a portfolio with no firms in it yet that month.
