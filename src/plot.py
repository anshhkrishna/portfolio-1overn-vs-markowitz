"""Draw the headline chart from the committed bootstrap sweep log.

Reads the per-(universe, window, strategy) Sharpe ratios and their 95%
block-bootstrap confidence intervals from `results/rigor.log` and plots Sharpe
against estimation window length for all four strategies, one subplot per
universe. Nothing is hand-placed: every point and error bar on the chart is
parsed from that log, so rerunning this script on the same log reproduces the
identical image.

Run with `python src/plot.py`.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RIGOR_LOG = PROJECT_ROOT / "results" / "rigor.log"
OUTPUT = PROJECT_ROOT / "results" / "headline.png"

ROW = re.compile(
    r"^(25 portfolios|49 industries), window=(\d+), ([\w /-]+?): "
    r"sharpe=(-?[\d.]+) 95% CI=\[(-?[\d.]+), (-?[\d.]+)\]"
)

STRATEGY_ORDER = ["1/N", "sample min-variance", "tangency", "shrinkage min-variance"]
COLORS = {
    "1/N": "#555555",
    "sample min-variance": "#c0392b",
    "tangency": "#8e6bbf",
    "shrinkage min-variance": "#2c6fbb",
}
MARKERS = {
    "1/N": "o",
    "sample min-variance": "s",
    "tangency": "^",
    "shrinkage min-variance": "D",
}


def parse_rigor_log(path=RIGOR_LOG):
    """Group Sharpe ratio and CI bounds by universe, then by strategy, then by window."""
    data = {"25 portfolios": {}, "49 industries": {}}
    for line in path.read_text().splitlines():
        match = ROW.match(line)
        if match is None:
            continue
        universe, window, strategy, sharpe, lo, hi = match.groups()
        bucket = data[universe].setdefault(strategy, {"windows": [], "sharpe": [], "lo": [], "hi": []})
        bucket["windows"].append(int(window))
        bucket["sharpe"].append(float(sharpe))
        bucket["lo"].append(float(lo))
        bucket["hi"].append(float(hi))

    if not data["25 portfolios"] or not data["49 industries"]:
        raise ValueError(f"no rows found for both universes in {path}")

    return data


def draw_universe(ax, universe_data, title):
    for strategy in STRATEGY_ORDER:
        series = universe_data[strategy]
        windows = series["windows"]
        sharpe = series["sharpe"]
        lo_err = [max(s - l, 0.0) for s, l in zip(sharpe, series["lo"])]
        hi_err = [max(h - s, 0.0) for h, s in zip(series["hi"], sharpe)]
        ax.errorbar(
            windows,
            sharpe,
            yerr=[lo_err, hi_err],
            color=COLORS[strategy],
            marker=MARKERS[strategy],
            markersize=6,
            linewidth=2,
            capsize=4,
            label=strategy,
        )

    ax.set_xticks([60, 120, 240])
    ax.set_xlabel("estimation window (months)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    ax.grid(False, axis="x")


def make_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(1600 / 150, 900 / 150), dpi=150, sharey=True)
    fig.patch.set_facecolor("white")

    draw_universe(axes[0], data["25 portfolios"], "25 size/book-to-market portfolios")
    draw_universe(axes[1], data["49 industries"], "49 industry portfolios")

    axes[0].set_ylabel("out-of-sample sharpe ratio (95% bootstrap ci)", fontsize=12)
    for ax in axes:
        ax.set_facecolor("white")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=11, loc="lower center", ncol=4, framealpha=1.0)
    fig.suptitle(
        "1/N holds on the industry universe, loses to min-variance on size/book-to-market",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    return fig


def main():
    data = parse_rigor_log()
    fig = make_figure(data)
    fig.savefig(OUTPUT, facecolor="white", dpi=150)
    print(f"wrote {OUTPUT.relative_to(PROJECT_ROOT)}")
    for universe, strategies in data.items():
        print(f"{universe}: {len(strategies)} strategies, windows {sorted(set(strategies['1/N']['windows']))}")


if __name__ == "__main__":
    main()
