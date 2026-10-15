"""Plot two vertically arranged depth distributions from an existing analysis CSV."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

if __package__:
    from .depthComplexity import SERIES, _ensure_arial, read_table
else:
    from depthComplexity import SERIES, _ensure_arial, read_table

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402


def make_figure(metadata, columns):
    """Two vertically arranged distributions with a shared depth axis."""
    _ensure_arial()
    maximum = int(metadata["max_depth"])
    bins = np.arange(0.5, maximum + 1.5)
    count = len(columns[SERIES[0][0]])
    weights = np.full(count, 1.0 / count)
    with matplotlib.rc_context({
        "font.family": "Arial", "font.size": 14.4,
        "axes.labelsize": 14.4, "xtick.labelsize": 14.4, "ytick.labelsize": 14.4,
    }):
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8.5, 6))
        fig.subplots_adjust(left=0.15, right=0.97, bottom=0.12, top=0.98, hspace=0)
        for ax, (key, label, color, *_) in zip(axes, SERIES, strict=True):
            ax.hist(
                columns[key], bins=bins, weights=weights, density=False,
                histtype="bar", stacked=False, color=color, edgecolor=color,
                linewidth=0.8, alpha=0.55, label=label, rasterized=False,
            )
            counts, _ = np.histogram(columns[key], bins=bins)
            # Integer arithmetic makes exact multiples advance to the next ten percent.
            upper = (10 * int(counts.max()) // count + 1) / 10
            ax.set(
                ylabel="Molecule proportion", xlim=(0.5, maximum + 0.5), ylim=(0, upper),
            )
            ax.set_yticks(np.arange(round(upper * 10) + 1) / 10)
            ax.set_xticks(np.arange(1, maximum + 1))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.20)
            ax.legend(loc="upper right", frameon=False, fontsize=12.96)
        for ax in axes[:-1]:
            ax.get_yticklabels()[0].set_visible(False)
        axes[-1].set_xlabel("Depth")
    return fig, axes


def plot_table(input_path, output_dir):
    metadata, columns = read_table(input_path)
    fig, _ = make_figure(metadata, columns)
    paths = []
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with matplotlib.rc_context({"svg.fonttype": "none", "font.family": "Arial"}):
            for extension in ("svg", "png"):
                path = output_dir / f"depth_distribution.{extension}"
                fig.savefig(path, dpi=300)
                paths.append(path)
    finally:
        plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        paths = plot_table(args.input, args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
