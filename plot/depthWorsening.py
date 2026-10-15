"""Plot adjacent-depth worsening proportions from an analysis CSV without model dependencies."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

if __package__:
    from .depthComplexity import _ensure_arial
else:
    from depthComplexity import _ensure_arial

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

COLOR = "#4370B4"
METADATA = (
    "checkpoint_id", "split_id", "split", "property", "unit", "supervision",
    "run_id", "seed", "weight_tying", "execution_mode",
    "inference_rule", "threshold", "max_depth", "sample_count", "error_metric",
    "error_units", "comparison", "population", "predictions_hash", "split_manifest_hash",
)
COUNTS = ("worse_count", "equal_count", "better_count")


def read_table(path):
    required = {*METADATA, *COUNTS, "depth", "next_depth", "worsening_proportion"}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Analysis CSV is missing required columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty analysis table")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed analysis table")
    if len({tuple(row[key] for key in METADATA) for row in rows}) != 1:
        raise ValueError("Plot one checkpoint, property, split and analysis protocol at a time")
    metadata = {key: rows[0][key] for key in METADATA}
    if any(not value for value in metadata.values()):
        raise ValueError("Missing analysis provenance")
    maximum, count = int(metadata["max_depth"]), int(metadata["sample_count"])
    if maximum < 2 or count < 1 or metadata["split"] not in ("validation", "test"):
        raise ValueError("Invalid maximum depth, sample count or split")
    for key, expected in {
        "execution_mode": "diagnostic", "inference_rule": "cdf",
        "error_metric": "absolute_error", "error_units": "original",
        "comparison": "strict_greater", "population": "all",
    }.items():
        if metadata[key] != expected:
            raise ValueError(f"Unexpected analysis protocol: {key}")
    threshold = float(metadata["threshold"])
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Invalid adaptive policy settings")
    rows.sort(key=lambda row: int(row["depth"]))
    if [int(row["depth"]) for row in rows] != list(range(1, maximum)):
        raise ValueError("Duplicate, missing or invalid depth")
    columns = {
        key: np.array([int(row[key]) for row in rows], dtype=np.int64)
        for key in ("depth", "next_depth", *COUNTS)
    }
    if not np.array_equal(columns["next_depth"], columns["depth"] + 1):
        raise ValueError("Expected consecutive depth pairs")
    for row in rows:
        counts = [int(row[key]) for key in COUNTS]
        if any(value < 0 or value > count for value in counts) or sum(counts) != count:
            raise ValueError("Counts must be nonnegative and sum to the total sample count")
    proportions = np.array([float(row["worsening_proportion"]) for row in rows])
    if (
        not np.isfinite(proportions).all() or (proportions < 0).any() or (proportions > 1).any()
        or not np.allclose(proportions, columns["worse_count"] / count, rtol=1e-12, atol=0)
    ):
        raise ValueError("Worsening proportions must agree with counts / sample_count")
    columns["worsening_proportion"] = proportions
    return metadata, columns


def make_figure(metadata, columns):
    _ensure_arial()
    maximum, count = int(metadata["max_depth"]), int(metadata["sample_count"])
    bins = np.arange(0.5, maximum + 0.5)
    upper = (10 * int(columns["worse_count"].max()) // count + 1) / 10
    with matplotlib.rc_context({
        "font.family": "Arial", "font.size": 14.4,
        "axes.labelsize": 14.4, "xtick.labelsize": 14.4, "ytick.labelsize": 14.4,
    }):
        fig, ax = plt.subplots(figsize=(8.5, 6))
        fig.subplots_adjust(left=0.16, right=0.97, bottom=0.13, top=0.97)
        ax.hist(
            columns["depth"], bins=bins, weights=columns["worsening_proportion"],
            density=False, histtype="bar", color=COLOR, edgecolor=COLOR,
            alpha=0.55, linewidth=0.8, rasterized=False,
        )
        ax.set(
            xlabel="Depth", ylabel="Worsening proportion",
            xlim=(0.5, maximum - 0.5), ylim=(0, upper),
        )
        ax.set_xticks(np.arange(1, maximum))
        ax.set_yticks(np.arange(round(upper * 10) + 1) / 10)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.20)
    return fig, ax


def plot_table(input_path, output_dir):
    metadata, columns = read_table(input_path)
    fig, _ = make_figure(metadata, columns)
    paths = []
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with matplotlib.rc_context({"svg.fonttype": "none", "font.family": "Arial"}):
            for extension in ("svg", "png"):
                path = output_dir / f"depth_worsening.{extension}"
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
    except (OSError, ValueError, OverflowError, csv.Error) as error:
        parser.error(str(error))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
