"""Plot an analysis CSV without importing PyTorch or loading models/datasets."""

import argparse
import csv
from pathlib import Path

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COMPLEXITIES = {
    "atom_count": ("Atom count", True),
    "heavy_atom_count": ("Heavy atom count", True),
    "radius_gyration": ("Radius of gyration (Å)", False),
    "element_count": ("Element count", True),
    "element_entropy": ("Element entropy (natural log)", False),
}
SERIES = (
    ("oracle_depth", "Error-optimal depth", "#C30078", "o", -0.18),
    ("cost_oracle_depth", "Cost-optimal depth", "#4370B4", "^", 0.0),
    ("halt_depth", "Model halt depth", "#529F9A", "s", 0.18),
)
METADATA = (
    "checkpoint_id", "split_id", "split", "property", "unit", "supervision",
    "normalization", "target_scale", "lambda_oracle", "lambda_compute", "threshold",
    "inference_rule", "execution_mode", "max_depth", "run_id", "seed", "weight_tying",
)


def read_table(path):
    required = {*METADATA, *COMPLEXITIES, *(item[0] for item in SERIES), "sample_id"}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Analysis CSV is missing required columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty analysis table")
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Malformed analysis table")
    if len({tuple(row[key] for key in METADATA) for row in rows}) != 1:
        raise ValueError("Plot one checkpoint, property, split and analysis protocol at a time")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample_id")
    rows.sort(key=lambda row: row["sample_id"])
    metadata = {key: rows[0][key] for key in METADATA}
    maximum = int(metadata["max_depth"])
    if maximum < 1 or metadata["split"] not in ("validation", "test"):
        raise ValueError("Invalid split or maximum depth")
    if metadata["execution_mode"] != "diagnostic" or metadata["inference_rule"] != "cdf":
        raise ValueError("Expected adaptive diagnostic analysis")
    for key in ("target_scale", "lambda_oracle", "lambda_compute", "threshold"):
        value = float(metadata[key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}")
    if float(metadata["target_scale"]) <= 0 or not 0 < float(metadata["threshold"]) <= 1:
        raise ValueError("Invalid target scale or threshold")
    columns = {
        key: np.array([float(row[key]) for row in rows])
        for key in (*COMPLEXITIES, *(item[0] for item in SERIES))
    }
    for key, values in columns.items():
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid numeric values in {key}")
    for key, *_ in SERIES:
        values = columns[key]
        if ((values < 1) | (values > maximum) | (values != np.floor(values))).any():
            raise ValueError(f"Invalid depth in {key}")
    for key, (_, discrete) in COMPLEXITIES.items():
        if discrete and (columns[key] != np.floor(columns[key])).any():
            raise ValueError(f"Non-integer counts in {key}")
    if (
        (columns["atom_count"] < 1).any() or (columns["element_count"] < 1).any()
        or (columns["element_count"] > columns["atom_count"]).any()
        or (columns["heavy_atom_count"] > columns["atom_count"]).any()
    ):
        raise ValueError("Invalid atom/element counts")
    return metadata, columns


def _ensure_arial():
    """Use installed Arial, including Windows fonts exposed to WSL."""
    properties = font_manager.FontProperties(family="Arial")
    try:
        font_manager.findfont(properties, fallback_to_default=False)
        return
    except ValueError:
        for directory in (Path("C:/Windows/Fonts"), Path("/mnt/c/Windows/Fonts")):
            for name in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"):
                path = directory / name
                if path.is_file():
                    font_manager.fontManager.addfont(str(path))
    try:
        font_manager.findfont(properties, fallback_to_default=False)
    except ValueError as error:
        raise ValueError("Arial is required; install Arial before plotting") from error


def make_figure(metadata, columns, metric, jitter_seed=42, jitter=True):
    _ensure_arial()
    with matplotlib.rc_context({"font.family": "Arial"}):
        return _make_figure(metadata, columns, metric, jitter_seed, jitter)


def _make_figure(metadata, columns, metric, jitter_seed=42, jitter=True):
    """Three vectorized scatter collections; input arrays are never mutated."""
    if metric not in COMPLEXITIES:
        raise ValueError(f"Unknown complexity: {metric}")
    if jitter_seed < 0:
        raise ValueError("jitter_seed must be nonnegative")
    label, discrete = COMPLEXITIES[metric]
    count = len(columns[metric])
    rng = np.random.default_rng(jitter_seed)
    x_noise = rng.uniform(-0.18, 0.18, count)
    y_noise = rng.uniform(-0.04, 0.04, count)
    x = columns[metric] + (x_noise if jitter and discrete else 0)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.91)
    for key, name, color, marker, offset in SERIES:
        y = columns[key] + (y_noise + offset if jitter else 0)
        ax.scatter(
            x, y, c=color, marker=marker, label=name, s=10, alpha=0.45,
            linewidths=0, rasterized=False, zorder=2,
        )
    maximum = int(metadata["max_depth"])
    ax.set(xlabel=label, ylabel="Depth", ylim=(0.5, maximum + 0.5))
    ax.set_yticks(np.arange(1, maximum + 1))
    if discrete:
        lower, upper = int(columns[metric].min()), int(columns[metric].max())
        ax.set_xticks(np.arange(lower, upper + 1))
        ax.set_xlim(lower - 0.5, upper + 0.5)
    ax.grid(axis="y", alpha=0.20, zorder=0)
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.015), ncol=3,
        frameon=False, fontsize=9, markerscale=2, borderaxespad=0,
    )
    return fig, ax


def plot_table(input_path, output_dir, complexity="all", jitter_seed=42, jitter=True):
    metadata, columns = read_table(input_path)
    metrics = list(COMPLEXITIES) if complexity == "all" else [complexity]
    if any(metric not in COMPLEXITIES for metric in metrics) or jitter_seed < 0:
        raise ValueError("Unknown complexity or negative jitter seed")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with matplotlib.rc_context({"svg.fonttype": "none", "font.family": "Arial"}):
        for metric in metrics:
            fig, _ = make_figure(metadata, columns, metric, jitter_seed, jitter)
            try:
                for extension in ("svg", "png"):
                    path = output_dir / f"depth_complexity_{metric}.{extension}"
                    fig.savefig(path, dpi=300)
                    paths.append(path)
            finally:
                plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--complexity", choices=["all", *COMPLEXITIES], default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jitter-seed", type=int, default=42)
    parser.add_argument("--no-jitter", action="store_true")
    args = parser.parse_args(argv)
    try:
        paths = plot_table(
            args.input, args.output_dir, args.complexity, args.jitter_seed, not args.no_jitter
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
