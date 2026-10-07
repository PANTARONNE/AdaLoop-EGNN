"""Offline curves only. No oracle depth or controller is implemented here."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_depth_curves(prediction_path, output_path, sample_ids=()):
    groups, individuals = defaultdict(list), defaultdict(dict)
    signatures, seen = set(), set()
    with Path(prediction_path).open() as handle:
        for row in csv.DictReader(handle):
            signatures.add(
                tuple(
                    row[key]
                    for key in (
                        "checkpoint_id",
                        "split_id",
                        "split",
                        "property",
                        "unit",
                        "supervision",
                    )
                )
            )
            key = (row["sample_id"], int(row["readout_depth"]))
            if key in seen:
                raise ValueError("Duplicate sample/depth row")
            seen.add(key)
            error = abs(float(row["y_pred"]) - float(row["y_true"]))
            groups[key[1]].append(error)
            if key[0] in sample_ids:
                individuals[key[0]][key[1]] = error
    if len(signatures) != 1:
        raise ValueError("Plot one checkpoint, split, property and supervision protocol at a time")
    sample_sets = {k: {sample_id for sample_id, depth in seen if depth == k} for k in groups}
    if any(ids != next(iter(sample_sets.values())) for ids in sample_sets.values()):
        raise ValueError("Every depth must contain the same samples")
    signature = next(iter(signatures))
    if any(sample_id not in individuals for sample_id in sample_ids):
        raise ValueError("Requested sample ID is absent")
    depths = sorted(groups)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(depths, [sum(groups[k]) / len(groups[k]) for k in depths], "o-", label="MAE")
    for sample_id, errors in individuals.items():
        ax.plot(sorted(errors), [errors[k] for k in sorted(errors)], alpha=0.65, label=sample_id)
    ax.set(
        xlabel="Readout depth",
        ylabel=f"Absolute error ({signature[4]})",
        title=f"{signature[3]} | supervision={signature[5]}",
    )
    ax.set_xticks(depths)
    ax.legend()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    args = parser.parse_args()
    plot_depth_curves(args.predictions, args.output, args.sample_id)


if __name__ == "__main__":
    main()
