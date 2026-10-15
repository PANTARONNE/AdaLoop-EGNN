"""Offline label-based oracle. Never called by a model or inference policy."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

from eqcompute.evaluation.predictionWriter import file_hash
from eqcompute.training.checkpoint import load_checkpoint


def oracle_depth(errors) -> int:
    if not errors or any(not math.isfinite(e) or e < 0 for e in errors):
        raise ValueError("errors must be a nonempty finite nonnegative sequence")
    # Python min preserves the first (shallowest) index on exact ties.
    return min(range(len(errors)), key=lambda i: errors[i]) + 1


def write_oracles(source, checkpoint_path, destination, overwrite=False):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Oracle output must not overwrite the prediction source")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    checkpoint = load_checkpoint(checkpoint_path)
    config, task = checkpoint["config"], checkpoint["task"]
    maximum = config["model"]["execution"]["depth"]
    scale = task["transform"]["scale"]
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid checkpoint target scale")
    expected = {
        "checkpoint_id": file_hash(checkpoint_path), "split_id": checkpoint["metadata"]["split_id"],
        "property": task["target"], "unit": task["unit"],
        "train_depth": str(maximum), "supervision": config["task"]["supervision"],
    }
    fields = [
        "checkpoint_id", "split_id", "split", "sample_id", "property", "unit",
        "supervision", "normalization", "target_scale", "max_depth",
        "oracle_depth", "min_error",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    pending, completed, split = {}, set(), None
    try:
        with source.open(newline="", encoding="utf-8") as handle, temporary.open(
            "w", newline="", encoding="utf-8"
        ) as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            reader = csv.DictReader(handle)
            if not (set(expected) | {"readout_depth", "sample_id", "split", "y_true", "y_pred"}
                    ).issubset(reader.fieldnames or []):
                raise ValueError("Oracle requires a complete by-depth prediction CSV")
            for row in reader:
                if any(row[key] != value for key, value in expected.items()):
                    raise ValueError("Prediction provenance does not match checkpoint")
                split = split or row["split"]
                if row["split"] != split or split not in ("train", "validation", "test"):
                    raise ValueError("Mixed or invalid prediction split")
                sample, depth = row["sample_id"], int(row["readout_depth"])
                target, prediction = float(row["y_true"]), float(row["y_pred"])
                if not math.isfinite(target) or not math.isfinite(prediction):
                    raise ValueError("Non-finite label or prediction")
                if sample in completed or not 1 <= depth <= maximum:
                    raise ValueError("Duplicate sample trajectory or invalid depth")
                record = pending.setdefault(sample, {"target": target, "errors": {}})
                if record["target"] != target or depth in record["errors"]:
                    raise ValueError("Inconsistent label or duplicate depth")
                record["errors"][depth] = abs(prediction - target) / scale
                if len(record["errors"]) == maximum:
                    errors = [record["errors"][k] for k in range(1, maximum + 1)]
                    best = oracle_depth(errors)
                    writer.writerow({
                        **{k: expected[k] for k in ("checkpoint_id", "split_id", "property", "unit",
                                                    "supervision")},
                        "split": split, "sample_id": sample,
                        "normalization": config["task"]["normalization"], "target_scale": scale,
                        "max_depth": maximum,
                        "oracle_depth": best, "min_error": errors[best - 1],
                    })
                    completed.add(sample)
                    del pending[sample]
            if pending or not completed:
                raise ValueError("Missing depths or empty predictions")
            manifest_path = Path(checkpoint_path).parent.parent / "split_ids.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest["split_id"] != expected["split_id"] or (
                    set(manifest["sample_ids"][split]) != completed
                ):
                    raise ValueError("Prediction samples do not cover the saved split exactly")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return len(completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    count = write_oracles(
        args.source, args.checkpoint, args.output, args.overwrite
    )
    print(f"Wrote {count} oracle rows: {args.output}")


if __name__ == "__main__":
    main()
