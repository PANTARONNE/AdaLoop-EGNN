"""Evaluate completed depth/seed jobs and summarize paired validation/test MAEs."""

import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from eqcompute.evaluation.predict import predict_checkpoint
from eqcompute.evaluation.predictionWriter import file_hash
from eqcompute.training.checkpoint import load_checkpoint


def _run_directory(sweep_dir, job):
    # Prefer the batch's own layout, allowing a sweep directory to be moved/copied.
    local = sweep_dir / "runs" / f"k{job['depth']}_s{job['seed']}"
    candidates = sorted(path for path in local.glob("*") if path.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Multiple run directories under {local}")
    if job.get("run_dir"):
        return (sweep_dir / job["run_dir"]).resolve()
    raise ValueError("No run directory recorded or found in this sweep")


def _read_run(sweep_dir, job):
    run_dir = _run_directory(sweep_dir, job)
    checkpoint_path = run_dir / "checkpoints/best.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    config, metadata = checkpoint["config"], checkpoint["metadata"]
    depth, seed = config["model"]["execution"]["depth"], config["seed"]
    tying = config["model"]["weight_tying"]
    if (depth, seed, tying) != (job["depth"], job["seed"], job["weight_tying"]):
        raise ValueError("best.pt depth/seed/weight_tying does not match jobs.json")
    val_mae = float(checkpoint["best_value"])
    if not math.isfinite(val_mae) or val_mae < 0:
        raise ValueError("best.pt has an invalid validation MAE")
    manifest = json.loads((run_dir / "split_ids.json").read_text(encoding="utf-8"))
    if manifest["split_id"] != metadata["split_id"]:
        raise ValueError("Saved sample split does not match best.pt")
    sample_ids = manifest["sample_ids"]["test"]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Test sample IDs must be nonempty and unique")
    expected = {
        "run_id": str(metadata["run_id"]),
        "checkpoint_id": file_hash(checkpoint_path),
        "split_id": str(metadata["split_id"]),
        "split": "test",
        "property": str(config["task"]["target"]),
        "unit": str(metadata["unit"]),
        "train_depth": str(depth),
        "weight_tying": str(tying),
        "supervision": str(config["task"]["supervision"]),
        "seed": str(seed),
    }
    signature = (
        expected["property"], expected["unit"], tying, expected["supervision"],
        expected["split_id"], config["model"]["name"],
    )
    return {
        "depth": depth, "seed": seed, "val_mae": val_mae,
        "checkpoint": checkpoint_path, "expected": expected,
        "sample_ids": set(sample_ids), "signature": signature,
    }


def _test_mae(path, run):
    """Validate provenance and complete final-depth coverage before reusing a CSV."""
    expected, depth = run["expected"], run["depth"]
    errors = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = set(expected) | {"sample_id", "readout_depth", "y_true", "y_pred"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Prediction CSV is missing required columns")
        for row in reader:
            if any(row[key] != value for key, value in expected.items()):
                raise ValueError("Prediction provenance does not match best.pt")
            readout_depth = int(row["readout_depth"])
            if not 1 <= readout_depth <= depth:
                raise ValueError("Prediction readout depth is outside the trained depth")
            if readout_depth != depth:
                continue
            sample_id = row["sample_id"]
            if sample_id in errors:
                raise ValueError("Duplicate final-depth prediction")
            target, prediction = float(row["y_true"]), float(row["y_pred"])
            error = abs(prediction - target)
            if not all(math.isfinite(value) for value in (target, prediction, error)):
                raise ValueError("Non-finite final-depth prediction or target")
            errors[sample_id] = error
    if set(errors) != run["sample_ids"]:
        raise ValueError("Prediction CSV does not cover exactly the saved test split")
    return statistics.mean(errors.values())


def _evaluate_test(run, device, batch_size):
    path = run["checkpoint"].parent.parent / "predictions/test_best_by_depth.csv"
    try:
        return _test_mae(path, run)
    except (OSError, ValueError, TypeError, csv.Error) as error:
        print(
            f"[evaluate] depth={run['depth']} seed={run['seed']}: {error}",
            file=sys.stderr, flush=True,
        )
    # Existing stale/incomplete files are replaced using the prediction writer's atomic save.
    predict_checkpoint(run["checkpoint"], "test", device, batch_size, overwrite=path.exists())
    return _test_mae(path, run)


def _mean_sd(values, precision):
    if not values:
        return "N/A"
    mean = f"{statistics.mean(values):.{precision}f}"
    sd = f"{statistics.stdev(values):.{precision}f}" if len(values) > 1 else "N/A"
    return f"{mean} ± {sd}"


def summarize_sweep(sweep_dir, device="auto", batch_size=None, precision=6):
    """Return the terminal table and skipped-job count; progress goes to stderr."""
    sweep_dir = Path(sweep_dir).resolve()
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0 <= precision <= 12:
        raise ValueError("precision must be between 0 and 12")
    jobs = json.loads((sweep_dir / "jobs.json").read_text(encoding="utf-8"))
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs.json must contain a nonempty list of sweep jobs")
    keys = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Each sweep job must be an object")
        if type(job.get("depth")) is not int or job["depth"] < 1:
            raise ValueError("Each sweep job must have a positive integer depth")
        if type(job.get("seed")) is not int or job["seed"] < 0:
            raise ValueError("Each sweep job must have a nonnegative integer seed")
        if type(job.get("weight_tying")) is not bool:
            raise ValueError("Each sweep job must have boolean weight_tying")
        key = (job["depth"], job["seed"])
        if key in keys:
            raise ValueError(f"Duplicate depth/seed entry: {key}")
        keys.add(key)
    if len({job["weight_tying"] for job in jobs}) != 1:
        raise ValueError("Cannot combine different weight_tying settings in one table")

    planned = Counter(job["depth"] for job in jobs)
    runs, skipped = [], 0

    def skip(job, reason):
        nonlocal skipped
        skipped += 1
        print(f"[skip] depth={job['depth']} seed={job['seed']}: {reason}", file=sys.stderr)

    for job in jobs:
        if job.get("status") != "succeeded" or job.get("exit_code") != 0:
            skip(job, f"status={job.get('status')}, exit_code={job.get('exit_code')}")
            continue
        try:
            runs.append(_read_run(sweep_dir, job))
        except Exception as error:
            skip(job, error)
    signatures = {run["signature"] for run in runs}
    if len(signatures) > 1:
        raise ValueError("Cannot combine different targets, units, splits, models or supervision")

    results = defaultdict(list)
    for run in runs:
        try:
            test_mae = _evaluate_test(run, device, batch_size)
            results[run["depth"]].append((run["val_mae"], test_mae))
        except Exception as error:
            skip(run, error)

    rows = [["Depth", "Seeds", "Val MAE", "Test MAE"]]
    for depth in sorted(planned):
        pairs = results[depth]
        rows.append([
            str(depth), f"{len(pairs)}/{planned[depth]}",
            _mean_sd([pair[0] for pair in pairs], precision),
            _mean_sd([pair[1] for pair in pairs], precision),
        ])
    widths = [max(len(row[column]) for row in rows) for column in range(4)]
    lines = ["  ".join(value.ljust(width) for value, width in zip(row, widths)) for row in rows]
    lines.insert(1, "  ".join("-" * width for width in widths))
    if signatures:
        target, unit, tying, supervision, _, model = next(iter(signatures))
        title = (
            f"Target: {target} | Unit: {unit} | Model: {model} | "
            f"weight_tying: {str(tying).lower()} | supervision: {supervision}"
        )
    else:
        title = "No readable completed checkpoints."
    note = "Mean ± sample SD (ddof=1); Seeds = evaluated/planned; checkpoint = best.pt."
    return "\n".join([title, note, *lines]), skipped
