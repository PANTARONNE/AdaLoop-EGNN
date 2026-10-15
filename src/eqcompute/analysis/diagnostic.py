"""Shared run provenance, complete diagnostic trajectories, and analysis cache helpers."""

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from eqcompute.evaluation.adaptiveWriter import DIAGNOSTIC_FIELDS
from eqcompute.evaluation.predict import predict_checkpoint
from eqcompute.evaluation.predictionWriter import file_hash
from eqcompute.training.checkpoint import load_checkpoint


def _csv_rows(path, required):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing required columns")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: malformed CSV row")
            yield row


def _finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite numeric value")
    return number


def _checked(path, function):
    try:
        return function()
    except (ValueError, KeyError, TypeError, csv.Error) as error:
        raise ValueError(f"{path}: {error}; regenerate with --overwrite") from error


def _predictions(path, expected, samples, maximum, scale):
    trajectories = {}
    for row in _csv_rows(path, DIAGNOSTIC_FIELDS):
        for key, value in expected.items():
            matches = (
                _finite(row[key]) == value if isinstance(value, float)
                else row[key] == str(value)
            )
            if not matches:
                raise ValueError(f"Prediction provenance mismatch: {key}")
        sample = row["sample_id"]
        depth, halt = int(row["readout_depth"]), int(row["halt_depth"])
        if sample not in samples or not 1 <= depth <= maximum or not 1 <= halt <= maximum:
            raise ValueError("Unexpected sample or invalid depth")
        if row["valid_depth"] != "True" or int(row["executed_depth"]) != maximum:
            raise ValueError("Expected complete diagnostic execution")
        target, prediction = _finite(row["y_true"]), _finite(row["y_pred"])
        for key in ("q", "p", "survival", "expected_depth"):
            _finite(row[key])
        record = trajectories.setdefault(sample, {"target": target, "halt": halt, "errors": {}})
        if record["target"] != target or record["halt"] != halt:
            raise ValueError("Inconsistent label or halt_depth within a molecule")
        if depth in record["errors"]:
            raise ValueError("Duplicate sample/depth row")
        record["errors"][depth] = _finite(abs(prediction - target) / scale)
    if set(trajectories) != samples:
        raise ValueError("Predictions do not cover the saved split exactly")
    for record in trajectories.values():
        if set(record["errors"]) != set(range(1, maximum + 1)):
            raise ValueError("Missing depths")
    return trajectories


def _compare_table(path, expected, key="sample_id"):
    fields = list(expected[0])
    actual = list(_csv_rows(path, fields))
    actual.sort(key=lambda row: row[key])
    expected = sorted(expected, key=lambda row: str(row[key]))
    wanted = [{key: str(value) for key, value in row.items()} for row in expected]
    if len(actual) != len(wanted) or any(
        {key: row[key] for key in fields} != target
        for row, target in zip(actual, wanted, strict=True)
    ):
        raise ValueError("Cached table differs from current inputs or analysis settings")


def _write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_summary(path, summary, overwrite):
    if path.exists() and not overwrite:
        def compare():
            if json.loads(path.read_text(encoding="utf-8")) != summary:
                raise ValueError("Cached summary differs from current inputs or settings")
        _checked(path, compare)
        return
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class DiagnosticRun:
    directory: Path
    checkpoint_path: Path
    config: dict
    metadata: dict
    provenance: dict
    policy: dict
    samples: set[str]
    maximum: int
    scale: float
    prediction_path: Path


def prepare_run(run_dir, split="validation", batch_size=None, *, minimum_depth=1):
    """Validate a saved adaptive run before loading data or running predictions."""
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    run_dir = Path(run_dir).resolve()
    checkpoint_path = run_dir / "checkpoints/best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing {checkpoint_path}; no fallback to last.pt")
    checkpoint = load_checkpoint(checkpoint_path)
    config, task, metadata = checkpoint["config"], checkpoint["task"], checkpoint["metadata"]
    execution = config["model"]["execution"]
    if execution.get("name") != "adaptive":
        raise ValueError("Diagnostic analysis requires an adaptive checkpoint")
    maximum = execution["depth"]
    if type(maximum) is not int or maximum < minimum_depth:
        raise ValueError(f"Maximum depth must be an integer >= {minimum_depth}")
    scale = _finite(task["transform"]["scale"])
    if scale <= 0:
        raise ValueError("Invalid target scale")
    threshold = _finite(execution["threshold"])
    if not 0 < threshold <= 1:
        raise ValueError("Invalid adaptive policy settings")
    if execution.get("inference_rule", "cdf") != "cdf":
        raise ValueError("Only the CDF inference rule is supported")
    manifest = json.loads((run_dir / "split_ids.json").read_text(encoding="utf-8"))
    ids = manifest["sample_ids"][split]
    samples = set(ids)
    if manifest["split_id"] != metadata["split_id"]:
        raise ValueError("Saved split_id does not match checkpoint")
    if not samples or len(samples) != len(ids) or any(not isinstance(i, str) for i in ids):
        raise ValueError("Saved sample IDs must be nonempty, unique strings")
    provenance = {
        "checkpoint_id": file_hash(checkpoint_path), "split_id": metadata["split_id"],
        "split": split, "property": task["target"], "unit": task["unit"],
        "supervision": config["task"]["supervision"],
    }
    policy = {
        "run_id": metadata["run_id"], "seed": config["seed"],
        "weight_tying": config["model"]["weight_tying"],
        "execution_mode": "diagnostic",
        "inference_rule": "cdf", "threshold": threshold,
    }
    return DiagnosticRun(
        run_dir, checkpoint_path, config, metadata, provenance, policy, samples, maximum, scale,
        run_dir / "predictions" / f"{split}_best_diagnostic_by_depth.csv",
    )


def load_predictions(run, device="auto", batch_size=None, overwrite=False, *, normalize=True):
    """Reuse verified full trajectories; optionally retain original error units."""
    if overwrite or not run.prediction_path.exists():
        predict_checkpoint(
            run.checkpoint_path, run.provenance["split"], device, batch_size,
            overwrite=overwrite, mode="diagnostic",
        )
    expected = {**run.provenance, **run.policy, "train_depth": run.maximum}
    return _checked(
        run.prediction_path,
        lambda: _predictions(
            run.prediction_path, expected, run.samples, run.maximum,
            run.scale if normalize else 1.0,
        ),
    )
