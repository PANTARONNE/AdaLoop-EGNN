"""Analyze one adaptive run using full diagnostic trajectories and existing oracles."""

import csv
import json
import math
import os
from collections import Counter
from itertools import combinations
from pathlib import Path

from eqcompute.analysis.complexity import write_complexity
from eqcompute.analysis.oracleDepth import oracle_depths, write_oracles
from eqcompute.evaluation.adaptiveWriter import DIAGNOSTIC_FIELDS
from eqcompute.evaluation.predict import load_evaluation, predict_checkpoint
from eqcompute.evaluation.predictionWriter import file_hash
from eqcompute.training.checkpoint import load_checkpoint

COMPLEXITIES = (
    "atom_count", "heavy_atom_count", "radius_gyration", "element_count", "element_entropy",
)
DEPTHS = ("oracle_depth", "cost_oracle_depth", "halt_depth")


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


def _complexities(path, split_id, split, samples):
    result = {}
    for row in _csv_rows(path, ("split_id", "split", "sample_id", *COMPLEXITIES)):
        if row["split_id"] != split_id or row["split"] != split:
            raise ValueError("Complexity split provenance mismatch")
        sample = row["sample_id"]
        if sample in result:
            raise ValueError("Duplicate complexity sample")
        values = {key: _finite(row[key]) for key in COMPLEXITIES}
        if any(value < 0 for value in values.values()):
            raise ValueError("Negative complexity")
        for key in ("atom_count", "heavy_atom_count", "element_count"):
            if not values[key].is_integer():
                raise ValueError("Non-integer atom/element count")
            values[key] = int(values[key])
        if not (
            1 <= values["element_count"] <= values["atom_count"]
            and values["heavy_atom_count"] <= values["atom_count"]
        ):
            raise ValueError("Invalid atom/element counts")
        result[sample] = values
    if set(result) != samples:
        raise ValueError("Complexity samples do not cover the saved split exactly")
    return result


def _compare_table(path, expected):
    fields = list(expected[0])
    actual = list(_csv_rows(path, fields))
    actual.sort(key=lambda row: row["sample_id"])
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


def analyze_run(
    run_dir, split="validation", device="auto", batch_size=None,
    lambda_oracle=None, overwrite=False,
):
    """Return a JSON-compatible summary; valid caches never trigger inference."""
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
        raise ValueError("Depth/complexity analysis requires an adaptive checkpoint")
    maximum = execution["depth"]
    if type(maximum) is not int or maximum < 1:
        raise ValueError("Invalid maximum depth")
    scale = _finite(task["transform"]["scale"])
    if scale <= 0:
        raise ValueError("Invalid target scale")
    weight = _finite(
        config["task"]["lambda_compute"] if lambda_oracle is None else lambda_oracle
    )
    oracle_depths([0.0], weight)
    threshold = _finite(execution["threshold"])
    training_weight = _finite(config["task"]["lambda_compute"])
    if not 0 < threshold <= 1 or training_weight < 0:
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
        "execution_mode": "diagnostic", "lambda_compute": training_weight,
        "inference_rule": "cdf", "threshold": threshold,
    }
    expected = {**provenance, **policy, "train_depth": maximum}
    destination = run_dir / "analysis/depth_complexity" / split
    prediction_path = run_dir / "predictions" / f"{split}_best_diagnostic_by_depth.csv"
    if overwrite or not prediction_path.exists():
        predict_checkpoint(
            checkpoint_path, split, device, batch_size, overwrite=overwrite, mode="diagnostic"
        )
    trajectories = _checked(
        prediction_path, lambda: _predictions(prediction_path, expected, samples, maximum, scale)
    )

    complexity_path = destination / "complexity.csv"
    if overwrite or not complexity_path.exists():
        _, _, _, _, loader, _, loaded = load_evaluation(
            checkpoint_path, split, "cpu", batch_size
        )
        write_complexity(
            loader, complexity_path, {key: loaded[key] for key in ("split_id", "split")},
            overwrite=overwrite,
        )
    complexities = _checked(
        complexity_path,
        lambda: _complexities(complexity_path, str(metadata["split_id"]), split, samples),
    )
    oracle_rows = []
    for sample in sorted(samples):
        record = trajectories[sample]
        errors = [record["errors"][k] for k in range(1, maximum + 1)]
        best, cost = oracle_depths(errors, weight)
        oracle_rows.append({
            **provenance, "sample_id": sample, "normalization": config["task"]["normalization"],
            "target_scale": scale, "lambda_oracle": weight, "max_depth": maximum,
            "oracle_depth": best, "cost_oracle_depth": cost,
            "min_error": errors[best - 1], "cost_oracle_error": errors[cost - 1],
        })
    oracle_path = destination / "oracle.csv"
    if overwrite or not oracle_path.exists():
        write_oracles(prediction_path, checkpoint_path, oracle_path, weight, overwrite)
    _checked(oracle_path, lambda: _compare_table(oracle_path, oracle_rows))

    merged = [
        {**row, **policy, "halt_depth": trajectories[row["sample_id"]]["halt"],
         **complexities[row["sample_id"]]}
        for row in oracle_rows
    ]
    merged_path = destination / "depth_complexity.csv"
    if merged_path.exists() and not overwrite:
        _checked(merged_path, lambda: _compare_table(merged_path, merged))
    else:
        _write_table(merged_path, merged)
    distributions = {}
    for name in DEPTHS:
        counts = Counter(row[name] for row in merged)
        distributions[name] = {
            "distinct_depths": len(counts),
            "counts": {str(k): counts[k] for k in range(1, maximum + 1)},
            "proportions": {str(k): counts[k] / len(merged) for k in range(1, maximum + 1)},
        }
    summary = {
        **provenance, **policy, "normalization": config["task"]["normalization"],
        "target_scale": scale, "lambda_oracle": weight, "max_depth": maximum,
        "sample_count": len(merged), "depths": distributions,
        "agreement": {
            f"{left}__{right}": sum(row[left] == row[right] for row in merged) / len(merged)
            for left, right in combinations(DEPTHS, 2)
        },
        "input_hashes": {
            "predictions": file_hash(prediction_path), "complexity": file_hash(complexity_path),
        },
    }
    _save_summary(destination / "summary.json", summary, overwrite)
    return summary
