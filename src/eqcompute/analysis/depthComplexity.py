"""Analyze one adaptive run using full diagnostic trajectories and existing oracles."""

from collections import Counter
from itertools import combinations

from eqcompute.analysis.complexity import write_complexity
from eqcompute.analysis.diagnostic import (
    _checked,
    _compare_table,
    _csv_rows,
    _finite,
    _save_summary,
    _write_table,
    load_predictions,
    prepare_run,
)
from eqcompute.analysis.oracleDepth import oracle_depths, write_oracles
from eqcompute.evaluation.predict import load_evaluation
from eqcompute.evaluation.predictionWriter import file_hash

COMPLEXITIES = (
    "atom_count", "heavy_atom_count", "radius_gyration", "element_count", "element_entropy",
)
DEPTHS = ("oracle_depth", "cost_oracle_depth", "halt_depth")


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


def analyze_run(
    run_dir, split="validation", device="auto", batch_size=None,
    lambda_oracle=None, overwrite=False,
):
    """Return a JSON-compatible summary; valid caches never trigger inference."""
    run = prepare_run(run_dir, split, batch_size)
    config, metadata = run.config, run.metadata
    checkpoint_path, prediction_path = run.checkpoint_path, run.prediction_path
    provenance, policy = run.provenance, run.policy
    samples, maximum, scale = run.samples, run.maximum, run.scale
    weight = _finite(
        config["task"]["lambda_compute"] if lambda_oracle is None else lambda_oracle
    )
    oracle_depths([0.0], weight)
    destination = run.directory / "analysis/depth_complexity" / split
    trajectories = load_predictions(run, device, batch_size, overwrite)

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
