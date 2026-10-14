"""Count per-molecule absolute-error worsening between adjacent diagnostic depths."""

import numpy as np

from eqcompute.analysis.diagnostic import (
    _checked,
    _compare_table,
    _save_summary,
    _write_table,
    load_predictions,
    prepare_run,
)
from eqcompute.evaluation.predictionWriter import file_hash


def transition_statistics(errors):
    """Compare all molecules at every transition, using strict comparisons without tolerance."""
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[0] == 0 or errors.shape[1] < 2:
        raise ValueError("Expected a nonempty molecule-by-depth matrix with at least two depths")
    if not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("Absolute errors must be finite and nonnegative")
    current, following = errors[:, :-1], errors[:, 1:]
    worse = np.count_nonzero(following > current, axis=0)
    equal = np.count_nonzero(following == current, axis=0)
    better = np.count_nonzero(following < current, axis=0)
    count = errors.shape[0]
    return [
        {
            "depth": k + 1, "next_depth": k + 2, "sample_count": count,
            "worse_count": int(worse[k]), "equal_count": int(equal[k]),
            "better_count": int(better[k]), "worsening_proportion": float(worse[k] / count),
        }
        for k in range(errors.shape[1] - 1)
    ]


def analyze_run(run_dir, split="validation", device="auto", batch_size=None, overwrite=False):
    """Load or export diagnostic predictions and save verified adjacent-depth statistics."""
    run = prepare_run(run_dir, split, batch_size, minimum_depth=2)
    trajectories = load_predictions(run, device, batch_size, overwrite, normalize=False)
    errors = np.array([
        [trajectories[sample]["errors"][k] for k in range(1, run.maximum + 1)]
        for sample in sorted(run.samples)
    ])
    transitions = transition_statistics(errors)
    input_hashes = {
        "predictions": file_hash(run.prediction_path),
        "split_manifest": file_hash(run.directory / "split_ids.json"),
    }
    metadata = {
        **run.provenance, **run.policy, "max_depth": run.maximum,
        "error_metric": "absolute_error", "error_units": "original",
        "comparison": "strict_greater", "population": "all",
        "predictions_hash": input_hashes["predictions"],
        "split_manifest_hash": input_hashes["split_manifest"],
    }
    rows = [{**metadata, **transition} for transition in transitions]
    destination = run.directory / "analysis/depth_worsening" / split
    table_path = destination / "depth_worsening.csv"
    if table_path.exists() and not overwrite:
        _checked(table_path, lambda: _compare_table(table_path, rows, key="depth"))
    else:
        _write_table(table_path, rows)
    summary = {
        **metadata, "sample_count": len(run.samples), "transitions": transitions,
        "input_hashes": input_hashes,
    }
    _save_summary(destination / "summary.json", summary, overwrite)
    return summary
