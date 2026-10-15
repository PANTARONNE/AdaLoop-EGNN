"""Separate early-exit exports from full, label-free diagnostic trajectories."""

import csv
import os
from pathlib import Path

import torch

from eqcompute.evaluation.predictionWriter import FIELDS

POLICY_FIELDS = ["execution_mode", "inference_rule", "threshold"]
SUMMARY_FIELDS = [
    key for key in FIELDS if key != "readout_depth"
] + POLICY_FIELDS + ["halt_depth", "executed_depth"]
DIAGNOSTIC_FIELDS = FIELDS + POLICY_FIELDS + [
    "q", "p", "survival", "expected_depth", "halt_depth", "executed_depth", "valid_depth",
]


@torch.inference_mode()
def write_adaptive_predictions(
    model, task, loader, device, destination, provenance, overwrite=False, mode="hard"
):
    if mode not in ("hard", "diagnostic"):
        raise ValueError("Export mode must be hard or diagnostic")
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Predictions exist: {destination}; explicitly enable overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    model.eval()
    seen, count = set(), 0
    policy = {
        "execution_mode": mode,
        "inference_rule": "cdf", "threshold": model.execution.threshold,
    }
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            fields = SUMMARY_FIELDS if mode == "hard" else DIAGNOSTIC_FIELDS
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for batch in loader:
                if len(set(batch.sample_id)) != len(batch.sample_id) or seen.intersection(
                    batch.sample_id
                ):
                    raise ValueError("Duplicate sample IDs during prediction export")
                seen.update(batch.sample_id)
                batch = batch.to(device)
                output = task.predict(model, batch, execution_mode=mode)
                info = output.adaptive
                targets = task.targets(batch).flatten().cpu().tolist()
                depths = info.halt_depth.cpu().tolist()
                executed = info.executed_depth.cpu().tolist()
                predictions = (
                    {None: output.prediction} if mode == "hard" else output.predictions_by_depth
                )
                if mode == "diagnostic":
                    q, p = info.halting_probabilities.cpu(), info.stop_probabilities.cpu()
                    survival = 1 - torch.cat([p.new_zeros(p.shape[0], 1), p[:, :-1].cumsum(1)], 1)
                    expected = info.expected_depth.cpu().tolist()
                for depth, normalized in predictions.items():
                    values = task.transform.decode(normalized).flatten().cpu()
                    if not torch.isfinite(values).all():
                        raise FloatingPointError("Non-finite adaptive prediction")
                    for index, (sample_id, target, value) in enumerate(zip(
                        batch.sample_id, targets, values.tolist(), strict=True
                    )):
                        row = {
                            **provenance, **policy, "sample_id": sample_id,
                            "property": task.context.name, "unit": task.unit,
                            "y_true": target, "y_pred": value,
                            "halt_depth": depths[index], "executed_depth": executed[index],
                        }
                        if mode == "diagnostic":
                            row.update(
                                readout_depth=depth, q=q[index, depth - 1].item(),
                                p=p[index, depth - 1].item(),
                                survival=survival[index, depth - 1].item(),
                                expected_depth=expected[index], valid_depth=True,
                            )
                        writer.writerow(row)
                        count += 1
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return count
