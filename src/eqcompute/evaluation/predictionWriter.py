import csv
import hashlib
import os
from pathlib import Path

import torch

from eqcompute.models.trace import TraceSpec

FIELDS = [
    "run_id",
    "checkpoint_id",
    "split_id",
    "split",
    "sample_id",
    "property",
    "train_depth",
    "readout_depth",
    "weight_tying",
    "supervision",
    "seed",
    "y_true",
    "y_pred",
    "unit",
]


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def write_predictions(model, task, loader, device, destination, provenance, overwrite=False):
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Predictions exist: {destination}; explicitly enable overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    model.eval()
    seen = set()
    row_count = 0
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for batch in loader:
            if len(set(batch.sample_id)) != len(batch.sample_id) or seen.intersection(
                batch.sample_id
            ):
                raise ValueError("Duplicate sample IDs during prediction export")
            seen.update(batch.sample_id)
            batch = batch.to(device)
            output = task.predict(model, batch, TraceSpec(predictions=True))
            targets = task.targets(batch).cpu().flatten().tolist()
            for depth, normalized in output.predictions_by_depth.items():
                values = task.transform.decode(normalized)
                if not torch.isfinite(values).all():
                    raise FloatingPointError("Non-finite prediction")
                for sample_id, target, value in zip(
                    batch.sample_id, targets, values.cpu().flatten().tolist(), strict=True
                ):
                    writer.writerow(
                        {
                            **provenance,
                            "sample_id": sample_id,
                            "property": task.context.name,
                            "readout_depth": depth,
                            "y_true": target,
                            "y_pred": value,
                            "unit": task.unit,
                        }
                    )
                    row_count += 1
    os.replace(temporary, destination)
    return row_count
