"""Initialize a fresh halting run from a trained predictor, without optimizer state."""

import hashlib
from pathlib import Path

from omegaconf import OmegaConf

from eqcompute.tasks.targetTransform import TargetTransform
from eqcompute.training.checkpoint import load_checkpoint


def initialize_halting(path, model, task, config, metadata):
    path = Path(path).resolve()
    checkpoint = load_checkpoint(path)
    saved = checkpoint["config"]
    current = OmegaConf.to_container(config, resolve=True)
    if task.config.phase != "halting" or saved["task"].get("phase") != "predictor":
        raise ValueError("init_from requires a predictor checkpoint and a halting destination")
    if checkpoint["task"].get("phase") != "predictor":
        raise ValueError("init_from checkpoint task phase mismatch")
    if saved["model"] != current["model"]:
        raise ValueError("init_from model configuration mismatch")
    for key in ("name", "target", "normalization"):
        if saved["task"][key] != current["task"][key]:
            raise ValueError(f"init_from task mismatch: {key}")
    for key in ("split_id", "dataset", "property", "unit", "target_convention"):
        if checkpoint["metadata"][key] != metadata[key]:
            raise ValueError(f"init_from metadata mismatch: {key}")
    state = checkpoint["task"]
    if (state["target"], state["index"], state["unit"]) != (
        task.context.name, task.target_index, task.unit,
    ):
        raise ValueError("init_from property metadata mismatch")
    for name in ("initializer", "operators", "readout", "head"):
        prefix = name + "."
        weights = {
            key[len(prefix):]: value for key, value in checkpoint["model"].items()
            if key.startswith(prefix)
        }
        getattr(model, name).load_state_dict(weights, strict=True)
    task.transform = TargetTransform(**state["transform"])
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"init_from": str(path), "init_from_sha256": digest}
