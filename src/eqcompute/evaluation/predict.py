from pathlib import Path

from omegaconf import OmegaConf

from eqcompute.application import make_loader, resolve_device
from eqcompute.evaluation.adaptiveWriter import write_adaptive_predictions
from eqcompute.evaluation.predictionWriter import file_hash, write_predictions
from eqcompute.factories import DATASETS, MODELS, TASKS, construct
from eqcompute.tasks.targetTransform import TargetTransform
from eqcompute.training.checkpoint import load_checkpoint


def load_evaluation(
    checkpoint_path, split="test", device="auto", batch_size=None
):
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    config = OmegaConf.create(checkpoint["config"])
    bundle = construct(DATASETS, config.dataset.name, config.dataset, config.task.target)
    if bundle.manifest["split_id"] != checkpoint["metadata"]["split_id"]:
        raise ValueError("Dataset identities/order or split changed since training")
    device = resolve_device(device)
    model = construct(MODELS, config.model.name, config.model, config.task.target).to(device)
    model.load_state_dict(checkpoint["model"])
    task = construct(
        TASKS, config.task.name, config.task, bundle.target_index, bundle.unit, TargetTransform()
    )
    task.load_state_dict(checkpoint["task"])
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        config.loader.batch_size = batch_size
    loader = make_loader(getattr(bundle, split), config.loader)
    provenance = {
        "run_id": checkpoint["metadata"]["run_id"],
        "checkpoint_id": file_hash(checkpoint_path),
        "split_id": bundle.manifest["split_id"],
        "split": split,
        "train_depth": config.model.execution.depth,
        "weight_tying": config.model.weight_tying,
        "supervision": config.task.supervision,
        "seed": config.seed,
    }
    return checkpoint_path, config, model, task, loader, device, provenance


def predict_checkpoint(
    checkpoint_path, split="test", device="auto", batch_size=None, overwrite=False, *, mode=None
):
    checkpoint_path, config, model, task, loader, device, provenance = load_evaluation(
        checkpoint_path, split, device, batch_size
    )
    if config.model.execution.name == "adaptive":
        mode = mode or "hard"
        if mode not in ("hard", "diagnostic"):
            raise ValueError("Adaptive prediction mode must be hard or diagnostic")
        suffix = "adaptive" if mode == "hard" else "diagnostic_by_depth"
        destination = (
            checkpoint_path.parent.parent / "predictions"
            / f"{split}_{checkpoint_path.stem}_{suffix}.csv"
        )
        count = write_adaptive_predictions(
            model, task, loader, device, destination, provenance, overwrite, mode
        )
        return destination, count
    if mode is not None:
        raise ValueError("Explicit execution mode is only supported for adaptive checkpoints")
    destination = (
        checkpoint_path.parent.parent
        / "predictions"
        / (f"{split}_{checkpoint_path.stem}_by_depth.csv")
    )
    count = write_predictions(model, task, loader, device, destination, provenance, overwrite)
    return destination, count
