from pathlib import Path

from omegaconf import OmegaConf

from eqcompute.application import make_loader, resolve_device
from eqcompute.evaluation.predictionWriter import file_hash, write_predictions
from eqcompute.factories import DATASETS, MODELS, TASKS, construct
from eqcompute.tasks.targetTransform import TargetTransform
from eqcompute.training.checkpoint import load_checkpoint


def predict_checkpoint(
    checkpoint_path, split="test", device="auto", batch_size=None, overwrite=False
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
    destination = (
        checkpoint_path.parent.parent
        / "predictions"
        / (f"{split}_{checkpoint_path.stem}_by_depth.csv")
    )
    count = write_predictions(model, task, loader, device, destination, provenance, overwrite)
    return destination, count
