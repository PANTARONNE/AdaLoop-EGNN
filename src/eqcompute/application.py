import hashlib
import importlib.metadata
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from eqcompute.data.collate import collate_molecules
from eqcompute.factories import DATASETS, MODELS, TASKS, construct
from eqcompute.metrics.regression import RegressionMetrics
from eqcompute.tasks.targetTransform import TargetTransform
from eqcompute.training.checkpoint import seed_everything
from eqcompute.training.initialization import initialize_halting
from eqcompute.training.optim import build_optimizer
from eqcompute.training.trainer import Trainer


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def make_loader(dataset, config, shuffle=False, generator=None):
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_molecules,
        generator=generator,
    )


def configure_logging(output_dir, level):
    logger = logging.getLogger("eqcompute")
    logger.disabled = False
    logger.setLevel(level)
    # Hydra's disabled logging config may disable loggers created during import.
    for name, child in logging.root.manager.loggerDict.items():
        if name.startswith("eqcompute.") and isinstance(child, logging.Logger):
            child.disabled = False
            child.setLevel(logging.NOTSET)
            child.propagate = True
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / "logs/train.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def train(config):
    seed_everything(config.seed)
    device = resolve_device(config.trainer.device)
    if config.trainer.epochs < 1 or config.loader.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if (
        config.trainer.gradient_clip is not None and config.trainer.gradient_clip <= 0
    ) or config.loader.num_workers < 0:
        raise ValueError("Invalid gradient_clip or num_workers")
    adaptive = config.model.execution.name == "adaptive"
    if adaptive != (config.task.name == "adaptive_regression"):
        raise ValueError("Adaptive execution and adaptive_regression must be used together")
    if config.init_from and config.resume:
        raise ValueError("init_from and resume are mutually exclusive")
    if config.init_from and (not adaptive or config.task.phase != "halting"):
        raise ValueError("init_from is only supported for the halting phase")
    if adaptive and config.task.phase == "halting" and not (config.init_from or config.resume):
        raise ValueError("The halting phase requires init_from or resume")
    # Construct and validate model before downloading data.
    model = construct(MODELS, config.model.name, config.model, config.task.target).to(device)
    config.dataset.root = str(Path(config.dataset.root).resolve())
    bundle = construct(DATASETS, config.dataset.name, config.dataset, config.task.target)
    transform = TargetTransform.fit(bundle.training_targets(), config.task.normalization)
    task = construct(
        TASKS, config.task.name, config.task, bundle.target_index, bundle.unit, transform
    )
    if adaptive:
        task.configure_model(model)
    optimizer = build_optimizer(
        config.optimizer, (p for p in model.parameters() if p.requires_grad),
    )
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        name: make_loader(
            getattr(bundle, name), config.loader, shuffle=name == "train", generator=generator
        )
        for name in ("train", "validation")
    }
    name = str(config.experiment_name)
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("experiment_name must be a directory name, not a path")
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    output_dir = Path(config.output_root).resolve() / f"{name}__{suffix}"
    output_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("checkpoints", "predictions", "logs"):
        (output_dir / directory).mkdir()
    configure_logging(output_dir, config.logging.level)
    resolved = OmegaConf.to_container(config, resolve=True)
    metadata = {
        "run_id": output_dir.name,
        "split_id": bundle.manifest["split_id"],
        "dataset": bundle.manifest.get("dataset", config.dataset.name),
        "property": config.task.target,
        "unit": bundle.unit,
        "target_convention": bundle.manifest.get("target_convention", "raw"),
        "device": str(device),
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "torch-geometric", "hydra-core", "numpy")
        },
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "resume_from": config.resume,
    }
    if config.init_from:
        metadata.update(initialize_halting(config.init_from, model, task, config, metadata))
    source_digest = hashlib.sha256()
    source_root = Path(__file__).parent
    for source in sorted(source_root.rglob("*.py")):
        source_digest.update(str(source.relative_to(source_root)).encode())
        source_digest.update(source.read_bytes())
    metadata["source_sha256"] = source_digest.hexdigest()
    OmegaConf.save(OmegaConf.create(resolved), output_dir / "config.yaml")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / "split_ids.json").write_text(json.dumps(bundle.manifest, indent=2))
    trainer = Trainer(
        model, task, optimizer, RegressionMetrics, device, output_dir, resolved, metadata, generator
    )
    if config.resume:
        trainer.restore(config.resume)
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logging.getLogger(__name__).info(
        "output=%s parameters=%d device=%s", output_dir, metadata["parameter_count"], device
    )
    trainer.fit(loaders["train"], loaders["validation"], config.trainer.epochs)
    return output_dir
