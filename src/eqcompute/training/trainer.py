import csv
import logging
import shutil
from pathlib import Path

import torch

from eqcompute.training.checkpoint import (
    capture_rng,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
)
from eqcompute.training.optim import build_scheduler

log = logging.getLogger(__name__)


class Trainer:
    """Owns optimization and epochs; knows neither the backbone nor execution depth."""

    def __init__(
        self,
        model,
        task,
        optimizer,
        metric_factory,
        device,
        output_dir,
        config,
        metadata,
        generator,
    ):
        self.model, self.task, self.optimizer = model, task, optimizer
        self.metric_factory = metric_factory
        self.device = device
        self.output_dir = Path(output_dir)
        self.config, self.metadata = config, metadata
        self.scheduler = build_scheduler(config["trainer"].get("scheduler"), optimizer)
        self.generator = generator
        self.start_epoch, self.best_value = 0, float("inf")

    def restore(self, path):
        path = Path(path)
        if path.name != "last.pt":
            raise ValueError("Resume from last.pt; best.pt is for evaluation")
        checkpoint = load_checkpoint(path)
        for key in ("model", "task", "dataset", "optimizer", "loader", "seed"):
            if checkpoint["config"][key] != self.config[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        if (
            checkpoint["config"]["trainer"]["gradient_clip"]
            != self.config["trainer"]["gradient_clip"]
        ):
            raise ValueError("Resume configuration mismatch: gradient_clip")
        if checkpoint["metadata"]["split_id"] != self.metadata["split_id"]:
            raise ValueError("Resume split mismatch")
        if checkpoint["config"]["trainer"].get("scheduler") != self.config["trainer"].get(
            "scheduler"
        ):
            raise ValueError("Resume scheduler mismatch; keep the original scheduler.t_max")
        if (checkpoint.get("scheduler") is None) != (self.scheduler is None):
            raise ValueError("Checkpoint scheduler state is missing or incompatible")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.task.load_state_dict(checkpoint["task"])
        restore_rng(checkpoint["rng"], self.generator)
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_value = checkpoint["best_value"]
        destination = self.output_dir / "checkpoints"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path.with_name("best.pt"), destination / "best.pt")

    def _epoch(self, loader, training):
        self.model.train(training)
        metric = self.metric_factory()
        total_loss, count = 0.0, 0
        with torch.set_grad_enabled(training):
            for batch in loader:
                batch = batch.to(self.device)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                result = self.task.run(self.model, batch, "train" if training else "validation")
                if not torch.isfinite(result.loss):
                    raise FloatingPointError("Non-finite loss")
                if training:
                    result.loss.backward()
                    clip = self.config["trainer"]["gradient_clip"]
                    if clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), clip, error_if_nonfinite=True
                        )
                    self.optimizer.step()
                total_loss += result.loss.item() * result.num_graphs
                count += result.num_graphs
                metric.update(result.prediction, result.target)
        return {"loss": total_loss / count, **metric.compute()}

    def fit(self, train_loader, validation_loader, epochs):
        if epochs <= self.start_epoch:
            raise ValueError("epochs must exceed the completed epoch count")
        if self.scheduler is not None and epochs > self.scheduler.T_max:
            raise ValueError("epochs must not exceed the fixed cosine scheduler.t_max")
        for epoch in range(self.start_epoch, epochs):
            learning_rate = self.optimizer.param_groups[0]["lr"]
            train = self._epoch(train_loader, True)
            validation = self._epoch(validation_loader, False)
            improved = validation["mae"] < self.best_value
            if improved:
                self.best_value = validation["mae"]
            # Exactly once per completed epoch, after optimization and validation.
            # Save the advanced state so resumed training uses the next epoch's LR.
            if self.scheduler is not None:
                self.scheduler.step()
            payload = {
                "format_version": 1,
                "epoch": epoch,
                "best_value": self.best_value,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "task": self.task.state_dict(),
                "rng": capture_rng(self.generator),
                "config": self.config,
                "metadata": self.metadata,
            }
            save_checkpoint(self.output_dir / "checkpoints/last.pt", payload)
            if improved:
                save_checkpoint(self.output_dir / "checkpoints/best.pt", payload)
            row = {
                "epoch": epoch + 1,
                "learning_rate": learning_rate,
                **{f"train_{k}": v for k, v in train.items()},
                **{f"validation_{k}": v for k, v in validation.items()},
            }
            if self.config["logging"]["csv"]:
                path = self.output_dir / "metrics.csv"
                exists = path.exists()
                with path.open("a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    if not exists:
                        writer.writeheader()
                    writer.writerow(row)
            log.info(
                "epoch=%d train_loss=%.6f validation_mae=%.6f %s",
                epoch + 1,
                train["loss"],
                validation["mae"],
                self.task.unit,
            )
