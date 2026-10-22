import math
from numbers import Real

import torch

from eqcompute.contracts import TaskResult
from eqcompute.tasks.regression import RegressionTask


def adaptive_loss(predictions, target, probabilities, loss="mae"):
    """Predictions [B,K,1], target [B,1], probabilities [B,K]."""
    errors = predictions - target[:, None, :]
    errors = errors.abs() if loss == "mae" else errors.square()
    return (probabilities * errors.mean(-1)).sum(1).mean()


def halting_entropy(probabilities):
    """Mean per-graph entropy of stop masses [B,K], in nats (not divided by log K)."""
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1).mean()


class AdaptiveRegressionTask(RegressionTask):
    supervision_modes = ("adaptive",)

    def __init__(self, config, target_index, unit, transform):
        super().__init__(config, target_index, unit, transform)
        if type(config.warmup_epochs) is not int or config.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be a nonnegative integer")
        beta = config.get("entropy_beta", 0.0)
        if (
            isinstance(beta, bool) or not isinstance(beta, Real)
            or not math.isfinite(beta) or beta < 0
        ):
            raise ValueError("entropy_beta must be a finite nonnegative number")
        self.entropy_beta = float(beta)
        self.epoch = 0

    def on_epoch_start(self, model, epoch):
        self.epoch = epoch
        model.execution.warmup = epoch < self.config.warmup_epochs

    def checkpoint_eligible(self, epoch):
        return epoch >= self.config.warmup_epochs

    def predict(self, model, batch, trace=None, *, execution_mode=None):
        return model(batch.graph, self.context, trace=trace, execution_mode=execution_mode)

    def run(self, model, batch, stage):
        warmup = stage == "train" and self.epoch < self.config.warmup_epochs
        output = self.predict(model, batch, execution_mode="warmup" if warmup else "full")
        target = self.targets(batch)
        normalized = self.transform.encode(target)
        info = output.adaptive
        if warmup:
            error = output.prediction - normalized
            prediction_loss = (error.abs() if self.config.loss == "mae" else error.square()).mean()
        else:
            values = torch.stack(list(output.predictions_by_depth.values()), 1)
            prediction_loss = adaptive_loss(
                values, normalized, info.stop_probabilities, self.config.loss,
            )
        p = info.stop_probabilities
        entropy = None if p is None else halting_entropy(p)
        entropy_loss = prediction_loss.new_zeros(())
        loss = prediction_loss
        if entropy is not None and self.entropy_beta > 0:
            entropy_loss = -self.entropy_beta * entropy
            loss = prediction_loss + entropy_loss
        statistics = {
            "prediction_loss": prediction_loss.detach().item(),
            "entropy_loss": entropy_loss.detach().item(),
            "expected_depth": (
                None if info.expected_depth is None else info.expected_depth.detach().mean().item()
            ),
            "halt_depth": info.halt_depth.float().mean().item(),
            "executed_depth": info.executed_depth.float().mean().item(),
            "halt_at_1_fraction": (info.halt_depth == 1).float().mean().item(),
            "halt_at_max_fraction": (
                (info.halt_depth == model.execution.depth).float().mean().item()
            ),
            "halting_entropy": (
                None if entropy is None else entropy.detach().item()
            ),
            "warmup_fraction": float(warmup),
        }
        return TaskResult(
            loss, self.transform.decode(output.prediction.detach()), target.detach(),
            batch.graph.num_graphs, statistics,
        )

    def state_dict(self):
        return {
            **super().state_dict(), "epoch": self.epoch,
            "warmup_epochs": self.config.warmup_epochs,
            "entropy_beta": self.entropy_beta,
        }

    def load_state_dict(self, state):
        if state["warmup_epochs"] != self.config.warmup_epochs:
            raise ValueError("Checkpoint warm-up schedule mismatch")
        if state.get("entropy_beta", 0.0) != self.entropy_beta:
            raise ValueError("Checkpoint entropy_beta mismatch")
        super().load_state_dict(state)
        self.epoch = state["epoch"]
