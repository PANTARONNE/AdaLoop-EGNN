import math
from numbers import Real

import torch

from eqcompute.contracts import TaskResult
from eqcompute.models.trace import TraceSpec
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
        if config.phase not in ("predictor", "halting"):
            raise ValueError("phase must be predictor or halting")
        beta = config.get("entropy_beta", 0.0)
        if (
            isinstance(beta, bool) or not isinstance(beta, Real)
            or not math.isfinite(beta) or beta < 0
        ):
            raise ValueError("entropy_beta must be a finite nonnegative number")
        self.entropy_beta = float(beta)

    def configure_model(self, model):
        """Apply before constructing the optimizer, including on resume."""
        model.requires_grad_(self.config.phase == "predictor")
        model.execution.controller.requires_grad_(self.config.phase == "halting")

    def _training_mode(self, model):
        if self.config.phase == "halting":
            # Frozen predictor buffers must also remain unchanged.
            for module in (model.initializer, model.operators, model.readout, model.head):
                module.eval()
        else:
            model.execution.controller.eval()

    def predict(self, model, batch, trace=None, *, execution_mode=None):
        return model(batch.graph, self.context, trace=trace, execution_mode=execution_mode)

    def run(self, model, batch, stage):
        if stage == "train":
            self._training_mode(model)
        if self.config.phase == "predictor":
            return self._run_predictor(model, batch, stage)
        output = self.predict(model, batch, execution_mode="full")
        target = self.targets(batch)
        normalized = self.transform.encode(target)
        info = output.adaptive
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
        }
        return TaskResult(
            loss, self.transform.decode(output.prediction.detach()), target.detach(),
            batch.graph.num_graphs, statistics,
        )

    def _run_predictor(self, model, batch, stage):
        target = self.targets(batch)
        normalized = self.transform.encode(target)
        statistics = {"entropy_loss": 0.0}
        if stage == "train":
            output = self.predict(model, batch, execution_mode="random_depth")
            prediction = output.prediction
            error = prediction - normalized
            statistics["executed_depth"] = output.adaptive.executed_depth.float().mean().item()
        else:
            output = self.predict(
                model, batch, TraceSpec(predictions=True), execution_mode="predictor_full",
            )
            values = torch.stack(list(output.predictions_by_depth.values()), 1)
            error = values - normalized[:, None, :]
            physical_error = (self.transform.decode(values) - target[:, None, :]).abs()
            # argmin returns the shallowest depth on ties.
            depths = physical_error.mean(-1).argmin(1)
            prediction = values[torch.arange(len(depths), device=depths.device), depths]
            statistics.update(
                mean_depth_mae=physical_error.mean().item(),
                final_depth_mae=physical_error[:, -1].mean().item(),
                oracle_depth=(depths + 1).float().mean().item(),
                executed_depth=float(model.execution.depth),
            )
        loss = (error.abs() if self.config.loss == "mae" else error.square()).mean()
        statistics["prediction_loss"] = loss.detach().item()
        return TaskResult(
            loss, self.transform.decode(prediction.detach()), target.detach(),
            batch.graph.num_graphs, statistics,
        )

    def state_dict(self):
        return {
            **super().state_dict(), "phase": self.config.phase,
            "entropy_beta": self.entropy_beta,
        }

    def load_state_dict(self, state):
        if state["phase"] != self.config.phase:
            raise ValueError("Checkpoint phase mismatch")
        if state["entropy_beta"] != self.entropy_beta:
            raise ValueError("Checkpoint entropy_beta mismatch")
        super().load_state_dict(state)
