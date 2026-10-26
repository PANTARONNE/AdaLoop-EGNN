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


def _validate_temperature(temperature):
    if (
        isinstance(temperature, bool) or not isinstance(temperature, Real)
        or not math.isfinite(temperature) or temperature <= 0
    ):
        raise ValueError("soft_cdf_temperature must be a finite positive number")
    return float(temperature)


def soft_cdf_weights(probabilities, threshold, temperature):
    """Smooth first CDF crossing; input stop masses and output weights are [B,K].

    The CDF must come from a valid stopping distribution. Fixing the endpoint
    crossing values to 0 and 1 preserves unit total mass, including when K=1.
    As temperature tends to zero this approaches hard CDF selection except at
    exact threshold ties, where sigmoid assigns half mass on either side.
    """
    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise ValueError("probabilities must have shape [B, K] with K >= 1")
    if (
        isinstance(threshold, bool) or not isinstance(threshold, Real)
        or not math.isfinite(threshold) or not 0 < threshold < 1
    ):
        raise ValueError("threshold must lie strictly between 0 and 1")
    temperature = _validate_temperature(temperature)
    cdf = probabilities[:, :-1].cumsum(1)
    reached = torch.sigmoid((cdf - threshold) / temperature)
    boundary_shape = (probabilities.shape[0], 1)
    reached = torch.cat([
        probabilities.new_zeros(boundary_shape), reached,
        probabilities.new_ones(boundary_shape),
    ], dim=1)
    return reached[:, 1:] - reached[:, :-1]


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
        # Missing fields describe checkpoints trained with the original objective.
        self.halting_loss = config.get("halting_loss", "expected")
        if self.halting_loss not in ("expected", "soft_cdf"):
            raise ValueError("halting_loss must be expected or soft_cdf")
        self.soft_cdf_temperature = _validate_temperature(
            config.get("soft_cdf_temperature", 0.1)
        )

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
        p = info.stop_probabilities
        expected_prediction_loss = adaptive_loss(values, normalized, p, self.config.loss)
        if self.halting_loss == "soft_cdf":
            weights = soft_cdf_weights(
                p, model.execution.threshold, self.soft_cdf_temperature,
            )
            prediction_loss = adaptive_loss(values, normalized, weights, self.config.loss)
        else:
            prediction_loss = expected_prediction_loss
        entropy = None if p is None else halting_entropy(p)
        entropy_loss = prediction_loss.new_zeros(())
        loss = prediction_loss
        if entropy is not None and self.entropy_beta > 0:
            entropy_loss = -self.entropy_beta * entropy
            loss = prediction_loss + entropy_loss
        statistics = {
            "prediction_loss": prediction_loss.detach().item(),
            "expected_prediction_loss": expected_prediction_loss.detach().item(),
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
            "halting_loss": self.halting_loss,
            "soft_cdf_temperature": self.soft_cdf_temperature,
        }

    def load_state_dict(self, state):
        if state["phase"] != self.config.phase:
            raise ValueError("Checkpoint phase mismatch")
        if state["entropy_beta"] != self.entropy_beta:
            raise ValueError("Checkpoint entropy_beta mismatch")
        if state.get("halting_loss", "expected") != self.halting_loss:
            raise ValueError("Checkpoint halting_loss mismatch")
        if state.get("soft_cdf_temperature", 0.1) != self.soft_cdf_temperature:
            raise ValueError("Checkpoint soft_cdf_temperature mismatch")
        super().load_state_dict(state)
