import torch
import torch.nn.functional as F

from eqcompute.contracts import PropertyContext, TaskResult
from eqcompute.models.trace import TraceSpec
from eqcompute.tasks.targetTransform import TargetTransform


class RegressionTask:
    supervision_modes = ("final", "all_steps")

    def __init__(self, config, target_index, unit, transform):
        if config.supervision not in self.supervision_modes:
            raise ValueError(f"supervision must be one of {self.supervision_modes}")
        if config.loss not in ("mae", "mse"):
            raise ValueError("loss must be mae or mse")
        self.config = config
        self.context = PropertyContext(config.target)
        self.target_index = target_index
        self.unit = unit
        self.transform = transform

    def targets(self, batch):
        return batch.target[:, self.target_index : self.target_index + 1]

    def predict(self, model, batch, trace=None):
        return model(batch.graph, self.context, trace=trace)

    def run(self, model, batch, stage):
        all_steps = self.config.supervision == "all_steps"
        output = self.predict(model, batch, TraceSpec(predictions=all_steps))
        target = self.targets(batch)
        normalized = self.transform.encode(target)
        predictions = (
            list(output.predictions_by_depth.values()) if all_steps else [output.prediction]
        )
        criterion = F.l1_loss if self.config.loss == "mae" else F.mse_loss
        loss = torch.stack([criterion(pred, normalized) for pred in predictions]).mean()
        return TaskResult(
            loss,
            self.transform.decode(output.prediction.detach()),
            target.detach(),
            batch.graph.num_graphs,
        )

    def state_dict(self):
        return {
            "target": self.context.name,
            "index": self.target_index,
            "unit": self.unit,
            "transform": self.transform.state_dict(),
        }

    def load_state_dict(self, state):
        if (state["target"], state["index"], state["unit"]) != (
            self.context.name,
            self.target_index,
            self.unit,
        ):
            raise ValueError("Checkpoint property metadata does not match the task")
        self.transform = TargetTransform(**state["transform"])
