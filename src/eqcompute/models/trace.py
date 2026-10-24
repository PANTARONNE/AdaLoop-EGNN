from dataclasses import dataclass, field

from torch import Tensor


@dataclass(frozen=True)
class TraceSpec:
    predictions: bool = False
    node_features: bool = False
    graph_features: bool = False
    positions: bool = False
    depths: tuple[int, ...] | None = None

    def selects(self, depth):
        return self.depths is None or depth in self.depths

    def validate(self, max_depth):
        if self.depths is not None and any(k < 1 or k > max_depth for k in self.depths):
            raise ValueError("Trace depths must lie in [1, execution depth]")


@dataclass
class AdaptiveInfo:
    mode: str
    halt_depth: Tensor
    executed_depth: Tensor
    valid_depth_mask: Tensor
    # Full distributions are unavailable after real early exit or during random-depth training.
    halting_probabilities: Tensor | None = None
    stop_probabilities: Tensor | None = None
    expected_depth: Tensor | None = None


@dataclass
class ModelOutput:
    prediction: Tensor
    # These stay differentiable when used by the Task for all-step supervision.
    predictions_by_depth: dict[int, Tensor] = field(default_factory=dict)
    # Analysis snapshots are detached copies on CPU, never used for the loss.
    trace: dict[int, dict[str, Tensor]] = field(default_factory=dict)
    adaptive: AdaptiveInfo | None = None


def snapshot(tensor):
    return tensor.detach().to("cpu", copy=True)
