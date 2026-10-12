from dataclasses import dataclass, field, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class GraphInputs:
    atomic_numbers: Tensor
    positions: Tensor
    batch_index: Tensor
    num_graphs: int
    # Optional supplied graph; the radius-graph EGNN builds its own topology.
    edge_index: Tensor | None = None
    edge_attr: Tensor | None = None

    def to(self, device: torch.device | str) -> "GraphInputs":
        return replace(
            self,
            **{
                key: value.to(device)
                for key, value in vars(self).items()
                if isinstance(value, Tensor)
            },
        )


@dataclass(frozen=True)
class SupervisedBatch:
    graph: GraphInputs
    target: Tensor
    sample_id: tuple[str, ...]

    def to(self, device: torch.device | str) -> "SupervisedBatch":
        return replace(self, graph=self.graph.to(device), target=self.target.to(device))


@dataclass(frozen=True)
class PropertyContext:
    name: str


@dataclass
class TaskResult:
    loss: Tensor
    prediction: Tensor  # Original physical units, detached.
    target: Tensor
    num_graphs: int
    # Per-graph means; None denotes an inapplicable statistic (e.g. warm-up entropy).
    statistics: dict[str, float | None] = field(default_factory=dict)
