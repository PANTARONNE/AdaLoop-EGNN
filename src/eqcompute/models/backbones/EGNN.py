from dataclasses import dataclass

import torch
from torch import Tensor, nn

from eqcompute.contracts import GraphInputs
from eqcompute.models.geometry import aggregate, radius_edges


@dataclass(frozen=True)
class EGNNState:
    node_features: Tensor
    positions: Tensor
    edge_index: Tensor


class EGNNInitializer(nn.Module):
    def __init__(self, hidden_dim: int, cutoff: float):
        super().__init__()
        self.embedding = nn.Embedding(119, hidden_dim)
        self.cutoff = cutoff

    def initialize(self, graph: GraphInputs) -> EGNNState:
        return EGNNState(
            self.embedding(graph.atomic_numbers), graph.positions, radius_edges(graph, self.cutoff)
        )


class EGNNStep(nn.Module):
    """One residual EGNN update with bounded, normalized coordinate displacements."""

    def __init__(self, hidden_dim, num_rbf, cutoff, coordinate_scale=0.1):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0, cutoff, num_rbf))
        self.rbf_width = cutoff / (num_rbf - 1)
        self.coordinate_scale = coordinate_scale
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coordinate = nn.Sequential(nn.Linear(hidden_dim, 1, bias=False), nn.Tanh())
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, state: EGNNState, graph: GraphInputs) -> EGNNState:
        h, x = state.node_features, state.positions
        src, dst = state.edge_index
        relative = x[src] - x[dst]
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        radial = torch.exp(-0.5 * ((distance - self.centers) / self.rbf_width).square())
        messages = self.message(torch.cat((h[src], h[dst], radial), dim=-1))
        pooled = aggregate(messages, dst, h.size(0), "mean")
        displacement = relative / (1 + distance) * self.coordinate(messages)
        next_x = x + self.coordinate_scale * aggregate(displacement, dst, x.size(0), "mean")
        next_h = self.norm(h + self.update(torch.cat((h, pooled), dim=-1)))
        return EGNNState(next_h, next_x, state.edge_index)
