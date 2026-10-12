"""Graph-invariant stopping statistics; no labels or raw coordinates as inputs."""

import torch
from torch import nn

from eqcompute.models.geometry import aggregate


class HaltingController(nn.Module):
    def __init__(self, hidden_dim, controller_dim=64, use_delta=True):
        super().__init__()
        if type(controller_dim) is not int or controller_dim < 1:
            raise ValueError("controller hidden_dim must be a positive integer")
        if type(use_delta) is not bool:
            raise ValueError("use_delta must be boolean")
        self.use_delta = use_delta
        self.network = nn.Sequential(
            nn.Linear(2 * hidden_dim + int(use_delta), controller_dim),
            nn.SiLU(), nn.Linear(controller_dim, 1),
        )

    def forward(self, current, previous, graph):
        index, count = graph.batch_index, graph.num_graphs
        mean = aggregate(current, index, count, "mean")
        variance = aggregate((current - mean[index]).square(), index, count, "mean")
        std = (variance + 1e-8).sqrt()
        parts = [mean, std]
        if self.use_delta:
            changes = torch.linalg.vector_norm(current - previous, dim=-1, keepdim=True)
            parts.append(aggregate(changes, index, count, "mean"))
        return self.network(torch.cat(parts, dim=-1)).squeeze(-1).sigmoid()


def stopping_distribution(hazards):
    """B x K hazards -> B x K masses; ignore the final learned hazard."""
    if hazards.ndim != 2 or hazards.shape[1] < 1:
        raise ValueError("hazards must have shape [B, K] with K >= 1")
    survival = hazards.new_ones(hazards.shape[0])
    masses = []
    for q in hazards[:, :-1].unbind(1):
        masses.append(survival * q)
        survival = survival * (1 - q)
    return torch.stack([*masses, survival], dim=1)


def cdf_depth(probabilities, threshold):
    reached = probabilities.cumsum(1) >= threshold
    # Enforce a last-step fallback even with cumulative floating-point roundoff.
    reached[:, -1] = True
    return reached.to(torch.int64).argmax(1) + 1
