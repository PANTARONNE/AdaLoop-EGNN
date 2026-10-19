"""QM9 EGNN from vgsatorras/egnn, adapted to packed graphs.

Reference: e9ca6c0c3e1d30a7598efbd66034121b4af8dccc, qm9/models.py and models/gcl.py.
See third_party/EGNN-LICENSE.txt. Padding and masked edges are omitted; aggregated
messages use LayerNorm before the node update. Coordinates remain fixed.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from eqcompute.models.geometry import aggregate


@dataclass(frozen=True)
class EGNNState:
    node_features: Tensor
    positions: Tensor
    edge_index: Tensor


class EGNNInitializer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.embedding = nn.Linear(15, hidden_dim)
        self.register_buffer("species", torch.tensor([1, 6, 7, 8, 9]), persistent=False)

    def initialize(self, graph):
        one_hot = graph.atomic_numbers[:, None] == self.species
        if not one_hot.any(dim=1).all():
            raise ValueError("EGNN supports only QM9 elements H, C, N, O, F")
        charges = graph.atomic_numbers.to(self.embedding.weight.dtype) / 9
        powers = charges[:, None].pow(torch.arange(3, device=charges.device))
        features = (one_hot[:, :, None] * powers[:, None, :]).flatten(1)
        edges = []
        for index in range(graph.num_graphs):
            nodes = torch.where(graph.batch_index == index)[0]
            row = nodes.repeat_interleave(nodes.numel())
            col = nodes.repeat(nodes.numel())
            keep = row != col
            edges.append(torch.stack((row[keep], col[keep])))
        return EGNNState(
            self.embedding(features), graph.positions, torch.cat(edges, dim=1)
        )


class EGNNStep(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.message_norm = nn.LayerNorm(hidden_dim)

    def forward(self, state, graph):
        h, x = state.node_features, state.positions
        row, col = state.edge_index
        radial = (x[row] - x[col]).square().sum(dim=1, keepdim=True)
        messages = self.edge_mlp(torch.cat((h[row], h[col], radial), dim=1))
        messages = messages * self.att_mlp(messages)
        pooled = aggregate(messages, row, h.size(0), "sum")
        pooled = self.message_norm(pooled)
        next_h = h + self.node_mlp(torch.cat((h, pooled), dim=1))
        return EGNNState(next_h, x, state.edge_index)


class EGNNReadout(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.node_dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, state, graph):
        return aggregate(
            self.node_dec(state.node_features), graph.batch_index, graph.num_graphs, "sum"
        )
