from torch import nn

from eqcompute.contracts import PropertyContext


class PredictionHead(nn.Module):
    def __init__(self, hidden_dim, property_name):
        super().__init__()
        self.property_name = property_name
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, graph_repr, context: PropertyContext):
        if context.name != self.property_name:
            raise ValueError(f"This checkpoint predicts {self.property_name}, not {context.name}")
        return self.network(graph_repr)
