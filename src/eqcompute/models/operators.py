from torch import nn


class OperatorBank(nn.Module):
    """Parameter ownership only; execution length belongs to the policy."""

    def __init__(self, step_factory, weight_tying: bool, capacity: int):
        super().__init__()
        self.weight_tying = weight_tying
        self.steps = nn.ModuleList([step_factory() for _ in range(1 if weight_tying else capacity)])

    def at(self, depth: int):
        if depth < 1 or (not self.weight_tying and depth > len(self.steps)):
            raise ValueError("Requested depth exceeds the untied operator bank")
        return self.steps[0 if self.weight_tying else depth - 1]
