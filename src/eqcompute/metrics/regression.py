import math


class RegressionMetrics:
    """Aggregate sufficient statistics, not unweighted batch averages."""

    def __init__(self):
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.count = 0

    def update(self, prediction, target):
        error = (prediction - target).detach().double()
        self.absolute_error += error.abs().sum().item()
        self.squared_error += error.square().sum().item()
        self.count += error.numel()

    def compute(self):
        if not self.count:
            raise ValueError("Cannot evaluate an empty loader")
        return {
            "mae": self.absolute_error / self.count,
            "rmse": math.sqrt(self.squared_error / self.count),
        }
