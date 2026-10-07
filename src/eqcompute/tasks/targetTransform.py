import torch


class TargetTransform:
    def __init__(self, mean=0.0, scale=1.0):
        self.mean = float(mean)
        self.scale = float(scale)
        if self.scale <= 0:
            raise ValueError("Target scale must be positive")

    @classmethod
    def fit(cls, train_targets, mode="standard"):
        if not torch.isfinite(train_targets).all():
            raise ValueError("Training labels contain non-finite values")
        if mode == "none":
            return cls()
        if mode != "standard":
            raise ValueError("normalization must be standard or none")
        return cls(train_targets.mean(), train_targets.std(unbiased=False).clamp_min(1e-8))

    def encode(self, values):
        return (values - self.mean) / self.scale

    def decode(self, values):
        return values * self.scale + self.mean

    def state_dict(self):
        return {"mean": self.mean, "scale": self.scale}
