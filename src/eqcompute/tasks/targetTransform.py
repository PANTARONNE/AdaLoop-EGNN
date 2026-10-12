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
        mean = train_targets.mean()
        if mode == "mad":
            scale = (train_targets - mean).abs().mean()
        elif mode == "standard":
            scale = train_targets.std(unbiased=False)
        else:
            raise ValueError("normalization must be standard, mad or none")
        return cls(mean, scale.clamp_min(1e-8))

    def encode(self, values):
        return (values - self.mean) / self.scale

    def decode(self, values):
        return values * self.scale + self.mean

    def state_dict(self):
        return {"mean": self.mean, "scale": self.scale}
