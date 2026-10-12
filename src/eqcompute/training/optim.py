import torch


def build_optimizer(config, parameters):
    choices = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    if config.name not in choices or config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("Invalid optimizer configuration")
    return choices[config.name](
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )


def build_scheduler(config, optimizer):
    if config is None:
        return None
    if (
        config["name"] != "cosine"
        or not isinstance(config["t_max"], int)
        or config["t_max"] < 1
        or config["eta_min"] < 0
        or config["eta_min"] > min(group["lr"] for group in optimizer.param_groups)
    ):
        raise ValueError("Invalid cosine scheduler configuration")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["t_max"], eta_min=config["eta_min"]
    )
