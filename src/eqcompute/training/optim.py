import torch


def build_optimizer(config, parameters):
    choices = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    if config.name not in choices or config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("Invalid optimizer configuration")
    return choices[config.name](
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
