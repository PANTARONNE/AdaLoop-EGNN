from eqcompute.data.QM9 import build_qm9
from eqcompute.models.iterative import build_egnn
from eqcompute.tasks.regression import RegressionTask

DATASETS = {"qm9": build_qm9}
MODELS = {"egnn": build_egnn}
TASKS = {"scalar_regression": RegressionTask}


def construct(registry, name, *args):
    if name not in registry:
        raise ValueError(f"Unknown component {name!r}; available: {list(registry)}")
    return registry[name](*args)
