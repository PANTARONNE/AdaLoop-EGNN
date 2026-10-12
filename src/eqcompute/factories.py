from eqcompute.data.QM9 import build_qm9
from eqcompute.data.QM9Orig import build_qm9_orig
from eqcompute.models.iterative import build_egnn, build_egnn_orig
from eqcompute.tasks.adaptiveRegression import AdaptiveRegressionTask
from eqcompute.tasks.regression import RegressionTask

DATASETS = {"qm9": build_qm9, "qm9_orig": build_qm9_orig}
MODELS = {"egnn": build_egnn, "egnn_orig": build_egnn_orig}
TASKS = {"scalar_regression": RegressionTask, "adaptive_regression": AdaptiveRegressionTask}


def construct(registry, name, *args):
    if name not in registry:
        raise ValueError(f"Unknown component {name!r}; available: {list(registry)}")
    return registry[name](*args)
