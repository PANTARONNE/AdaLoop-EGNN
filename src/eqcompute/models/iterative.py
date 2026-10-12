from torch import nn

from eqcompute.models.adaptiveExecution import AdaptiveExecution
from eqcompute.models.backbones.EGNN import EGNNInitializer, EGNNStep
from eqcompute.models.backbones.EGNNOrig import (
    EGNNOrigInitializer,
    EGNNOrigReadout,
    EGNNOrigStep,
)
from eqcompute.models.execution import FixedDepthExecution
from eqcompute.models.heads import PredictionHead, ScalarReadout
from eqcompute.models.operators import OperatorBank
from eqcompute.models.trace import TraceSpec


class IterativeModel(nn.Module):
    def __init__(self, initializer, operators, execution, readout, head):
        super().__init__()
        self.initializer = initializer
        self.operators = operators
        self.execution = execution
        self.readout = readout
        self.head = head

    def initialize(self, graph):
        return self.initializer.initialize(graph)

    def step(self, state, graph, *, depth=1):
        return self.operators.at(depth)(state, graph)

    def forward(self, graph, context, *, trace=None, execution_mode=None):
        extra = {}
        if execution_mode is not None:
            if not isinstance(self.execution, AdaptiveExecution):
                raise ValueError("execution_mode is only supported by adaptive models")
            extra["execution_mode"] = execution_mode
        return self.execution.run(
            self.initializer,
            self.operators,
            self.readout,
            self.head,
            graph,
            context,
            trace or TraceSpec(),
            **extra,
        )


def build_egnn(config, property_name):
    if config.hidden_dim < 2 or config.num_rbf < 2 or config.graph.cutoff <= 0:
        raise ValueError("hidden_dim >= 2, num_rbf >= 2 and cutoff > 0 are required")
    if config.graph.topology != "initial_radius" or config.execution.name != "fixed":
        raise ValueError("Experiment 0 supports initial_radius topology and fixed execution")
    if config.head.pooling not in ("sum", "mean"):
        raise ValueError("pooling must be sum or mean")
    if not isinstance(config.weight_tying, bool):
        raise ValueError("weight_tying must be a boolean")
    execution = FixedDepthExecution(config.execution.depth)
    return IterativeModel(
        EGNNInitializer(config.hidden_dim, config.graph.cutoff),
        OperatorBank(
            lambda: EGNNStep(
                config.hidden_dim, config.num_rbf, config.graph.cutoff, config.coordinate_scale
            ),
            config.weight_tying,
            execution.depth,
        ),
        execution,
        ScalarReadout(config.head.pooling),
        PredictionHead(config.hidden_dim, property_name),
    )


def build_egnn_orig(config, property_name):
    if config.hidden_dim < 2 or config.execution.name not in ("fixed", "adaptive"):
        raise ValueError("EGNNOrig requires hidden_dim >= 2 and fixed/adaptive execution")
    if not isinstance(config.weight_tying, bool):
        raise ValueError("weight_tying must be a boolean")
    if config.execution.name == "adaptive":
        if not config.weight_tying:
            raise ValueError("Adaptive EGNNOrig requires weight_tying=true")
        if config.execution.inference_rule != "cdf":
            raise ValueError("Only cdf inference is supported")
        execution = AdaptiveExecution(
            config.execution.depth, config.hidden_dim, config.halting.hidden_dim,
            config.execution.threshold, config.halting.use_delta,
        )
    else:
        execution = FixedDepthExecution(config.execution.depth)
    return IterativeModel(
        EGNNOrigInitializer(config.hidden_dim),
        OperatorBank(
            lambda: EGNNOrigStep(config.hidden_dim), config.weight_tying, execution.depth
        ),
        execution,
        EGNNOrigReadout(config.hidden_dim),
        PredictionHead(config.hidden_dim, property_name),
    )
