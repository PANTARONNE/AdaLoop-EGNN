from eqcompute.models.trace import ModelOutput, snapshot


class FixedDepthExecution:
    def __init__(self, depth):
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
            raise ValueError("depth must be a positive integer")
        self.depth = depth

    def run(self, initializer, operators, readout, head, graph, context, trace):
        trace.validate(self.depth)
        # Validate capacity before doing any work.
        operators.at(self.depth)
        state = initializer.initialize(graph)
        predictions, snapshots = {}, {}
        for depth in range(1, self.depth + 1):
            state = operators.at(depth)(state, graph)
            selected = trace.selects(depth)
            needs_prediction = depth == self.depth or (trace.predictions and selected)
            if needs_prediction or (trace.graph_features and selected):
                graph_repr = readout(state, graph)
            if needs_prediction:
                prediction = head(graph_repr, context)
                if trace.predictions and selected:
                    predictions[depth] = prediction
            if selected:
                record = {}
                if trace.node_features:
                    record["node_features"] = snapshot(state.node_features)
                if trace.positions:
                    record["positions"] = snapshot(state.positions)
                if trace.graph_features:
                    record["graph_features"] = snapshot(graph_repr)
                if record:
                    snapshots[depth] = record
        return ModelOutput(prediction, predictions, snapshots)
