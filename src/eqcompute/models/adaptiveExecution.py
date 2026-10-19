"""Shared EGNN execution: random-depth warm-up, exact expectation, or online exit."""

import math

import torch
from torch import nn

from eqcompute.models.activeGraphs import select_graphs
from eqcompute.models.execution import FixedDepthExecution
from eqcompute.models.halting import HaltingController, cdf_depth, stopping_distribution
from eqcompute.models.trace import AdaptiveInfo, ModelOutput, snapshot


class AdaptiveExecution(nn.Module):
    def __init__(self, depth, hidden_dim, controller_dim=64, threshold=0.5, use_delta=True):
        super().__init__()
        FixedDepthExecution(depth)  # Reuse the fixed-policy depth contract.
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("threshold must lie strictly between 0 and 1")
        self.depth, self.threshold = depth, float(threshold)
        self.controller = HaltingController(hidden_dim, controller_dim, use_delta)
        self.warmup = False

    def run(
        self, initializer, operators, readout, head, graph, context, trace, *, execution_mode=None
    ):
        if not operators.weight_tying:
            raise ValueError("Adaptive execution requires shared operators")
        trace.validate(self.depth)
        mode = execution_mode or (
            ("warmup" if self.warmup else "full") if self.training else "hard"
        )
        if mode not in ("warmup", "full", "diagnostic", "hard"):
            raise ValueError(f"Unknown adaptive execution mode: {mode}")
        if mode == "warmup":
            if not self.training:
                raise ValueError("warmup mode is only available during training")
            depth = int(torch.randint(1, self.depth + 1, ()))
            output = FixedDepthExecution(depth).run(
                initializer, operators, readout, head, graph, context, trace
            )
            depths = torch.full(
                (graph.num_graphs,), depth, device=graph.positions.device, dtype=torch.long
            )
            valid = torch.arange(1, self.depth + 1, device=depths.device)[None] <= depths[:, None]
            output.adaptive = AdaptiveInfo(mode, depths, depths.clone(), valid)
            return output
        if mode == "hard":
            if self.training or torch.is_grad_enabled():
                raise ValueError("hard mode requires model.eval() and inference_mode/no_grad")
            if trace.predictions or trace.node_features or trace.graph_features or trace.positions:
                raise ValueError("Full traces require execution_mode='diagnostic'")
            return self._hard(initializer, operators, readout, head, graph, context)
        state = initializer.initialize(graph)
        predictions, records, hazards = {}, {}, []
        for depth in range(1, self.depth + 1):
            previous = state.node_features
            state = operators.at(depth)(state, graph)
            representation = readout(state, graph)
            predictions[depth] = head(representation, context)
            hazards.append(
                self.controller(state.node_features, previous, graph)
                if depth < self.depth else state.node_features.new_ones(graph.num_graphs)
            )
            if trace.selects(depth):
                record = {}
                for enabled, key, value in (
                    (trace.node_features, "node_features", state.node_features),
                    (trace.positions, "positions", state.positions),
                    (trace.graph_features, "graph_features", representation),
                ):
                    if enabled:
                        record[key] = snapshot(value)
                if record:
                    records[depth] = record
        q = torch.stack(hazards, 1)
        p = stopping_distribution(q)
        depths = cdf_depth(p, self.threshold)
        steps = torch.arange(1, self.depth + 1, device=p.device)
        values = torch.stack(list(predictions.values()), 1)
        prediction = values[torch.arange(graph.num_graphs, device=p.device), depths - 1]
        info = AdaptiveInfo(
            mode, depths, torch.full_like(depths, self.depth),
            torch.ones_like(p, dtype=torch.bool), q, p, (p * steps).sum(1),
        )
        return ModelOutput(prediction, predictions, records, info)

    def _hard(self, initializer, operators, readout, head, graph, context):
        state = initializer.initialize(graph)
        count, device = graph.num_graphs, graph.positions.device
        active_ids = torch.arange(count, device=device)
        survival = state.node_features.new_ones(count)
        # Sum masses in the same order as the full reference CDF.
        cumulative = torch.zeros_like(survival)
        result = state.node_features.new_empty(count, 1)
        depths = torch.empty(count, device=device, dtype=torch.long)
        for depth in range(1, self.depth + 1):
            previous = state.node_features
            state = operators.at(depth)(state, graph)
            prediction = head(readout(state, graph), context)
            q = (
                self.controller(state.node_features, previous, graph)
                if depth < self.depth else survival.new_ones(graph.num_graphs)
            )
            cumulative = cumulative + survival * q
            survival = survival * (1 - q)
            stop = cumulative >= self.threshold
            if depth == self.depth:
                stop = torch.ones_like(stop)
            result[active_ids[stop]] = prediction[stop]
            depths[active_ids[stop]] = depth
            keep = ~stop
            if not keep.any():
                break
            if stop.any():
                state, graph = select_graphs(state, graph, keep)
                active_ids = active_ids[keep]
                survival, cumulative = survival[keep], cumulative[keep]
        valid = torch.arange(1, self.depth + 1, device=device)[None] <= depths[:, None]
        return ModelOutput(result, adaptive=AdaptiveInfo("hard", depths, depths.clone(), valid))
