from dataclasses import replace

import pytest
import torch

from eqcompute.contracts import PropertyContext
from eqcompute.models.execution import FixedDepthExecution
from eqcompute.models.geometry import radius_edges
from eqcompute.models.iterative import build_egnn
from eqcompute.models.trace import TraceSpec


@pytest.mark.parametrize("tied", [True, False])
def test_forward_and_manual_steps(config, batch, tied):
    config.model.weight_tying = tied
    model = build_egnn(config.model, "mu")
    context = PropertyContext("mu")
    output = model(batch.graph, context, trace=TraceSpec(predictions=True))
    state = model.initialize(batch.graph)
    for k in range(1, 4):
        state = model.step(state, batch.graph, depth=k)
        manual = model.head(model.readout(state, batch.graph), context)
        torch.testing.assert_close(manual, output.predictions_by_depth[k])
    assert output.prediction.shape == (4, 1)
    output.prediction.square().mean().backward()
    assert model.initializer.embedding.weight.grad is not None
    assert torch.isfinite(output.prediction).all()


def test_sharing_parameter_count_and_gradients(config, batch):
    model = build_egnn(config.model, "mu")
    assert model.operators.at(1) is model.operators.at(3)
    count = sum(p.numel() for p in model.parameters())
    model.execution = FixedDepthExecution(5)
    assert count == sum(p.numel() for p in model.parameters())
    calls = []
    handle = model.operators.at(1).register_forward_hook(lambda *args: calls.append(1))
    model(batch.graph, PropertyContext("mu")).prediction.sum().backward()
    handle.remove()
    assert len(calls) == 5
    assert model.operators.at(1).message[0].weight.grad.abs().sum() > 0
    config.model.weight_tying = False
    model = build_egnn(config.model, "mu")
    assert model.operators.at(1) is not model.operators.at(2)
    ids = [{id(p) for p in step.parameters()} for step in model.operators.steps]
    assert not (ids[0] & ids[1])
    assert sum(p.numel() for p in model.parameters()) > count
    model.execution = FixedDepthExecution(4)
    with pytest.raises(ValueError):
        model(batch.graph, PropertyContext("mu"))


def test_trace_is_optional_detached_and_result_preserving(config, batch):
    model = build_egnn(config.model, "mu")
    context = PropertyContext("mu")
    plain = model(batch.graph, context)
    assert plain.trace == plain.predictions_by_depth == {}
    output = model(
        batch.graph,
        context,
        trace=TraceSpec(
            predictions=True, node_features=True, graph_features=True, positions=True, depths=(1, 3)
        ),
    )
    torch.testing.assert_close(plain.prediction, output.prediction)
    assert set(output.predictions_by_depth) == {1, 3}
    assert output.predictions_by_depth[1].requires_grad
    assert not output.trace[1]["node_features"].requires_grad
    assert output.trace[1]["graph_features"].shape == (4, 32)
    assert (
        output.trace[1]["node_features"].data_ptr() != output.trace[3]["node_features"].data_ptr()
    )
    with pytest.raises(ValueError):
        model(batch.graph, context, trace=TraceSpec(depths=(4,)))
    with pytest.raises(ValueError):
        model(batch.graph, PropertyContext("gap"))


@pytest.mark.parametrize("reflection", [False, True])
def test_euclidean_symmetry(config, batch, reflection):
    model = build_egnn(config.model, "mu").double()
    graph = replace(batch.graph, positions=batch.graph.positions.double())
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if (torch.linalg.det(rotation) < 0).item() != reflection:
        rotation[:, 0] *= -1
    translation = torch.tensor([3.0, -2.0, 0.7], dtype=torch.float64)
    transformed = replace(graph, positions=graph.positions @ rotation.T + translation)
    spec = TraceSpec(predictions=True, positions=True)
    a = model(graph, PropertyContext("mu"), trace=spec)
    b = model(transformed, PropertyContext("mu"), trace=spec)
    for depth in range(1, 4):
        torch.testing.assert_close(
            a.predictions_by_depth[depth], b.predictions_by_depth[depth], atol=1e-9, rtol=1e-9
        )
        torch.testing.assert_close(
            a.trace[depth]["positions"] @ rotation.T + translation,
            b.trace[depth]["positions"],
            atol=1e-9,
            rtol=1e-9,
        )


def test_permutation_batch_isolation_and_empty_neighbors(config, batch, samples):
    from eqcompute.data.collate import collate_molecules

    model = build_egnn(config.model, "mu")
    graph = batch.graph
    context = PropertyContext("mu")
    baseline = model(graph, context).prediction
    permutation = torch.randperm(graph.positions.size(0))
    permuted = replace(
        graph,
        positions=graph.positions[permutation],
        atomic_numbers=graph.atomic_numbers[permutation],
        batch_index=graph.batch_index[permutation],
    )
    torch.testing.assert_close(baseline, model(permuted, context).prediction)
    alone = collate_molecules([samples[0]])
    torch.testing.assert_close(baseline[:1], model(alone.graph, context).prediction)
    edges = radius_edges(graph, config.model.graph.cutoff)
    assert torch.equal(graph.batch_index[edges[0]], graph.batch_index[edges[1]])
    assert not (edges[0] == edges[1]).any()
    singleton = collate_molecules([samples[2]])
    assert model.initialize(singleton.graph).edge_index.shape == (2, 0)
    assert torch.isfinite(model(singleton.graph, context).prediction).all()
