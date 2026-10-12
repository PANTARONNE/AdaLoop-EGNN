from dataclasses import replace

import pytest
import torch
from torch import nn

from eqcompute.contracts import PropertyContext
from eqcompute.data.collate import collate_molecules
from eqcompute.models.activeGraphs import select_graphs
from eqcompute.models.iterative import build_egnn_orig
from eqcompute.models.trace import TraceSpec


class SizeHazards(nn.Module):
    """Test-only prescribed stops: N=3 -> 1; N=5 -> 2; others -> 4."""

    def forward(self, current, previous, graph):
        counts = torch.bincount(graph.batch_index, minlength=graph.num_graphs)
        return torch.where(counts == 3, 0.8, torch.where(counts == 5, 0.3, 0.02))


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable"
    )),
])
def test_compact_matches_full_and_skips_real_message_work(adaptive_config, batch, device):
    batch = batch.to(device)
    model = build_egnn_orig(adaptive_config.model, "mu").to(device).eval()
    model.execution.controller = SizeHazards()
    work = []
    hook = model.operators.at(1).register_forward_pre_hook(
        lambda _, args: work.append((len(args[0].node_features), args[0].edge_index.shape[1]))
    )
    with torch.inference_mode():
        reference = model(batch.graph, PropertyContext("mu"), execution_mode="diagnostic")
        work.clear()
        hard = model(batch.graph, PropertyContext("mu"))
    hook.remove()
    torch.testing.assert_close(hard.prediction, reference.prediction)
    assert hard.adaptive.halt_depth.tolist() == [1, 2, 4, 4]
    assert work == [(13, 38), (10, 32), (5, 12), (5, 12)]
    assert reference.adaptive.executed_depth.tolist() == [4] * 4
    assert hard.adaptive.executed_depth.tolist() == [1, 2, 4, 4]
    assert hard.adaptive.expected_depth is None
    assert hard.adaptive.stop_probabilities is None
    assert hard.adaptive.valid_depth_mask.sum(1).tolist() == [1, 2, 4, 4]


def test_controller_registered_shared_gradients_and_depth_capacity(adaptive_config, batch):
    model = build_egnn_orig(adaptive_config.model, "mu")
    names = dict(model.named_parameters())
    assert any(name.startswith("execution.controller.") for name in names)
    output = model(batch.graph, PropertyContext("mu"))
    loss = (output.adaptive.stop_probabilities * torch.stack([
        value.square().flatten() for value in output.predictions_by_depth.values()
    ], 1)).sum()
    loss.backward()
    for module in (model.execution.controller, model.operators.at(1), model.head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert model.operators.at(1) is model.operators.at(12)
    size = sum(p.numel() for p in model.parameters())
    adaptive_config.model.execution.depth = 12
    assert sum(p.numel() for p in build_egnn_orig(adaptive_config.model, "mu").parameters()) == size


def test_adaptive_symmetries_and_batch_isolation(adaptive_config, batch, samples):
    model = build_egnn_orig(adaptive_config.model, "mu").double().eval()
    graph = replace(batch.graph, positions=batch.graph.positions.double())
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.double))
    reflection = torch.diag(torch.tensor([-1., 1., 1.], dtype=torch.double))
    with torch.inference_mode():
        original = model(graph, PropertyContext("mu"), execution_mode="diagnostic")
        for matrix in (rotation, reflection):
            transformed = replace(graph, positions=graph.positions @ matrix + 4)
            changed = model(transformed, PropertyContext("mu"), execution_mode="diagnostic")
            torch.testing.assert_close(original.prediction, changed.prediction)
            torch.testing.assert_close(
                original.adaptive.stop_probabilities, changed.adaptive.stop_probabilities
            )
            assert torch.equal(original.adaptive.halt_depth, changed.adaptive.halt_depth)
        permutation = torch.randperm(len(graph.positions))
        permuted = replace(
            graph, positions=graph.positions[permutation],
            atomic_numbers=graph.atomic_numbers[permutation],
            batch_index=graph.batch_index[permutation],
        )
        changed = model(permuted, PropertyContext("mu"), execution_mode="diagnostic")
        torch.testing.assert_close(original.prediction, changed.prediction)
        assert torch.equal(original.adaptive.halt_depth, changed.adaptive.halt_depth)
        for index, sample in enumerate(samples):
            single = collate_molecules([sample]).graph
            single = replace(single, positions=single.positions.double())
            output = model(single, PropertyContext("mu"))
            torch.testing.assert_close(output.prediction[0], original.prediction[index])


def test_compaction_remaps_supplied_edges_and_preserves_state(adaptive_config, batch):
    model = build_egnn_orig(adaptive_config.model, "mu")
    state = model.initialize(batch.graph)
    graph = replace(
        batch.graph, edge_index=state.edge_index,
        edge_attr=torch.arange(state.edge_index.shape[1])[:, None],
    )
    keep = torch.tensor([False, True, False, True])
    compact, graph = select_graphs(state, graph, keep)
    assert graph.num_graphs == 2
    assert graph.batch_index.tolist() == [0] * 5 + [1] * 4
    torch.testing.assert_close(compact.edge_index, graph.edge_index)
    assert graph.edge_attr.flatten().tolist() == list(range(6, 38))
    torch.testing.assert_close(
        compact.node_features, state.node_features[keep[batch.graph.batch_index]]
    )


def test_modes_do_not_mutate_policy_and_single_step(adaptive_config, batch):
    adaptive_config.model.execution.depth = 1
    model = build_egnn_orig(adaptive_config.model, "mu").eval()
    with torch.inference_mode():
        full = model(batch.graph, PropertyContext("mu"), execution_mode="diagnostic")
        hard = model(batch.graph, PropertyContext("mu"))
        torch.testing.assert_close(full.prediction, hard.prediction)
        assert hard.adaptive.mode == "hard"
        torch.testing.assert_close(full.adaptive.stop_probabilities, torch.ones(4, 1))
        with pytest.raises(ValueError, match="diagnostic"):
            model(batch.graph, PropertyContext("mu"), trace=TraceSpec(predictions=True))
    with pytest.raises(ValueError, match="inference_mode"):
        model(batch.graph, PropertyContext("mu"))


@pytest.mark.parametrize("field,value", [
    ("weight_tying", False), ("execution.depth", 0), ("execution.threshold", 0),
    ("execution.threshold", float("nan")), ("execution.inference_rule", "hazard"),
])
def test_invalid_adaptive_config(adaptive_config, field, value):
    from omegaconf import OmegaConf

    OmegaConf.update(adaptive_config.model, field, value)
    with pytest.raises(ValueError):
        build_egnn_orig(adaptive_config.model, "mu")
