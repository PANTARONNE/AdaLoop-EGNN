import math
from dataclasses import replace

import pytest
import torch

from eqcompute.models.halting import HaltingController, cdf_depth, stopping_distribution
from eqcompute.tasks.adaptiveRegression import adaptive_loss, halting_entropy


def test_stopping_probabilities_boundaries_and_gradients():
    q = torch.tensor([[0.2, 0.5, 0.7], [0.0, 1.0, 0.0]], requires_grad=True)
    p = stopping_distribution(q)
    torch.testing.assert_close(p, torch.tensor([[0.2, 0.4, 0.4], [0.0, 1.0, 0.0]]))
    torch.testing.assert_close(p.sum(1), torch.ones(2))
    assert (p >= 0).all()
    assert cdf_depth(p, 0.5).tolist() == [2, 2]
    (p * torch.arange(1., 4.)).sum().backward()
    assert q.grad[:, -1].eq(0).all()
    assert q.grad[:, :-1].abs().sum() > 0
    torch.testing.assert_close(stopping_distribution(torch.rand(2, 1)), torch.ones(2, 1))


@pytest.mark.parametrize("criterion,expected", [("mae", 1.25), ("mse", 2.5)])
def test_expected_loss_is_per_sample_not_mixture(criterion, expected):
    predictions = torch.tensor([[[0.], [4.]], [[3.], [1.]]], requires_grad=True)
    target = torch.tensor([[2.], [1.]])
    p = torch.tensor([[0.75, 0.25], [0.25, 0.75]], requires_grad=True)
    loss = adaptive_loss(predictions, target, p, criterion)
    torch.testing.assert_close(loss, torch.tensor(expected))
    loss.backward()
    assert predictions.grad.abs().sum() > 0
    assert p.grad.abs().sum() > 0


@pytest.mark.parametrize("criterion,expected", [("mae", 2.0), ("mse", 4.0)])
def test_equal_errors_have_no_preference_for_shallow_depths(criterion, expected):
    predictions = torch.tensor([[[2.], [-2.], [2.]]])
    target = torch.zeros(1, 1)
    for p in (torch.tensor([[1., 0., 0.]]), torch.tensor([[0., 0., 1.]])):
        torch.testing.assert_close(
            adaptive_loss(predictions, target, p, criterion), torch.tensor(expected)
        )


def test_controller_graph_statistics_permutation_and_single_node(batch):
    h0, h1 = torch.randn(13, 8), torch.randn(13, 8, requires_grad=True)
    controller = HaltingController(8, 5)
    q = controller(h1, h0, batch.graph)
    permutation = torch.randperm(13)
    graph = replace(batch.graph, batch_index=batch.graph.batch_index[permutation])
    torch.testing.assert_close(q, controller(h1[permutation], h0[permutation], graph))
    assert torch.isfinite(q).all()
    q.sum().backward()
    assert torch.isfinite(h1.grad).all()
    assert h1.grad.abs().sum() > 0


@pytest.mark.parametrize("q", [torch.zeros(3, 12), torch.ones(3, 12)])
def test_extreme_hazards(q):
    p = stopping_distribution(q)
    assert torch.isfinite(p).all()
    torch.testing.assert_close(p.sum(1), torch.ones(3))


@pytest.mark.parametrize("masses,expected", [
    ([[0.25, 0.25, 0.25, 0.25]], math.log(4)),
    ([[1.0, 0.0], [0.0, 1.0]], 0.0),
    ([[1.0], [1.0]], 0.0),
    ([[0.5, 0.5], [1.0, 0.0]], math.log(2) / 2),
])
def test_halting_entropy_values_and_finite_gradients(masses, expected):
    p = torch.tensor(masses, dtype=torch.float64, requires_grad=True)
    entropy = halting_entropy(p)
    torch.testing.assert_close(entropy, torch.tensor(expected, dtype=torch.float64))
    entropy.backward()
    assert torch.isfinite(p.grad).all()
