"""Mathematical and training contracts for the differentiable CDF surrogate."""

from copy import deepcopy

import pytest
import torch
from omegaconf import OmegaConf

from eqcompute.application import make_loader
from eqcompute.metrics.regression import RegressionMetrics
from eqcompute.models.halting import cdf_depth
from eqcompute.models.iterative import build_egnn
from eqcompute.tasks.adaptiveRegression import (
    AdaptiveRegressionTask,
    adaptive_loss,
    soft_cdf_weights,
)
from eqcompute.tasks.targetTransform import TargetTransform
from eqcompute.training.checkpoint import load_checkpoint, seed_everything
from eqcompute.training.optim import build_optimizer
from eqcompute.training.trainer import Trainer


def select_loss(config, name="soft_cdf", temperature=0.1):
    config.task.phase = "halting"
    OmegaConf.update(config, "task.halting_loss", name, force_add=True)
    OmegaConf.update(config, "task.soft_cdf_temperature", temperature, force_add=True)
    return config


def make_task(config):
    return AdaptiveRegressionTask(config.task, 0, "D", TargetTransform(2.5, 2.0))


def test_soft_cdf_matches_known_cumulative_probabilities():
    p = torch.tensor([[0.2, 0.4, 0.4]], dtype=torch.float64)
    actual = soft_cdf_weights(p, threshold=0.5, temperature=0.1)
    transitions = torch.sigmoid(torch.tensor([-3.0, 1.0], dtype=p.dtype))
    reference = torch.stack((transitions[0], transitions[1] - transitions[0],
                             1 - transitions[1]))[None]
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("threshold", [0.05, 0.5, 0.95])
@pytest.mark.parametrize("temperature", [1e-5, 0.1, 10.0])
def test_weights_form_a_distribution_for_boundary_stop_masses(threshold, temperature):
    p = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.], [.2, .3, .5]])
    weights = soft_cdf_weights(p, threshold, temperature)
    assert weights.shape == p.shape
    assert weights.dtype == p.dtype
    assert weights.device == p.device
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    torch.testing.assert_close(weights.sum(1), torch.ones(len(p)))


def test_single_depth_has_all_mass_and_keeps_dtype():
    p = torch.ones(3, 1, dtype=torch.float64)
    torch.testing.assert_close(soft_cdf_weights(p, 0.7, 0.1), p, rtol=0, atol=0)


@pytest.mark.parametrize("threshold", [0.25, 0.5, 0.75])
def test_low_temperature_approaches_hard_cdf_away_from_ties(threshold):
    p = torch.tensor([[.2, .4, .4], [.8, .1, .1], [.1, .2, .7]], dtype=torch.float64)
    selected = cdf_depth(p, threshold) - 1
    expected = torch.nn.functional.one_hot(selected, p.shape[1]).to(p.dtype)
    torch.testing.assert_close(soft_cdf_weights(p, threshold, 1e-3), expected,
                               rtol=0, atol=1e-8)


def test_exact_threshold_tie_is_a_soft_half_split_not_hard_tie_breaking():
    p = torch.tensor([[.5, .25, .25]], dtype=torch.float64)
    weights = soft_cdf_weights(p, 0.5, 1e-5)
    torch.testing.assert_close(weights, torch.tensor([[.5, .5, 0.]], dtype=p.dtype),
                               rtol=0, atol=1e-12)
    assert cdf_depth(p, 0.5).item() == 1


def test_soft_cdf_gradcheck_through_normalized_stop_probabilities():
    logits = torch.tensor([[.1, -.4, .3], [.8, -.2, -.7]],
                          dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda values: soft_cdf_weights(values.softmax(-1), 0.4, 0.2), (logits,),
        eps=1e-6, atol=1e-5, rtol=1e-4,
    )


def test_equal_depth_errors_give_zero_selection_gradient():
    logits = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    weights = soft_cdf_weights(logits.softmax(-1), 0.5, 0.1)
    loss = (weights * torch.tensor([[2.], [5.]], dtype=weights.dtype)).sum(1).mean()
    gradient, = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(loss, torch.tensor(3.5, dtype=weights.dtype))
    torch.testing.assert_close(gradient, torch.zeros_like(logits), rtol=0, atol=1e-12)


@pytest.mark.parametrize("criterion,expected", [("mae", 2.0), ("mse", 4.0)])
def test_error_is_weighted_after_loss_so_predictions_cannot_cancel(criterion, expected):
    predictions = torch.tensor([[[0.], [4.]]])
    target = torch.tensor([[2.]])
    weights = soft_cdf_weights(torch.tensor([[.5, .5]]), 0.5, 0.1)
    result = adaptive_loss(predictions, target, weights, criterion)
    assert result.item() == pytest.approx(expected)
    assert ((weights[..., None] * predictions).sum(1) - target).abs().item() == 0


@pytest.mark.parametrize("beta", [0.0, 0.01])
@pytest.mark.parametrize("criterion", ["mae", "mse"])
def test_task_uses_soft_cdf_gradient_but_reports_hard_prediction(
    adaptive_config, batch, beta, criterion,
):
    config = select_loss(adaptive_config, temperature=0.17)
    config.task.entropy_beta = beta
    config.task.loss = criterion
    config.model.execution.threshold = 0.73
    model = build_egnn(config.model, config.task.target)
    task = make_task(config)
    task.configure_model(model)
    model.train()
    result = task.run(model, batch, "train")
    output = task.predict(model, batch, execution_mode="full")
    p = output.adaptive.stop_probabilities
    values = torch.stack(list(output.predictions_by_depth.values()), 1)
    target = task.transform.encode(task.targets(batch))
    cumulative = p.cumsum(1)[:, :-1]
    transitions = torch.cat((p.new_zeros(len(p), 1),
                             torch.sigmoid((cumulative - 0.73) / 0.17),
                             p.new_ones(len(p), 1)), dim=1)
    weights = transitions[:, 1:] - transitions[:, :-1]
    errors = values - target[:, None, :]
    errors = errors.abs() if criterion == "mae" else errors.square()
    prediction_loss = (weights * errors.mean(-1)).sum(1).mean()
    expected_loss = (p * errors.mean(-1)).sum(1).mean()
    entropy_loss = beta * (p * p.clamp_min(1e-12).log()).sum(1).mean()
    reference = prediction_loss + entropy_loss
    torch.testing.assert_close(result.loss, reference)
    assert result.statistics["prediction_loss"] == pytest.approx(prediction_loss.item())
    assert result.statistics["expected_prediction_loss"] == pytest.approx(expected_loss.item())
    assert result.statistics["entropy_loss"] == pytest.approx(entropy_loss.item())
    torch.testing.assert_close(result.prediction, task.transform.decode(output.prediction))
    assert not result.prediction.requires_grad

    parameters = tuple(model.execution.controller.parameters())
    actual = torch.autograd.grad(result.loss, parameters, retain_graph=True)
    expected = torch.autograd.grad(reference, parameters)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)
    assert sum(gradient.abs().sum().item() for gradient in actual) > 1e-7
    result.loss.backward()
    for name, parameter in model.named_parameters():
        if name.startswith("execution.controller."):
            assert parameter.requires_grad and parameter.grad is not None
        else:
            assert not parameter.requires_grad and parameter.grad is None
    model.eval()
    with torch.no_grad():
        validation = task.run(model, batch, "validation")
        hard = task.predict(model, batch, execution_mode="hard")
    torch.testing.assert_close(validation.prediction, task.transform.decode(hard.prediction),
                               rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("value", [0, -0.1, float("nan"), float("inf"), -float("inf"),
                                  True, False, "0.1", None])
def test_invalid_soft_cdf_temperature_is_rejected(adaptive_config, value):
    select_loss(adaptive_config, temperature=value)
    with pytest.raises(ValueError, match="temperature"):
        make_task(adaptive_config)


@pytest.mark.parametrize("value", ["hard_cdf", "softmax", "", None, 0, True])
def test_invalid_halting_loss_is_rejected(adaptive_config, value):
    select_loss(adaptive_config, name=value)
    with pytest.raises(ValueError, match="halting_loss"):
        make_task(adaptive_config)


@pytest.mark.parametrize("threshold", [0., 1., -0.1, float("nan"), float("inf")])
def test_helper_rejects_invalid_cdf_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        soft_cdf_weights(torch.tensor([[.3, .7]]), threshold, 0.1)


def test_soft_cdf_task_state_roundtrip(adaptive_config):
    task = make_task(select_loss(adaptive_config, temperature=0.07))
    state = task.state_dict()
    assert state["halting_loss"] == "soft_cdf"
    assert state["soft_cdf_temperature"] == 0.07
    restored = make_task(adaptive_config)
    restored.load_state_dict(state)
    assert restored.state_dict() == state


@pytest.mark.parametrize("key,value", [("halting_loss", "expected"),
                                      ("soft_cdf_temperature", 0.2)])
def test_soft_cdf_task_rejects_changed_loss_settings(adaptive_config, key, value):
    task = make_task(select_loss(adaptive_config))
    state = deepcopy(task.state_dict())
    state[key] = value
    with pytest.raises(ValueError, match=key):
        task.load_state_dict(state)


def test_legacy_state_defaults_to_expected_and_cannot_be_loaded_as_soft_cdf(adaptive_config):
    select_loss(adaptive_config, name="expected")
    expected_task = make_task(adaptive_config)
    legacy = expected_task.state_dict()
    legacy.pop("halting_loss", None)
    legacy.pop("soft_cdf_temperature", None)
    expected_task.load_state_dict(legacy)
    assert expected_task.state_dict()["halting_loss"] == "expected"
    soft_task = make_task(select_loss(adaptive_config))
    with pytest.raises(ValueError, match="halting_loss"):
        soft_task.load_state_dict(legacy)


def test_legacy_config_without_new_keys_uses_expected_loss(adaptive_config):
    values = OmegaConf.to_container(adaptive_config.task, resolve=True)
    values["phase"] = "halting"
    values.pop("halting_loss", None)
    values.pop("soft_cdf_temperature", None)
    task = AdaptiveRegressionTask(OmegaConf.create(values), 0, "D", TargetTransform())
    assert task.state_dict()["halting_loss"] == "expected"


def assemble_trainer(config, samples, directory):
    seed_everything(config.seed)
    directory.mkdir()
    model = build_egnn(config.model, config.task.target)
    task = make_task(config)
    task.configure_model(model)
    optimizer = build_optimizer(config.optimizer,
                                (parameter for parameter in model.parameters()
                                 if parameter.requires_grad))
    generator = torch.Generator().manual_seed(config.seed)
    trainer = Trainer(
        model, task, optimizer, RegressionMetrics, "cpu", directory,
        OmegaConf.to_container(config, resolve=True),
        {"run_id": directory.name, "split_id": "softcdf-fixture"}, generator,
    )
    return trainer, make_loader(samples, config.loader, True, generator), make_loader(
        samples, config.loader,
    )


def test_soft_cdf_full_training_matches_checkpoint_resume(adaptive_config, samples, tmp_path):
    config = select_loss(adaptive_config, temperature=0.07)
    config.loader.batch_size = 3
    config.task.entropy_beta = 0.0
    full, training, validation = assemble_trainer(config, samples, tmp_path / "full")
    full.fit(training, validation, 3)
    partial, training, validation = assemble_trainer(config, samples, tmp_path / "partial")
    partial.fit(training, validation, 1)
    resumed, training, validation = assemble_trainer(config, samples, tmp_path / "resumed")
    resumed.restore(tmp_path / "partial/checkpoints/last.pt")
    resumed.fit(training, validation, 3)
    for name, value in full.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
    assert full.best_value == resumed.best_value
    assert resumed.task.state_dict()["halting_loss"] == "soft_cdf"
    assert resumed.task.state_dict()["soft_cdf_temperature"] == 0.07


def test_legacy_expected_checkpoint_resumes_with_new_explicit_defaults(
    adaptive_config, samples, tmp_path,
):
    config = select_loss(adaptive_config, name="expected")
    config.loader.batch_size = 3
    full, training, validation = assemble_trainer(config, samples, tmp_path / "full")
    full.fit(training, validation, 3)
    partial, training, validation = assemble_trainer(config, samples, tmp_path / "legacy")
    partial.fit(training, validation, 1)
    path = tmp_path / "legacy/checkpoints/last.pt"
    state = load_checkpoint(path)
    for key in ("halting_loss", "soft_cdf_temperature"):
        state["config"]["task"].pop(key)
        state["task"].pop(key)
    torch.save(state, path)
    resumed, training, validation = assemble_trainer(config, samples, tmp_path / "resumed")
    resumed.restore(path)
    resumed.fit(training, validation, 3)
    for name, value in full.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
    assert full.best_value == resumed.best_value
    assert resumed.task.state_dict()["halting_loss"] == "expected"
