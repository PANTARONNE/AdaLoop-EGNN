import csv
import math

import pytest
import torch
from omegaconf import OmegaConf

from eqcompute.application import make_loader
from eqcompute.metrics.regression import RegressionMetrics
from eqcompute.models.iterative import build_egnn
from eqcompute.tasks.adaptiveRegression import AdaptiveRegressionTask, adaptive_loss
from eqcompute.tasks.targetTransform import TargetTransform
from eqcompute.training.checkpoint import load_checkpoint, seed_everything
from eqcompute.training.optim import build_optimizer
from eqcompute.training.trainer import Trainer


def assemble(config, samples, directory):
    seed_everything(config.seed)
    model = build_egnn(config.model, config.task.target)
    task = AdaptiveRegressionTask(config.task, 0, "D", TargetTransform(2.5, 1.0))
    optimizer = build_optimizer(config.optimizer, model.parameters())
    generator = torch.Generator().manual_seed(config.seed)
    directory.mkdir()
    trainer = Trainer(
        model, task, optimizer, RegressionMetrics, "cpu", directory,
        OmegaConf.to_container(config, resolve=True),
        {"run_id": "fixture", "split_id": "fixture"}, generator,
    )
    train = make_loader(samples, config.loader, True, generator)
    validation = make_loader(samples, config.loader)
    return trainer, train, validation


@pytest.mark.parametrize("pause_after", [1, 2])
@pytest.mark.parametrize("beta", [0.0, 0.01])
def test_resume_across_warmup_and_adaptive(adaptive_config, samples, tmp_path, pause_after, beta):
    adaptive_config.task.entropy_beta = beta
    adaptive_config.loader.batch_size = 2
    full, train, validation = assemble(adaptive_config, samples, tmp_path / "full")
    full.fit(train, validation, 3)
    part, train, validation = assemble(adaptive_config, samples, tmp_path / "part")
    part.fit(train, validation, pause_after)
    assert (tmp_path / "part/checkpoints/best.pt").exists() is (pause_after > 1)
    resumed, train, validation = assemble(adaptive_config, samples, tmp_path / "resumed")
    resumed.restore(tmp_path / "part/checkpoints/last.pt")
    resumed.fit(train, validation, 3)
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    assert full.best_value == resumed.best_value
    saved = load_checkpoint(tmp_path / "resumed/checkpoints/last.pt")
    assert saved["task"] == resumed.task.state_dict()
    assert saved["task"]["warmup_epochs"] == 1
    assert saved["task"]["entropy_beta"] == saved["config"]["task"]["entropy_beta"] == beta
    assert "lambda_compute" not in saved["config"]["task"]
    assert "_lambda" not in saved["config"]["experiment_name"]
    with (tmp_path / "full/metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    for stage in ("train", "validation"):
        assert f"{stage}_compute_loss" not in rows[0]
        for row in rows:
            prediction_loss = float(row[f"{stage}_prediction_loss"])
            entropy_loss = float(row[f"{stage}_entropy_loss"])
            assert float(row[f"{stage}_loss"]) == pytest.approx(prediction_loss + entropy_loss)
            if row[f"{stage}_halting_entropy"]:
                assert entropy_loss == pytest.approx(
                    -beta * float(row[f"{stage}_halting_entropy"]), abs=1e-8,
                )
            if beta == 0:
                assert float(row[f"{stage}_loss"]) == prediction_loss
    assert rows[0]["train_expected_depth"] == ""
    assert rows[0]["train_halting_entropy"] == ""
    assert float(rows[0]["train_entropy_loss"]) == 0
    assert float(rows[1]["train_expected_depth"]) > 0
    assert float(rows[1]["train_executed_depth"]) == 4


def test_warmup_coverage_no_controller_update_and_adaptive_updates(adaptive_config, batch):
    model = build_egnn(adaptive_config.model, "mu")
    task = AdaptiveRegressionTask(adaptive_config.task, 0, "D", TargetTransform())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    task.on_epoch_start(model, 0)
    original = {
        key: value.clone() for key, value in model.execution.controller.state_dict().items()
    }
    observed = set()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        result = task.run(model, batch, "train")
        assert result.statistics["entropy_loss"] == 0
        assert result.statistics["halting_entropy"] is None
        observed.add(result.statistics["executed_depth"])
        result.loss.backward()
        assert all(p.grad is None for p in model.execution.controller.parameters())
        optimizer.step()
    assert observed == {1, 2, 3, 4}
    for key, value in original.items():
        torch.testing.assert_close(
            value, model.execution.controller.state_dict()[key], rtol=0, atol=0
        )
    task.on_epoch_start(model, 1)
    optimizer.zero_grad(set_to_none=True)
    task.run(model, batch, "train").loss.backward()
    optimizer.step()
    assert any(not torch.equal(value, model.execution.controller.state_dict()[key])
               for key, value in original.items())


def test_small_adaptive_problem_learns(adaptive_config, batch):
    adaptive_config.task.warmup_epochs = 0
    model = build_egnn(adaptive_config.model, "mu")
    task = AdaptiveRegressionTask(adaptive_config.task, 0, "D", TargetTransform())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    initial = task.run(model, batch, "train").statistics["prediction_loss"]
    for _ in range(180):
        optimizer.zero_grad()
        result = task.run(model, batch, "train")
        result.loss.backward()
        optimizer.step()
    final = task.run(model, batch, "validation").statistics["prediction_loss"]
    assert final < initial * 0.2, (initial, final)


@pytest.mark.parametrize("value", [-1, 0.5, True])
def test_invalid_task(adaptive_config, value):
    adaptive_config.task.warmup_epochs = value
    with pytest.raises(ValueError):
        AdaptiveRegressionTask(adaptive_config.task, 0, "D", TargetTransform())


@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf"), -float("inf"),
                                  True, False, "0.01", None])
def test_invalid_entropy_beta(adaptive_config, value):
    adaptive_config.task.entropy_beta = value
    with pytest.raises(ValueError, match="entropy_beta"):
        AdaptiveRegressionTask(adaptive_config.task, 0, "D", TargetTransform())


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("criterion", ["mae", "mse"])
def test_zero_entropy_preserves_loss_and_gradients(adaptive_config, batch, legacy, criterion):
    adaptive_config.task.entropy_beta = 0.0
    if legacy:
        del adaptive_config.task.entropy_beta
    adaptive_config.task.loss = criterion
    model = build_egnn(adaptive_config.model, "mu")
    task = AdaptiveRegressionTask(adaptive_config.task, 0, "D", TargetTransform())
    task.on_epoch_start(model, adaptive_config.task.warmup_epochs)
    result = task.run(model, batch, "train")
    output = task.predict(model, batch, execution_mode="full")
    expected = adaptive_loss(
        torch.stack(list(output.predictions_by_depth.values()), 1),
        task.transform.encode(task.targets(batch)), output.adaptive.stop_probabilities, criterion,
    )
    torch.testing.assert_close(result.loss, expected, rtol=0, atol=0)
    parameters = tuple(model.parameters())
    actual_grads = torch.autograd.grad(result.loss, parameters, allow_unused=True)
    expected_grads = torch.autograd.grad(expected, parameters, allow_unused=True)
    for actual, expected_grad in zip(actual_grads, expected_grads, strict=True):
        if expected_grad is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected_grad, rtol=0, atol=0)
    assert result.statistics["entropy_loss"] == 0
    assert task.entropy_beta == 0


@pytest.mark.parametrize("epoch,stage", [(0, "train"), (0, "validation"),
                                        (1, "train"), (1, "validation")])
def test_entropy_phase_and_loss_accounting(adaptive_config, batch, epoch, stage):
    model = build_egnn(adaptive_config.model, "mu")
    task = AdaptiveRegressionTask(adaptive_config.task, 0, "D", TargetTransform())
    task.on_epoch_start(model, epoch)
    result = task.run(model, batch, stage)
    stats = result.statistics
    if epoch == 0 and stage == "train":
        assert stats["halting_entropy"] is None
        assert stats["entropy_loss"] == 0
    else:
        assert stats["halting_entropy"] > 0
        assert stats["entropy_loss"] == pytest.approx(-0.01 * stats["halting_entropy"])
    assert result.loss.item() == pytest.approx(stats["prediction_loss"] + stats["entropy_loss"])


def test_entropy_updates_controller_toward_uniform_with_equal_errors(adaptive_config, batch):
    adaptive_config.model.execution.depth = 2
    model = build_egnn(adaptive_config.model, "mu")
    task = AdaptiveRegressionTask(adaptive_config.task, 0, "D", TargetTransform())
    task.on_epoch_start(model, adaptive_config.task.warmup_epochs)
    gate = model.execution.controller.network[-1]
    batch.target.zero_()
    with torch.no_grad():
        # Equal predictions at both depths remove the prediction-loss gate signal.
        for parameter in model.head.parameters():
            parameter.zero_()
        gate.weight.zero_()
        gate.bias.fill_(math.log(4))  # q_1=0.8 -> p=(0.8, 0.2).
    optimizer = torch.optim.SGD(model.execution.controller.parameters(), lr=1.0)
    before = task.run(model, batch, "train")
    assert before.statistics["prediction_loss"] == 0
    assert before.loss.item() < 0  # A valid total loss when the entropy reward exceeds the error.
    before.loss.backward()
    torch.testing.assert_close(
        gate.bias.grad,
        torch.full_like(gate.bias, task.entropy_beta * 0.8 * 0.2 * math.log(4)),
        rtol=1e-4, atol=1e-7,
    )
    optimizer.step()
    after = task.run(model, batch, "train")
    assert after.statistics["halting_entropy"] > before.statistics["halting_entropy"]


@pytest.mark.parametrize("beta", [0.0, 0.01])
def test_task_state_rejects_changed_entropy_beta(adaptive_config, beta):
    adaptive_config.task.entropy_beta = beta
    task = AdaptiveRegressionTask(adaptive_config.task, 0, "D", TargetTransform())
    state = task.state_dict()
    state["entropy_beta"] = beta + 0.01
    with pytest.raises(ValueError, match="entropy_beta"):
        task.load_state_dict(state)
    del state["entropy_beta"]
    if beta == 0:
        task.load_state_dict(state)
    else:
        with pytest.raises(ValueError, match="entropy_beta"):
            task.load_state_dict(state)


@pytest.mark.parametrize("beta", [0.0, 0.01])
def test_legacy_resume_defaults_entropy_to_zero(adaptive_config, samples, tmp_path, beta):
    adaptive_config.task.entropy_beta = 0.0
    original, train, validation = assemble(adaptive_config, samples, tmp_path / "original")
    original.fit(train, validation, 1)
    path = tmp_path / "original/checkpoints/last.pt"
    saved = load_checkpoint(path)
    del saved["config"]["task"]["entropy_beta"]
    del saved["task"]["entropy_beta"]
    torch.save(saved, path)
    adaptive_config.task.entropy_beta = beta
    resumed, train, validation = assemble(adaptive_config, samples, tmp_path / "resumed")
    if beta > 0:
        with pytest.raises(ValueError, match="configuration mismatch: task"):
            resumed.restore(path)
    else:
        resumed.restore(path)
        assert resumed.task.entropy_beta == 0
        resumed.fit(train, validation, 2)
        updated = load_checkpoint(tmp_path / "resumed/checkpoints/last.pt")
        assert updated["task"]["entropy_beta"] == 0


def test_resume_still_rejects_other_task_changes(adaptive_config, samples, tmp_path):
    original, train, validation = assemble(adaptive_config, samples, tmp_path / "original")
    original.fit(train, validation, 1)
    adaptive_config.task.loss = "mse"
    resumed, _, _ = assemble(adaptive_config, samples, tmp_path / "resumed")
    with pytest.raises(ValueError, match="configuration mismatch: task"):
        resumed.restore(tmp_path / "original/checkpoints/last.pt")
