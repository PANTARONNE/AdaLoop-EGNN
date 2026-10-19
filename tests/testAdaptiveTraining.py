import csv

import pytest
import torch
from omegaconf import OmegaConf

from eqcompute.application import make_loader
from eqcompute.metrics.regression import RegressionMetrics
from eqcompute.models.iterative import build_egnn
from eqcompute.tasks.adaptiveRegression import AdaptiveRegressionTask
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
def test_resume_across_warmup_and_adaptive(adaptive_config, samples, tmp_path, pause_after):
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
    assert "lambda_compute" not in saved["config"]["task"]
    assert "_lambda" not in saved["config"]["experiment_name"]
    with (tmp_path / "full/metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    for stage in ("train", "validation"):
        assert f"{stage}_compute_loss" not in rows[0]
        for row in rows:
            assert float(row[f"{stage}_loss"]) == float(row[f"{stage}_prediction_loss"])
    assert rows[0]["train_expected_depth"] == ""
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
    initial = task.run(model, batch, "train").loss.item()
    for _ in range(180):
        optimizer.zero_grad()
        result = task.run(model, batch, "train")
        result.loss.backward()
        optimizer.step()
    final = task.run(model, batch, "validation").loss.item()
    assert final < initial * 0.2, (initial, final)


@pytest.mark.parametrize("value", [-1, 0.5, True])
def test_invalid_task(adaptive_config, value):
    adaptive_config.task.warmup_epochs = value
    with pytest.raises(ValueError):
        AdaptiveRegressionTask(adaptive_config.task, 0, "D", TargetTransform())
