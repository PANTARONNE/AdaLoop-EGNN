import csv

import pytest
import torch
from omegaconf import OmegaConf

from eqcompute.application import make_loader
from eqcompute.evaluation.prediction_writer import write_predictions
from eqcompute.metrics.regression import RegressionMetrics
from eqcompute.models.iterative import build_egnn
from eqcompute.models.trace import TraceSpec
from eqcompute.tasks.regression import RegressionTask
from eqcompute.tasks.target_transform import TargetTransform
from eqcompute.training.checkpoint import load_checkpoint, seed_everything
from eqcompute.training.optim import build_optimizer
from eqcompute.training.trainer import Trainer


def assemble(config, samples, directory):
    seed_everything(config.seed)
    model = build_egnn(config.model, "mu")
    task = RegressionTask(config.task, 0, "D", TargetTransform.fit(torch.arange(1.0, 5.0)))
    optimizer = build_optimizer(config.optimizer, model.parameters())
    generator = torch.Generator().manual_seed(config.seed)
    directory.mkdir()
    trainer = Trainer(
        model,
        task,
        optimizer,
        RegressionMetrics,
        "cpu",
        directory,
        OmegaConf.to_container(config, resolve=True),
        {"run_id": "fixture", "split_id": "fixture"},
        generator,
    )
    train = make_loader(samples, config.loader, True, generator)
    validation = make_loader(samples, config.loader, generator=generator)
    return trainer, train, validation


@pytest.mark.parametrize("supervision", ["final", "all_steps"])
def test_loss_protocol_and_overfit(config, batch, supervision):
    config.task.supervision = supervision
    model = build_egnn(config.model, "mu")
    task = RegressionTask(config.task, 0, "D", TargetTransform())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    initial = task.run(model, batch, "train").loss.item()
    for _ in range(160):
        optimizer.zero_grad()
        result = task.run(model, batch, "train")
        result.loss.backward()
        optimizer.step()
    final = task.run(model, batch, "validation").loss.item()
    assert final < initial * 0.15, (initial, final)


def test_checkpoint_epoch_resume_matches_uninterrupted(config, samples, tmp_path):
    config.loader.batch_size = 2
    full, train, validation = assemble(config, samples, tmp_path / "full")
    full.fit(train, validation, 3)
    part, train, validation = assemble(config, samples, tmp_path / "part")
    part.fit(train, validation, 1)
    resumed, train, validation = assemble(config, samples, tmp_path / "resumed")
    resumed.restore(tmp_path / "part/checkpoints/last.pt")
    resumed.fit(train, validation, 3)
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    assert full.best_value == resumed.best_value
    saved = load_checkpoint(tmp_path / "resumed/checkpoints/last.pt")
    assert saved["epoch"] == 2
    assert saved["task"] == resumed.task.state_dict()
    assert (tmp_path / "resumed/checkpoints/best.pt").exists()


def test_prediction_export_identity_units_and_completeness(config, samples, tmp_path):
    trainer, train, validation = assemble(config, samples, tmp_path / "run")
    trainer.fit(train, validation, 1)
    path = tmp_path / "predictions.csv"
    provenance = dict(
        run_id="fixture",
        checkpoint_id="hash",
        split_id="split",
        split="test",
        train_depth=3,
        weight_tying=True,
        supervision="final",
        seed=0,
    )
    count = write_predictions(trainer.model, trainer.task, validation, "cpu", path, provenance)
    rows = list(csv.DictReader(path.open()))
    assert count == len(rows) == 12
    assert {(r["sample_id"], int(r["readout_depth"])) for r in rows} == {
        (s.sample_id, k) for s in samples for k in range(1, 4)
    }
    assert all(r["unit"] == "D" for r in rows)
    assert {float(r["y_true"]) for r in rows} == {1, 2, 3, 4}
    with pytest.raises(FileExistsError):
        write_predictions(trainer.model, trainer.task, validation, "cpu", path, provenance)


def test_metrics_weight_samples_not_batches():
    metric = RegressionMetrics()
    metric.update(torch.tensor([[0.0], [0.0], [0.0]]), torch.zeros(3, 1))
    metric.update(torch.tensor([[4.0]]), torch.zeros(1, 1))
    assert metric.compute() == {"mae": 1.0, "rmse": 2.0}


def test_all_steps_loss_has_direct_gradient_at_every_readout(config, batch):
    config.task.supervision = "all_steps"
    model = build_egnn(config.model, "mu")
    task = RegressionTask(config.task, 0, "D", TargetTransform())
    output = task.predict(model, batch, TraceSpec(predictions=True))
    for prediction in output.predictions_by_depth.values():
        prediction.retain_grad()
    result = task.run(lambda *args, **kwargs: output, batch, "train")
    expected = torch.stack(
        [(p - task.targets(batch)).abs().mean() for p in output.predictions_by_depth.values()]
    ).mean()
    torch.testing.assert_close(result.loss, expected)
    result.loss.backward()
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0 for p in output.predictions_by_depth.values()
    )


def test_checkpoint_rejects_incompatible_experiment(config, samples, tmp_path):
    trainer, train, validation = assemble(config, samples, tmp_path / "source")
    trainer.fit(train, validation, 1)
    config.task.supervision = "all_steps"
    other, _, _ = assemble(config, samples, tmp_path / "other")
    with pytest.raises(ValueError, match="mismatch: task"):
        other.restore(tmp_path / "source/checkpoints/last.pt")
