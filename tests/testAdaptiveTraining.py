import csv
import math

import pytest
import torch
from omegaconf import OmegaConf

from eqcompute.application import make_loader, train
from eqcompute.data.QM9 import DatasetBundle
from eqcompute.factories import DATASETS
from eqcompute.metrics.regression import RegressionMetrics
from eqcompute.models.iterative import build_egnn
from eqcompute.models.trace import ModelOutput
from eqcompute.tasks.adaptiveRegression import AdaptiveRegressionTask, adaptive_loss
from eqcompute.tasks.targetTransform import TargetTransform
from eqcompute.training.checkpoint import load_checkpoint, seed_everything
from eqcompute.training.initialization import initialize_halting
from eqcompute.training.optim import build_optimizer
from eqcompute.training.trainer import Trainer


def assemble(config, samples, directory):
    seed_everything(config.seed)
    model = build_egnn(config.model, config.task.target)
    task = AdaptiveRegressionTask(config.task, 0, "D", TargetTransform(2.5, 2.0))
    task.configure_model(model)
    optimizer = build_optimizer(
        config.optimizer, (p for p in model.parameters() if p.requires_grad),
    )
    generator = torch.Generator().manual_seed(config.seed)
    directory.mkdir()
    trainer = Trainer(
        model, task, optimizer, RegressionMetrics, "cpu", directory,
        OmegaConf.to_container(config, resolve=True),
        {"run_id": "fixture", "split_id": "fixture", "dataset": "qm9", "property": "mu",
         "unit": "D", "target_convention": "raw"}, generator,
    )
    return (trainer, make_loader(samples, config.loader, True, generator),
            make_loader(samples, config.loader))


@pytest.mark.parametrize("phase", ["predictor", "halting"])
@pytest.mark.parametrize("pause_after", [1, 2])
@pytest.mark.parametrize("beta", [0.0, 0.01])
def test_resume_each_phase(adaptive_config, samples, tmp_path, phase, pause_after, beta):
    adaptive_config.task.phase = phase
    adaptive_config.task.entropy_beta = beta
    adaptive_config.loader.batch_size = 3
    full, training, validation = assemble(adaptive_config, samples, tmp_path / "full")
    full.fit(training, validation, 3)
    part, training, validation = assemble(adaptive_config, samples, tmp_path / "part")
    part.fit(training, validation, pause_after)
    assert (tmp_path / "part/checkpoints/best.pt").exists()
    resumed, training, validation = assemble(adaptive_config, samples, tmp_path / "resumed")
    resumed.restore(tmp_path / "part/checkpoints/last.pt")
    resumed.fit(training, validation, 3)
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    assert full.best_value == resumed.best_value
    saved = load_checkpoint(tmp_path / "resumed/checkpoints/last.pt")
    assert saved["task"] == resumed.task.state_dict()
    assert saved["task"]["phase"] == phase
    with (tmp_path / "full/metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert full.best_value == min(float(row["validation_mae"]) for row in rows)
    for stage in ("train", "validation"):
        for row in rows:
            assert float(row[f"{stage}_loss"]) == pytest.approx(
                float(row[f"{stage}_prediction_loss"]) + float(row[f"{stage}_entropy_loss"]),
            )
            if phase == "predictor":
                assert float(row[f"{stage}_entropy_loss"]) == 0
            else:
                assert float(row[f"{stage}_entropy_loss"]) == pytest.approx(
                    -beta * float(row[f"{stage}_halting_entropy"]), abs=1e-8,
                )
    if phase == "predictor":
        assert "validation_mean_depth_mae" in rows[0]
        assert "validation_final_depth_mae" in rows[0]


@pytest.mark.parametrize("phase", ["predictor", "halting"])
def test_only_selected_parameters_update(adaptive_config, batch, phase):
    adaptive_config.task.phase = phase
    model = build_egnn(adaptive_config.model, "mu")
    task = AdaptiveRegressionTask(adaptive_config.task, 0, "D", TargetTransform())
    task.configure_model(model)
    optimizer = build_optimizer(adaptive_config.optimizer,
                                (p for p in model.parameters() if p.requires_grad))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    observed = set()
    for _ in range(40):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = task.run(model, batch, "train")
        observed.add(result.statistics["executed_depth"])
        result.loss.backward()
        for parameter in model.parameters():
            if not parameter.requires_grad:
                assert parameter.grad is None
        optimizer.step()
    changed = []
    for key, value in model.state_dict().items():
        controller = key.startswith("execution.controller.")
        frozen = controller if phase == "predictor" else not controller
        if frozen:
            torch.testing.assert_close(before[key], value, rtol=0, atol=0)
        else:
            changed.append(not torch.equal(before[key], value))
    assert any(changed)
    assert observed == ({1, 2, 3, 4} if phase == "predictor" else {4})
    if phase == "halting":
        assert not model.operators.training
        assert model.execution.controller.training


def test_predictor_batch_shares_one_depth_and_validation_skips_controller(
    adaptive_config, batch, monkeypatch,
):
    model = build_egnn(adaptive_config.model, "mu")
    task = AdaptiveRegressionTask(adaptive_config.task, 0, "D", TargetTransform())
    task.configure_model(model)

    def forbidden(*args):
        pytest.fail("Predictor training/validation must not call the controller")

    monkeypatch.setattr(model.execution.controller, "forward", forbidden)
    for _ in range(10):
        output = task.predict(model, batch, execution_mode="random_depth")
        assert output.adaptive.executed_depth.unique().numel() == 1
        assert output.adaptive.halt_depth.equal(output.adaptive.executed_depth)
    model.eval()
    with torch.no_grad():
        result = task.run(model, batch, "validation")
    assert result.statistics["executed_depth"] == 4
    assert result.statistics["entropy_loss"] == 0


def test_oracle_metrics_denormalization_ties_and_uneven_batches(
    adaptive_config, samples, tmp_path, monkeypatch,
):
    adaptive_config.loader.batch_size = 3
    trainer, _, validation = assemble(adaptive_config, samples, tmp_path / "oracle")
    errors = torch.tensor([[1., -1., 3., 4.], [5., 0., -2., 3.],
                           [4., 3., -1., 2.], [2., 4., 6., 8.]])

    def predictions(model, batch, trace=None, *, execution_mode=None):
        assert execution_mode == "predictor_full"
        assert trace.predictions
        target = trainer.task.targets(batch)
        ids = target[:, 0].long() - 1
        values = trainer.task.transform.encode(target + errors[ids])
        return ModelOutput(values[:, -1:], {k + 1: values[:, k:k + 1] for k in range(4)})

    monkeypatch.setattr(trainer.task, "predict", predictions)
    metrics = trainer._epoch(validation, False)
    assert metrics["mae"] == pytest.approx(1.)
    assert metrics["rmse"] == pytest.approx(math.sqrt(1.5))
    assert metrics["oracle_depth"] == pytest.approx(1.75)
    assert metrics["mean_depth_mae"] == pytest.approx(errors.abs().mean().item())
    assert metrics["final_depth_mae"] == pytest.approx(errors[:, -1].abs().mean().item())
    assert metrics["loss"] == pytest.approx(errors.abs().mean().item() / 2.)


@pytest.mark.parametrize("beta", [0., 0.01])
def test_halting_loss_and_cdf_metric(adaptive_config, batch, beta):
    adaptive_config.task.phase = "halting"
    adaptive_config.task.entropy_beta = beta
    model = build_egnn(adaptive_config.model, "mu")
    task = AdaptiveRegressionTask(adaptive_config.task, 0, "D", TargetTransform(2.5, 2.))
    task.configure_model(model)
    result = task.run(model, batch, "train")
    output = task.predict(model, batch, execution_mode="full")
    p = output.adaptive.stop_probabilities
    expected = adaptive_loss(torch.stack(list(output.predictions_by_depth.values()), 1),
                             task.transform.encode(task.targets(batch)), p)
    expected = expected + beta * (p * p.clamp_min(1e-12).log()).sum(1).mean()
    torch.testing.assert_close(result.loss, expected)
    torch.testing.assert_close(result.prediction, task.transform.decode(output.prediction))
    parameters = tuple(model.execution.controller.parameters())
    actual = torch.autograd.grad(result.loss, parameters)
    reference = torch.autograd.grad(expected, parameters)
    for left, right in zip(actual, reference, strict=True):
        torch.testing.assert_close(left, right)
    model.eval()
    with torch.no_grad():
        validation = task.run(model, batch, "validation")
        hard = task.predict(model, batch, execution_mode="hard")
    torch.testing.assert_close(validation.prediction, task.transform.decode(hard.prediction),
                               rtol=1e-5, atol=1e-5)


def test_entropy_updates_controller_toward_uniform_with_equal_errors(adaptive_config, batch):
    adaptive_config.task.phase = "halting"
    adaptive_config.model.execution.depth = 2
    model = build_egnn(adaptive_config.model, "mu")
    task = AdaptiveRegressionTask(adaptive_config.task, 0, "D", TargetTransform())
    task.configure_model(model)
    gate = model.execution.controller.network[-1]
    batch.target.zero_()
    with torch.no_grad():
        for parameter in model.head.parameters():
            parameter.zero_()
        gate.weight.zero_()
        gate.bias.fill_(math.log(4))
    optimizer = torch.optim.SGD(model.execution.controller.parameters(), lr=1.0)
    before = task.run(model, batch, "train")
    assert before.statistics["prediction_loss"] == 0
    assert before.loss.item() < 0
    before.loss.backward()
    torch.testing.assert_close(gate.bias.grad,
                               torch.full_like(gate.bias, 0.01 * 0.8 * 0.2 * math.log(4)),
                               rtol=1e-4, atol=1e-7)
    optimizer.step()
    assert task.run(model, batch, "train").statistics["halting_entropy"] > (
        before.statistics["halting_entropy"]
    )


@pytest.mark.parametrize("value", ["joint", "warmup", "", None])
def test_invalid_phase(adaptive_config, value):
    adaptive_config.task.phase = value
    with pytest.raises(ValueError, match="phase"):
        AdaptiveRegressionTask(adaptive_config.task, 0, "D", TargetTransform())


@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf"), -float("inf"),
                                  True, False, "0.01", None])
def test_invalid_entropy_beta(adaptive_config, value):
    adaptive_config.task.entropy_beta = value
    with pytest.raises(ValueError, match="entropy_beta"):
        AdaptiveRegressionTask(adaptive_config.task, 0, "D", TargetTransform())


@pytest.mark.parametrize("key,value", [("phase", "halting"), ("entropy_beta", 0.1)])
def test_task_state_rejects_changes(adaptive_config, key, value):
    task = AdaptiveRegressionTask(adaptive_config.task, 0, "D", TargetTransform())
    state = task.state_dict()
    state[key] = value
    with pytest.raises(ValueError, match=key):
        task.load_state_dict(state)


def test_cross_stage_initialization(adaptive_config, samples, tmp_path):
    source, training, validation = assemble(adaptive_config, samples, tmp_path / "source")
    source.fit(training, validation, 1)
    adaptive_config.task.phase = "halting"
    destination, _, _ = assemble(adaptive_config, samples, tmp_path / "destination")
    controller = {
        k: v.clone() for k, v in destination.model.execution.controller.state_dict().items()
    }
    path = tmp_path / "source/checkpoints/best.pt"
    # Deliberately different controller proves that its source weights are ignored.
    saved = load_checkpoint(path)
    for key in saved["model"]:
        if key.startswith("execution.controller."):
            saved["model"][key].fill_(42.)
    torch.save(saved, path)
    provenance = initialize_halting(path, destination.model, destination.task, adaptive_config,
                                    destination.metadata)
    for key, value in destination.model.state_dict().items():
        if not key.startswith("execution.controller."):
            torch.testing.assert_close(value, saved["model"][key], rtol=0, atol=0)
    for key, value in controller.items():
        assert torch.equal(value, destination.model.execution.controller.state_dict()[key])
    assert destination.task.transform.state_dict() == saved["task"]["transform"]
    assert destination.optimizer.state_dict()["state"] == {}
    assert destination.start_epoch == 0 and destination.best_value == float("inf")
    assert provenance["init_from"] == str(path.resolve())
    assert len(provenance["init_from_sha256"]) == 64


@pytest.mark.parametrize("section,key,value", [
    ("task", "phase", "halting"), ("task", "normalization", "standard"),
    ("task", "target", "alpha"), ("model", "hidden_dim", 99),
    ("metadata", "split_id", "other"), ("metadata", "unit", "eV"),
])
def test_initialization_rejects_mismatches(adaptive_config, samples, tmp_path, section, key, value):
    source, training, validation = assemble(adaptive_config, samples, tmp_path / "source")
    source.fit(training, validation, 1)
    path = tmp_path / "source/checkpoints/last.pt"
    saved = load_checkpoint(path)
    (saved["metadata"] if section == "metadata" else saved["config"][section])[key] = value
    torch.save(saved, path)
    adaptive_config.task.phase = "halting"
    destination, _, _ = assemble(adaptive_config, samples, tmp_path / "destination")
    with pytest.raises(ValueError):
        initialize_halting(path, destination.model, destination.task, adaptive_config,
                           destination.metadata)


@pytest.mark.parametrize("halting_loss", ["expected", "soft_cdf"])
def test_two_stage_application_and_resume(
    adaptive_config, samples, tmp_path, monkeypatch, halting_loss,
):
    config = adaptive_config
    config.output_root = str(tmp_path)
    # A fixed scheduler horizon permits resuming from epoch 1 to epoch 2.
    config.trainer.scheduler.t_max = 3
    config.trainer.epochs = 1
    bundle = DatasetBundle(samples[:2], samples[2:], samples[2:],
                           {"split_id": "fixture", "dataset": "qm9"}, 0, "D")
    monkeypatch.setitem(DATASETS, "qm9", lambda *args: bundle)
    source = train(config)
    config.task.phase = "halting"
    config.task.halting_loss = halting_loss
    with pytest.raises(ValueError, match="requires init_from or resume"):
        train(config)
    config.init_from = str(source / "checkpoints/best.pt")
    destination = train(config)
    saved = load_checkpoint(destination / "checkpoints/last.pt")
    assert saved["task"]["halting_loss"] == halting_loss
    original = load_checkpoint(config.init_from)
    for key, value in original["model"].items():
        if not key.startswith("execution.controller."):
            assert torch.equal(value, saved["model"][key])
    assert saved["metadata"]["init_from"] == config.init_from
    config.resume = str(destination / "checkpoints/last.pt")
    with pytest.raises(ValueError, match="mutually exclusive"):
        train(config)
    config.init_from = None
    config.trainer.epochs = 2
    resumed = train(config)
    saved_resume = load_checkpoint(resumed / "checkpoints/last.pt")
    assert saved_resume["epoch"] == 1
    assert saved_resume["metadata"]["init_from"] == saved["metadata"]["init_from"]
    config.task.phase = "predictor"
    with pytest.raises(ValueError, match="configuration mismatch: task"):
        train(config)
