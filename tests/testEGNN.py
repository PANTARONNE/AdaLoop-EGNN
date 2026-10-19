import csv
import importlib.util
import json
import math
import runpy
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from eqcompute.application import train
from eqcompute.contracts import PropertyContext
from eqcompute.evaluation.depthSweep import summarize_sweep
from eqcompute.evaluation.predict import predict_checkpoint
from eqcompute.models.execution import FixedDepthExecution
from eqcompute.models.iterative import build_egnn
from eqcompute.models.trace import TraceSpec
from eqcompute.tasks.targetTransform import TargetTransform
from eqcompute.training.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("normalization", ["mad", "standard"])
@pytest.mark.parametrize("preset", [
    "qm9_loop_egnn_diagnostic", "qm9_adaloop_egnn_smoke",
])
def test_pyg_presets_train_and_predict(samples, tmp_path, monkeypatch, normalization, preset):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        config = compose(config_name="config", overrides=[f"experiment={preset}"])
    assert config.dataset.name == "qm9"
    config.dataset.root = str(tmp_path / "local_qm9")
    config.dataset.train_size, config.dataset.val_size = 2, 1
    config.dataset.test_size, config.dataset.subset_size = 1, 4
    source = [sample.clone() for sample in samples]
    for index, sample in enumerate(source):
        sample.idx = index
    monkeypatch.setattr("eqcompute.data.QM9.QM9", lambda root: source)
    config.output_root = str(tmp_path / "runs")
    config.trainer.device = "cpu"
    config.trainer.epochs = 2
    config.loader.batch_size = 4
    config.model.hidden_dim = 8
    config.model.execution.depth = 3
    config.task.normalization = normalization
    output = train(config)
    checkpoint = load_checkpoint(output / "checkpoints/best.pt")
    assert checkpoint["config"]["task"]["normalization"] == normalization
    assert checkpoint["metadata"]["dataset"] == "qm9_pyg"
    assert checkpoint["metadata"]["target_convention"] == "raw"
    assert checkpoint["metadata"]["split_id"]
    assert checkpoint["task"]["unit"] == "D"
    path, count = predict_checkpoint(output / "checkpoints/best.pt", device="cpu")
    adaptive = config.model.execution.name == "adaptive"
    assert count == (1 if adaptive else 3)
    assert path.name == ("test_best_adaptive.csv" if adaptive else "test_best_by_depth.csv")
    if adaptive:
        path, count = predict_checkpoint(
            output / "checkpoints/best.pt", device="cpu", mode="diagnostic"
        )
        assert count == 3
        assert path.name == "test_best_diagnostic_by_depth.csv"


@pytest.fixture
def egnn_config():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(config_name="config", overrides=[
            "experiment=qm9_egnn",
        ])


def reference_class(monkeypatch):
    """Load untouched upstream snapshots without installing their project."""
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, ROOT / "tests/reference" / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    with monkeypatch.context() as context:
        context.setitem(sys.modules, "models", types.ModuleType("models"))
        context.setitem(sys.modules, "models.gcl", load("models.gcl", "GCL.py"))
        return load("upstream_qm9", "QM9.py").EGNN


def parameter_pairs(model, reference):
    modules = [
        (model.initializer.embedding, reference.embedding),
        (model.readout.node_dec, reference.node_dec),
        (model.head.network, reference.graph_dec),
    ] + [
        (step, getattr(reference, f"gcl_{i}")) for i, step in enumerate(model.operators.steps)
    ]
    pairs = []
    for ours, upstream in modules:
        source = dict(upstream.named_parameters())
        assert set(dict(ours.named_parameters())) == set(source)
        pairs.extend((param, source[name]) for name, param in ours.named_parameters())
    return pairs


def padded_inputs(samples, dtype):
    batch_size, size = len(samples), max(len(sample.z) for sample in samples)
    charges = torch.zeros(batch_size, size, dtype=torch.long)
    positions = torch.zeros(batch_size, size, 3, dtype=dtype)
    for i, sample in enumerate(samples):
        charges[i, : len(sample.z)] = sample.z
        positions[i, : len(sample.z)] = sample.pos
    mask = charges > 0
    edge_mask = mask[:, :, None] * mask[:, None, :]
    edge_mask *= ~torch.eye(size, dtype=torch.bool)[None]
    edges = torch.tensor([
        [b * size + i, b * size + j]
        for b in range(batch_size) for i in range(size) for j in range(size)
    ]).T
    one_hot = charges[:, :, None] == torch.tensor([1, 6, 7, 8, 9])
    powers = (charges.to(dtype)[:, :, None] / 9).pow(torch.arange(3))
    features = (one_hot[:, :, :, None] * powers[:, :, None, :]).reshape(-1, 15)
    return dict(
        h0=features, x=positions.reshape(-1, 3), edges=edges,
        edge_attr=None, node_mask=mask.reshape(-1, 1).to(dtype),
        edge_mask=edge_mask.reshape(-1, 1).to(dtype), n_nodes=size,
    )


@pytest.mark.parametrize("depth", [1, 3, 7])
def test_upstream_prediction_gradients_and_adam_update_without_message_norm(
    egnn_config, batch, samples, monkeypatch, depth
):
    egnn_config.model.hidden_dim = 16
    egnn_config.model.execution.depth = depth
    model = build_egnn(egnn_config.model, "mu").double()
    # The upstream oracle has no message normalization.
    for step in model.operators.steps:
        step.message_norm = torch.nn.Identity()
    reference = reference_class(monkeypatch)(
        15, 0, 16, n_layers=depth, attention=True, node_attr=0
    ).double()
    pairs = parameter_pairs(model, reference)
    with torch.no_grad():
        for ours, upstream in pairs:
            ours.copy_(upstream)
    graph = replace(batch.graph, positions=batch.graph.positions.double())
    reference_inputs = padded_inputs(samples, torch.float64)
    pred = model(graph, PropertyContext("mu")).prediction[:, 0]
    expected = reference(**reference_inputs)
    torch.testing.assert_close(pred, expected, atol=1e-11, rtol=1e-10)
    target = torch.arange(len(samples), dtype=torch.float64)
    (pred - target).abs().mean().backward()
    (expected - target).abs().mean().backward()
    for ours, upstream in pairs:
        torch.testing.assert_close(ours.grad, upstream.grad, atol=1e-11, rtol=1e-9)
    for parameters in (model.parameters(), reference.parameters()):
        torch.optim.Adam(parameters, lr=0.0005, weight_decay=1e-16).step()
    for ours, upstream in pairs:
        torch.testing.assert_close(ours, upstream, atol=1e-11, rtol=1e-9)


@pytest.mark.parametrize("tied", [False, True])
def test_egnn_symmetry_trace_and_batch_isolation(egnn_config, batch, samples, tied):
    from eqcompute.data.collate import collate_molecules

    egnn_config.model.hidden_dim = 8
    egnn_config.model.weight_tying = tied
    model = build_egnn(egnn_config.model, "mu").double()
    graph = replace(batch.graph, positions=batch.graph.positions.double())
    context = PropertyContext("mu")
    spec = TraceSpec(predictions=True, positions=True)
    baseline = model(graph, context, trace=spec)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    transformed = replace(graph, positions=graph.positions @ rotation + 2)
    torch.testing.assert_close(baseline.prediction, model(transformed, context).prediction)
    permutation = torch.randperm(len(graph.positions))
    permuted = replace(
        graph, positions=graph.positions[permutation],
        atomic_numbers=graph.atomic_numbers[permutation],
        batch_index=graph.batch_index[permutation],
    )
    torch.testing.assert_close(baseline.prediction, model(permuted, context).prediction)
    alone = collate_molecules(samples[:1]).graph
    alone = replace(alone, positions=alone.positions.double())
    torch.testing.assert_close(baseline.prediction[:1], model(alone, context).prediction)
    for state in baseline.trace.values():
        torch.testing.assert_close(state["positions"], graph.positions, rtol=0, atol=0)
    assert len(baseline.predictions_by_depth) == 7


def test_tied_egnn_matches_identical_untied_layers_and_accumulates_gradients(egnn_config, batch):
    egnn_config.model.hidden_dim = 8
    egnn_config.model.execution.depth = 3
    untied = build_egnn(egnn_config.model, "mu")
    egnn_config.model.weight_tying = True
    tied = build_egnn(egnn_config.model, "mu")
    tied.initializer.load_state_dict(untied.initializer.state_dict())
    tied.readout.load_state_dict(untied.readout.state_dict())
    tied.head.load_state_dict(untied.head.state_dict())
    tied_step = tied.operators.at(1)
    tied_step.load_state_dict(untied.operators.at(1).state_dict())
    for step in untied.operators.steps:
        step.load_state_dict(tied_step.state_dict())
    assert len(tied.operators.steps) == 1
    assert tied.operators.at(1) is tied.operators.at(3)
    assert untied.operators.at(1) is not untied.operators.at(3)
    parameter_count = sum(p.numel() for p in tied.parameters())
    assert parameter_count < sum(p.numel() for p in untied.parameters())
    context = PropertyContext("mu")
    a = tied(batch.graph, context, trace=TraceSpec(predictions=True))
    b = untied(batch.graph, context, trace=TraceSpec(predictions=True))
    for depth in (1, 2, 3):
        torch.testing.assert_close(a.predictions_by_depth[depth], b.predictions_by_depth[depth])
    a.prediction.square().mean().backward()
    b.prediction.square().mean().backward()
    separate = [dict(step.named_parameters()) for step in untied.operators.steps]
    for name, parameter in tied_step.named_parameters():
        torch.testing.assert_close(parameter.grad, sum(step[name].grad for step in separate))
        assert torch.isfinite(parameter.grad).all()
    tied.execution = FixedDepthExecution(7)
    assert sum(p.numel() for p in tied.parameters()) == parameter_count
    deeper = tied(batch.graph, context, trace=TraceSpec(predictions=True))
    assert len(deeper.predictions_by_depth) == 7


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_egnn_rejects_non_boolean_tying(egnn_config, value):
    egnn_config.model.weight_tying = value
    with pytest.raises(ValueError, match="must be a boolean"):
        build_egnn(egnn_config.model, "mu")


@pytest.fixture
def local_pyg_qm9(egnn_config, samples, tmp_path, monkeypatch):
    """Exercise the retained PyG adapter without downloading or changing real data."""
    source = [sample.clone() for sample in samples]
    for index, sample in enumerate(source):
        sample.idx = index
    monkeypatch.setattr("eqcompute.data.QM9.QM9", lambda root: source)
    egnn_config.dataset.train_size = 2
    egnn_config.dataset.val_size = 1
    egnn_config.dataset.test_size = 1
    egnn_config.dataset.subset_size = 4
    return tmp_path / "local_qm9"


def test_train_only_mad():
    values = torch.tensor([0.0, 0.0, 9.0])
    transform = TargetTransform.fit(values, "mad")
    assert transform.mean == 3 and transform.scale == 4
    held_out = torch.tensor([500.0])
    torch.testing.assert_close(transform.decode(transform.encode(held_out)), held_out)
    assert transform.scale == 4


@pytest.mark.parametrize("preset,tied", [
    ("qm9_egnn", False), ("qm9_loop_egnn", True),
])
def test_egnn_config_and_sweep_plan(preset, tied):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        preset_config = compose(config_name="config", overrides=[f"experiment={preset}"])
    assert preset_config.dataset.name == "qm9"
    assert preset_config.dataset.root == "data/qm9"
    assert preset_config.model.weight_tying is tied
    assert preset_config.model.execution.depth == 7
    assert f"tied{tied}_k7_final" in preset_config.experiment_name
    cli = runpy.run_path(str(ROOT / "scripts/sweepDepth.py"))
    args = cli["parse_args"]([
        "--gpus", "0", "--depths", "2,7", "--seeds", "0,1", "--weight-tying", str(tied).lower(),
        "--", f"experiment={preset}", "task.target=homo",
    ])
    jobs, _, _, config = cli["make_plan"](args)
    assert len(jobs) == 4
    assert config.optimizer.learning_rate == 0.001
    config.task.target = "mu"
    assert config.optimizer.learning_rate == 0.0005
    assert config.loader.batch_size == 96
    assert config.trainer.scheduler.t_max == 1000
    assert config.trainer.gradient_clip == 10.0
    assert config.task.normalization == "mad"
    assert config.model.weight_tying is tied
    assert all(job.parameters["weight_tying"] is tied for job in jobs)
    model = build_egnn(config.model, "mu")
    assert (model.operators.at(1) is model.operators.at(config.model.execution.depth)) is tied


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable"
    )),
])
@pytest.mark.parametrize("tied", [False, True])
def test_egnn_training_prediction_sweep_and_scheduler(
    egnn_config, local_pyg_qm9, tmp_path, device, tied
):
    egnn_config.dataset.root = str(local_pyg_qm9)
    egnn_config.output_root = str(tmp_path / "outputs")
    egnn_config.trainer.device = device
    egnn_config.trainer.epochs = 2
    egnn_config.loader.batch_size = 4
    egnn_config.model.hidden_dim = 8
    egnn_config.model.execution.depth = 2
    egnn_config.model.weight_tying = tied
    run = train(egnn_config)
    checkpoint = load_checkpoint(run / "checkpoints/last.pt")
    assert checkpoint["scheduler"]["last_epoch"] == 2
    assert checkpoint["optimizer"]["param_groups"][0]["lr"] == 0
    assert checkpoint["metadata"]["target_convention"] == "raw"
    with (run / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["learning_rate"]) for row in rows] == [0.0005, 0.00025]
    path, count = predict_checkpoint(run / "checkpoints/best.pt", device="cpu")
    assert count == 2  # One test molecule, two readout depths.
    with path.open() as handle:
        predictions = list(csv.DictReader(handle))
    assert all(row["unit"] == "D" for row in predictions)
    assert all(row["weight_tying"] == str(tied) for row in predictions)
    assert all(math.isfinite(float(row["y_pred"])) for row in predictions)
    jobs = [{
        "depth": 2, "seed": 0, "weight_tying": tied, "status": "succeeded",
        "exit_code": 0, "run_dir": str(run),
    }]
    (tmp_path / "jobs.json").write_text(json.dumps(jobs))
    table, skipped = summarize_sweep(tmp_path, device="cpu")
    assert skipped == 0 and "Model: egnn" in table and "1/1" in table
    assert f"weight_tying: {str(tied).lower()}" in table
    # Evaluation must not advance the saved scheduler.
    assert load_checkpoint(run / "checkpoints/last.pt")["scheduler"] == checkpoint["scheduler"]


def test_tied_egnn_checkpoint_resume(egnn_config, local_pyg_qm9, tmp_path):
    egnn_config.dataset.root = str(local_pyg_qm9)
    egnn_config.output_root = str(tmp_path / "outputs")
    egnn_config.trainer.device = "cpu"
    egnn_config.trainer.epochs = 3
    egnn_config.trainer.scheduler.t_max = 3
    egnn_config.loader.batch_size = 4
    egnn_config.model.hidden_dim = 8
    egnn_config.model.execution.depth = 3
    egnn_config.model.weight_tying = True
    full = train(egnn_config)
    egnn_config.trainer.epochs = 1
    part = train(egnn_config)
    egnn_config.trainer.epochs = 3
    egnn_config.resume = str(part / "checkpoints/last.pt")
    resumed = train(egnn_config)
    a = load_checkpoint(full / "checkpoints/last.pt")
    b = load_checkpoint(resumed / "checkpoints/last.pt")
    for name, value in a["model"].items():
        torch.testing.assert_close(value, b["model"][name], rtol=0, atol=0)
    assert a["scheduler"] == b["scheduler"]
    assert a["best_value"] == b["best_value"]


def test_cosine_resume_matches_uninterrupted(config, samples, tmp_path):
    from testTraining import assemble

    config.loader.batch_size = 2
    config.trainer.gradient_clip = None
    OmegaConf.update(
        config, "trainer.scheduler", {"name": "cosine", "t_max": 3, "eta_min": 0.0}, force_add=True
    )
    full, training, validation = assemble(config, samples, tmp_path / "full")
    full.fit(training, validation, 3)
    part, training, validation = assemble(config, samples, tmp_path / "part")
    part.fit(training, validation, 1)
    resumed, training, validation = assemble(config, samples, tmp_path / "resumed")
    resumed.restore(tmp_path / "part/checkpoints/last.pt")
    resumed.fit(training, validation, 3)
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    assert resumed.scheduler.state_dict() == full.scheduler.state_dict()
    assert resumed.best_value == full.best_value
    config.trainer.scheduler.t_max = 4
    other, _, _ = assemble(config, samples, tmp_path / "incompatible")
    with pytest.raises(ValueError, match="scheduler mismatch"):
        other.restore(tmp_path / "part/checkpoints/last.pt")
