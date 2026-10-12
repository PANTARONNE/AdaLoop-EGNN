import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from eqcompute.analysis.complexity import graph_complexity, write_complexity
from eqcompute.analysis.oracleDepth import oracle_depths, write_oracles
from eqcompute.application import make_loader
from eqcompute.evaluation.predictionWriter import file_hash, write_predictions
from eqcompute.models.iterative import build_egnn_orig
from eqcompute.tasks.regression import RegressionTask
from eqcompute.tasks.targetTransform import TargetTransform
from eqcompute.training.checkpoint import save_checkpoint


def test_diagnostic_preset_all_steps_gradients_and_orig_defaults(batch):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.3"
    ):
        config = compose(config_name="config", overrides=[
            "experiment=qm9_loop_egnn_orig_diagnostic", "model.hidden_dim=8",
            "model.execution.depth=3",
        ])
        original = compose(config_name="config", overrides=["experiment=qm9_egnn_orig"])
    assert original.task.supervision == "final"
    assert original.task.normalization == config.task.normalization == "mad"
    assert config.model.weight_tying
    model = build_egnn_orig(config.model, "mu")
    task = RegressionTask(config.task, 0, "D", TargetTransform())
    from eqcompute.models.trace import TraceSpec

    output = task.predict(model, batch, TraceSpec(predictions=True))
    for value in output.predictions_by_depth.values():
        value.retain_grad()
    result = task.run(lambda *a, **kw: output, batch, "train")
    result.loss.backward()
    assert all(value.grad is not None and value.grad.abs().sum() > 0
               for value in output.predictions_by_depth.values())


def test_oracle_ties_and_cost():
    assert oracle_depths([0.4, 0.1, 0.1]) == (2, 2)
    assert oracle_depths([0.2, 0.1, 0.0], 0.9) == (3, 1)
    assert oracle_depths([2.0], 1.0) == (1, 1)
    with pytest.raises(ValueError):
        oracle_depths([float("nan")])


def test_oracle_export_scale_provenance_missing_and_duplicate(
    adaptive_config, samples, tmp_path
):
    config = adaptive_config
    config.model.execution.name = "fixed"
    config.task.name = "scalar_regression"
    config.task.supervision = "all_steps"
    model = build_egnn_orig(config.model, "mu")
    task = RegressionTask(config.task, 0, "D", TargetTransform(2., 2.))
    checkpoint = tmp_path / "checkpoints/best.pt"
    save_checkpoint(checkpoint, {
        "format_version": 1, "config": OmegaConf.to_container(config, resolve=True),
        "task": task.state_dict(), "metadata": {"split_id": "split"},
    })
    (tmp_path / "split_ids.json").write_text(json.dumps({
        "split_id": "split", "sample_ids": {"validation": [s.sample_id for s in samples]},
    }))
    source, destination = tmp_path / "depth.csv", tmp_path / "oracle.csv"
    provenance = dict(run_id="r", checkpoint_id=file_hash(checkpoint), split_id="split",
                      split="validation", train_depth=4, weight_tying=True,
                      supervision="all_steps", seed=0)
    write_predictions(model, task, make_loader(samples, config.loader), "cpu", source, provenance)
    assert write_oracles(source, checkpoint, destination, 0.03) == 4
    with destination.open() as handle:
        results = list(csv.DictReader(handle))
    assert all(float(row["target_scale"]) == 2 for row in results)
    with source.open() as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    for record in results:
        errors = [abs(float(row["y_pred"]) - float(row["y_true"])) / 2
                  for row in rows if row["sample_id"] == record["sample_id"]]
        assert (int(record["oracle_depth"]), int(record["cost_oracle_depth"])) == oracle_depths(
            errors, 0.03
        )
    for invalid in (
        rows[:-1], [rows[0], *rows],
        [row for row in rows if row["sample_id"] != samples[-1].sample_id],
    ):
        with source.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows(invalid)
        previous = destination.read_bytes()
        with pytest.raises(ValueError):
            write_oracles(source, checkpoint, destination, overwrite=True)
        assert destination.read_bytes() == previous
    rows[0]["checkpoint_id"] = "wrong"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="provenance"):
        write_oracles(source, checkpoint, destination, overwrite=True)


def test_complexity_known_geometry_and_invariance(batch, samples, adaptive_config, tmp_path):
    graph = replace(batch.graph, positions=batch.graph.positions.double())
    records = graph_complexity(graph)
    assert [r["atom_count"] for r in records] == [3, 5, 1, 4]
    assert [r["heavy_atom_count"] for r in records] == [2, 4, 0, 3]
    assert records[2]["radius_gyration"] == records[2]["element_entropy"] == 0
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.double))
    changed = graph_complexity(replace(graph, positions=graph.positions @ rotation + 12))
    for before, after in zip(records, changed):
        assert before == pytest.approx(after)
    loader = make_loader(samples, adaptive_config.loader)
    destination = tmp_path / "complexity.csv"
    assert write_complexity(loader, destination, {"split_id": "x", "split": "test"}) == 4
    with pytest.raises(FileExistsError):
        write_complexity(loader, destination, {"split_id": "x", "split": "test"})
