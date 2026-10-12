import csv
import json
import math
import runpy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from eqcompute.application import train
from eqcompute.data.QM9 import DatasetBundle
from eqcompute.evaluation import depthSweep
from eqcompute.evaluation.predictionWriter import FIELDS, file_hash
from eqcompute.factories import DATASETS


def make_run(root, depth, seed, val_mae, errors=(1.0, 3.0), cache=True):
    run = root / "runs" / f"k{depth}_s{seed}" / "run"
    (run / "checkpoints").mkdir(parents=True)
    metadata = {"run_id": f"k{depth}_s{seed}", "split_id": "same-split", "unit": "D"}
    checkpoint = {
        "format_version": 1, "best_value": val_mae, "epoch": 2, "metadata": metadata,
        "config": {
            "seed": seed, "model": {
                "name": "egnn", "execution": {"depth": depth}, "weight_tying": False,
            },
            "task": {"target": "mu", "supervision": "final"},
        },
    }
    torch.save(checkpoint, run / "checkpoints/best.pt")
    (run / "split_ids.json").write_text(json.dumps({
        "split_id": "same-split", "sample_ids": {"test": ["test:0", "test:1"]},
    }))
    # The last epoch is deliberately much worse; evaluation must use best.pt.
    (run / "metrics.csv").write_text("epoch,validation_mae\n4,999\n")
    rows = []
    for readout in range(1, depth + 1):
        for index, error in enumerate(errors):
            rows.append({
                "run_id": metadata["run_id"],
                "checkpoint_id": file_hash(run / "checkpoints/best.pt"),
                "split_id": "same-split", "split": "test", "sample_id": f"test:{index}",
                "property": "mu", "unit": "D", "train_depth": depth, "readout_depth": readout,
                "weight_tying": False, "supervision": "final", "seed": seed,
                "y_true": 0.0, "y_pred": error if readout == depth else 1000,
            })
    if cache:
        write_csv(run, rows)
    job = {
        "depth": depth, "seed": seed, "weight_tying": False,
        "status": "succeeded", "exit_code": 0,
        "run_dir": "/old/machine/moved/run",  # Must resolve the local sweep layout.
    }
    return job, run, rows


def write_csv(run, rows):
    destination = run / "predictions/test_best_by_depth.csv"
    destination.parent.mkdir(exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_jobs(root, jobs):
    (root / "jobs.json").write_text(json.dumps(jobs))


def no_inference(*args, **kwargs):
    pytest.fail("A matching complete prediction cache must not trigger inference")


def test_adaptive_checkpoint_is_not_reported_as_fixed(tmp_path, monkeypatch, capsys):
    job, run, _ = make_run(tmp_path, 2, 0, 1.0)
    path = run / "checkpoints/best.pt"
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["config"]["model"]["execution"]["name"] = "adaptive"
    torch.save(checkpoint, path)
    write_jobs(tmp_path, [job])
    monkeypatch.setattr(depthSweep, "predict_checkpoint", no_inference)
    _, skipped = depthSweep.summarize_sweep(tmp_path)
    assert skipped == 1
    assert "require fixed execution" in capsys.readouterr().err


def test_paired_seed_statistics_use_best_checkpoint_and_final_depth(tmp_path, monkeypatch):
    a, _, _ = make_run(tmp_path, 2, 0, 1.0, (0.0, 2.0))
    b, _, _ = make_run(tmp_path, 2, 1, 3.0, (4.0, 6.0))
    c, _, _ = make_run(tmp_path, 4, 0, 9.0, (2.0, 4.0))
    write_jobs(tmp_path, [c, b, a])
    monkeypatch.setattr(depthSweep, "predict_checkpoint", no_inference)
    table, skipped = depthSweep.summarize_sweep(tmp_path)
    assert skipped == 0
    assert "Target: mu | Unit: D" in table
    assert "2.000000 ± 1.414214" in table  # Sample SD, not population SD / SEM.
    assert "3.000000 ± 2.828427" in table  # Final-depth errors only.
    assert "9.000000 ± N/A" in table
    assert "3.000000 ± N/A" in table
    assert "2/2" in table and "1/1" in table
    lines = table.splitlines()
    assert lines[-2].startswith("2 ") and lines[-1].startswith("4 ")


@pytest.mark.parametrize("corruption", [
    "hash", "seed", "split", "missing_sample", "duplicate", "nan", "truncated", "missing",
])
def test_invalid_predictions_are_regenerated_once(tmp_path, monkeypatch, corruption):
    job, run, valid_rows = make_run(tmp_path, 2, 0, 0.5)
    rows = [row.copy() for row in valid_rows]
    if corruption == "hash":
        rows[0]["checkpoint_id"] = "old-checkpoint"
    elif corruption == "seed":
        rows[0]["seed"] = 99
    elif corruption == "split":
        rows[0]["split"] = "validation"
    elif corruption == "missing_sample":
        rows.pop()
    elif corruption == "duplicate":
        rows.append(rows[-1].copy())
    elif corruption == "nan":
        rows[-1]["y_pred"] = "nan"
    write_csv(run, rows)
    path = run / "predictions/test_best_by_depth.csv"
    if corruption == "truncated":
        path.write_text("sample_id,y_pred\ntest:0,1\n")
    elif corruption == "missing":
        path.unlink()
    calls = []

    def predict(checkpoint, split, device, batch_size, overwrite):
        calls.append((checkpoint, split, device, batch_size, overwrite))
        write_csv(run, valid_rows)

    monkeypatch.setattr(depthSweep, "predict_checkpoint", predict)
    write_jobs(tmp_path, [job])
    table, skipped = depthSweep.summarize_sweep(tmp_path, device="cpu", batch_size=8)
    assert skipped == 0 and "2.000000 ± N/A" in table
    assert calls == [(run / "checkpoints/best.pt", "test", "cpu", 8, corruption != "missing")]
    monkeypatch.setattr(depthSweep, "predict_checkpoint", no_inference)
    assert depthSweep.summarize_sweep(tmp_path)[1] == 0


def test_failures_exclude_both_metrics_and_report_empty_depth(tmp_path, monkeypatch, capsys):
    a, _, _ = make_run(tmp_path, 2, 0, 1.0)
    b, _, _ = make_run(tmp_path, 2, 1, 99.0, cache=False)
    c = {"depth": 4, "seed": 0, "weight_tying": False, "status": "failed", "exit_code": 1}
    write_jobs(tmp_path, [a, b, c])

    def fail(*args, **kwargs):
        raise RuntimeError("test inference failed")

    monkeypatch.setattr(depthSweep, "predict_checkpoint", fail)
    table, skipped = depthSweep.summarize_sweep(tmp_path)
    assert skipped == 2
    assert "1.000000 ± N/A" in table and "1/2" in table
    assert "99.000000" not in table
    assert "0/1" in table
    assert "test inference failed" in capsys.readouterr().err


def test_duplicate_seed_entries_are_rejected(tmp_path):
    job, _, _ = make_run(tmp_path, 2, 0, 1.0)
    write_jobs(tmp_path, [job, job])
    with pytest.raises(ValueError, match="Duplicate depth/seed"):
        depthSweep.summarize_sweep(tmp_path)


def test_mixed_protocols_are_rejected_before_inference(tmp_path, monkeypatch):
    a, _, _ = make_run(tmp_path, 2, 0, 1.0)
    b, run, _ = make_run(tmp_path, 2, 1, 1.0, cache=False)
    checkpoint = torch.load(run / "checkpoints/best.pt", weights_only=False)
    checkpoint["config"]["task"]["supervision"] = "all_steps"
    torch.save(checkpoint, run / "checkpoints/best.pt")
    write_jobs(tmp_path, [a, b])
    monkeypatch.setattr(depthSweep, "predict_checkpoint", no_inference)
    with pytest.raises(ValueError, match="Cannot combine"):
        depthSweep.summarize_sweep(tmp_path)


def test_missing_checkpoint_is_reported(tmp_path, capsys):
    job, run, _ = make_run(tmp_path, 2, 0, 1.0)
    (run / "checkpoints/best.pt").unlink()
    write_jobs(tmp_path, [job])
    table, skipped = depthSweep.summarize_sweep(tmp_path)
    assert skipped == 1 and "0/1" in table
    assert "No readable completed checkpoints" in table
    assert "best.pt" in capsys.readouterr().err


def test_cli_runs_real_checkpoint_prediction_and_then_reuses_cache(
    config, samples, tmp_path, monkeypatch, capsys,
):
    config.trainer.device = "cpu"
    config.trainer.epochs = 1
    config.model.execution.depth = 2
    config.model.weight_tying = False
    config.output_root = str(tmp_path / "runs/k2_s0")
    config.seed = 0
    manifest = {
        "split_id": "fixture-split", "sample_ids": {
            "train": [sample.sample_id for sample in samples[:2]],
            "validation": [samples[2].sample_id], "test": [samples[3].sample_id],
        },
    }
    bundle = DatasetBundle(samples[:2], samples[2:3], samples[3:], manifest, 0, "D")
    monkeypatch.setitem(DATASETS, "qm9", lambda *args: bundle)
    run = train(config)
    write_jobs(tmp_path, [{
        "depth": 2, "seed": 0, "weight_tying": False,
        "status": "succeeded", "exit_code": 0, "run_dir": str(run),
    }])
    cli = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/evalDepth.py"))
    assert cli["main"](["--sweep-dir", str(tmp_path), "--device", "cpu"]) == 0
    assert "Val MAE" in capsys.readouterr().out
    path = run / "predictions/test_best_by_depth.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(math.isfinite(float(row["y_pred"])) for row in rows)
    assert OmegaConf.load(run / "config.yaml").model.execution.depth == 2
    before = path.read_bytes()
    monkeypatch.setattr(depthSweep, "predict_checkpoint", no_inference)
    assert cli["main"](["--sweep-dir", str(tmp_path), "--device", "cpu"]) == 0
    assert path.read_bytes() == before
