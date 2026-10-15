import csv

import pytest
import torch

from eqcompute.application import make_loader
from eqcompute.evaluation.adaptiveWriter import write_adaptive_predictions
from eqcompute.models.iterative import build_egnn_orig
from eqcompute.tasks.adaptiveRegression import AdaptiveRegressionTask
from eqcompute.tasks.targetTransform import TargetTransform


def test_hard_and_diagnostic_exports(adaptive_config, samples, tmp_path):
    model = build_egnn_orig(adaptive_config.model, "mu")
    task = AdaptiveRegressionTask(adaptive_config.task, 0, "D", TargetTransform(2.5, 2.))
    loader = make_loader(samples, adaptive_config.loader)
    provenance = dict(run_id="r", checkpoint_id="hash", split_id="split", split="test",
                      train_depth=4, weight_tying=True, supervision="adaptive", seed=0)
    hard, full = tmp_path / "hard.csv", tmp_path / "full.csv"
    assert write_adaptive_predictions(model, task, loader, "cpu", hard, provenance) == 4
    assert write_adaptive_predictions(
        model, task, loader, "cpu", full, provenance, mode="diagnostic"
    ) == 16
    with hard.open() as handle:
        hard_rows = list(csv.DictReader(handle))
    with full.open() as handle:
        full_rows = list(csv.DictReader(handle))
    assert "lambda_compute" not in hard_rows[0]
    assert "lambda_compute" not in full_rows[0]
    assert all(float(row["expected_depth"]) >= 1 for row in full_rows)
    for row in hard_rows:
        trajectory = [r for r in full_rows if r["sample_id"] == row["sample_id"]]
        chosen = trajectory[int(row["halt_depth"]) - 1]
        assert float(row["y_pred"]) == pytest.approx(float(chosen["y_pred"]), abs=1e-5)
        assert row["executed_depth"] == row["halt_depth"]
        assert all(int(r["executed_depth"]) == 4 for r in trajectory)
        assert sum(float(r["p"]) for r in trajectory) == pytest.approx(1)
    assert {float(r["y_true"]) for r in hard_rows} == {1, 2, 3, 4}
    with pytest.raises(FileExistsError):
        write_adaptive_predictions(model, task, loader, "cpu", hard, provenance)
    with pytest.raises(ValueError, match="Duplicate"):
        write_adaptive_predictions(
            model, task, make_loader([samples[0], samples[0]], adaptive_config.loader),
            "cpu", tmp_path / "duplicate.csv", provenance,
        )
    assert not (tmp_path / "duplicate.csv").exists()
    assert torch.is_grad_enabled()  # The export decorator restores caller grad mode.
