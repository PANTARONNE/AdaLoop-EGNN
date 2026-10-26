"""Local PyG QM9 smoke through the standard AdaLoop training entry point."""

import argparse
import csv
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from eqcompute.application import train
from eqcompute.evaluation.adaptiveWriter import write_adaptive_predictions
from eqcompute.evaluation.predict import load_evaluation
from eqcompute.metrics.regression import RegressionMetrics
from eqcompute.training.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def run_smoke(data_root, output_root, device="auto", normalization="mad"):
    data_root = Path(data_root).resolve()
    if not (data_root / "processed/data_v3.pt").is_file():
        raise FileNotFoundError("Smoke requires an existing local PyG QM9 processed/data_v3.pt")
    torch.set_num_threads(1)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        config = compose(config_name="config", overrides=[
            "experiment=qm9_adaloop_egnn_smoke",
        ])
    config.dataset.root = str(data_root)
    config.output_root = str(Path(output_root).resolve())
    config.task.normalization = normalization
    config.task.target = "mu"
    config.trainer.device = device
    config.trainer.epochs = 1
    predictor_output = train(config)
    config.task.phase = "halting"
    config.task.halting_loss = "soft_cdf"
    config.trainer.epochs = 2
    config.init_from = str(predictor_output / "checkpoints/best.pt")
    output = train(config)
    predictor = load_checkpoint(config.init_from)
    halting = load_checkpoint(output / "checkpoints/last.pt")
    for key, value in predictor["model"].items():
        if not key.startswith("execution.controller."):
            torch.testing.assert_close(value, halting["model"][key], rtol=0, atol=0)
    checkpoint_path = output / "checkpoints/best.pt"
    _, config, model, task, test, device, provenance = load_evaluation(
        checkpoint_path, "test", device
    )
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    model.eval()
    maximum_difference = 0.0
    metric = RegressionMetrics()
    with torch.inference_mode():
        for batch in test:
            batch = batch.to(device)
            full = task.predict(model, batch, execution_mode="diagnostic")
            hard = task.predict(model, batch, execution_mode="hard")
            torch.testing.assert_close(full.prediction, hard.prediction, atol=1e-5, rtol=1e-5)
            if not torch.equal(full.adaptive.halt_depth, hard.adaptive.halt_depth):
                raise AssertionError("Compact and full stopping depths differ")
            maximum_difference = max(
                maximum_difference, (full.prediction - hard.prediction).abs().max().item()
            )
            metric.update(task.transform.decode(hard.prediction), task.targets(batch))
    for mode, suffix in (("hard", "adaptive"), ("diagnostic", "diagnostic_by_depth")):
        write_adaptive_predictions(
            model, task, test, device, output / f"predictions/test_best_{suffix}.csv",
            provenance, mode=mode,
        )
    with (output / "predictions/test_best_adaptive.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    summary = {
        **metadata, "predictor_epochs": 1, "halting_epochs": 2,
        "predictor_output": str(predictor_output),
        "predictor_validation_oracle_mae": predictor["best_value"],
        "predictor_unchanged": True,
        "train_samples": 128, "validation_samples": 32, "test_samples": 32,
        "normalization": normalization,
        "halting_loss": config.task.halting_loss,
        "soft_cdf_temperature": config.task.soft_cdf_temperature,
        "validation_mae": load_checkpoint(checkpoint_path)["best_value"],
        "purpose": "local_real_data_acceptance",
        "test": metric.compute(), "max_normalized_full_hard_difference": maximum_difference,
        "halting_consistency": 1.0,
        "average_halt_depth": sum(int(r["halt_depth"]) for r in rows) / len(rows),
    }
    (output / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(ROOT / "data/qm9"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--normalization", choices=["mad", "standard", "none"], default="mad")
    args = parser.parse_args()
    output, summary = run_smoke(
        args.data_root, args.output_root, args.device, args.normalization
    )
    print(json.dumps(summary, indent=2))
    print(f"Smoke output: {output}")


if __name__ == "__main__":
    main()
