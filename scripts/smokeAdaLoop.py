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
            "experiment=qm9_adaloop_egnn_orig_smoke",
        ])
    config.dataset.root = str(data_root)
    config.output_root = str(Path(output_root).resolve())
    config.task.normalization = normalization
    config.task.target = "mu"
    config.trainer.device = device
    output = train(config)
    checkpoint_path = output / "checkpoints/best.pt"
    _, config, model, task, test, device, provenance = load_evaluation(
        checkpoint_path, "test", device
    )
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    model.eval()
    maximum_difference, hard_work, full_work = 0.0, [], []
    metric = RegressionMetrics()
    work = []
    hook = model.operators.at(1).register_forward_pre_hook(
        lambda _, args: work.append((len(args[0].node_features), args[0].edge_index.shape[1]))
    )
    with torch.inference_mode():
        for batch in test:
            batch = batch.to(device)
            work.clear()
            full = task.predict(model, batch, execution_mode="diagnostic")
            full_work.extend(work)
            work.clear()
            hard = task.predict(model, batch, execution_mode="hard")
            hard_work.extend(work)
            torch.testing.assert_close(full.prediction, hard.prediction, atol=1e-5, rtol=1e-5)
            if not torch.equal(full.adaptive.halt_depth, hard.adaptive.halt_depth):
                raise AssertionError("Compact and full stopping depths differ")
            maximum_difference = max(
                maximum_difference, (full.prediction - hard.prediction).abs().max().item()
            )
            metric.update(task.transform.decode(hard.prediction), task.targets(batch))
    hook.remove()
    for mode, suffix in (("hard", "adaptive"), ("diagnostic", "diagnostic_by_depth")):
        write_adaptive_predictions(
            model, task, test, device, output / f"predictions/test_best_{suffix}.csv",
            provenance, mode=mode,
        )
    with (output / "predictions/test_best_adaptive.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    summary = {
        **metadata, "epochs": 3, "warmup_epochs": 1,
        "train_samples": 128, "validation_samples": 32, "test_samples": 32,
        "normalization": normalization,
        "validation_mae": load_checkpoint(checkpoint_path)["best_value"],
        "purpose": "local_real_data_acceptance",
        "test": metric.compute(), "max_normalized_full_hard_difference": maximum_difference,
        "halting_consistency": 1.0,
        "average_halt_depth": sum(int(r["halt_depth"]) for r in rows) / len(rows),
        "full_node_steps": sum(n for n, _ in full_work),
        "hard_node_steps": sum(n for n, _ in hard_work),
        "full_edge_steps": sum(e for _, e in full_work),
        "hard_edge_steps": sum(e for _, e in hard_work),
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
