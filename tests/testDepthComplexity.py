import csv
import json
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch

from eqcompute.analysis import depthComplexity as analysis
from eqcompute.analysis import depthWorsening, diagnostic
from eqcompute.application import make_loader, train
from eqcompute.data.QM9 import DatasetBundle
from eqcompute.evaluation.adaptiveWriter import DIAGNOSTIC_FIELDS
from eqcompute.evaluation.predictionWriter import file_hash
from eqcompute.factories import DATASETS

ROOT = Path(__file__).resolve().parents[1]
plot = runpy.run_path(str(ROOT / "plot/depthComplexity.py"))
cli = runpy.run_path(str(ROOT / "scripts/depthComplexity.py"))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def run_fixture(tmp_path, monkeypatch, samples, adaptive_config):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/best.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({
        "format_version": 1,
        "config": {
            "seed": 0,
            "model": {"weight_tying": True, "execution": {
                "name": "adaptive", "depth": 3, "threshold": 0.5, "inference_rule": "cdf",
            }},
            "task": {"supervision": "adaptive", "normalization": "mad"},
        },
        "task": {"target": "mu", "unit": "D", "transform": {"scale": 2.0}},
        "metadata": {"run_id": "fixture-run", "split_id": "fixture-split"},
    }, checkpoint)
    (run / "split_ids.json").write_text(json.dumps({
        "split_id": "fixture-split", "sample_ids": {
            "validation": [sample.sample_id for sample in samples[:2]],
            "test": [sample.sample_id for sample in samples[:2]],
        },
    }))
    calls = {"predict": 0, "load": 0}

    def predict(path, split, device, batch_size, overwrite, mode):
        assert Path(path) == checkpoint and mode == "diagnostic"
        calls["predict"] += 1
        rows = []
        for i, errors in enumerate(([0.4, 0.2, 0.0], [0.0, 0.0, 0.4])):
            for depth, error in enumerate(errors, 1):
                rows.append({
                    "run_id": "fixture-run", "checkpoint_id": file_hash(checkpoint),
                    "split_id": "fixture-split", "split": split, "sample_id": samples[i].sample_id,
                    "property": "mu", "unit": "D", "supervision": "adaptive",
                    "train_depth": 3, "weight_tying": True, "seed": 0,
                    "execution_mode": "diagnostic",
                    "inference_rule": "cdf", "threshold": 0.5,
                    "readout_depth": depth, "y_true": 0.0, "y_pred": error,
                    "halt_depth": 2 if i == 0 else 1, "executed_depth": 3, "valid_depth": True,
                    "q": 0.5, "p": 1 / 3, "survival": (4 - depth) / 3, "expected_depth": 2.0,
                })
        target = run / "predictions" / f"{split}_best_diagnostic_by_depth.csv"
        write_csv(target, rows, DIAGNOSTIC_FIELDS)
        return target, len(rows)

    def load(path, split, device, batch_size):
        calls["load"] += 1
        return (
            path, None, None, None, make_loader(samples[:2], adaptive_config.loader), device,
            {"split_id": "fixture-split", "split": split},
        )

    monkeypatch.setattr(diagnostic, "predict_checkpoint", predict)
    monkeypatch.setattr(analysis, "load_evaluation", load)
    return run, calls


def test_all_results_cache_repair_and_overwrite(run_fixture, capsys):
    run, calls = run_fixture
    assert cli["main"](["--run-dir", str(run)]) == 0
    assert "N=2" in capsys.readouterr().out
    output = run / "analysis/depth_complexity/validation"
    rows = read_csv(output / "depth_complexity.csv")
    assert [tuple(int(row[key]) for key in analysis.DEPTHS) for row in rows] == [
        (3, 2), (1, 1),
    ]
    summary = json.loads((output / "summary.json").read_text())
    removed = {"lambda_compute", "lambda_oracle", "cost_oracle_depth", "cost_oracle_error"}
    assert not removed & rows[0].keys()
    assert not removed & summary.keys()
    assert set(summary["depths"]) == {"oracle_depth", "halt_depth"}
    assert summary["depths"]["oracle_depth"]["distinct_depths"] == 2
    assert summary["depths"]["oracle_depth"]["proportions"]["3"] == 0.5
    assert summary["agreement"] == {"oracle_depth__halt_depth": 0.5}
    original = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
    analysis.analyze_run(run)
    assert calls == {"predict": 1, "load": 1}
    assert original == {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()
    }
    (output / "oracle.csv").unlink()
    (output / "depth_complexity.csv").unlink()
    analysis.analyze_run(run)
    assert calls == {"predict": 1, "load": 1}
    assert (output / "oracle.csv").read_bytes() == original[output / "oracle.csv"][0]
    updated = analysis.analyze_run(run, overwrite=True)
    assert calls == {"predict": 2, "load": 2}
    assert updated == summary


def test_cli_rejects_removed_cost_option(run_fixture, capsys):
    run, calls = run_fixture
    with pytest.raises(SystemExit) as error:
        cli["main"](["--run-dir", str(run), "--lambda-oracle", "0"])
    assert error.value.code == 2
    assert "unrecognized arguments: --lambda-oracle" in capsys.readouterr().err
    assert calls == {"predict": 0, "load": 0}


@pytest.mark.parametrize("corruption", [
    "checkpoint", "split", "duplicate", "missing_depth", "missing_sample",
    "nan", "halt", "label", "hard", "invalid_depth", "complexity_missing",
    "complexity_duplicate", "complexity_split", "complexity_nan", "merged", "summary",
])
def test_invalid_caches_fail_without_silent_overwrite(run_fixture, corruption):
    run, calls = run_fixture
    analysis.analyze_run(run)
    output = run / "analysis/depth_complexity/validation"
    path = run / "predictions/validation_best_diagnostic_by_depth.csv"
    if corruption.startswith("complexity"):
        path = output / "complexity.csv"
    elif corruption in ("merged", "summary"):
        path = output / ("summary.json" if corruption == "summary" else "depth_complexity.csv")
    if corruption == "summary":
        summary = json.loads(path.read_text())
        summary["sample_count"] = 99
        path.write_text(json.dumps(summary))
    else:
        rows = read_csv(path)
        if corruption in ("duplicate", "complexity_duplicate"):
            rows.append(rows[0].copy())
        elif corruption in ("missing_depth", "complexity_missing"):
            rows.pop()
        elif corruption == "missing_sample":
            rows = [row for row in rows if row["sample_id"] != "fixture:1"]
        else:
            key, value = {
                "checkpoint": ("checkpoint_id", "stale"),
                "split": ("split", "test"), "nan": ("y_pred", "nan"),
                "halt": ("halt_depth", "1"), "label": ("y_true", "1"),
                "hard": ("execution_mode", "hard"), "invalid_depth": ("readout_depth", "4"),
                "complexity_split": ("split_id", "wrong"),
                "complexity_nan": ("radius_gyration", "nan"),
                "merged": ("oracle_depth", "2"),
            }[corruption]
            rows[0][key] = value
        write_csv(path, rows)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="--overwrite"):
        analysis.analyze_run(run)
    assert path.read_bytes() == before
    assert calls == {"predict": 1, "load": 1}
    analysis.analyze_run(run, overwrite=True)
    assert calls == {"predict": 2, "load": 2}


def test_bad_run_checkpoint_manifest_and_arguments(run_fixture):
    run, _ = run_fixture
    with pytest.raises(FileNotFoundError, match="best.pt"):
        analysis.analyze_run(run / "absent")
    for kwargs in ({"split": "train"}, {"batch_size": 0}):
        with pytest.raises(ValueError):
            analysis.analyze_run(run, **kwargs)
    checkpoint = run / "checkpoints/best.pt"
    payload = torch.load(checkpoint, weights_only=False)
    payload["config"]["model"]["execution"]["name"] = "fixed"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="adaptive"):
        analysis.analyze_run(run)
    payload["config"]["model"]["execution"]["name"] = "adaptive"
    torch.save(payload, checkpoint)
    manifest = run / "split_ids.json"
    value = json.loads(manifest.read_text())
    value["sample_ids"]["validation"].append("fixture:0")
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="unique"):
        analysis.analyze_run(run)
    checkpoint.rename(checkpoint.with_name("last.pt"))
    with pytest.raises(FileNotFoundError, match="no fallback"):
        analysis.analyze_run(run)


@pytest.mark.parametrize("metric", list(plot["COMPLEXITIES"]))
def test_scatter_colors_offsets_ticks_and_vector_exports(run_fixture, metric, tmp_path):
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from PIL import Image

    run, _ = run_fixture
    analysis.analyze_run(run)
    path = run / "analysis/depth_complexity/validation/depth_complexity.csv"
    original = path.read_bytes()
    metadata, columns = plot["read_table"](path)
    untouched = {key: value.copy() for key, value in columns.items()}
    fig, ax = plot["make_figure"](metadata, columns, metric)
    other, other_ax = plot["make_figure"](metadata, columns, metric)
    try:
        assert len(ax.collections) == 2
        assert not fig.texts
        assert ax.get_title() == ""
        labels = [
            ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(),
            *ax.get_legend().get_texts(),
        ]
        assert all(label.get_fontfamily() == ["Arial"] for label in labels)
        np.testing.assert_array_equal(ax.get_yticks(), [1, 2, 3])
        for index, (key, label, color, marker, offset) in enumerate(plot["SERIES"]):
            points = ax.collections[index]
            assert points.get_offsets().shape == (2, 2)
            assert points.get_label() == label
            assert colors.to_hex(points.get_facecolors()[0]) == color.lower()
            assert not points.get_rasterized()
            np.testing.assert_array_equal(
                points.get_offsets(), other_ax.collections[index].get_offsets()
            )
            y_noise = points.get_offsets()[:, 1] - columns[key] - offset
            assert np.abs(y_noise).max() <= 0.04
            x_noise = points.get_offsets()[:, 0] - columns[metric]
            assert np.abs(x_noise).max() <= (
                0.18 if plot["COMPLEXITIES"][metric][1] else 0
            )
        for key in columns:
            np.testing.assert_array_equal(columns[key], untouched[key])
    finally:
        plt.close(fig)
        plt.close(other)
    fig, ax = plot["make_figure"](metadata, columns, metric, jitter=False)
    try:
        for item, (key, *_) in zip(ax.collections, plot["SERIES"], strict=True):
            np.testing.assert_array_equal(item.get_offsets()[:, 0], columns[metric])
            np.testing.assert_array_equal(item.get_offsets()[:, 1], columns[key])
    finally:
        plt.close(fig)
    paths = plot["plot_table"](path, tmp_path / "figures", metric)
    assert {item.suffix for item in paths} == {".svg", ".png"}
    svg = ET.parse(paths[0]).getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    texts = svg.findall(".//svg:text", ns)
    assert texts
    assert all("Arial" in text.attrib.get("style", "") for text in texts)
    assert not any("Checkpoint:" in (text.text or "") for text in texts)
    assert svg.findall(".//svg:use", ns)
    assert not svg.findall(".//svg:image", ns)
    for color in ("#c30078", "#529f9a"):
        assert color in paths[0].read_text().lower()
    assert "#4370b4" not in paths[0].read_text().lower()
    with Image.open(paths[1]) as image:
        assert image.size == (2550, 1800)
        assert image.info["dpi"][0] == pytest.approx(300, abs=0.1)
        image.verify()
    assert path.read_bytes() == original


def test_plot_all_and_reject_bad_tables(run_fixture, tmp_path):
    run, _ = run_fixture
    analysis.analyze_run(run)
    path = run / "analysis/depth_complexity/validation/depth_complexity.csv"
    assert len(plot["plot_table"](path, tmp_path / "all")) == 10
    rows = read_csv(path)
    for key, value in (("checkpoint_id", "other"), ("sample_id", rows[1]["sample_id"]),
                       ("oracle_depth", "1.5"), ("atom_count", "nan")):
        corrupted = [row.copy() for row in rows]
        corrupted[0][key] = value
        write_csv(path, corrupted)
        with pytest.raises(ValueError):
            plot["read_table"](path)


def test_real_training_directory_end_to_end(adaptive_config, samples, tmp_path, monkeypatch):
    config = adaptive_config
    config.trainer.epochs = 2
    config.task.warmup_epochs = 1
    config.output_root = str(tmp_path / "runs")
    manifest = {
        "split_id": "fixture-split", "sample_ids": {
            "train": [sample.sample_id for sample in samples[:2]],
            "validation": [samples[2].sample_id], "test": [samples[3].sample_id],
        },
    }
    bundle = DatasetBundle(samples[:2], samples[2:3], samples[3:], manifest, 0, "D")
    monkeypatch.setitem(DATASETS, "qm9", lambda *args: bundle)
    run = train(config)
    for split in ("validation", "test"):
        result = depthWorsening.analyze_run(run, split=split, device="cpu")
        assert result["sample_count"] == 1
        assert len(result["transitions"]) == config.model.execution.depth - 1
        assert cli["main"](["--run-dir", str(run), "--split", split, "--device", "cpu"]) == 0
        output = run / "analysis/depth_complexity" / split
        assert all((output / name).is_file() for name in (
            "complexity.csv", "oracle.csv", "depth_complexity.csv", "summary.json",
        ))
        summary = json.loads((output / "summary.json").read_text())
        assert summary["sample_count"] == 1
        assert read_csv(output / "depth_complexity.csv")[0]["sample_id"] == (
            manifest["sample_ids"][split][0]
        )
    def forbidden(*args, **kwargs):
        pytest.fail("Complete caches must not load a dataset or run inference")
    monkeypatch.setattr(analysis, "load_evaluation", forbidden)
    monkeypatch.setattr(diagnostic, "predict_checkpoint", forbidden)
    analysis.analyze_run(run)


@pytest.fixture
def distribution_plot(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "plot"))
    return runpy.run_path(str(ROOT / "plot/depthDistribution.py"))


def test_depth_histogram_proportions_and_panels(run_fixture, distribution_plot):
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    run, _ = run_fixture
    summary = analysis.analyze_run(run)
    source = run / "analysis/depth_complexity/validation/depth_complexity.csv"
    metadata, columns = distribution_plot["read_table"](source)
    before = {key: value.copy() for key, value in columns.items()}
    fig, axes = distribution_plot["make_figure"](metadata, columns)
    try:
        assert not fig.texts and len(axes) == 2
        assert [ax.get_xlabel() for ax in axes] == ["", "Depth"]
        assert axes[0].get_position().y0 > axes[1].get_position().y0
        fig.canvas.draw()
        assert axes[-1].xaxis.label.get_window_extent().y0 >= 0
        for ax, (key, label, color, *_) in zip(
            axes, distribution_plot["SERIES"], strict=True
        ):
            assert not ax.get_title()
            assert ax.get_ylabel() == "Molecule proportion"
            assert ax.get_shared_x_axes().joined(ax, axes[-1])
            assert ax.get_xlim() == axes[-1].get_xlim()
            expected_upper = 0.6
            assert ax.get_ylim() == pytest.approx((0, expected_upper))
            np.testing.assert_allclose(
                ax.get_yticks(), np.arange(round(expected_upper * 10) + 1) / 10,
            )
            np.testing.assert_array_equal(ax.get_xticks(), [1, 2, 3])
            assert len(ax.containers) == 1
            assert [text.get_text() for text in ax.get_legend().get_texts()] == [label]
            container = ax.containers[0]
            heights = np.array([patch.get_height() for patch in container])
            expected = [summary["depths"][key]["proportions"][str(k)] for k in (1, 2, 3)]
            np.testing.assert_allclose(heights, expected)
            np.testing.assert_allclose(
                heights * summary["sample_count"],
                [summary["depths"][key]["counts"][str(k)] for k in (1, 2, 3)],
            )
            assert heights.sum() == pytest.approx(1)
            np.testing.assert_allclose([patch.get_x() for patch in container], [0.5, 1.5, 2.5])
            for patch in container:
                assert patch.get_width() == pytest.approx(1)
                assert patch.get_alpha() == 0.55
                assert patch.get_facecolor()[3] == 0.55
                assert colors.to_hex(patch.get_facecolor()) == color.lower()
                assert colors.to_hex(patch.get_edgecolor()) == color.lower()
                assert patch.get_linewidth() == 0.8
                assert not patch.get_rasterized()
            assert "%" in ax.yaxis.get_major_formatter()(0.5)
            texts = [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(),
                     *ax.get_yticklabels(), *ax.get_legend().get_texts()]
            assert all(text.get_fontfamily() == ["Arial"] for text in texts)
        assert not axes[0].get_xticklabels()
        assert axes[-1].get_xticklabels()
        assert axes[0].containers[0][1].get_height() == 0
        assert axes[1].containers[0][2].get_height() == 0
        for key in columns:
            np.testing.assert_array_equal(before[key], columns[key])
    finally:
        plt.close(fig)

def test_depth_histogram_cli_exports(run_fixture, distribution_plot, tmp_path):
    from PIL import Image

    run, _ = run_fixture
    analysis.analyze_run(run)
    source = run / "analysis/depth_complexity/validation/depth_complexity.csv"
    original = source.read_bytes()
    destination = tmp_path / "histogram"
    assert distribution_plot["main"]([
        "--input", str(source), "--output-dir", str(destination),
    ]) == 0
    assert {path.name for path in destination.iterdir()} == {
        "depth_distribution.svg", "depth_distribution.png",
    }
    svg = ET.parse(destination / "depth_distribution.svg").getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    texts = svg.findall(".//svg:text", ns)
    assert texts and all("Arial" in text.attrib.get("style", "") for text in texts)
    assert not svg.findall(".//svg:image", ns)
    with Image.open(destination / "depth_distribution.png") as image:
        assert image.size == (2550, 1800)
        assert image.info["dpi"][0] == pytest.approx(300, abs=0.1)
        image.verify()
    assert source.read_bytes() == original


@pytest.mark.parametrize("corruption", ["empty", "duplicate", "depth", "mixed"])
def test_depth_histogram_rejects_invalid_input(
    run_fixture, distribution_plot, tmp_path, corruption
):
    run, _ = run_fixture
    analysis.analyze_run(run)
    source = run / "analysis/depth_complexity/validation/depth_complexity.csv"
    rows = read_csv(source)
    fields = list(rows[0])
    if corruption == "empty":
        rows = []
    elif corruption == "duplicate":
        rows.append(rows[0].copy())
    elif corruption == "depth":
        rows[0]["halt_depth"] = "4"
    else:
        rows[0]["checkpoint_id"] = "other"
    write_csv(source, rows, fields)
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError):
        distribution_plot["plot_table"](source, destination)
    assert not destination.exists()
