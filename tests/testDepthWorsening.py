import json
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch
from testDepthComplexity import read_csv, write_csv
from testDepthComplexity import run_fixture as run_fixture

from eqcompute.analysis import depthWorsening as analysis

ROOT = Path(__file__).resolve().parents[1]
cli = runpy.run_path(str(ROOT / "scripts/depthWorsening.py"))


@pytest.fixture
def plotting(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "plot"))
    return runpy.run_path(str(ROOT / "plot/depthWorsening.py"))


def test_transition_statistics_strict_comparisons_and_boundaries():
    rows = analysis.transition_statistics([[3, 2, 4, 4], [1, 1, 2, 0]])
    assert [row["worse_count"] for row in rows] == [0, 2, 0]
    assert [row["equal_count"] for row in rows] == [1, 0, 1]
    assert [row["better_count"] for row in rows] == [1, 0, 1]
    assert [row["worsening_proportion"] for row in rows] == [0, 1, 0]
    tiny = analysis.transition_statistics([[1, np.nextafter(1., np.inf)]])
    assert tiny[0]["worsening_proportion"] == 1  # No tolerance around ties.
    for errors in ([], [[1]], [[-1, 0]], [[0, np.nan]], [[0, np.inf]], [1, 2]):
        with pytest.raises(ValueError):
            analysis.transition_statistics(errors)


def test_analysis_raw_errors_all_molecules_cache_repair_and_overwrite(
    run_fixture, monkeypatch, capsys,
):
    run, calls = run_fixture
    original = analysis.transition_statistics
    def capture(errors):
        np.testing.assert_array_equal(errors, [[0.4, 0.2, 0], [0, 0, 0.4]])
        return original(errors)
    monkeypatch.setattr(analysis, "transition_statistics", capture)
    assert cli["main"](["--run-dir", str(run)]) == 0
    assert "N=2" in capsys.readouterr().out
    output = run / "analysis/depth_worsening/validation"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["population"] == "all" and summary["error_units"] == "original"
    assert [row["worsening_proportion"] for row in summary["transitions"]] == [0, 0.5]
    assert all(row["sample_count"] == 2 for row in summary["transitions"])
    assert calls == {"predict": 1, "load": 0}
    assert not (run / "analysis/depth_complexity").exists()
    rows = read_csv(output / "depth_worsening.csv")
    assert [int(row["worse_count"]) for row in rows] == [0, 1]
    # The worsening molecule halted at depth 1, but still counts in transition 2 -> 3.
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
    analysis.analyze_run(run)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
    for name in ("depth_worsening.csv", "summary.json"):
        (output / name).unlink()
        analysis.analyze_run(run)
        assert (output / name).read_bytes() == before[output / name][0]
    assert calls == {"predict": 1, "load": 0}
    analysis.analyze_run(run, split="test")
    test_path = run / "analysis/depth_worsening/test/depth_worsening.csv"
    test_before = test_path.read_bytes(), test_path.stat().st_mtime_ns
    analysis.analyze_run(run, overwrite=True)
    assert calls == {"predict": 3, "load": 0}
    assert test_before == (test_path.read_bytes(), test_path.stat().st_mtime_ns)


@pytest.mark.parametrize("corruption", [
    "checkpoint", "split", "duplicate", "missing", "nan", "halt", "table", "summary", "hash",
])
def test_invalid_caches_require_overwrite(run_fixture, corruption):
    run, calls = run_fixture
    analysis.analyze_run(run)
    output = run / "analysis/depth_worsening/validation"
    path = run / "predictions/validation_best_diagnostic_by_depth.csv"
    if corruption == "summary":
        path = output / "summary.json"
        data = json.loads(path.read_text())
        data["sample_count"] = 99
        path.write_text(json.dumps(data))
    elif corruption == "hash":
        # Same numeric predictions and statistics, changed input file identity.
        path.write_text(path.read_text().replace("0.4,", "0.400,"))
    else:
        if corruption == "table":
            path = output / "depth_worsening.csv"
        rows = read_csv(path)
        if corruption == "duplicate":
            rows.append(rows[0].copy())
        elif corruption == "missing":
            rows.pop()
        else:
            key, value = {
                "checkpoint": ("checkpoint_id", "wrong"), "split": ("split", "test"),
                "nan": ("y_pred", "nan"), "halt": ("halt_depth", "1"),
                "table": ("worsening_proportion", "0.123"),
            }[corruption]
            rows[0][key] = value
        write_csv(path, rows)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="--overwrite"):
        analysis.analyze_run(run)
    assert path.read_bytes() == before and calls["predict"] == 1
    analysis.analyze_run(run, overwrite=True)
    assert calls["predict"] == 2


@pytest.mark.parametrize("problem", ["missing", "fixed", "single_depth", "split", "empty"])
def test_invalid_run_rejected_before_inference(run_fixture, problem):
    run, calls = run_fixture
    checkpoint = run / "checkpoints/best.pt"
    if problem == "missing":
        checkpoint.rename(checkpoint.with_name("last.pt"))
    elif problem in ("fixed", "single_depth"):
        payload = torch.load(checkpoint, weights_only=False)
        execution = payload["config"]["model"]["execution"]
        execution["name" if problem == "fixed" else "depth"] = (
            "fixed" if problem == "fixed" else 1
        )
        torch.save(payload, checkpoint)
    else:
        path = run / "split_ids.json"
        data = json.loads(path.read_text())
        if problem == "split":
            data["split_id"] = "wrong"
        else:
            data["sample_ids"]["validation"] = []
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        analysis.analyze_run(run)
    assert calls == {"predict": 0, "load": 0}


@pytest.mark.parametrize("corruption", [
    "empty", "duplicate", "missing", "mixed", "negative", "fractional", "total",
    "proportion", "nan", "pair", "protocol", "columns",
])
def test_plot_rejects_invalid_input(run_fixture, plotting, tmp_path, corruption):
    run, _ = run_fixture
    analysis.analyze_run(run)
    source = run / "analysis/depth_worsening/validation/depth_worsening.csv"
    rows = read_csv(source)
    fields = list(rows[0])
    if corruption == "empty":
        rows = []
    elif corruption == "duplicate":
        rows.append(rows[0].copy())
    elif corruption == "missing":
        rows.pop()
    elif corruption == "columns":
        fields.remove("depth")
        for row in rows:
            del row["depth"]
    else:
        key, value = {
            "mixed": ("checkpoint_id", "other"), "negative": ("worse_count", "-1"),
            "fractional": ("worse_count", "0.5"), "total": ("equal_count", "0"),
            "proportion": ("worsening_proportion", "0.2"),
            "nan": ("worsening_proportion", "nan"), "pair": ("next_depth", "3"),
            "protocol": ("population", "active"),
        }[corruption]
        rows[0][key] = value
    write_csv(source, rows, fields)
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError):
        plotting["plot_table"](source, destination)
    assert not destination.exists()


def test_histogram_values_style_and_exports(run_fixture, plotting, tmp_path):
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from PIL import Image

    run, _ = run_fixture
    analysis.analyze_run(run)
    source = run / "analysis/depth_worsening/validation/depth_worsening.csv"
    original = source.read_bytes()
    metadata, columns = plotting["read_table"](source)
    fig, ax = plotting["make_figure"](metadata, columns)
    try:
        assert len(ax.containers) == 1 and not ax.get_title() and not fig.texts
        assert ax.get_legend() is None
        assert ax.get_xlabel() == "Depth" and ax.get_ylabel() == "Worsening proportion"
        assert ax.get_ylim() == pytest.approx((0, 0.6))
        np.testing.assert_array_equal(ax.get_xticks(), [1, 2])
        patches = ax.containers[0]
        np.testing.assert_allclose([p.get_height() for p in patches], [0, 0.5])
        np.testing.assert_allclose([p.get_x() for p in patches], [0.5, 1.5])
        for patch in patches:
            assert patch.get_width() == 1 and patch.get_alpha() == 0.55
            assert colors.to_hex(patch.get_facecolor()) == "#4370b4"
            assert colors.to_hex(patch.get_edgecolor()) == "#4370b4"
            assert patch.get_linewidth() == 0.8 and not patch.get_rasterized()
        assert ax.yaxis.get_major_formatter()(0.5) == "50%"
        labels = [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]
        assert all(t.get_fontfamily() == ["Arial"] and t.get_fontsize() == 14.4 for t in labels)
    finally:
        plt.close(fig)
    destination = tmp_path / "figures"
    assert plotting["main"](["--input", str(source), "--output-dir", str(destination)]) == 0
    svg = ET.parse(destination / "depth_worsening.svg").getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    texts = svg.findall(".//svg:text", ns)
    assert texts and all("Arial" in t.attrib.get("style", "") for t in texts)
    assert not svg.findall(".//svg:image", ns) and svg.findall(".//svg:path", ns)
    with Image.open(destination / "depth_worsening.png") as image:
        assert image.size == (2550, 1800)
        assert image.info["dpi"][0] == pytest.approx(300, abs=0.1)
        image.verify()
    assert source.read_bytes() == original
    # Independent transition probabilities must not be renormalized to sum to 1.
    for counts, upper in (([0, 0], 0.1), ([2, 2], 1.1)):
        columns["worse_count"] = np.array(counts)
        columns["worsening_proportion"] = np.array(counts) / 2
        fig, ax = plotting["make_figure"](metadata, columns)
        try:
            assert ax.get_ylim() == pytest.approx((0, upper))
            np.testing.assert_allclose(
                [p.get_height() for p in ax.containers[0]], columns["worsening_proportion"],
            )
        finally:
            plt.close(fig)
