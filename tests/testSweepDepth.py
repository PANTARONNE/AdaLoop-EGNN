import json
import os
import runpy
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from eqcompute.factories import MODELS, construct
from eqcompute.launching.GPUQueue import Job, gpu_devices, run_queue

ROOT = Path(__file__).resolve().parents[1]
CLI = runpy.run_path(str(ROOT / "scripts/sweepDepth.py"))
BASE_ARGS = ["--gpus", "0,1", "--depths", "2,4", "--seeds", "0,7", "--weight-tying", "false"]


@pytest.mark.parametrize("preset,batch_size,tied", [
    ("qm9_smoke", 32, False),
    ("qm9_egnn", 96, False),
    ("qm9_loop_egnn", 96, False),
    ("qm9_loop_egnn_diagnostic", 96, False),
    ("qm9_adaloop_egnn", 96, True),
    ("qm9_adaloop_egnn_smoke", 16, True),
])
def test_plan_cartesian_product_presets_and_output_paths(
    tmp_path, monkeypatch, preset, batch_size, tied
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    output = tmp_path / "outputs with spaces"
    arguments = BASE_ARGS.copy()
    arguments[-1] = str(tied).lower()
    args = CLI["parse_args"](
        arguments + ["--", f"experiment={preset}", f"output_root={json.dumps(str(output))}"]
    )
    jobs, devices, directory, _ = CLI["make_plan"](args)
    assert devices == ["0", "1"]
    assert len(jobs) == 4
    assert {(job.parameters["depth"], job.parameters["seed"]) for job in jobs} == {
        (2, 0), (2, 7), (4, 0), (4, 7)
    }
    assert directory.parent == output / "_depth_sweeps"
    assert not directory.exists()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        for job in jobs:
            config = compose(config_name="config", overrides=job.command[3:])
            assert config.model.execution.depth == job.parameters["depth"]
            assert config.seed == job.parameters["seed"]
            assert config.model.weight_tying is tied  # The CLI takes precedence over the preset.
            assert config.trainer.device == "cuda:0"
            assert config.loader.batch_size == batch_size
            assert Path(config.output_root) == job.output_root
            assert config.dataset.split_seed == 42
            construct(MODELS, config.model.name, config.model, config.task.target)


@pytest.mark.parametrize(
    "override",
    ["seed=5", "++trainer.device=cpu", "model.weight_tying=true", "resume=null",
     "hydra.mode=MULTIRUN", "optimizer.lr=0.001,0.01"],
)
def test_reject_overrides_that_break_single_job_contract(override):
    args = CLI["parse_args"](BASE_ARGS + ["--", override])
    with pytest.raises(ValueError):
        CLI["make_plan"](args)


@pytest.mark.parametrize(
    "option,value",
    [("--depths", "0,2"), ("--seeds", "4294967296"), ("--gpus", "0,0"),
     ("--seeds", "-1"), ("--depths", "2,,4"), ("--weight-tying", "yes")],
)
def test_invalid_cli_lists(option, value):
    args = BASE_ARGS.copy()
    args[args.index(option) + 1] = value
    with pytest.raises(SystemExit):
        CLI["parse_args"](args)


def test_weight_tying_is_required():
    with pytest.raises(SystemExit):
        CLI["parse_args"](BASE_ARGS[:-2])


def test_existing_gpu_mask_is_respected():
    assert gpu_devices([1, 0], {"CUDA_VISIBLE_DEVICES": "3,7"}) == ["7", "3"]
    assert gpu_devices([0], {"CUDA_VISIBLE_DEVICES": "GPU-abc"}) == ["GPU-abc"]
    with pytest.raises(ValueError):
        gpu_devices([2], {"CUDA_VISIBLE_DEVICES": "3,7"})
    with pytest.raises(ValueError):
        gpu_devices([0], {"CUDA_VISIBLE_DEVICES": ""})


def test_dry_run_does_not_create_outputs_or_prepare_data(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def unexpected(*args, **kwargs):
        pytest.fail("dry-run must not prepare data or launch training")

    monkeypatch.setitem(CLI["main"].__globals__, "prepare_data", unexpected)
    monkeypatch.setitem(CLI["main"].__globals__, "run_queue", unexpected)
    code = CLI["main"](
        BASE_ARGS + ["--dry-run", "--", f"output_root={json.dumps(str(tmp_path))}"]
    )
    assert code == 0
    assert "Jobs: 4" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_queue_real_processes_refill_free_gpu_and_continue_after_failure(tmp_path, monkeypatch):
    # A waits until C starts. B fails while A is running, so C must reuse B's GPU.
    worker = tmp_path / "worker.py"
    worker.write_text(
        """import json, os, sys, time
from pathlib import Path
name, output, shared = sys.argv[1:]
shared = Path(shared)
gpu = os.environ['CUDA_VISIBLE_DEVICES']
lock = shared / (gpu + '.lock')
with lock.open('x'):
    pass
print(json.dumps({'gpu': gpu, 'pid': os.getpid()}), flush=True)
run = Path(output) / 'actual_run'
run.mkdir()
(run / 'metadata.json').write_text('{}')
deadline = time.monotonic() + 15
if name == 'a':
    (shared / 'a_started').touch()
    while not (shared / 'release_a').exists():
        assert time.monotonic() < deadline, 'free GPU was not refilled'
        time.sleep(0.02)
elif name == 'b':
    while not (shared / 'a_started').exists():
        assert time.monotonic() < deadline, 'jobs did not overlap'
        time.sleep(0.02)
elif name == 'c':
    assert (shared / 'device-a.lock').exists(), 'A should still be running'
    (shared / 'release_a').touch()
lock.unlink()
sys.exit(7 if name == 'b' else 0)
""", encoding="utf-8"
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-mask")
    jobs = [
        Job(name, [sys.executable, str(worker), name, str(tmp_path / name), str(tmp_path)],
            tmp_path / name, {})
        for name in ("a", "b", "c")
    ]
    directory = tmp_path / "sweep"
    assert run_queue(jobs, ["device-a", "device-b"], directory, tmp_path) == 1
    records = json.loads((directory / "jobs.json").read_text())
    assert [row["status"] for row in records] == ["succeeded", "failed", "succeeded"]
    assert [row["exit_code"] for row in records] == [0, 7, 0]
    assert [row["gpu"] for row in records] == ["device-a", "device-b", "device-b"]
    for row in records:
        assert Path(row["run_dir"]).name == "actual_run"
        assert json.loads(Path(row["log"]).read_text())["gpu"] == row["gpu"]
    assert len({row["pid"] for row in records}) == 3
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-mask"


def test_process_launch_failure_does_not_block_later_jobs(tmp_path):
    jobs = [
        Job("bad", [str(tmp_path / "missing_executable")], tmp_path / "bad", {}),
        Job("good", [sys.executable, "-c", "print('ok')"], tmp_path / "good", {}),
    ]
    directory = tmp_path / "sweep"
    assert run_queue(jobs, ["0"], directory, tmp_path) == 1
    records = json.loads((directory / "jobs.json").read_text())
    assert records[0]["status"] == "failed"
    assert "error" in records[0]
    assert records[1]["status"] == "succeeded"


@pytest.mark.skipif(os.name != "posix", reason="Linux process group cleanup")
def test_sigterm_stops_workers_and_cancels_pending_jobs(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        """import os, signal, subprocess, sys, time
from pathlib import Path
def stop(signum, frame):
    child.wait(timeout=3)
    Path('stopped').touch()
    sys.exit(0)
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
signal.signal(signal.SIGTERM, stop)
Path('started').write_text(str(os.getpid()))
time.sleep(60)
""", encoding="utf-8"
    )
    runner = tmp_path / "runner.py"
    runner.write_text(
        """import sys
from pathlib import Path
from eqcompute.launching.GPUQueue import Job, run_queue
root = Path.cwd()
jobs = [Job(str(i), [sys.executable, str(root / 'worker.py')], root / str(i), {})
        for i in range(2)]
sys.exit(run_queue(jobs, ['0'], root / 'sweep', root))
""", encoding="utf-8"
    )
    process = subprocess.Popen(
        [sys.executable, str(runner)], cwd=tmp_path,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 15
        while not (tmp_path / "started").exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.05)
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=12)
        assert process.returncode == 130, output
        assert (tmp_path / "stopped").exists()  # The descendant also received SIGTERM.
        records = json.loads((tmp_path / "sweep/jobs.json").read_text())
        assert [row["status"] for row in records] == ["interrupted", "cancelled"]
        pid = int((tmp_path / "started").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=12)
