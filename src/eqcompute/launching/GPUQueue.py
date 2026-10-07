import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Job:
    name: str
    command: list[str]
    output_root: Path
    parameters: dict


def gpu_devices(indices, environ=None):
    """Map visible CUDA ordinals to devices, respecting an existing visibility mask."""
    environ = os.environ if environ is None else environ
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [str(index) for index in indices]
    devices = [item.strip() for item in visible.split(",")]
    if any(not item or item == "-1" for item in devices):
        raise ValueError("CUDA_VISIBLE_DEVICES does not expose usable GPUs")
    if max(indices) >= len(devices):
        raise ValueError("--gpus indexes exceed the existing CUDA_VISIBLE_DEVICES list")
    selected = [devices[index] for index in indices]
    if len(set(selected)) != len(selected):
        raise ValueError("Selected GPU slots refer to duplicate devices")
    return selected


def _now():
    return datetime.now(timezone.utc).isoformat()


def _save(path, records):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _signal_process(process, sig):
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        elif process.poll() is None:
            process.terminate() if sig == signal.SIGTERM else process.kill()
    except ProcessLookupError:
        pass


def _stop_processes(processes):
    # Signal the entire session, including any DataLoader worker processes.
    for process in processes:
        _signal_process(process, signal.SIGTERM)
    deadline = time.monotonic() + 5
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        _signal_process(process, signal.SIGKILL if os.name == "posix" else signal.SIGTERM)
        process.wait()


def run_queue(jobs, devices, directory, cwd):
    """Run at most one fresh subprocess per GPU; failures do not stop the queue."""
    if not jobs or not devices or len(set(devices)) != len(devices):
        raise ValueError("Supply jobs and a nonempty list of unique GPU devices")
    directory = Path(directory)
    (directory / "logs").mkdir(parents=True, exist_ok=True)
    manifest = directory / "jobs.json"
    records = [
        {
            "job": job.name,
            **job.parameters,
            "command": job.command,
            "cwd": str(cwd),
            "output_root": str(job.output_root),
            "run_dir": None,
            "log": str(directory / "logs" / f"{job.name}.log"),
            "gpu": None,
            "status": "pending",
            "exit_code": None,
        }
        for job in jobs
    ]
    active = {}
    next_job = 0
    interrupted = False

    def finish(index, status, code):
        record = records[index]
        record.update(status=status, exit_code=code, finished_at=_now())
        runs = list(jobs[index].output_root.glob("*/metadata.json"))
        if len(runs) == 1:
            record["run_dir"] = str(runs[0].parent)
        print(f"[{status}] {record['job']} GPU={record['gpu']} exit={code}", flush=True)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous_term = signal.signal(signal.SIGTERM, interrupt)
    previous_int = signal.getsignal(signal.SIGINT)
    try:
        _save(manifest, records)
        while next_job < len(jobs) or active:
            for device, (process, index) in list(active.items()):
                code = process.poll()
                if code is not None:
                    finish(index, "succeeded" if code == 0 else "failed", code)
                    del active[device]
                    _save(manifest, records)
            for device in devices:
                if device in active or next_job == len(jobs):
                    continue
                index = next_job
                next_job += 1
                job, record = jobs[index], records[index]
                record.update(gpu=device, started_at=_now())
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PYTHONUNBUFFERED="1")
                try:
                    job.output_root.mkdir(parents=True, exist_ok=False)
                    with Path(record["log"]).open("w", encoding="utf-8") as log:
                        process = subprocess.Popen(
                            job.command,
                            cwd=cwd,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=os.name == "posix",
                        )
                    active[device] = (process, index)
                    record.update(status="running", pid=process.pid)
                    print(f"[running] {job.name} GPU={device}", flush=True)
                except OSError as error:
                    record["error"] = str(error)
                    finish(index, "failed", None)
                _save(manifest, records)
            if active:
                time.sleep(0.2)
    except KeyboardInterrupt:
        interrupted = True
        print("Stopping running jobs; pending jobs will be cancelled.", flush=True)
    finally:
        # Also clean up children if the scheduler itself encounters an exception.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            _stop_processes([process for process, _ in active.values()])
            for process, index in active.values():
                finish(index, "interrupted", process.returncode)
            for record in records:
                if record["status"] == "pending":
                    record.update(status="cancelled", finished_at=_now())
            _save(manifest, records)
        finally:
            signal.signal(signal.SIGINT, previous_int)
            signal.signal(signal.SIGTERM, previous_term)
    failed = sum(record["status"] == "failed" for record in records)
    succeeded = sum(record["status"] == "succeeded" for record in records)
    print(f"Finished: {succeeded} succeeded, {failed} failed. Manifest: {manifest}", flush=True)
    return 130 if interrupted else int(failed > 0)
