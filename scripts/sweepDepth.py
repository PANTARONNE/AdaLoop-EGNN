import argparse
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from uuid import uuid4

from hydra import compose, initialize_config_dir
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import OmegaConf

from eqcompute.launching.GPUQueue import Job, gpu_devices, run_queue

ROOT = Path(__file__).resolve().parents[1]


def integers(value, minimum=0, maximum=None):
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use comma-separated integers") from error
    if any(item < minimum or (maximum is not None and item > maximum) for item in result):
        bound = f" and <= {maximum}" if maximum is not None else ""
        raise argparse.ArgumentTypeError(f"Values must be >= {minimum}{bound}")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("Duplicate values are not allowed")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run depth x seed experiments, one independent training per GPU.",
        epilog="Pass common Hydra overrides after --. GPU indices refer to visible CUDA devices.",
    )
    parser.add_argument("--gpus", type=integers, required=True)
    parser.add_argument("--depths", type=lambda value: integers(value, minimum=1), required=True)
    parser.add_argument(
        "--seeds", type=lambda value: integers(value, maximum=2**32 - 1), required=True
    )
    parser.add_argument("--weight-tying", choices=("true", "false"), required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print jobs without preparing/training"
    )
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.overrides[:1] == ["--"]:
        args.overrides = args.overrides[1:]
    return args


def make_plan(args):
    owned = {"model.execution.depth", "seed", "model.weight_tying", "trainer.device", "resume"}
    for override in OverridesParser.create().parse_overrides(args.overrides):
        key = override.key_or_group
        if key in owned or key == "hydra" or key.startswith(("hydra.", "hydra/")):
            raise ValueError(f"{key} is controlled by sweepDepth.py; use its options or train.py")
        if override.is_sweep_override():
            raise ValueError("Only --depths and --seeds are scanned; use single-valued overrides")

    devices = gpu_devices(args.gpus)
    jobs, reference_dataset = [], None
    directory = None
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        for depth, seed in product(args.depths, args.seeds):
            overrides = args.overrides + [
                f"model.execution.depth={depth}",
                f"seed={seed}",
                f"model.weight_tying={args.weight_tying}",
                "trainer.device=cuda:0",
            ]
            config = compose(config_name="config", overrides=overrides)
            if config.resume is not None:
                raise ValueError("A sweep starts fresh runs; resume individual jobs with train.py")
            dataset = OmegaConf.to_container(config.dataset, resolve=True)
            if reference_dataset is None:
                reference_dataset = dataset
                preparation_config = config
                suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                directory = (
                    (ROOT / config.output_root).resolve()
                    / "_depth_sweeps"
                    / f"{suffix}_{uuid4().hex[:8]}"
                )
            elif dataset != reference_dataset:
                raise ValueError("All depth/seed jobs must share the same dataset configuration")
            name = f"k{depth}_s{seed}"
            output_root = directory / "runs" / name
            # Quote the Hydra string value, not the shell: Popen receives an argument list.
            overrides += [f"output_root={json.dumps(str(output_root))}"]
            jobs.append(
                Job(
                    name,
                    [sys.executable, "-u", str(ROOT / "scripts/train.py"), *overrides],
                    output_root,
                    {"depth": depth, "seed": seed, "weight_tying": args.weight_tying == "true"},
                )
            )
    return jobs, devices, directory, preparation_config


def prepare_data(config):
    # Reuse the dataset factory on CPU once before workers can write the shared cache.
    from eqcompute.factories import DATASETS, construct

    config.dataset.root = str((ROOT / config.dataset.root).resolve())
    construct(DATASETS, config.dataset.name, config.dataset, config.task.target)


def main(argv=None):
    args = parse_args(argv)
    jobs, devices, directory, config = make_plan(args)
    print(f"Jobs: {len(jobs)}; GPUs: {', '.join(devices)}; weight_tying={args.weight_tying}")
    print(f"Sweep directory: {directory}")
    if args.dry_run:
        print(f"Working directory: {ROOT}")
        for job in jobs:
            print(f"{job.name}: {shlex.join(job.command)}")
        print("GPUs are assigned when slots become free; no data or CUDA validation was performed.")
        return 0
    os.chdir(ROOT)
    print("Preparing the shared dataset before starting training workers...", flush=True)
    prepare_data(config)
    return run_queue(jobs, devices, directory, ROOT)


if __name__ == "__main__":
    sys.exit(main())
