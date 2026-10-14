"""Analyze adjacent-depth prediction worsening for one adaptive training output directory."""

import argparse
from pathlib import Path

from eqcompute.analysis.depthWorsening import analyze_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = analyze_run(
            args.run_dir, args.split, args.device, args.batch_size, args.overwrite,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"{summary['property']} | {summary['split']} | N={summary['sample_count']}")
    print("All molecules; strict increase in absolute error; ties are not worsening.")
    for row in summary["transitions"]:
        print(
            f"Depth {row['depth']} -> {row['next_depth']}: "
            f"{row['worse_count']}/{row['sample_count']} "
            f"({row['worsening_proportion']:.1%}) worse; "
            f"{row['equal_count']} equal; {row['better_count']} better"
        )
    destination = Path(args.run_dir).resolve() / "analysis/depth_worsening" / args.split
    for name in ("depth_worsening.csv", "summary.json"):
        print(destination / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
