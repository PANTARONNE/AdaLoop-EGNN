"""Analyze all depth/complexity results for one adaptive training output directory."""

import argparse
import sys
from pathlib import Path

from eqcompute.analysis.depthComplexity import analyze_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lambda-oracle", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = analyze_run(
            args.run_dir, args.split, args.device, args.batch_size,
            args.lambda_oracle, args.overwrite,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"{summary['property']} | {summary['split']} | N={summary['sample_count']}")
    print(f"lambda_oracle={summary['lambda_oracle']} | threshold={summary['threshold']}")
    for name, stats in summary["depths"].items():
        print(f"{name}: {stats['distinct_depths']} distinct depths")
        print("  " + ", ".join(
            f"{depth}: {count} ({stats['proportions'][depth]:.1%})"
            for depth, count in stats["counts"].items()
        ))
    for pair, fraction in summary["agreement"].items():
        print(f"Agreement {pair}: {fraction:.1%}")
    destination = Path(args.run_dir).resolve() / "analysis/depth_complexity" / args.split
    print(f"Analysis: {destination}")
    for name in ("complexity.csv", "oracle.csv", "depth_complexity.csv", "summary.json"):
        print(f"  {destination / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
