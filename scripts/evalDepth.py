import argparse
import sys

from eqcompute.evaluation.depthSweep import summarize_sweep


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize a depth sweep as mean ± sample SD")
    parser.add_argument("--sweep-dir", required=True, help="Sweep directory containing jobs.json")
    parser.add_argument("--device", default="auto", help="Device for test inference")
    parser.add_argument(
        "--batch-size", type=int, help="Evaluation batch size (default: training size)"
    )
    parser.add_argument("--precision", type=int, default=6, help="Decimal places, from 0 to 12")
    args = parser.parse_args(argv)
    try:
        table, skipped = summarize_sweep(
            args.sweep_dir, args.device, args.batch_size, args.precision
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(table)
    return int(skipped > 0)


if __name__ == "__main__":
    sys.exit(main())
