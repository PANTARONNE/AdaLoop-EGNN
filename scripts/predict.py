import argparse

from eqcompute.evaluation.predict import predict_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Export fixed-depth or adaptive predictions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mode", choices=["hard", "diagnostic"])
    args = parser.parse_args()
    path, count = predict_checkpoint(
        args.checkpoint, args.split, args.device, args.batch_size, args.overwrite, mode=args.mode
    )
    print(f"Wrote {count} rows: {path}")


if __name__ == "__main__":
    main()
