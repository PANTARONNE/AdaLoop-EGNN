"""Geometry/composition diagnostics only; never supplied to the controller."""

import argparse
import csv
import os
from pathlib import Path

from eqcompute.evaluation.predict import load_evaluation


def graph_complexity(graph):
    records = []
    for index in range(graph.num_graphs):
        mask = graph.batch_index == index
        z, x = graph.atomic_numbers[mask], graph.positions[mask]
        if not z.numel():
            raise ValueError("Complexity requires nonempty molecules")
        _, counts = z.unique(return_counts=True)
        fractions = counts.to(x.dtype) / z.numel()
        records.append({
            "atom_count": z.numel(), "heavy_atom_count": int((z > 1).sum()),
            "radius_gyration": (x - x.mean(0)).square().sum(1).mean().sqrt().item(),
            "element_count": counts.numel(),
            "element_entropy": -(fractions * fractions.log()).sum().item(),
        })
    return records


def write_complexity(loader, destination, provenance, overwrite=False):
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    seen = set()
    fields = ["split_id", "split", "sample_id", "atom_count", "heavy_atom_count",
              "radius_gyration", "element_count", "element_entropy"]
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for batch in loader:
                records = graph_complexity(batch.graph)
                for sample, record in zip(batch.sample_id, records, strict=True):
                    if sample in seen:
                        raise ValueError("Duplicate sample in complexity export")
                    seen.add(sample)
                    writer.writerow({**provenance, "sample_id": sample, **record})
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    _, _, _, _, loader, _, provenance = load_evaluation(args.checkpoint, args.split, "cpu")
    count = write_complexity(
        loader, args.output, {k: provenance[k] for k in ("split_id", "split")}, args.overwrite
    )
    print(f"Wrote {count} complexity rows: {args.output}")


if __name__ == "__main__":
    main()
