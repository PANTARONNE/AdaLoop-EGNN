import hashlib
import json

import torch


def make_splits(ids, train_size, val_size, test_size=None, subset_size=None, seed=42):
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset sample IDs must be unique")
    total = len(ids) if subset_size is None else int(subset_size)
    test_size = total - train_size - val_size if test_size is None else test_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("All splits must be nonempty")
    if total > len(ids) or train_size + val_size + test_size != total:
        raise ValueError("Split sizes must sum to the selected dataset size")
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed))[:total]
    parts = {
        "train": order[:train_size].tolist(),
        "validation": order[train_size : train_size + val_size].tolist(),
        "test": order[train_size + val_size :].tolist(),
    }
    manifest = {name: [ids[i] for i in indices] for name, indices in parts.items()}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return parts, {"split_id": digest, "seed": seed, "sample_ids": manifest}
