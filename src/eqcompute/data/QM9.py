from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.datasets import QM9

from eqcompute.data.splits import make_splits

# Units are those of PyG's processed QM9, after its built-in conversions.
TARGETS = {
    "mu": (0, "D"),
    "alpha": (1, "a0^3"),
    "homo": (2, "eV"),
    "lumo": (3, "eV"),
    "gap": (4, "eV"),
    "r2": (5, "a0^2"),
    "zpve": (6, "eV"),
    "U0": (7, "eV"),
    "U": (8, "eV"),
    "H": (9, "eV"),
    "G": (10, "eV"),
    "Cv": (11, "cal/(mol K)"),
    "U0_atom": (12, "eV"),
    "U_atom": (13, "eV"),
    "H_atom": (14, "eV"),
    "G_atom": (15, "eV"),
    "A": (16, "GHz"),
    "B": (17, "GHz"),
    "C": (18, "GHz"),
}


class QM9Adapter(Dataset):
    def __init__(self, root):
        self.source = QM9(root=str(root))
        self.ids = [f"qm9:{int(self.source[i].idx)}" for i in range(len(self.source))]

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        data = self.source[index].clone()
        data.sample_id = self.ids[index]
        return data


@dataclass
class DatasetBundle:
    train: Dataset
    validation: Dataset
    test: Dataset
    manifest: dict
    target_index: int
    unit: str

    def training_targets(self):
        return torch.cat(
            [sample.y[:, self.target_index : self.target_index + 1] for sample in self.train], dim=0
        )


def build_qm9(config, target):
    if target not in TARGETS:
        raise ValueError(f"Unknown QM9 property {target!r}; choose from {list(TARGETS)}")
    dataset = QM9Adapter(config.root)
    parts, manifest = make_splits(
        dataset.ids,
        config.train_size,
        config.val_size,
        config.test_size,
        config.subset_size,
        config.split_seed,
    )
    manifest["dataset"] = "qm9_pyg"
    index, unit = TARGETS[target]
    return DatasetBundle(
        **{name: Subset(dataset, indices) for name, indices in parts.items()},
        manifest=manifest,
        target_index=index,
        unit=unit,
    )
