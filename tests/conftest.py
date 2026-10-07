from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch_geometric.data import Data

from eqcompute.data.collate import collate_molecules


@pytest.fixture(autouse=True)
def deterministic_small_tests():
    torch.set_num_threads(1)
    torch.manual_seed(7)


@pytest.fixture
def config():
    directory = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=directory, version_base="1.3"):
        return compose(config_name="config", overrides=["experiment=qm9_smoke"])


@pytest.fixture
def samples():
    result = []
    for i, count in enumerate((3, 5, 1, 4)):
        result.append(
            Data(
                z=torch.tensor([1, 6, 8, 7, 9][:count]),
                pos=torch.randn(count, 3),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 4),
                y=torch.full((1, 19), float(i + 1)),
                sample_id=f"fixture:{i}",
            )
        )
    return result


@pytest.fixture
def batch(samples):
    return collate_molecules(samples)
