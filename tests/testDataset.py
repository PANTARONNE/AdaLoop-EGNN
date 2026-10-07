import os
from pathlib import Path

import pytest
import torch

from eqcompute.data.collate import collate_molecules
from eqcompute.data.QM9 import TARGETS, QM9Adapter
from eqcompute.data.splits import make_splits
from eqcompute.tasks.targetTransform import TargetTransform


def test_collation_identity_and_shapes(samples, batch):
    assert batch.graph.positions.shape == (13, 3)
    assert batch.target.shape == (4, 19)
    assert batch.sample_id == tuple(s.sample_id for s in samples)
    assert batch.graph.batch_index.tolist() == [0] * 3 + [1] * 5 + [2] + [3] * 4
    assert collate_molecules(samples[::-1]).sample_id == batch.sample_id[::-1]


def test_split_disjoint_stable_and_invalid():
    ids = [f"qm9:{i}" for i in range(30)]
    parts, manifest = make_splits(ids, 10, 5, 5, 20, seed=42)
    assert (parts, manifest) == make_splits(ids, 10, 5, 5, 20, seed=42)
    a, b, c = (set(v) for v in parts.values())
    assert not (a & b or a & c or b & c)
    assert len(a | b | c) == 20
    with pytest.raises(ValueError):
        make_splits(ids, 10, 5, 5)
    with pytest.raises(ValueError):
        make_splits(["same"] * 30, 10, 10)


def test_target_units_and_train_only_transform():
    assert TARGETS["U0"] == (7, "eV")
    assert TARGETS["U0_atom"] == (12, "eV")
    train = torch.tensor([[1.0], [3.0]])
    transform = TargetTransform.fit(train)
    assert transform.mean == 2 and transform.scale == 1
    held_out = torch.tensor([[100.0]])
    torch.testing.assert_close(transform.decode(transform.encode(held_out)), held_out)
    assert transform.mean == 2  # Held-out values never refit the transform.


@pytest.mark.integration
def test_real_qm9_smoke():
    root = os.environ.get("QM9_ROOT")
    if not root or not (Path(root) / "processed/data_v3.pt").exists():
        pytest.skip("Set QM9_ROOT to an existing processed dataset; tests never download it")
    dataset = QM9Adapter(root)
    assert len(dataset) > 120000
    batch = collate_molecules([dataset[0], dataset[1]])
    assert batch.target.shape == (2, 19)
    assert len(set(dataset.ids)) == len(dataset)
    assert torch.isfinite(batch.graph.positions).all()
