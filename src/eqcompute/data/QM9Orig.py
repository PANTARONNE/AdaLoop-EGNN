"""Offline QM9 XYZ preprocessing matching the official EGNN/Cormorant pipeline.

Reference: vgsatorras/egnn@e9ca6c0c3e1d30a7598efbd66034121b4af8dccc.
See third_party/EGNN-LICENSE.txt. Raw labels are rounded to float32 BEFORE
subtracting atom references and converting units, as in the reference loader.
"""

import hashlib
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from eqcompute.data.QM9 import DatasetBundle

UPSTREAM_COMMIT = "e9ca6c0c3e1d30a7598efbd66034121b4af8dccc"
TOTAL_MOLECULES = 133885
EXCLUDED_COUNT = 3054
TRAIN_SIZE = 100000
CACHE_VERSION = 1
ELEMENTS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
PROPERTIES = ("mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv")
THERMO_PROPERTIES = ("zpve", "U0", "U", "H", "G", "Cv")
UNITS = ("D", "a0^3", "eV", "eV", "eV", "a0^2", "meV", "eV", "eV", "eV", "eV", "cal/(mol K)")
CONVERSIONS = {name: 27.2114 for name in ("homo", "lumo", "gap", "U0", "U", "H", "G")}
CONVERSIONS["zpve"] = 27211.4
RAW_FILES = ("dsgdb9nsd.xyz.tar.bz2", "uncharacterized.txt", "atomref.txt")


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def original_splits(excluded):
    """Return archive indices in the same order as official process_xyz_files."""
    excluded = set(excluded)
    if len(excluded) != EXCLUDED_COUNT or not excluded.issubset(range(TOTAL_MOLECULES)):
        raise ValueError("Invalid QM9 exclusion list")
    included = np.array(sorted(set(range(TOTAL_MOLECULES)) - excluded))
    order = np.random.RandomState(0).permutation(len(included))
    test_size = int(0.1 * len(included))
    val_end = len(included) - test_size
    selections = np.split(included[order], [TRAIN_SIZE, val_end])
    # Upstream filters tar members by split membership, preserving archive order.
    return {
        name: sorted(indices.tolist())
        for name, indices in zip(("train", "validation", "test"), selections)
    }


def parse_xyz(text):
    lines = text.splitlines()
    count = int(lines[0])
    props = lines[1].split()
    if len(props) != 17 or count < 1 or count > 29:
        raise ValueError("Invalid QM9 XYZ record")
    charges, positions = [], []
    for line in lines[2 : count + 2]:
        element, x, y, z, _ = line.replace("*^", "e").split()
        charges.append(ELEMENTS[element])
        positions.append([float(x), float(y), float(z)])
    return (
        int(props[1]), torch.tensor(charges, dtype=torch.long),
        torch.tensor(positions, dtype=torch.float32),
        torch.tensor([float(value) for value in props[5:]], dtype=torch.float32),
    )


def read_atomrefs(path):
    references = torch.zeros(10, len(PROPERTIES), dtype=torch.float64)
    found = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] in ELEMENTS:
            if len(fields) != 7 or fields[0] in found:
                raise ValueError("Invalid QM9 atom reference table")
            found.add(fields[0])
            for name, value in zip(THERMO_PROPERTIES, fields[1:]):
                references[ELEMENTS[fields[0]], PROPERTIES.index(name)] = float(value)
    if found != set(ELEMENTS):
        raise ValueError("QM9 atom reference table must contain H, C, N, O, F")
    return references


def prepare_qm9_orig(root):
    root = Path(root)
    cache = root / "processed/qm9Orig.pt"
    manifest_path = root / "processed/qm9Orig.json"
    if cache.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_version") != CACHE_VERSION:
            raise ValueError("Unsupported QM9Orig cache version; prepare a new dataset.root")
        if file_hash(cache) != manifest["cache_sha256"]:
            raise ValueError("QM9Orig cache checksum mismatch")
        # Raw files may be left behind on the preparation machine. If present, verify them.
        for name, checksum in manifest["raw_sha256"].items():
            raw = root / "raw" / name
            if raw.exists() and file_hash(raw) != checksum:
                raise ValueError(f"QM9Orig raw file changed: {raw}")
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        return payload, manifest

    raw = root / "raw"
    missing = [str(raw / name) for name in RAW_FILES if not (raw / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "QM9Orig requires offline raw files (or a complete processed cache). Missing: "
            + ", ".join(missing)
        )
    excluded = []
    for line in (raw / "uncharacterized.txt").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            excluded.append(int(fields[0]) - 1)
    if len(excluded) != len(set(excluded)):
        raise ValueError("Duplicate IDs in QM9 exclusion list")
    archive_splits = original_splits(excluded)
    included = sorted(set(range(TOTAL_MOLECULES)) - set(excluded))
    archive_to_cached = {index: i for i, index in enumerate(included)}
    size = len(included)
    charges = torch.zeros(size, 29, dtype=torch.long)
    positions = torch.zeros(size, 29, 3, dtype=torch.float32)
    labels = torch.empty(size, len(PROPERTIES), dtype=torch.float32)
    counts = torch.empty(size, dtype=torch.long)
    sample_indices = torch.empty(size, dtype=torch.long)
    seen = 0
    # Read members without extracting anything to the filesystem.
    with tarfile.open(raw / RAW_FILES[0], "r|bz2") as archive:
        for archive_index, member in enumerate(archive):
            if not member.isfile() or not member.name.endswith(".xyz"):
                raise ValueError("Unexpected member in QM9 archive")
            with archive.extractfile(member) as handle:
                molecule_id, z, pos, y = parse_xyz(handle.read().decode("utf-8"))
            if molecule_id != archive_index + 1:
                raise ValueError("QM9 archive order differs from official molecular IDs")
            seen += 1
            if archive_index not in archive_to_cached:
                continue
            index = archive_to_cached[archive_index]
            counts[index] = len(z)
            charges[index, : len(z)] = z
            positions[index, : len(z)] = pos
            labels[index] = y
            sample_indices[index] = molecule_id
    if seen != TOTAL_MOLECULES:
        raise ValueError(f"Expected {TOTAL_MOLECULES} XYZ records, found {seen}")
    references = read_atomrefs(raw / "atomref.txt")
    thermo = torch.zeros_like(labels, dtype=torch.float64)
    # Match upstream float64 count * reference, then sum by atomic species.
    for charge in ELEMENTS.values():
        thermo += (charges == charge).sum(1, keepdim=True).double() * references[charge]
    payload = {
        "charges": charges, "positions": positions, "labels": labels,
        "thermo": thermo.float(), "counts": counts, "indices": sample_indices,
        "splits": {
            name: [archive_to_cached[index] for index in indices]
            for name, indices in archive_splits.items()
        },
    }
    manifest = {
        "cache_version": CACHE_VERSION, "upstream_commit": UPSTREAM_COMMIT,
        "raw_sha256": {name: file_hash(raw / name) for name in RAW_FILES},
        "split_seed": 0, "num_molecules": size,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, cache)
    manifest["cache_sha256"] = file_hash(cache)
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)
    return payload, manifest


class QM9OrigDataset(Dataset):
    def __init__(self, payload, labels, indices):
        self.payload, self.labels, self.indices = payload, labels, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        count = int(self.payload["counts"][row])
        return Data(
            z=self.payload["charges"][row, :count],
            pos=self.payload["positions"][row, :count],
            y=self.labels[row : row + 1],
            sample_id=f"qm9orig:{int(self.payload['indices'][row])}",
        )


def build_qm9_orig(config, target):
    if target not in PROPERTIES:
        raise ValueError(f"Unknown original QM9 property {target!r}; choose from {PROPERTIES}")
    payload, provenance = prepare_qm9_orig(config.root)
    labels = payload["labels"].clone()
    if config.subtract_thermo:
        labels -= payload["thermo"]
    for name, factor in CONVERSIONS.items():
        labels[:, PROPERTIES.index(name)] *= factor
    parts = {}
    for name in ("train", "validation", "test"):
        limit = config.get(f"{name}_limit")
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise ValueError("Dataset limits must be positive integers or null")
        parts[name] = payload["splits"][name][:limit]
    ids = {
        name: [f"qm9orig:{int(payload['indices'][index])}" for index in indices]
        for name, indices in parts.items()
    }
    manifest = {
        "dataset": "qm9_orig", "seed": 0, "sample_ids": ids,
        "preprocessing": provenance, "subtract_thermo": bool(config.subtract_thermo),
        "target_convention": "atom_reference_subtracted" if config.subtract_thermo else "raw",
    }
    # Includes data/target provenance so prediction and evaluation cannot mix caches.
    manifest["split_id"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    index = PROPERTIES.index(target)
    return DatasetBundle(
        **{name: QM9OrigDataset(payload, labels, indices) for name, indices in parts.items()},
        manifest=manifest, target_index=index, unit=UNITS[index],
    )
