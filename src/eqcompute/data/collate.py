from torch_geometric.data import Batch

from eqcompute.contracts import GraphInputs, SupervisedBatch


def collate_molecules(samples) -> SupervisedBatch:
    if not samples or any(sample.num_nodes == 0 for sample in samples):
        raise ValueError("A batch must contain nonempty molecules")
    batch = Batch.from_data_list(samples)
    return SupervisedBatch(
        graph=GraphInputs(
            atomic_numbers=batch.z,
            positions=batch.pos,
            batch_index=batch.batch,
            num_graphs=batch.num_graphs,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
        ),
        target=batch.y,
        sample_id=tuple(sample.sample_id for sample in samples),
    )
