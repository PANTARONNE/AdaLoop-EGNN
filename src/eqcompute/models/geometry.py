import torch
from torch import Tensor

from eqcompute.contracts import GraphInputs


def radius_edges(graph: GraphInputs, cutoff: float) -> Tensor:
    """Exact per-molecule radius graph; no neighbor cap or optional CUDA extension.

    Quadratic per molecule, appropriate for QM9. Replace this helper for large graphs.
    Edges are directed (source, destination); self loops and cross-graph edges excluded.
    """
    edges = []
    for graph_id in range(graph.num_graphs):
        nodes = torch.where(graph.batch_index == graph_id)[0]
        xyz = graph.positions[nodes]
        distances = torch.cdist(xyz, xyz, compute_mode="donot_use_mm_for_euclid_dist")
        mask = distances < cutoff
        mask.fill_diagonal_(False)
        src, dst = mask.nonzero(as_tuple=True)
        edges.append(torch.stack((nodes[src], nodes[dst])))
    return torch.cat(edges, dim=1)


def aggregate(values: Tensor, index: Tensor, size: int, reduction="sum") -> Tensor:
    result = values.new_zeros((size, values.size(-1))).index_add(0, index, values)
    if reduction == "mean":
        counts = values.new_zeros(size).index_add(0, index, values.new_ones(index.numel()))
        result = result / counts.clamp_min(1).unsqueeze(-1)
    elif reduction != "sum":
        raise ValueError(f"Unsupported reduction: {reduction}")
    return result
