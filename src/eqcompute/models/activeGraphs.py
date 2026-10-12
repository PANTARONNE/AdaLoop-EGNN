"""Compact complete molecules while retaining their current recurrent state."""

from dataclasses import replace

import torch


def select_graphs(state, graph, keep):
    if keep.dtype != torch.bool or keep.shape != (graph.num_graphs,):
        raise ValueError("keep must be a boolean mask over graphs")
    node_keep = keep[graph.batch_index]
    node_map = torch.full_like(graph.batch_index, -1)
    node_map[node_keep] = torch.arange(node_keep.sum(), device=node_map.device)
    graph_map = keep.long().cumsum(0) - 1

    def select_edges(edges):
        mask = node_keep[edges[0]] & node_keep[edges[1]]
        return node_map[edges[:, mask]], mask

    state_edges, _ = select_edges(state.edge_index)
    supplied_edges, supplied_attrs = None, None
    if graph.edge_index is not None:
        supplied_edges, edge_keep = select_edges(graph.edge_index)
        if graph.edge_attr is not None:
            supplied_attrs = graph.edge_attr[edge_keep]
    elif graph.edge_attr is not None:
        raise ValueError("edge_attr requires edge_index")
    next_graph = replace(
        graph,
        atomic_numbers=graph.atomic_numbers[node_keep],
        positions=graph.positions[node_keep],
        batch_index=graph_map[graph.batch_index[node_keep]],
        num_graphs=int(keep.sum()),
        edge_index=supplied_edges,
        edge_attr=supplied_attrs,
    )
    next_state = replace(
        state, node_features=state.node_features[node_keep],
        positions=state.positions[node_keep], edge_index=state_edges,
    )
    return next_state, next_graph
