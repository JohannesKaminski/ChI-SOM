"""
U-Distance graph construction: builds a networkx graph of SOM grid positions
connected by toroidal grid-neighbor edges, weighted by the high-dimensional
(feature-space) distance between the corresponding codebook vectors. Reuses
the same per-neighbor distance computation as U-Matrix construction (see
`chisom._core.cpu.umatrix`).
"""

from collections.abc import Callable

import networkx as nx
import numpy as np

from chisom._core.cpu.distance import make_universal_distance_func
from chisom._core.cpu.umatrix import compute_neighbor_distances, NeighborDistances
from chisom._core.types import Codebook

def u_graph_from_neighbor_distances(distances: NeighborDistances, rows: int, columns: int) -> nx.Graph:
    """
    Build the U-Distance graph for a codebook.

    Nodes are grid positions (row, column) as native int tuples (all
    rows * columns positions). Edges connect toroidal grid neighbors
    (2 * rows * columns edges total, no duplicates), each weighted
    (the "weight" edge attribute) by the high-dimensional distance
    between the corresponding codebook vectors.

    CAVE: This function assumes a toroidal topology

    Parameters
    ----------
    distances : NeighborDistances
        The neighbor distances computed from the codebook.
    rows : int
        The number of rows in the codebook.
    columns : int
        The number of columns in the codebook.

    Returns
    -------
    nx.Graph
        The U-Distance graph.
    """
    # TODO: Add option for non-toroidal topology
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from((r, c) for r in range(rows) for c in range(columns))

    # Every vertical grid edge is uniquely represented by (r, c) -> (r-1, c)
    # via prev_row_distance, and every horizontal grid edge uniquely by
    # (r, c) -> (r, c-1) via prev_col_distance. Together these cover all
    # 2 * rows * columns toroidal grid edges exactly once (see
    # compute_neighbor_distances for the exact offset semantics).
    row_idx, col_idx = np.meshgrid(
        np.arange(rows), np.arange(columns), indexing="ij"
    )
    row_idx = row_idx.ravel()
    col_idx = col_idx.ravel()
    prev_row_idx = (row_idx - 1) % rows
    prev_col_idx = (col_idx - 1) % columns

    nodes = list(zip(row_idx.tolist(), col_idx.tolist()))

    vertical_edges = zip(
        nodes,
        zip(prev_row_idx.tolist(), col_idx.tolist()),
        distances.prev_row_distance.ravel().astype(float).tolist(),
    )
    horizontal_edges = zip(
        nodes,
        zip(row_idx.tolist(), prev_col_idx.tolist()),
        distances.prev_col_distance.ravel().astype(float).tolist(),
    )

    graph.add_weighted_edges_from(vertical_edges)
    graph.add_weighted_edges_from(horizontal_edges)

    return graph


def make_u_graph_calculation(vector_dist_norm: str) -> Callable[[Codebook], nx.Graph]:
    """
    Create a function to build the U-Distance graph for a codebook.

    Parameters
    ----------
    vector_dist_norm : str
        The type of vector distance norm to use for the feature-space edge
        weights.

    Returns
    -------
    Callable[[Codebook], nx.Graph]
        A function that builds an undirected, edge-weighted networkx graph
        from a codebook.
    """
    vector_dist_func = make_universal_distance_func(vector_dist_norm)

    def calculate_u_graph(codebook: Codebook) -> nx.Graph:
        rows, columns = codebook.shape[:2]
        distances = compute_neighbor_distances(codebook, vector_dist_func)
        return u_graph_from_neighbor_distances(distances, rows, columns)

    return calculate_u_graph
