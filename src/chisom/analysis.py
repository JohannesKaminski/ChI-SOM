"""
Post-training analysis of a trained SOM: distance and (future) ranking
queries over the SOM's U-Distance graph (see `chisom.Som.u_graph`).
"""

from typing import Tuple, Union, Optional, Collection, cast

import networkx as nx
import numpy as np

PositionLike = Union[Tuple[int, int], np.ndarray]


def u_distance(
    graph: nx.Graph,
    source: Optional[PositionLike | Collection[PositionLike]],
    target: PositionLike,
) -> dict[tuple[int, int], float]:
    """
    Compute the u-distance(s) from source position(s) to a target position.

    The u-distance is the shortest-path length between a source and
    `target` through a U-Distance graph (see `chisom.Som.u_graph`),
    where edges are weighted by the high-dimensional distance between
    neighboring codebook vectors. Build the graph once via
    `chisom.Som.u_graph` and reuse it across many `u_distance` calls
    (e.g. for many BMU pairs) rather than rebuilding it per call.

    Parameters
    ----------
    graph : nx.Graph
        U-Distance graph, as returned by `chisom.Som.u_graph`.
    source
        Grid position(s) to compute the distance from:
            - `None`: every node in `graph`.
            - A single position (tuple/array of 2 ints): that one node.
            - A collection of positions: each given source node.
    target
        Grid position (row, column) to compute the distance to. Same
        accepted forms as a single `source` position.

    Returns
    -------
    dict[tuple[int, int], float]
        Mapping from each queried source node (as an int `(row, column)`
        tuple) to its u-distance to `target`. Has one entry per node in
        `graph` when `source is None`, a single entry for a single source
        position, or one entry per given source for a collection thereof.
    """
    target_row, target_col = target
    target_node = (int(target_row), int(target_col))

    if source is None:
        return nx.shortest_path_length(graph, target=target_node, weight="weight")

    if not isinstance(source, Collection) or len(source) == 0:
        raise ValueError(
            "source must be None, a PositionLike, or a non-empty collection thereof"
        )

    first = next(iter(source))
    if isinstance(first, Collection):
        source_nodes = []
        for s in cast(Collection[PositionLike], source):
            row, col = s
            source_nodes.append((int(row), int(col)))
    else:
        row, col = cast(PositionLike, source)
        source_nodes = [(int(row), int(col))]

    return {
        s: nx.shortest_path_length(graph, source=s, target=target_node, weight="weight")
        for s in source_nodes
    }



# Extension point for the planned "minimal u-ranking" feature: it will need
# many/all-pairs shortest-path queries over the same graph built once by
# Som.u_graph (e.g. ranking BMU positions by u-distance to a reference
# set). Prefer nx.multi_source_dijkstra or single-source
# nx.shortest_path_length(graph, source=..., weight="weight") for batches,
# rather than calling u_distance() in a loop, to avoid recomputing
# shortest-path trees redundantly. Not implemented here.
