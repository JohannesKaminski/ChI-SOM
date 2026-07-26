import networkx as nx
import numpy as np
import pytest

from chisom._core.cpu.distance import make_universal_distance_func
from chisom._core.cpu.umatrix import compute_neighbor_distances
from chisom._core.graph import make_u_graph_calculation
from chisom.analysis import u_distance


def _make_codebook() -> np.ndarray:
    # rows=3, cols=3, features=2 (feature values duplicated to make
    # manhattan distance = 2 * |v1 - v2|, trivial to hand-check).
    # Column values: col0=0, col1=10, col2=0 -- makes the toroidal
    # col0<->col2 wraparound edge cheap (weight 0) and the "reading order"
    # path through col1 expensive (weight 20 each hop).
    col_values = np.array([0.0, 10.0, 0.0], dtype=np.float32)
    codebook = np.tile(col_values[np.newaxis, :, np.newaxis], (3, 1, 2))
    return np.ascontiguousarray(codebook, dtype=np.float32)


@pytest.fixture
def codebook():
    return _make_codebook()


@pytest.fixture
def graph(codebook):
    return make_u_graph_calculation("manhattan")(codebook)


def test_graph_has_expected_nodes_and_edges(graph):
    assert graph.number_of_nodes() == 9
    assert graph.number_of_edges() == 2 * 9
    assert set(graph.nodes) == {(r, c) for r in range(3) for c in range(3)}


def test_vertical_edges_are_zero_weight(graph):
    # all rows are identical -> every row-direction edge weight is 0
    for c in range(3):
        for r in range(3):
            assert graph[(r, c)][((r + 1) % 3, c)]["weight"] == pytest.approx(0.0)


def test_toroidal_column_wraparound_is_cheap(graph):
    # col0 <-> col2 wraps around and both have value 0 -> weight 0
    assert graph[(0, 0)][(0, 2)]["weight"] == pytest.approx(0.0)
    # col0 <-> col1 and col1 <-> col2 are both expensive (|10-0| * 2 = 20)
    assert graph[(0, 0)][(0, 1)]["weight"] == pytest.approx(20.0)
    assert graph[(0, 1)][(0, 2)]["weight"] == pytest.approx(20.0)


def test_u_distance_between_adjacent_nodes_equals_edge_weight(graph):
    assert u_distance(graph, (0, 0), (0, 1)) == pytest.approx({(0, 0): 20.0})


def test_u_distance_finds_cheaper_multihop_over_direct_reading_order(graph):
    # (0,0) -> (0,2): direct toroidal wraparound edge (weight 0) is cheaper
    # than the "reading order" 2-hop path via (0,1) (weight 20 + 20 = 40).
    assert u_distance(graph, (0, 0), (0, 2)) == pytest.approx({(0, 0): 0.0})


def test_u_distance_accepts_numpy_bmu_like_input(graph):
    source = np.array([0, 0], dtype=np.uint16)
    target = np.array([0, 2], dtype=np.uint16)
    assert u_distance(graph, source, target) == pytest.approx({(0, 0): 0.0})


def test_u_distance_all_sources_returns_one_entry_per_node(graph):
    target = (0, 0)
    result = u_distance(graph, None, target)
    expected = {
        node: nx.shortest_path_length(graph, source=node, target=target, weight="weight")
        for node in graph.nodes()
    }
    assert result == pytest.approx(expected)


def test_u_distance_multiple_sources_returns_one_entry_per_source(graph):
    sources = [(0, 0), (1, 1)]
    target = (0, 2)
    result = u_distance(graph, sources, target)
    assert result == pytest.approx(
        {
            (0, 0): u_distance(graph, sources[0], target)[(0, 0)],
            (1, 1): u_distance(graph, sources[1], target)[(1, 1)],
        }
    )


def test_u_distance_rejects_empty_source(graph):
    with pytest.raises(ValueError):
        u_distance(graph, [], (0, 0))


def test_incident_edge_weights_match_umatrix_pre_normalization_sum(codebook, graph):
    # Cross-check: the graph's 4 incident edge weights for a node must sum
    # to the same pre-normalization value make_umatrix_calculation sums
    # before its min-max scaling -- recomputed here via the shared
    # compute_neighbor_distances helper that umatrix.py itself calls
    # internally.
    vector_dist_func = make_universal_distance_func("manhattan")
    distances = compute_neighbor_distances(codebook, vector_dist_func)

    for r in range(3):
        for c in range(3):
            expected = (
                distances.prev_row_distance[r, c]
                + distances.next_row_distance[r, c]
                + distances.prev_col_distance[r, c]
                + distances.next_col_distance[r, c]
            )
            incident_sum = sum(
                graph[(r, c)][neighbor]["weight"]
                for neighbor in graph.neighbors((r, c))
            )
            assert incident_sum == pytest.approx(float(expected))
