from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from chisom._core.cpu.distance import make_universal_distance_func
from chisom._core.cpu.types import BroadcastingDistanceFunction
from chisom._core.types import Codebook, MapValues, UMatrix

FASTMATH_FLAG = False


class NeighborDistances(NamedTuple):
    """
    High-dimensional distances from every grid position to each of its 4
    toroidal grid neighbors.

    All fields have shape (rows, columns) and dtype float32.
    """

    prev_row_distance: MapValues
    """prev_row_distance[r, c] = dist(codebook[r, c], codebook[(r-1) % rows, c])"""
    next_row_distance: MapValues
    """next_row_distance[r, c] = dist(codebook[r, c], codebook[(r+1) % rows, c])"""
    prev_col_distance: MapValues
    """prev_col_distance[r, c] = dist(codebook[r, c], codebook[r, (c-1) % columns])"""
    next_col_distance: MapValues
    """next_col_distance[r, c] = dist(codebook[r, c], codebook[r, (c+1) % columns])"""


def compute_neighbor_distances(
    codebook: Codebook, vector_dist_func: BroadcastingDistanceFunction
) -> NeighborDistances:
    """
    Compute high-dimensional distances between every neuron and its 4
    toroidal grid neighbors (previous/next row, previous/next column).

    Shared by U-Matrix construction (`make_umatrix_calculation`) and
    U-Distance graph construction (`chisom._core.graph.make_u_graph_calculation`)
    so the per-neighbor distance computation is not duplicated between the two.

    CAVE: This function assumes a toroidal topology.

    Parameters
    ----------
    codebook : Codebook
        3D array of shape (rows, columns, features) representing the codebook.
    vector_dist_func : BroadcastingDistanceFunction
        Broadcasting distance function, as returned by `make_universal_distance_func`.

    Returns
    -------
    NeighborDistances
        The 4 directional neighbor-distance arrays, each shape (rows, columns).
    """
    # TODO: Add option for non-toroidal topology
    prev_row_matrix = np.concat((codebook[-1:, :, :], codebook[:-1, :, :]), axis=0)
    prev_row_distance = vector_dist_func(codebook, prev_row_matrix)
    next_row_distance = np.concat(
        (prev_row_distance[1:, :], prev_row_distance[:1, :]), axis=0
    )

    prev_col_matrix = np.concat((codebook[:, -1:, :], codebook[:, :-1, :]), axis=1)
    prev_col_distance = vector_dist_func(codebook, prev_col_matrix)
    next_col_distance = np.concat(
        (prev_col_distance[:, 1:], prev_col_distance[:, :1]), axis=1
    )

    return NeighborDistances(
        prev_row_distance, next_row_distance, prev_col_distance, next_col_distance
    )

def umatrix_from_neighbor_distances(neighbor_distances: NeighborDistances) -> UMatrix:
    umatrix = (
        neighbor_distances.prev_row_distance + neighbor_distances.next_col_distance +
        neighbor_distances.next_row_distance + neighbor_distances.prev_col_distance
    )
    u_min = np.min(umatrix)
    u_max = np.max(umatrix)
    umatrix = (umatrix - u_min) / (u_max - u_min)
    return np.astype(umatrix, np.float16)

def make_umatrix_calculation(vector_dist_norm: str) -> Callable[[Codebook], UMatrix]:
    """
    Create a function to calculate the U-Matrix for a given distance norm.

    Parameters
    ----------
    vector_dist_norm : str
        The type of vector distance norm to use for U-Matrix calculation.

    Returns
    -------
    Callable[[Codebook], MapValues]
        A function that takes a 3D array of shape (height, width, dimensions)
        and returns the U-Matrix as a 2D array of shape (height, width).
        The U-Matrix is normalized to the range [0, 1].
    """
    vector_dist_func = make_universal_distance_func(vector_dist_norm)

    def calculate_umatrix(codebook: Codebook) -> UMatrix:
        """
        Calculate the U-Matrix for a given 3D array of shape (height, width, dimensions).
        This is done by calculating the distances to the neighboring neurons
        and normalizing the resulting matrix to the range [0, 1].
        CAVE: This function assumes a toroidal topology

        Parameters
        ----------
        codebook : Codebook
            3D array of shape (height, width, dimensions) representing the data.

        Returns
        -------
        NDArray[np.float16]
            2D array of shape (height, width) representing the U-Matrix,
            normalized to the range [0, 1].
        """
        distances = compute_neighbor_distances(codebook, vector_dist_func)

        return umatrix_from_neighbor_distances(distances)

    return calculate_umatrix
