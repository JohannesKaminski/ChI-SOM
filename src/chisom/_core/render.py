from typing import Collection, Union

import numpy as np
from numba import jit, prange
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

EARTH_COLORS = [
    (0, 0, 245),
    (0, 0, 245),
    (55, 125, 34),
    (55, 125, 34),
    (62, 131, 38),
    (22, 131, 38),
    (93, 144, 54),
    (93, 144, 54),
    (142, 160, 78),
    (142, 160, 78),
    (189, 176, 101),
    (189, 176, 101),
    (188, 165, 95),
    (188, 165, 95),
    (173, 149, 82),
    (173, 149, 82),
    (156, 130, 68),
    (156, 130, 68),
    (136, 109, 52),
    (136, 109, 52),
    (105, 77, 28),
    (105, 77, 28),
    (233, 229, 223),
    (233, 229, 223),
    (255, 255, 255),
    (255, 255, 255),
]
CYCLIC_COLORS = ["#FFFFFF", "#268086", "#FFFFFF"]


def create_stops(n_classes: int):
    """
    Create the stops necessary to create a non-continuous colormap with n_classes

    Parameters
    ----------
    n_classes : int
        number of discrete colors

    Returns
    -------
    ndarray[np.int32]
        Stops for the color values
    """

    color_start = np.linspace(0, 1, n_classes + 1)
    color_end = np.linspace(0, 1, n_classes + 1) - 0.00001
    stops = np.empty((n_classes * 2), dtype=np.float32)

    for i in range(len(color_start) - 1):
        stops[(i * 2)] = color_start[i]
        stops[(i * 2) + 1] = color_end[i + 1]
    stops[-1] = color_start[-1]

    return stops


EARTH_POS = create_stops(13)
CYCLIC_POS = [0, 0.5, 1]


def min_max(array: NDArray) -> tuple[NDArray, float, float]:
    minimum = np.min(array)
    maximum = np.max(array)

    return (array - minimum) / (maximum - minimum), minimum, maximum


def interpolate_matrix(
    matrix: NDArray[np.float16], scaling: int
) -> NDArray[np.float16]:
    """
    Interpolates a matrix by a given scaling factor.

    Parameters
    ----------
    matrix : NDArray[np.float16]
        matrix to interpolate
    scaling : int
        scaling factor

    Returns
    -------
    NDArray[np.float16]
        Interpolated umatrix
    """
    # Double the matrix to handle border interpolation
    if scaling < 2:
        return matrix

    fourfold_matrix = np.tile(matrix, (2, 2))

    # create places of known values on grid for interpolation
    rows, cols = fourfold_matrix.shape
    row_steps = np.linspace(0, rows * scaling, rows, dtype=int, endpoint=False)
    col_steps = np.linspace(0, cols * scaling, cols, dtype=int, endpoint=False)

    # create spline function for interpolation
    interpolation_function = RegularGridInterpolator(
        (row_steps, col_steps), fourfold_matrix
    )

    new_rows, new_cols = np.meshgrid(
        range(row_steps.max() + 1),
        range(col_steps.max() + 1),
        indexing="ij",
    )

    # interpolate umatrix
    interpolated_matrix = interpolation_function((new_rows, new_cols))

    # cut interpolated umatrix to orginal projection area
    interpolated_matrix_cut = interpolated_matrix[
        : matrix.shape[0] * scaling, : matrix.shape[1] * scaling
    ]

    # shift interpolated umatrix to have the origins of the original points in the center of an interpolated area
    padding = scaling // 2
    original_rows, original_cols = matrix.shape
    final_interpolated_matrix = np.empty(
        (original_rows * scaling, original_cols * scaling), dtype=np.float16
    )
    # upper left corner
    final_interpolated_matrix[:padding, :padding] = interpolated_matrix_cut[
        -padding:, -padding:
    ]
    # left edge
    final_interpolated_matrix[padding:, :padding] = interpolated_matrix_cut[
        :-padding, -padding:
    ]
    # top edge
    final_interpolated_matrix[:padding, padding:] = interpolated_matrix_cut[
        -padding:, :-padding
    ]
    # rest of map
    final_interpolated_matrix[padding:, padding:] = interpolated_matrix_cut[
        :-padding, :-padding
    ]

    return final_interpolated_matrix


@jit(cache=True, parallel=True)
def create_bmu_composition(
    bmu_id_for_datapoint: NDArray[np.uint32],
    class_as_id: NDArray[np.uint16],
    num_bmus: int,
    num_bins: int,
) -> NDArray[np.uint16]:
    occurances = np.zeros((num_bmus, num_bins), dtype=np.uint16)

    for unique_color in prange(num_bins):
        mask = class_as_id == unique_color

        group_values = bmu_id_for_datapoint[mask]
        occurances[:, unique_color] = np.bincount(group_values, minlength=num_bmus)[
            np.newaxis, ...
        ]

    return occurances


def bmu_raw_to_map_coordinates(
    bmu_coordinates: NDArray[np.uint16],
    scaling_factor: int,
) -> NDArray[np.float16]:

    return (bmu_coordinates * scaling_factor + scaling_factor // 2 + 0.5).astype(
        np.float16
    )


class RatioWeighting:
    @staticmethod
    def gini_coefficient(x, axis=1):
        """
        Compute Gini coefficient along specified axis of a 2D matrix,
        considering only non-zero values. Uses vectorized operations for efficiency.

        Parameters:
        -----------
        x : 2D numpy array or array-like
            Input matrix
        axis : int, default=1
            Axis along which to compute the Gini coefficient (0 for rows, 1 for columns)

        Returns:
        --------
        NDArray
            Array of Gini coefficients for each row/column
        """
        x = np.asarray(x)

        # Create a mask for non-zero values
        nonzero_mask = x != 0

        # Count non-zero elements along the specified axis
        n_nonzero = np.sum(nonzero_mask, axis=axis)

        # Create arrays to store results
        result = np.ones(x.shape[1 - axis])

        # For rows/slices with at least 2 non-zero values
        valid_indices = np.where(n_nonzero >= 2)[0]

        if len(valid_indices) > 0:
            # Process each valid row/column
            for idx in valid_indices:
                if axis == 1:
                    # Get the non-zero values for this row
                    values = x[idx][nonzero_mask[idx]]
                else:
                    # Get the non-zero values for this column
                    values = x[:, idx][nonzero_mask[:, idx]]

                # Efficient computation of all pairwise absolute differences
                diff_matrix = np.abs(np.subtract.outer(values, values))

                # Calculate the Gini coefficient
                gini = np.sum(diff_matrix) / (2 * len(values) * np.sum(values))
                result[idx] = gini

        return result

    @staticmethod
    def excess_coefficient_absolute(x, axis=1):
        # Get indices that sort the array along the specified axis
        if x.shape[axis] < 2:
            return np.ones(len(x))
        sorted_x = np.argsort(x, axis=axis)
        # Get the largest and second largest values along the specified axis
        largest_x = np.take_along_axis(x, sorted_x[:, -1:], axis=axis)
        second_largest_x = np.take_along_axis(x, sorted_x[:, -2:-1], axis=axis)
        # Calculate the excess
        excess = largest_x - second_largest_x
        excess[excess < 0] = 0
        return excess.flatten()

    @staticmethod
    def excess_coefficient_relative(x, axis=1):
        # Get indices that sort the array along the specified axis
        sorted_x = np.argsort(x, axis=axis)
        # Get the largest and second largest values along the specified axis
        largest_x = np.take_along_axis(x, sorted_x[:, -1:], axis=axis)
        second_largest_x = np.take_along_axis(x, sorted_x[:, -2:-1], axis=axis)
        # Calculate the excess
        excess = 1 - second_largest_x / largest_x
        # Set negative excess values to 0
        # This is done to avoid negative excess values, which can occur if the second largest value is larger than the largest value
        excess[excess < 0] = 0
        return excess.flatten()

    SCHEMES = {
        "Gini Coefficient": gini_coefficient,
        "Excess Coefficient (Absolute)": excess_coefficient_absolute,
        "Excess Coefficient (Relative)": excess_coefficient_relative,
    }
    DEFAULT_SCHEME = "Excess Coefficient (Absolute)"

    @staticmethod
    def average_for_coordinate(
        values: Union[NDArray, Collection], coordinate_id: NDArray
    ) -> NDArray:
        # Use bincount with weights to calculate sums for each unique index
        _values = np.asarray(values)
        sums = np.bincount(coordinate_id, weights=_values)
        # Use bincount to calculate counts for each unique index
        counts = np.bincount(coordinate_id)

        # Calculate average by dividing sums by counts
        return np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)

    @staticmethod
    def ratio_for_coordinate(
        values: Union[NDArray, Collection],
        bmu_id_for_datapoint: NDArray[np.uint32],
        bins: list,
        num_bmus: int,
    ) -> NDArray[np.float16]:
        # Initialize result array
        num_bins = len(bins)
        occurances = np.zeros((num_bmus, num_bins), dtype=np.uint16)
        _values = np.asarray(values)

        _, class_as_id = np.unique_inverse(_values)
        class_as_id = np.astype(class_as_id, np.uint16)

        # Process each bmu
        occurances = create_bmu_composition(
            bmu_id_for_datapoint, class_as_id, num_bmus, num_bins
        )

        counts = np.sum(occurances, axis=1)
        ratios = np.divide(
            occurances,
            counts[:, np.newaxis],
            out=np.zeros_like(occurances, dtype=np.float16),
        )
        ratios[np.isnan(ratios)] = 0

        return ratios
