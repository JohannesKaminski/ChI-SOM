"""
Headless, matplotlib-based plotting of trained SOMs for use in scripts,
without the need for the interactive Qt viewer.
"""

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from pandas import DataFrame

from chisom._core.render import (
    EARTH_COLORS,
    EARTH_POS,
    RatioWeighting,
    bmu_raw_to_map_coordinates,
    interpolate_matrix,
)
from chisom._core.types import UMatrix
from chisom.io.datastores import DatasetBase

# The same stepped topographic colormap the Qt viewer uses as default
EarthColorMap = LinearSegmentedColormap.from_list(
    "chisom_earth",
    list(zip(EARTH_POS.tolist(), (np.asarray(EARTH_COLORS) / 255).tolist())),
)


def plot_som(
    umatrix: UMatrix,
    bmu_coordinates: Optional[NDArray[np.uint16]] = None,
    data: Optional[Union[DatasetBase, DataFrame]] = None,
    color_by: Optional[str] = None,
    *,
    categorical: Optional[bool] = None,
    category_colors: Optional[Mapping[Any, Any]] = None,
    cmap: Union[str, Colormap] = "viridis",
    umatrix_cmap: Optional[Union[str, Colormap]] = None,
    alpha_scheme: Optional[str] = "excess_absolute",
    layer: int = -1,
    scaling_factor: int = 3,
    marker_size: float = 10.0,
    figsize: Optional[tuple[float, float]] = None,
    dpi: int = 150,
    ax: Optional[Axes] = None,
    save_as: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plot a trained SOM as a static matplotlib figure.

    Draws the interpolated U-matrix as background image and, if BMU
    coordinates are given, the BMUs as scatter markers on top — optionally
    colored by a property of the underlying data. Reproduces the default
    look of the interactive viewer, but works headless (e.g. with the
    Agg backend) and can be written directly to file.

    Parameters
    ----------
    umatrix : UMatrix
        U-matrix as returned by ``Som.get_umatrix()``. Either 2D
        (rows, columns) or 3D (layers, rows, columns).
    bmu_coordinates : NDArray[np.uint16], optional
        (N, 2) array of (row, column) BMU coordinates as returned by
        ``Som.predict()``. If None, only the U-matrix is drawn.
    data : DatasetBase or DataFrame, optional
        Data source holding the property values referenced by `color_by`.
        Must have one entry per row in `bmu_coordinates`.
    color_by : str, optional
        Name of the column in `data` to color the BMUs by. If None, BMUs
        are drawn in plain black.
    categorical : bool, optional
        Force categorical (True) or continuous (False) coloring. By default
        this is taken from the column properties of `data` (for a plain
        DataFrame the heuristic of the viewer is used: at most 10 unique
        values within the first 100 rows counts as categorical).
    category_colors : Mapping, optional
        Mapping of category value to matplotlib color, used for categorical
        coloring. By default colors are assigned from the ``tab10``/``tab20``
        palettes in sorted category order.
    cmap : str or Colormap
        Matplotlib colormap for continuous property coloring.
    umatrix_cmap : str or Colormap, optional
        Matplotlib colormap for the U-matrix. Defaults to the Earth
        colormap of the viewer.
    alpha_scheme : {"gini", "excess_absolute", "excess_relative"}, optional
        Weighting scheme used to encode the dominance of the primary
        category as marker opacity (see ``RatioWeighting``). If None,
        markers are fully opaque. Ignored for continuous coloring.
    layer : int
        Layer to display for a 3D U-matrix.
    scaling_factor : int
        Upscaling factor for the U-matrix interpolation.
    marker_size : float
        BMU marker diameter in points.
    figsize : tuple of float, optional
        Figure size in inches, passed to matplotlib.
    dpi : int
        Figure resolution, passed to matplotlib.
    ax : Axes, optional
        Existing axes to draw into. If given, `figsize` and `dpi` are
        ignored and the legend is placed inside the axes.
    save_as : str or Path, optional
        If given, the figure is saved to this path; the format is inferred
        from the suffix (e.g. ``.pdf``).

    Returns
    -------
    Figure
        The matplotlib figure, for further customization.
    """
    _umatrix = np.asarray(umatrix)
    if _umatrix.ndim == 2:
        _umatrix = _umatrix[np.newaxis, ...]
    elif _umatrix.ndim != 3:
        raise ValueError(
            "umatrix must be 2D (rows, columns) or 3D (layers, rows, columns)"
        )
    if scaling_factor < 1:
        raise ValueError("Scaling factor must be greater than 0")

    selected_values = _umatrix[layer]
    rows, columns = selected_values.shape

    if ax is None:
        own_figure = True
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, layout="constrained")
    else:
        own_figure = False
        root_figure = ax.get_figure(root=True)
        if not isinstance(root_figure, Figure):
            raise ValueError("ax must belong to a Figure")
        fig = root_figure

    scaled_values = interpolate_matrix(selected_values, scaling_factor)
    image = ax.imshow(
        scaled_values,
        cmap=EarthColorMap if umatrix_cmap is None else umatrix_cmap,
        # With this extent, pixel centers sit at half-integers, so BMU
        # markers land exactly where the viewer places them
        extent=(0, columns * scaling_factor, rows * scaling_factor, 0),
        origin="upper",
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, label="U-height")

    if bmu_coordinates is not None:
        unique_bmu_coordinates, index_to_unique_mapping = np.unique(
            bmu_coordinates, axis=0, return_inverse=True
        )
        map_coordinates = bmu_raw_to_map_coordinates(
            np.astype(unique_bmu_coordinates, np.uint16), scaling_factor
        )

        face_colors: Union[str, NDArray] = "black"
        if color_by is not None:
            if data is None:
                raise ValueError("color_by requires a data source")
            values = _get_column_values(data, color_by)
            if len(values) != len(bmu_coordinates):
                raise ValueError(
                    f"data has {len(values)} rows but bmu_coordinates has {len(bmu_coordinates)}"
                )

            if _is_categorical(data, color_by, values, categorical):
                face_colors, categories, resolved_colors = _categorical_face_colors(
                    values,
                    np.astype(index_to_unique_mapping, np.uint32),
                    len(unique_bmu_coordinates),
                    category_colors,
                    alpha_scheme,
                )
                handles = [
                    Line2D(
                        [],
                        [],
                        linestyle="",
                        marker="o",
                        markerfacecolor=resolved_colors[category],
                        markeredgecolor="black",
                        label=str(category),
                    )
                    for category in categories
                ]
                if own_figure:
                    fig.legend(
                        handles=handles, title=color_by, loc="outside right upper"
                    )
                else:
                    ax.legend(handles=handles, title=color_by)
            else:
                property_cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
                face_colors, norm = _continuous_face_colors(
                    values, np.astype(index_to_unique_mapping, np.uint32), property_cmap
                )
                fig.colorbar(
                    ScalarMappable(norm=norm, cmap=property_cmap), ax=ax, label=color_by
                )

        ax.scatter(
            map_coordinates[:, 1],
            map_coordinates[:, 0],
            s=marker_size**2,
            facecolors=face_colors,
            edgecolors="black",
            linewidths=1.5,
        )

    if save_as is not None:
        fig.savefig(save_as)

    return fig


def _get_column_values(
    data: Union[DatasetBase, DataFrame], column_name: str
) -> NDArray:
    if isinstance(data, DataFrame):
        return data[column_name].to_numpy()
    if column_name not in data.columns_with_properties:
        raise KeyError(
            f"Column '{column_name}' not found; available: {data.column_names}"
        )
    values = data.get_values_for_column(column_name)
    return np.asarray(values)


def _is_categorical(
    data: Union[DatasetBase, DataFrame],
    column_name: str,
    values: NDArray,
    override: Optional[bool],
) -> bool:
    if override is not None:
        return override
    if isinstance(data, DataFrame):
        # Same heuristic as the viewer's DataFrameSource
        return len(set(values[:100])) <= 10

    value_type = data.columns_with_properties[column_name].value_type
    if value_type == "na":
        raise ValueError(
            f"Column '{column_name}' has no usable value type for coloring"
        )
    return value_type == "categorical"


def _continuous_face_colors(
    values: NDArray, index_to_unique_mapping: NDArray[np.uint32], cmap: Colormap
) -> tuple[NDArray, Normalize]:
    averages = RatioWeighting.average_for_coordinate(values, index_to_unique_mapping)

    minimum = float(np.min(averages))
    maximum = float(np.max(averages))
    if minimum == maximum:
        normalized = np.full(len(averages), 0.5)
    else:
        normalized = (averages - minimum) / (maximum - minimum)

    return cmap(normalized), Normalize(minimum, maximum)


def _categorical_face_colors(
    values: NDArray,
    index_to_unique_mapping: NDArray[np.uint32],
    num_bmus: int,
    category_colors: Optional[Mapping[Any, Any]],
    alpha_scheme: Optional[str],
) -> tuple[NDArray, NDArray, Mapping[Any, Any]]:
    # Sorted, matching the class order np.unique_inverse produces in ratio_for_coordinate
    categories = np.unique(np.asarray(values))

    if category_colors is None:
        if len(categories) > 20:
            raise ValueError(
                f"Column has {len(categories)} categories; pass explicit category_colors"
            )
        palette = plt.get_cmap("tab10" if len(categories) <= 10 else "tab20")
        category_colors = {
            category: palette(i) for i, category in enumerate(categories)
        }
    else:
        missing = [c for c in categories if c not in category_colors]
        if missing:
            raise ValueError(f"category_colors is missing entries for {missing}")

    ratios = RatioWeighting.ratio_for_coordinate(
        values, index_to_unique_mapping, list(categories), num_bmus
    )
    primary_category = np.argmax(ratios, axis=1)

    rgba = np.asarray([to_rgba(category_colors[c]) for c in categories])
    face_colors = rgba[primary_category]

    if alpha_scheme is not None:
        if alpha_scheme not in RatioWeighting.SCHEMES:
            raise ValueError(
                f"Unknown alpha_scheme '{alpha_scheme}', expected one of {list(RatioWeighting.SCHEMES)}"
            )
        weighting = RatioWeighting.SCHEMES[alpha_scheme]
        face_colors[:, 3] = np.clip(weighting(np.astype(ratios, np.float32)), 0, 1)

    return face_colors, categories, category_colors
