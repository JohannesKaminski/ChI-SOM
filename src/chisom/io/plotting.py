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

# Tunable layout defaults for `plot_som`. These only take effect where the
# corresponding keyword argument is left at its default (None/unset);
# override the keyword argument on a given call rather than editing these.
DEFAULT_CELL_SIZE_IN = 0.2  # inches per SOM grid cell, used to size `figsize`
CHROME_ALLOWANCE_IN = 1.6  # inches reserved for the colorbar(s)/outside legend
DEFAULT_COLORBAR_FRACTION = 0.15  # matplotlib's own colorbar default
DEFAULT_COLORBAR_PAD = 0.05  # matplotlib's own colorbar default
DEFAULT_LEGEND_FONTSIZE_PT = 9.0


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
    marker_size: Optional[float] = None,
    marker_cell_fraction: float = 0.7,
    legend_ncol: Optional[int] = None,
    legend_label_maxlen: int = 24,
    chrome_scale: float = 1.0,
    figsize: Optional[tuple[float, float]] = None,
    cell_size_in: float = DEFAULT_CELL_SIZE_IN,
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

    Sizing of the map image, the BMU markers, and the colorbar(s)/legend is
    coupled by default: the figure is sized from the SOM's actual grid shape
    (`cell_size_in`), the axes box is aspect-locked to that same grid shape
    (`rows / columns`), and marker size is derived from how large a single
    grid cell actually renders once that layout is resolved. This keeps the
    map, its markers, and its chrome proportionate to each other regardless
    of how big or how non-square a given SOM's lattice is. Pass explicit
    `figsize`/`marker_size` to opt back into fixed, grid-independent sizing.

    Parameters
    ----------
    umatrix : UMatrix
        U-matrix as returned by the ``Som.umatrix`` property. Either 2D
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
    marker_size : float, optional
        BMU marker diameter in points. By default (None), it is derived
        from the rendered size of one SOM grid cell (see
        `marker_cell_fraction`) so markers stay proportionate to the map
        regardless of grid shape or figure size. Pass a float to pin an
        exact, grid-independent size instead.
    marker_cell_fraction : float
        When `marker_size` is auto-derived, the fraction of one rendered
        grid cell's size (the smaller of its width/height in points) that
        the marker diameter should occupy. Ignored if `marker_size` is set.
    legend_ncol : int, optional
        Number of columns for the categorical-coloring legend. By default
        (None), chosen automatically so the legend wraps into additional
        columns rather than growing arbitrarily tall for large category
        counts (see `_auto_legend_ncol`).
    legend_label_maxlen : int
        Maximum rendered length of a categorical legend label before it is
        elided with "…", so a few very long category names can't blow up
        the legend's (and therefore the map's) reserved width.
    chrome_scale : float
        Single multiplier applied together to the auto-derived marker size,
        the colorbar's `fraction`/`pad`, and the legend/colorbar font size,
        and to the chrome width reserved in an auto-derived `figsize`. Use
        this to scale all non-map chrome up or down in one step (e.g. for a
        much larger or smaller `figsize`) instead of tuning each piece of
        chrome separately. Does not affect an explicitly-passed
        `marker_size` or `figsize`.
    figsize : tuple of float, optional
        Figure size in inches, passed to matplotlib. By default (None),
        derived from the SOM's grid shape via `cell_size_in` instead of
        matplotlib's own grid-independent default.
    cell_size_in : float
        Inches per SOM grid cell, used to derive `figsize` when `figsize`
        is not given. Ignored if `figsize` is set explicitly.
    dpi : int
        Figure resolution, passed to matplotlib.
    ax : Axes, optional
        Existing axes to draw into. If given, `figsize`, `dpi`, and
        `cell_size_in` are ignored and the legend is placed inside the axes.
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
        if figsize is None:
            figsize = (
                columns * cell_size_in + CHROME_ALLOWANCE_IN * chrome_scale,
                rows * cell_size_in,
            )
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, layout="compressed")
    else:
        own_figure = False
        root_figure = ax.get_figure(root=True)
        if not isinstance(root_figure, Figure):
            raise ValueError("ax must belong to a Figure")
        fig = root_figure

    ax.set_box_aspect(rows / columns)

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
    fig.colorbar(
        image,
        ax=ax,
        label="U-height",
        fraction=DEFAULT_COLORBAR_FRACTION * chrome_scale,
        pad=DEFAULT_COLORBAR_PAD * chrome_scale,
    )

    if bmu_coordinates is not None:
        unique_bmu_coordinates, index_to_unique_mapping = np.unique(
            bmu_coordinates, axis=0, return_inverse=True
        )
        map_coordinates = bmu_raw_to_map_coordinates(
            np.astype(unique_bmu_coordinates, np.uint16), scaling_factor
        )

        face_colors: Union[str, NDArray] = "black"
        legend_fontsize = DEFAULT_LEGEND_FONTSIZE_PT * chrome_scale
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
                        label=_truncate_label(str(category), legend_label_maxlen),
                    )
                    for category in categories
                ]
                ncol = (
                    legend_ncol
                    if legend_ncol is not None
                    else _auto_legend_ncol(len(categories))
                )
                if own_figure:
                    fig.legend(
                        handles=handles,
                        title=color_by,
                        loc="outside right upper",
                        ncol=ncol,
                        fontsize=legend_fontsize,
                    )
                else:
                    ax.legend(
                        handles=handles,
                        title=color_by,
                        ncol=ncol,
                        fontsize=legend_fontsize,
                    )
            else:
                property_cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
                face_colors, norm = _continuous_face_colors(
                    values, np.astype(index_to_unique_mapping, np.uint32), property_cmap
                )
                fig.colorbar(
                    ScalarMappable(norm=norm, cmap=property_cmap),
                    ax=ax,
                    label=color_by,
                    fraction=DEFAULT_COLORBAR_FRACTION * chrome_scale,
                    pad=DEFAULT_COLORBAR_PAD * chrome_scale,
                )

        if marker_size is None:
            cell_pt = _cell_size_points(fig, ax, rows, columns)
            effective_marker_size = cell_pt * marker_cell_fraction * chrome_scale
        else:
            effective_marker_size = marker_size
        # Keep the marker edge visible but proportionate; a fixed 1.5pt
        # stroke (the old default) can swamp a small auto-derived marker.
        effective_linewidth = float(np.clip(effective_marker_size * 0.12, 0.5, 2.0))

        ax.scatter(
            map_coordinates[:, 1],
            map_coordinates[:, 0],
            s=effective_marker_size**2,
            facecolors=face_colors,
            edgecolors="black",
            linewidths=effective_linewidth,
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


def _truncate_label(label: str, max_len: int) -> str:
    """Elide an over-long legend label so the legend's rendered width stays bounded."""
    if max_len <= 1 or len(label) <= max_len:
        return label
    return label[: max_len - 1] + "…"


def _auto_legend_ncol(n_categories: int, max_rows: int = 12) -> int:
    """Wrap the legend into additional columns once it would exceed `max_rows` entries."""
    return max(1, -(-n_categories // max_rows))  # ceil division


def _cell_size_points(fig: Figure, ax: Axes, rows: int, columns: int) -> float:
    """
    Physical size, in points, of one SOM grid cell as actually laid out in
    `fig`.

    Forces a layout pass (so the constrained/compressed layout solver has
    already accounted for the colorbar(s)/legend added so far) and reads
    back the axes' rendered pixel extent, converting to points via the
    figure's dpi. Used to auto-derive `marker_size` proportionate to how
    large the map is actually rendered, rather than a fixed constant.
    """
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    width_pt = bbox.width / fig.dpi * 72.0
    height_pt = bbox.height / fig.dpi * 72.0
    return min(width_pt / columns, height_pt / rows)
