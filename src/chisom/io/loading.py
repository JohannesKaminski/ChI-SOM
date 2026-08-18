"""
Loading of viewer artefacts (U-matrix, BMU coordinates, datasets) from file paths.

Kept free of any GUI import, so the command line interface can use it without the
optional ``gui`` extra installed.
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import tables
from numpy.typing import NDArray
from pandas import DataFrame

from chisom.io.datastores import DatasetBase, HDF5Dataset

__all__ = [
    "DATASET_SUFFIXES",
    "dataset_column_names",
    "inspect_hdf5_groups",
    "load_bmu_coordinates",
    "load_dataset",
    "load_umatrix",
]

HDF5_SUFFIXES = (".h5", ".hdf5")
CSV_SUFFIXES = (".csv",)
TSV_SUFFIXES = (".tsv", ".txt")
PARQUET_SUFFIXES = (".parquet", ".pq")

DATASET_SUFFIXES = HDF5_SUFFIXES + CSV_SUFFIXES + TSV_SUFFIXES + PARQUET_SUFFIXES


def load_umatrix(filepath: Union[str, Path]) -> NDArray:
    """Load a U-matrix from a ``.npy`` file.

    Parameters
    ----------
    filepath
        Path to a NumPy array holding a 2D (single layer) or 3D (layered) U-matrix.

    Returns
    -------
    NDArray
        The U-matrix, always as a 3D array with the layer as first axis.

    Raises
    ------
    ValueError
        Raised if the file does not contain a 2D or 3D array.
    """
    umatrix = _load_npy(filepath, "U-matrix")

    if umatrix.ndim == 2:
        return umatrix[np.newaxis, :, :]
    if umatrix.ndim == 3:
        return umatrix
    raise ValueError(
        f"'{filepath}' does not hold a U-matrix: expected a 2D or 3D array, "
        f"got {umatrix.ndim} dimensions."
    )


def load_bmu_coordinates(filepath: Union[str, Path]) -> NDArray:
    """Load BMU coordinates from a ``.npy`` file.

    Parameters
    ----------
    filepath
        Path to a NumPy array of shape ``(n_datapoints, 2)`` holding the row/column
        coordinate of the best matching unit of every datapoint.

    Returns
    -------
    NDArray
        The BMU coordinates.

    Raises
    ------
    ValueError
        Raised if the array does not have the expected shape or dtype.
    """
    bmu_coordinates = _load_npy(filepath, "BMU coordinates")

    if bmu_coordinates.ndim != 2 or bmu_coordinates.shape[1] != 2:
        raise ValueError(
            f"'{filepath}' does not hold BMU coordinates: expected an array of shape "
            f"(n_datapoints, 2), got {bmu_coordinates.shape}."
        )
    if not np.issubdtype(bmu_coordinates.dtype, np.integer):
        raise ValueError(
            f"'{filepath}' does not hold BMU coordinates: expected an integer dtype, "
            f"got {bmu_coordinates.dtype}."
        )
    return bmu_coordinates


def load_dataset(
    filepath: Union[str, Path],
    group_subset: Optional[List[str]] = None,
) -> Union[DatasetBase, DataFrame]:
    """Load a dataset of datapoint properties, dispatching on the file extension.

    Parameters
    ----------
    filepath
        Path to the dataset. Supported are HDF5 stores created with
        [`HDF5Creator`][chisom.io.HDF5Creator] (`.h5`, `.hdf5`), delimited text
        (`.csv`, `.tsv`, `.txt`) and Parquet (`.parquet`, `.pq`).
    group_subset
        HDF5 only: the groups to include, by default all of them.

    Returns
    -------
    Union[DatasetBase, DataFrame]
        An [`HDF5Dataset`][chisom.io.HDF5Dataset] for HDF5 stores, a `DataFrame` otherwise.

    Raises
    ------
    ValueError
        Raised if the file extension is not supported.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in HDF5_SUFFIXES:
        return HDF5Dataset(str(filepath), group_subset=group_subset)
    if suffix in CSV_SUFFIXES:
        return pd.read_csv(filepath)
    if suffix in TSV_SUFFIXES:
        return pd.read_csv(filepath, sep="\t")
    if suffix in PARQUET_SUFFIXES:
        try:
            return pd.read_parquet(filepath)
        except ImportError as exc:
            # pandas needs pyarrow or fastparquet for this, neither is a dependency
            raise ValueError(
                f"Cannot read '{filepath}': reading Parquet needs an additional "
                f"engine. Install it with `pip install pyarrow`. ({exc})"
            ) from exc

    raise ValueError(
        f"Cannot load '{filepath}': unsupported dataset format '{suffix}'. "
        f"Supported are {', '.join(DATASET_SUFFIXES)}."
    )


def inspect_hdf5_groups(filepath: Union[str, Path]) -> List[str]:
    """Read the group names of an HDF5 store without loading the dataset.

    Parameters
    ----------
    filepath
        Path to the HDF5 store.

    Returns
    -------
    List[str]
        The names of the groups at the root of the store, sorted like
        [`HDF5Dataset`][chisom.io.HDF5Dataset] sorts them.
    """
    with tables.open_file(str(filepath), mode="r") as file:
        return sorted(str(node._v_name) for node in file.list_nodes("/"))


def dataset_column_names(data: Union[DatasetBase, DataFrame]) -> List[str]:
    """Return the column names of either dataset flavour.

    Parameters
    ----------
    data
        A dataset as returned by [`load_dataset`][chisom.io.load_dataset].

    Returns
    -------
    List[str]
        The column names.
    """
    if isinstance(data, DataFrame):
        return [str(column) for column in data.columns]
    return list(data.column_names)


def _load_npy(filepath: Union[str, Path], description: str) -> NDArray:
    filepath = Path(filepath)
    if filepath.suffix.lower() != ".npy":
        raise ValueError(
            f"Cannot load the {description} from '{filepath}': expected a NumPy "
            f"array file (.npy), got '{filepath.suffix}'."
        )
    try:
        array = np.load(filepath)
    except ValueError as exc:
        raise ValueError(f"Cannot read '{filepath}' as a NumPy array: {exc}") from exc

    if not isinstance(array, np.ndarray):
        raise ValueError(
            f"'{filepath}' does not hold a plain NumPy array, but a {type(array).__name__}."
        )
    return array
