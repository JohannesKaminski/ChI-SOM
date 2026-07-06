from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from PIL import ImageQt
from pyqtgraph import mkBrush
from PySide6 import QtCore
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QBrush, QPixmap
from rdkit import Chem
from rdkit.Chem import Draw as rdDraw

from chisom.io.datastores import DatasetBase

from ._types import ColumnProperties, TabularDatasource, bmu_type
from .helpers import create_bmu_composition, min_max


class RatioWeightingSchemes:
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
        numpy.ndarray
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



class DataFrameSource(QObject):
    def __init__(self, *args, **kwargs):
        self._df = pd.DataFrame(*args, **kwargs)

        # Gather info about columns, similar to ChemDataset, but slow
        self.columns_with_properties: dict[str, ColumnProperties] = {}
        for column in self._df.columns:
            data = self._df[column].to_numpy()

            if len(set(data[:100])) <= 10:
                self.set_categorical(data, column)
            else:
                self.set_continuous(data, column)

    @property
    def column_names(self) -> list[str]:
        return list(self._df.columns)

    @Slot(str)
    def set_column_continuous(self, column_name: str) -> None:
        data = self._df[column_name].to_numpy()
        self.set_continuous(data, column_name)

    def set_continuous(self, data: NDArray, column_name: str) -> None:
        data_type = np.dtype(data)

        try:
            value_range = [np.min(data), np.max(data)]

        except Exception:
            try:
                value_range = [min(data), max(data)]
            except Exception:
                data_type = None
                value_type = "na"
                value_range = []
            else:
                value_type = "continuous"
        else:
            value_type = "continuous"

        self.columns_with_properties[column_name] = ColumnProperties(
            data_type, value_type, value_range
        )

    @Slot(str)
    def set_column_categorical(self, column_name: str) -> None:
        data = self._df[column_name].to_numpy()
        self.set_categorical(data, column_name)

    def set_categorical(self, data: NDArray, column_name: str) -> None:
        data_type = np.dtype(data)

        try:
            unique = np.unique(data)
        except Exception:
            unique = set(data)

        value_type = "categorical"
        value_range = list(unique)

        self.columns_with_properties[column_name] = ColumnProperties(
            data_type, value_type, value_range
        )

    def __len__(self) -> int:
        return len(self._df.index)

    def get_values_for_column(self, column_name: str) -> NDArray:
        return self._df[column_name].to_numpy()

    def get_value(self, row_idc: Union[int, list[int]], column_idx: int) -> Any:
        return self._df.iloc[row_idc, column_idx]


class BMUMap(QObject):
    """
    A class to map BMU coordinates to data indices and scatterplot item indices.
    Also provides signals to notify when BMUs and/or scaling change.

    """

    map_bmu_coordinates_changed = Signal(np.ndarray)

    def __init__(
        self,
        bmu_raw_coordinates: Optional[NDArray[np.float16]],
        scaling_factor: int,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent=parent)
        self.bmu_map_coordinates: NDArray[np.float16] = np.empty(
            (0, 2), dtype=np.float16
        )
        self.bmu_raw_coordinates_rec: NDArray = np.empty((0, 2), dtype=bmu_type)

        self.unique_bmu_coordinates: NDArray[np.uint16] = np.empty(
            (0, 2), dtype=np.uint16
        )
        self.unique_bmu_coordinates_set: set[tuple[int, int]] = set()
        self.unique_bmu_coordinates_rec: NDArray = np.rec.array(
            np.empty(0, dtype=bmu_type)
        )
        self.index_to_unique_mapping: NDArray[np.int32] = np.empty(0, dtype=np.int32)

        self.scaling_factor: int = scaling_factor
        self.padding: int = 0

        self.bmu_state: int  # Used to track the visibility of the BMUs in the scatterplot and button position in the control widget

        if bmu_raw_coordinates is not None:
            self.bmu_raw_coordinates_rec = np.rec.array(
                bmu_raw_coordinates, dtype=bmu_type
            )
            # Returns the unique coordinates of the BMUs and for each original BMU to which unique bmus it correspond with.
            _unique_bmu_coordinates, _index_to_unique_mapping = np.unique(
                bmu_raw_coordinates, axis=0, return_inverse=True
            )
            self.unique_bmu_coordinates = np.astype(_unique_bmu_coordinates, np.uint16)
            self.index_to_unique_mapping = np.astype(_index_to_unique_mapping, np.int32)
            # Transform to set of tuples for faster lookup
            self.unique_bmu_coordinates_set = set(
                tuple(coord) for coord in self.unique_bmu_coordinates
            )

            self.unique_bmu_coordinates_rec = np.rec.array(
                self.unique_bmu_coordinates, dtype=bmu_type
            )
            self.bmu_state = 2
        else:
            self.bmu_state = 1

        self.calculate_bmu_map_coordinates()

    def __len__(self) -> int:
        return len(self.unique_bmu_coordinates)

    def get_bmu_info_from_map_coordinates(
        self, map_coordinates: NDArray
    ) -> tuple[NDArray, NDArray]:
        # Transform the coordinates to tuples and convert to a set for faster membership testing
        raw_coordinates = np.empty_like(map_coordinates, dtype=np.float32)

        # Use floor division, as each pixel in the map corresponds to a square of size scaling_factor x scaling_factor in the raw coordinates
        raw_coordinates[:, 0] = (
            map_coordinates[:, 0] - self.padding
        ) // self.scaling_factor
        raw_coordinates[:, 1] = (
            map_coordinates[:, 1] - self.padding
        ) // self.scaling_factor
        # Clip to zero, to avoid coordinates included due to padding
        np.clip(raw_coordinates, 0, None, out=raw_coordinates)
        _raw_coordinates = set(tuple(coord) for coord in raw_coordinates.tolist())
        bmu_coordinates_in_selection = (
            _raw_coordinates & self.unique_bmu_coordinates_set
        )

        # For each coordinate in the selection, retrieve the indices of the scatterplot items coresponding to the coordinates
        # as well as the indices of the data points that are mapped to this coordinate
        bmu_coordinates_in_selection_rec = np.rec.array(
            list(bmu_coordinates_in_selection), dtype=bmu_type
        )
        data_idx = np.argwhere(
            np.isin(
                self.bmu_raw_coordinates_rec, bmu_coordinates_in_selection_rec
            ).flatten()
        )
        scatterplot_idx = np.argwhere(
            np.isin(
                self.unique_bmu_coordinates_rec, bmu_coordinates_in_selection_rec
            ).flatten()
        )
        return scatterplot_idx, data_idx

    def calculate_bmu_map_coordinates(self) -> None:
        """
        Returns the BMU coordinates in the map space.
        This is used to display the BMUs in the scatterplot.
        """
        if len(self.unique_bmu_coordinates) > 0:
            self.padding = self.scaling_factor // 2
            margin = self.padding + 0.5
            self.bmu_map_coordinates = np.empty(
                (len(self.unique_bmu_coordinates), 2), dtype=np.float16
            )
            self.bmu_map_coordinates[:, 0] = (
                self.unique_bmu_coordinates[:, 0] * self.scaling_factor + margin
            )
            self.bmu_map_coordinates[:, 1] = (
                self.unique_bmu_coordinates[:, 1] * self.scaling_factor + margin
            )

            self.map_bmu_coordinates_changed.emit(self.bmu_map_coordinates)

    @Slot(int)
    def set_scaling_factor(self, value: int) -> None:
        """
        Sets the scaling factor used for the BMU coordinates.
        This is used to scale the BMU coordinates to the map space.
        """
        if value < 1:
            raise ValueError("Scaling factor must be greater than 0")
        self.scaling_factor = value
        self.calculate_bmu_map_coordinates()


class CommonDataModel(QtCore.QAbstractTableModel):
    """
    Provides a unified interface to access data from different sources
    (e.g. ChemDataset, DataFrame) and their corresponding BMU coordinates.
    """

    selection_changed = Signal()

    def __init__(
        self,
        datasource: Optional[Union[DatasetBase, DataFrame]] = None,
        structure_info_column: Optional[str] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        """
        Initialize the DataSource with BMU coordinates and data.

        Parameters
        ----------
        data : Optional[Union[DatasetBase, DataFrame]], optional
            The dataset containing the data entries, which can be a ChemDataset or a DataFrame, by default None

        Raises
        ------
        ValueError
            Raised if the data type is not supported.
        """
        super().__init__(parent=parent)
        self.columns_with_properties: Dict[str, ColumnProperties]
        self.column_back_map: Dict[int, int] = {}

        self.data_source: TabularDatasource = (
            datasource
            if isinstance(datasource, DatasetBase)
            else DataFrameSource(datasource)
        )
        self.get_values_for_column = self.data_source.get_values_for_column

        # Store columns used in the table. Size determines number of columns
        # Copy to remove unwanted columns (e.g., "fingerprint")
        self.columns = [column for column in self.data_source.column_names]

        # Store columns with properties for color selector
        # Copy to remove unwanted columns for color selector
        self.columns_with_properties = {
            name: ColumnProperties(*properties)
            for name, properties in self.data_source.columns_with_properties.items()
        }

        # "fingerprint is unwanted in either"
        if "fingerprint" in self.columns:
            self.columns.remove("fingerprint")
            self.columns_with_properties.pop("fingerprint")

        # Remove unwanted columns and info for colorselector
        to_pop = []  # cannot change size of dictionary during iterating over it
        for name, property in self.columns_with_properties.items():
            if property.value_type == "na":
                to_pop.append(name)
        for name in to_pop:
            self.columns_with_properties.pop(name)

        # Map the new column indices back to the sources column indices
        if structure_info_column in self.data_source.column_names:
            self.columns.append("Structure")
            for i, name in enumerate(self.columns):
                if name == "Structure":
                    # Used indexing into the data source's column names to map the new column index to the source column index
                    self.column_back_map[i] = self.data_source.column_names.index(
                        structure_info_column
                    )
                    self.structure_column_id = i

                else:
                    self.column_back_map[i] = self.data_source.column_names.index(name)
            self.structure_info_column_id = self.data_source.column_names.index(
                structure_info_column
            )
        else:
            for i, name in enumerate(self.columns):
                self.column_back_map[i] = self.data_source.column_names.index(name)
            self.structure_column_id = None
            self.structure_info_column_id = None

    def rowCount(self, /, parent=...) -> int:
        return len(self.data_source)

    def columnCount(self, /, parent=...) -> int:
        return len(self.columns)

    def headerData(self, section, orientation, /, role=...):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Horizontal:
                return self.columns[section] if section < len(self.columns) else None
            if orientation == QtCore.Qt.Orientation.Vertical:
                return section + 1  # Return row number for vertical header

    def data(self, index, /, role=...):
        row = index.row()
        column = index.column()

        if not index.isValid():
            return None

        data_column = self.column_back_map[column]
        datapoint = self.data_source.get_value(row, data_column)

        if (
            role == QtCore.Qt.ItemDataRole.DecorationRole
            and column == self.structure_column_id
        ):
            compound_image = self.create_CompoundImage(datapoint)
            return compound_image

        elif (
            role == QtCore.Qt.ItemDataRole.DisplayRole
            and column != self.structure_column_id
        ):
            return str(
                datapoint
            )  # Conversion to string not optimal, as orignial dtype might be needed later
        else:
            return None

    def flags(self, index):
        return QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable

    @staticmethod
    def create_CompoundImage(smiles: str) -> QPixmap:
        """Create a QPixmap from a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        img = rdDraw.MolToImage(mol, size=(200, 150))
        return QPixmap.fromImage(ImageQt.ImageQt(img))


class FilterModel(QtCore.QAbstractProxyModel):
    selection_changed = Signal()

    def __init__(self, sourceModel: CommonDataModel, parent=None) -> None:
        super().__init__(parent=parent)
        super().setSourceModel(sourceModel)
        self.columns = self.sourceModel().columns
        self.columns_with_properties: Dict[str, ColumnProperties] = (
            self.sourceModel().columns_with_properties
        )
        self.selected_rows: List[int] = []

        self.structure_column_id = self.sourceModel().structure_column_id
        self.structure_info_column_id = self.sourceModel().structure_info_column_id
        self.get_values_for_column = self.sourceModel().get_values_for_column

    def parent(self, child):
        return QtCore.QModelIndex()

    def mapToSource(self, proxyIndex):
        if not proxyIndex.isValid() or self.sourceModel() is None:
            return QtCore.QModelIndex()
        row = self.selected_rows[proxyIndex.row()]
        return self.sourceModel().index(row, proxyIndex.column())

    def mapFromSource(self, sourceIndex):
        if not sourceIndex.isValid() or self.sourceModel() is None:
            return QtCore.QModelIndex()
        try:
            row = self.selected_rows.index(sourceIndex.row() - 1)
            return self.index(row, sourceIndex.column())
        except ValueError:
            return QtCore.QModelIndex()

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        return len(self.selected_rows)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        return self.sourceModel().columnCount(parent)

    def index(self, row, column, /, parent=...):
        return self.createIndex(row, column)

    def data(self, proxyIndex, /, role=...):
        base_index = self.mapToSource(proxyIndex)
        return self.sourceModel().data(base_index, role)

    @Slot(list)
    def set_selected_rows(self, rows: NDArray[np.int64]) -> None:
        self.beginResetModel()
        self.selected_rows = rows.flatten().tolist()
        self.endResetModel()
        self.selection_changed.emit()


class BMUColors(QObject):
    colors_updated = Signal(list)
    cmap_updated = Signal(list)

    def __init__(self, datamodel: FilterModel, bmu_map: BMUMap):
        super().__init__()
        self.datamodel = datamodel
        self.bmu_map = bmu_map
        self.properies = self.datamodel.columns_with_properties

        self.current_colors: List[QBrush] = [mkBrush("k")] * len(
            self.bmu_map.bmu_map_coordinates
        )

        # For each column, store the distribution results to safe time
        self.bmu_ratio_mapping = {}

    @Slot(list)
    def update_bmu_colors_gradient(self, property_info):
        property_name, property_cmap = property_info
        bmu_id_for_datapoint = self.bmu_map.index_to_unique_mapping

        if property_name not in self.bmu_ratio_mapping:
            data = self.datamodel.get_values_for_column(property_name)

            bmu_average = self.average_for_coordinate(data, bmu_id_for_datapoint)
            # MinMax, so it is usable by colormap
            bmu_average, minimum, maximum = min_max(bmu_average)
            self.bmu_ratio_mapping[property_name] = (bmu_average, minimum, maximum)

        bmu_average, minimum, maximum = self.bmu_ratio_mapping[property_name]
        colors = property_cmap.map(bmu_average)

        self.cmap_updated.emit([property_cmap, minimum, maximum, property_name])
        self.recolor_bmus(colors)

    @Slot(list)
    def update_bmu_colors_categorical(self, property_name, category_to_color_mapping):
        bmu_id_for_datapoint = self.bmu_map.index_to_unique_mapping

        # dont use prediefinde bins and reconstruct to ensure order of category and selected color is kept
        category_bins = []
        category_colors = []
        for category, color in category_to_color_mapping.items():
            category_bins.append(category)
            category_colors.append(color.getRgb()[:3])
        category_colors = np.asarray(category_colors, dtype=np.uint8)

        if property_name not in self.bmu_ratio_mapping.keys():
            data = self.datamodel.get_values_for_column(property_name)

            num_bmus = len(self.bmu_map)

            # Calculate how many datapoint of each category fall onto every BMU
            bmu_composition_ratio = self.ratio_for_coordinate(
                data, bmu_id_for_datapoint, category_bins, num_bmus
            )

            # Select the most common category for each coordinate to determine primary color
            primary_catergory = np.argmax(bmu_composition_ratio, axis=1)

            # Calculate the alphas for the classes based on how much stronger the strongest category is than the others
            alpha = RatioWeightingSchemes.excess_coefficient_absolute(
                bmu_composition_ratio
            )
            # Map alpha values to ints in the range [0, 255]
            alpha = np.clip((alpha * 255), 0, 255).astype(np.uint8)

            self.bmu_ratio_mapping[property_name] = (primary_catergory, alpha)

        primary_catergory, alpha = self.bmu_ratio_mapping[property_name]
        colors = np.zeros((len(primary_catergory), 3), dtype=np.uint8)

        # Assign colors based on class_labels
        np.take(category_colors, primary_catergory, axis=0, out=colors)
        # Add alpha channel
        colors = np.hstack([colors, alpha[:, np.newaxis]])

        self.recolor_bmus(colors)

    def recolor_bmus(self, colors):
        # create QBrush object for each color to use the same QBrush for all points of the same color, as individual QBrushes for each point decrese performance
        unique_colors, mapping = np.unique(colors, axis=0, return_inverse=True)
        brushes = [mkBrush(color) for color in unique_colors]
        self.current_colors = [brushes[i] for i in mapping]
        self.colors_updated.emit(self.current_colors)

    @staticmethod
    def average_for_coordinate(values: NDArray, coordinate_id: NDArray) -> NDArray:
        # Use bincount with weights to calculate sums for each unique index
        sums = np.bincount(coordinate_id, weights=values)
        # Use bincount to calculate counts for each unique index
        counts = np.bincount(coordinate_id)

        # Calculate average by dividing sums by counts
        return np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)

    @staticmethod
    def ratio_for_coordinate(
        values: NDArray,
        bmu_id_for_datapoint: NDArray,
        bins: list,
        num_bmus: int,
    ) -> NDArray:
        # Initialize result array
        num_bins = len(bins)
        occurances = np.zeros((num_bmus, num_bins), dtype=int)

        _, class_as_id = np.unique_inverse(values)
        class_as_id = np.astype(class_as_id, np.int32)

        # Process each bmu
        occurances = create_bmu_composition(
            bmu_id_for_datapoint, class_as_id, num_bmus, num_bins
        )

        counts = np.sum(occurances, axis=1)
        ratios = occurances / counts[:, np.newaxis]
        ratios[np.isnan(ratios)] = 0

        return ratios
