import os

from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
import PySide6.QtWidgets as W
from numpy.typing import NDArray
from pandas import DataFrame
from pyqtgraph.functions import mkPen
from PySide6.QtCore import QObject, QPoint, QSize, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QKeySequence

from chisom._core.render import interpolate_matrix
from chisom._interface._types import ColumnProperties
from chisom._interface.models import (
    BMUColors,
    BMUFilter,
    BMUMap,
    CommonDataModel,
    CyclicGreen,
    EarthColorMap,
    FilterModel,
)
from chisom.io import loading
from chisom.io.datastores import DatasetBase

# NOTE: Ideally, used QtPixmapCache to store a certain amout of images in memory

pg.setConfigOption("useNumba", True)
pg.setConfigOption("imageAxisOrder", "row-major")
pg.setConfigOption("background", "w")


class UMap(QObject):
    def __init__(
        self,
        image: NDArray,
        scaling_factor: int = 3,
        layer: int = -1,
        parent: Optional[QObject] = None,
        *args,
        **kwargs,
    ):
        super().__init__(parent=parent)
        self.layer = layer
        self.max_layer = image.shape[0] if image.ndim == 3 else 1
        self.raw_values = image
        self.scaling_factor = scaling_factor  # Default scaling factor
        self.ImageItem = pg.ImageItem()
        self.set_umatrix(self.raw_values)

    def set_umatrix(self, image: NDArray, *args, **kwargs):
        """
        Set the image of the UMap to a new image.
        This is used to update the U-matrix.
        """
        self.selected_values = image[self.layer]
        self.scaled_values = np.astype(
            interpolate_matrix(self.selected_values, self.scaling_factor), np.float32
        )
        self.ImageItem.setImage(image=self.scaled_values, *args, **kwargs)

    @Slot(int)
    def set_scaling_factor(self, scaling: int):
        """
        Rescale the UMap to a new scaling factor.
        This is used to update the U-matrix with a new scaling factor.
        """
        self.scaling_factor = scaling
        self.scaled_values = interpolate_matrix(
            self.selected_values, self.scaling_factor
        )
        self.ImageItem.setImage(image=self.scaled_values)

    @Slot(int)
    def set_layer(self, layer: int):
        """
        Set the layer of the UMap to a new layer.
        This is used to update the U-matrix with a new layer.
        """
        if layer < (self.max_layer * -1) or layer >= self.max_layer:
            raise ValueError(
                f"Layer {layer} out of bounds for UMap with {self.raw_values.shape[0]} layers."
            )
        self.layer = layer
        self.set_umatrix(self.raw_values)


class ImageDelegate(W.QStyledItemDelegate):
    def paint(self, painter, option, index):
        pixmap = index.data(Qt.ItemDataRole.DecorationRole)
        if pixmap:
            # Center-align the image
            pixmap_rect = option.rect
            pixmap_rect.setWidth(pixmap.width())
            pixmap_rect.setHeight(pixmap.height())
            pixmap_rect.moveCenter(option.rect.center())

            painter.drawPixmap(pixmap_rect, pixmap)
            return
        super().paint(painter, option, index)

    def sizeHint(self, option, index):
        pixmap = index.data(Qt.ItemDataRole.DecorationRole)
        if pixmap:
            return QSize(pixmap.width(), pixmap.height())
        return super().sizeHint(option, index)


class CompoundTable(W.QTableView):
    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent=parent)

        self.contextMenu = W.QMenu()
        self.contextMenu.addAction("Copy Selection").triggered.connect(self.copySel)
        self.contextMenu.addAction("Copy All").triggered.connect(self.copyAll)
        self.contextMenu.addAction("Save Selection").triggered.connect(self.saveSel)
        self.contextMenu.addAction("Save All").triggered.connect(self.saveAll)

    def setModel(self, model):
        super().setModel(model)

        if (
            hasattr(model, "structure_column_id")
            and model.structure_column_id is not None
        ):
            self.model_has_structure_column = True
            structure_column = model.structure_column_id
            self.structure_info_column_id = model.structure_info_column_id
            self.setItemDelegateForColumn(structure_column, ImageDelegate())

        else:
            self.model_has_structure_column = False

        self.resize_to_contents()

    @Slot()
    def resize_to_contents(self):
        header = self.horizontalHeader()
        self.resizeRowsToContents()
        self.resizeColumnsToContents()

        if self.model_has_structure_column:
            # Now make the structure info column expand to fill remaining space.
            header.setSectionResizeMode(
                self.structure_info_column_id, W.QHeaderView.ResizeMode.Stretch
            )

            # Ensure the last section does not steal extra space unless it's the
            # designated stretch column.
            header.setStretchLastSection(False)
        else:
            header.setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch)
            header.setStretchLastSection(True)

    def serialize(self, useSelection=False):
        """Convert entire table (or just selected area) into tab-separated text values"""
        # Adapted from pyqtgraph TableWidget
        model = self.model()
        if useSelection:
            selection = self.selectedIndexes()
            rows = {index.row() for index in selection}
            columns = {index.column() for index in selection}
        else:
            rows = list(range(model.rowCount()))
            n_columns = model.columnCount()
            if self.model_has_structure_column:
                n_columns -= 1  # Account for the structure column
            columns = list(range(n_columns))
        data = np.empty(
            (len(rows) + 1, len(columns)), dtype="U240"
        )  # account for header row

        for i, c in enumerate(columns):
            data[0, i] = model.headerData(
                c, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
            )

        for i, r in enumerate(rows):
            for j, c in enumerate(columns):
                index = model.index(r, c)
                data[i + 1, j] = model.data(index, Qt.ItemDataRole.DisplayRole)

        s = ""
        for row in data:
            s += "\t".join(row) + "\n"
        return s

    @Slot()
    def copySel(self):
        """Copy selected data to clipboard."""
        # Adapted from pyqtgraph TableWidget
        W.QApplication.clipboard().setText(self.serialize(useSelection=True))

    @Slot()
    def copyAll(self):
        """Copy all data to clipboard."""
        W.QApplication.clipboard().setText(self.serialize(useSelection=False))

    @Slot()
    def saveSel(self):
        """Save selected data to file."""
        self.save(self.serialize(useSelection=True))

    @Slot()
    def saveAll(self):
        """Save all data to file."""
        self.save(self.serialize(useSelection=False))

    def save(self, data):
        fileName, _ = W.QFileDialog.getSaveFileName(
            self,
            "Save As...",
            "",
            "Tab-separated values (*.tsv)",
        )
        if not fileName:
            return
        with open(fileName, "w") as fd:
            fd.write(data)

    def contextMenuEvent(self, ev):
        self.contextMenu.popup(ev.globalPos())

    def keyPressEvent(self, ev):
        if ev.matches(QKeySequence.StandardKey.Copy):
            ev.accept()
            self.copySel()
        else:
            super().keyPressEvent(ev)


class MoleculeCard(W.QWidget):
    """A single molecule's structure thumbnail and property list, used inside a BmuHoverPopup."""

    THUMBNAIL_SIZE = (220, 165)

    def __init__(
        self,
        smiles: Optional[str],
        properties: list[tuple[str, str]],
        parent=None,
    ):
        super().__init__(parent=parent)

        layout = W.QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        thumbnail_label = W.QLabel(self)
        if smiles:
            pixmap = CommonDataModel.create_CompoundImage(
                smiles, size=self.THUMBNAIL_SIZE
            )
            thumbnail_label.setPixmap(pixmap)
        else:
            thumbnail_label.setFixedSize(*self.THUMBNAIL_SIZE)
            thumbnail_label.setText("No structure")
            thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(thumbnail_label, 0)

        props_layout = W.QVBoxLayout()
        props_layout.setSpacing(2)
        for name, value in properties:
            label = W.QLabel(f"<b>{name}:</b> {value}")
            label.setWordWrap(True)
            props_layout.addWidget(label)
        props_layout.addStretch()
        layout.addLayout(props_layout, 1)


class BmuHoverPopup(W.QWidget):
    """
    Floating, transient panel listing the molecules at a hovered BMU cell.

    Stays open for as long as the source BMU point is hovered. Once the
    cursor leaves that point, a short grace window gives the user a chance
    to move onto the popup itself; if they don't (or if they do and then
    leave the popup), it closes.
    """

    POPUP_WIDTH = 380
    VISIBLE_ENTRIES = 2.5
    # Must stay shorter than UpperView's hover-dwell time, so a popup for a
    # point the cursor has left is gone by the time a new one could open.
    LEAVE_GRACE_MS = 400

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedWidth(self.POPUP_WIDTH)

        self.scroll_area = W.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.setFrameShape(W.QFrame.Shape.StyledPanel)

        self.content = W.QWidget()
        self.content.setFixedWidth(self.POPUP_WIDTH - 4)
        self.content_layout = W.QVBoxLayout(self.content)
        self.content_layout.setSpacing(6)
        self.content_layout.addStretch()
        self.scroll_area.setWidget(self.content)

        outer = W.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.scroll_area)

        self._cards: list[MoleculeCard] = []
        self._grace_timer = QTimer(self)
        self._grace_timer.setSingleShot(True)
        self._grace_timer.timeout.connect(self._grace_expired)
        self._mouse_is_over = False

    def set_molecules(
        self, molecule_rows: list[tuple[Optional[str], list[tuple[str, str]]]]
    ):
        while self.content_layout.count() > 1:
            item = self.content_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        self._cards = []
        for smiles, properties in molecule_rows:
            card = MoleculeCard(smiles, properties, parent=self.content)
            self.content_layout.insertWidget(self.content_layout.count() - 1, card)
            self._cards.append(card)

    def show_near(self, global_pos: QPoint):
        self.adjustSize()
        self.resize(self.POPUP_WIDTH, self._visible_height())
        self.move(global_pos.x() + 16, global_pos.y() + 16)
        self.show()

    def _visible_height(self) -> int:
        content_height = self.content.sizeHint().height()
        if not self._cards:
            return content_height
        card_height = self._cards[0].sizeHint().height() + self.content_layout.spacing()
        return min(content_height, int(card_height * self.VISIBLE_ENTRIES))

    def bmu_left(self):
        """The cursor left the BMU point this popup was opened for."""
        if not self._mouse_is_over:
            self._grace_timer.start(self.LEAVE_GRACE_MS)

    def bmu_still_hovered(self):
        """The cursor is still on the BMU point this popup was opened for."""
        self._grace_timer.stop()

    def _grace_expired(self):
        if not self._mouse_is_over:
            self.close()

    def enterEvent(self, event):
        self._mouse_is_over = True
        self._grace_timer.stop()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._mouse_is_over = False
        self.close()
        super().leaveEvent(event)


class CatergoryPair(W.QWidget):
    """
    Stores the assosiation between a columns class and the color button instance
    """

    def __init__(self, text):
        super().__init__()
        self.main_layout = W.QHBoxLayout()
        self.category = text
        self.label = W.QLabel(text)
        self.button = pg.ColorButton()
        self.main_layout.addWidget(self.label)
        self.main_layout.addWidget(self.button)
        self.setLayout(self.main_layout)


class ColorCategoryWidget(W.QGroupBox):
    category_to_color_mapping_set = Signal(dict)

    def __init__(self, data_columns: dict[str, ColumnProperties], parent=None):
        super().__init__(parent=parent)

        self.currently_selected = None
        self.data_columns = data_columns  # Available columns to color bty
        self.know_columns = {}
        self.category_list = W.QVBoxLayout()
        self.emit_button = W.QPushButton("Set Colors")
        self.setLayout(self.category_list)
        self.emit_button.pressed.connect(self._property_set)

    @Slot(str)
    def select_property(self, name: str):
        """
        Called when a new column to color by is selected and it is categorical
        """
        self.currently_selected = name
        # If this columns has been selected previously, use those values
        if name in self.know_columns:
            self.update_selection(self.know_columns[name])
        # else, create a list of the column available categories and a colorbutton instance
        elif name in self.data_columns:
            color_list = []
            for category in self.data_columns[name].value_range:
                pair = CatergoryPair(category)
                color_list.append(pair)
            self.know_columns[name] = (
                color_list  # Store for later to keep button instances available
            )
            self.update_selection(self.know_columns[name])
        else:
            raise ValueError("Selected Property unknown")

    def update_selection(self, color_list):
        # Update the currently visible selection in the layout

        # first, remove everything
        while not self.category_list.isEmpty():
            item = self.category_list.itemAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    self.category_list.removeWidget(widget)
                    # only set invisible, as it is the same object as in self.known_columns
                    widget.setVisible(False)

        # rebuild layout from list
        for item in color_list:
            item.setVisible(True)
            self.category_list.addWidget(item)
        # add button at the end
        self.category_list.addWidget(self.emit_button)
        self.emit_button.setVisible(True)

    @Slot(bool)
    def _property_set(self):
        catergory_to_color_mapping = {}
        for widget in self.know_columns[self.currently_selected]:
            catergory_to_color_mapping[widget.category] = widget.button._color
        self.category_to_color_mapping_set.emit(catergory_to_color_mapping)


class CategoryFilterPair(W.QWidget):
    """
    Stores the association between a column's category and its checkbox
    """

    def __init__(self, text):
        super().__init__()
        self.main_layout = W.QHBoxLayout()
        self.category = text
        self.label = W.QLabel(text)
        self.checkbox = W.QCheckBox()
        self.checkbox.setChecked(True)
        self.main_layout.addWidget(self.label)
        self.main_layout.addWidget(self.checkbox)
        self.setLayout(self.main_layout)


class FilterCategoryWidget(W.QGroupBox):
    category_filter_set = Signal(set)

    def __init__(self, data_columns: dict[str, ColumnProperties], parent=None):
        super().__init__(parent=parent)

        self.currently_selected = None
        self.data_columns = data_columns  # Available columns to filter by
        self.know_columns = {}
        self.category_list = W.QVBoxLayout()
        self.emit_button = W.QPushButton("Apply Filter")
        self.setLayout(self.category_list)
        self.emit_button.pressed.connect(self._property_set)

    @Slot(str)
    def select_property(self, name: str):
        """
        Called when a new column to filter by is selected and it is categorical
        """
        self.currently_selected = name
        # If this column has been selected previously, use those checkboxes
        if name in self.know_columns:
            self.update_selection(self.know_columns[name])
        # else, create a list of the column's available categories and a checkbox instance
        elif name in self.data_columns:
            checkbox_list = []
            for category in self.data_columns[name].value_range:
                pair = CategoryFilterPair(category)
                checkbox_list.append(pair)
            self.know_columns[name] = (
                checkbox_list  # Store for later to keep checkbox instances available
            )
            self.update_selection(self.know_columns[name])
        else:
            raise ValueError("Selected Property unknown")

    def update_selection(self, checkbox_list):
        # Update the currently visible selection in the layout

        # first, remove everything
        while not self.category_list.isEmpty():
            item = self.category_list.itemAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    self.category_list.removeWidget(widget)
                    # only set invisible, as it is the same object as in self.know_columns
                    widget.setVisible(False)

        # rebuild layout from list
        for item in checkbox_list:
            item.setVisible(True)
            self.category_list.addWidget(item)
        # add button at the end
        self.category_list.addWidget(self.emit_button)
        self.emit_button.setVisible(True)

    def reset_selection(self):
        # Re-check every category for every known column, restoring the "no filter" state
        for checkbox_list in self.know_columns.values():
            for pair in checkbox_list:
                pair.checkbox.setChecked(True)

    @Slot(bool)
    def _property_set(self):
        selected_categories = {
            widget.category
            for widget in self.know_columns[self.currently_selected]
            if widget.checkbox.isChecked()
        }
        self.category_filter_set.emit(selected_categories)


class ControlWidget(W.QGroupBox):
    colormap_changed = Signal(str)
    colorbar_visible = Signal(bool)
    bmus_toggled = Signal(bool)
    bmus_resized = Signal(int)
    continuous_color_selected = Signal(str, object)
    categorical_color_selected = Signal(str, dict)
    continuous_filter_selected = Signal(str, float, float)
    categorical_filter_selected = Signal(str, set)
    filter_cleared = Signal()

    def __init__(
        self,
        cmaps: dict[str, pg.ColorMap],
        data_columns: dict[str, ColumnProperties],
        bmu_colors: BMUColors,
        parent=None,
    ):
        super().__init__("Controls", parent=parent)

        self.bmu_colors = bmu_colors
        self.cmaps = cmaps
        self.main_layout = W.QVBoxLayout(self)
        self.data_columns = data_columns

        # Colormap
        cmap_layout = W.QHBoxLayout()
        self.cmap_label = W.QLabel("Colormap:")
        self.cmap_selector = W.QComboBox()
        self.cmap_selector.setEditable(False)
        self.cmap_selector.addItems(list(self.cmaps.keys()))
        self.cmap_selector.currentTextChanged.connect(self.change_colormap)
        cmap_layout.addWidget(self.cmap_label)
        cmap_layout.addWidget(self.cmap_selector)
        self.main_layout.addLayout(cmap_layout)
        self.main_layout.addWidget(W.QFrame(frameShape=W.QFrame.Shape.HLine))

        # BMU control
        bmu_layout = W.QGridLayout()
        self.bmu_visibility_label = W.QLabel("BMUs")
        self.bmu_visibility_toggle = W.QCheckBox("show")
        self.bmu_visibility_toggle.checkStateChanged.connect(self.toggle_bmus)
        self.bmu_size_label = W.QLabel("Size:")
        self.bmu_size_selector = W.QSpinBox(parent=self)
        self.bmu_size_selector.setRange(1, 200)
        self.bmu_size_selector.setSingleStep(1)
        self.bmu_size_selector.valueChanged.connect(self.resize_bmus)
        self.bmu_color_by_label = W.QLabel("Color by:")
        self.bmu_color_by_selector = W.QComboBox()
        self.bmu_color_by_selector.setEditable(False)
        self.bmu_color_by_selector.addItems(list(data_columns.keys()))
        self.bmu_color_by_selector.textActivated.connect(self.select_property)

        bmu_layout.addWidget(self.bmu_visibility_label, 0, 0)
        bmu_layout.addWidget(self.bmu_visibility_toggle, 0, 1)
        bmu_layout.addWidget(self.bmu_size_label, 1, 0)
        bmu_layout.addWidget(self.bmu_size_selector, 1, 1)
        bmu_layout.addWidget(self.bmu_color_by_label, 2, 0)
        bmu_layout.addWidget(self.bmu_color_by_selector, 2, 1)
        self.main_layout.addLayout(bmu_layout)

        self.category_color = ColorCategoryWidget(data_columns)
        self.category_color.setVisible(False)
        self.main_layout.addWidget(self.category_color)
        self.continuous_color = W.QComboBox(self)
        self.continuous_color.setEditable(False)
        self.continuous_color.addItems(list(self.cmaps.keys()))
        self.continuous_color.setVisible(False)
        self.main_layout.addWidget(self.continuous_color)

        self.category_color.category_to_color_mapping_set.connect(
            self.color_selected_categorical
        )
        self.continuous_color.textActivated.connect(self.color_selected_continuous)

        self.main_layout.addWidget(W.QFrame(frameShape=W.QFrame.Shape.HLine))

        # Filter control
        filter_layout = W.QGridLayout()
        self.filter_by_label = W.QLabel("Filter by:")
        self.filter_by_selector = W.QComboBox()
        self.filter_by_selector.setEditable(False)
        self.filter_by_selector.addItems(list(data_columns.keys()))
        self.filter_by_selector.textActivated.connect(self.select_filter_property)
        filter_layout.addWidget(self.filter_by_label, 0, 0)
        filter_layout.addWidget(self.filter_by_selector, 0, 1)
        self.main_layout.addLayout(filter_layout)

        self.filter_category = FilterCategoryWidget(data_columns)
        self.filter_category.setVisible(False)
        self.main_layout.addWidget(self.filter_category)

        self.filter_continuous = W.QWidget()
        filter_continuous_layout = W.QGridLayout(self.filter_continuous)
        self.filter_min_label = W.QLabel("Min:")
        self.filter_min_spin = W.QDoubleSpinBox()
        self.filter_max_label = W.QLabel("Max:")
        self.filter_max_spin = W.QDoubleSpinBox()
        self.filter_apply_button = W.QPushButton("Apply Filter")
        self.filter_apply_button.pressed.connect(self.emit_continuous_filter)
        filter_continuous_layout.addWidget(self.filter_min_label, 0, 0)
        filter_continuous_layout.addWidget(self.filter_min_spin, 0, 1)
        filter_continuous_layout.addWidget(self.filter_max_label, 1, 0)
        filter_continuous_layout.addWidget(self.filter_max_spin, 1, 1)
        filter_continuous_layout.addWidget(self.filter_apply_button, 2, 0, 1, 2)
        self.filter_continuous.setVisible(False)
        self.main_layout.addWidget(self.filter_continuous)

        self.filter_clear_button = W.QPushButton("Clear Filter")
        self.filter_clear_button.pressed.connect(self.clear_filter)
        self.main_layout.addWidget(self.filter_clear_button)

        self.filter_category.category_filter_set.connect(self.emit_categorical_filter)

        self.main_layout.addStretch()

        self.setLayout(self.main_layout)

    @Slot(str)
    def color_selected_continuous(self, cmap: str):
        current_property = self.bmu_color_by_selector.currentText()
        selected_cmap = self.cmaps[cmap]
        self.colorbar_visible.emit(True)
        self.continuous_color_selected.emit(current_property, selected_cmap)

    @Slot(dict)
    def color_selected_categorical(self, category_to_color_mapping: dict[str, str]):
        current_property = self.bmu_color_by_selector.currentText()
        self.colorbar_visible.emit(False)
        self.categorical_color_selected.emit(
            current_property, category_to_color_mapping
        )

    @Slot(str)
    def select_property(self, name):
        value_type = self.data_columns[name].value_type
        if value_type == "categorical":
            self.continuous_color.setVisible(False)
            self.category_color.select_property(name)
            self.category_color.setVisible(True)
        elif value_type == "continuous":
            self.category_color.setVisible(False)
            self.continuous_color.setVisible(True)
        else:
            self.category_color.setVisible(False)
            self.continuous_color.setVisible(False)

    @Slot(str)
    def select_filter_property(self, name):
        value_type = self.data_columns[name].value_type
        if value_type == "categorical":
            self.filter_continuous.setVisible(False)
            self.filter_category.select_property(name)
            self.filter_category.setVisible(True)
        elif value_type == "continuous":
            self.filter_category.setVisible(False)
            minimum, maximum = self.data_columns[name].value_range
            self.filter_min_spin.setRange(minimum, maximum)
            self.filter_max_spin.setRange(minimum, maximum)
            self.filter_min_spin.setValue(minimum)
            self.filter_max_spin.setValue(maximum)
            self.filter_continuous.setVisible(True)
        else:
            self.filter_category.setVisible(False)
            self.filter_continuous.setVisible(False)

    @Slot()
    def emit_continuous_filter(self):
        current_property = self.filter_by_selector.currentText()
        self.continuous_filter_selected.emit(
            current_property, self.filter_min_spin.value(), self.filter_max_spin.value()
        )

    @Slot(set)
    def emit_categorical_filter(self, categories: set):
        current_property = self.filter_by_selector.currentText()
        self.categorical_filter_selected.emit(current_property, categories)

    @Slot()
    def clear_filter(self):
        self.filter_category.reset_selection()
        name = self.filter_by_selector.currentText()
        if name and self.data_columns[name].value_type == "continuous":
            minimum, maximum = self.data_columns[name].value_range
            self.filter_min_spin.setValue(minimum)
            self.filter_max_spin.setValue(maximum)
        self.filter_cleared.emit()

    @Slot(int, int)
    def set_bmu_state(self, bmu_state: Qt.CheckState, bmu_size: int):
        self.bmu_visibility_toggle.setCheckState(Qt.CheckState(bmu_state))
        self.bmu_size_selector.setValue(bmu_size)

    @Slot(int)
    def toggle_bmus(self, state: Qt.CheckState):
        if state == Qt.CheckState.Checked:
            self.bmus_toggled.emit(True)
        if state == Qt.CheckState.Unchecked:
            self.bmus_toggled.emit(False)

    @Slot(int)
    def resize_bmus(self, size: int):
        self.bmus_resized.emit(size)

    @Slot(str)
    def change_colormap(self, cmap: str):
        self.cmap_selector.setCurrentText(cmap)
        self.colormap_changed.emit(cmap)


class Roi(pg.PolyLineROI):
    """
    A custom ROI class that emits a signal when the ROI is changed.
    This is used to update the BMU selection based on the ROI.
    """

    roi_changed = Signal()

    def __init__(self, positions=[], closed=True, *args, **kwargs):
        super().__init__(positions=positions, closed=closed, *args, **kwargs)
        self.sigRegionChangeFinished.connect(self._roi_changed)
        self.previous_positions = []

    def addPoint(self, point: tuple[float, float]):
        self.previous_positions = [
            tuple(handle["item"].pos()) for handle in self.handles
        ]
        new_positions = self.previous_positions + [point]
        self.setPoints(new_positions)

    def clear(self):
        """
        Clear the ROI points.
        """
        self.previous_positions = []
        super().setPoints(self.previous_positions)

    def _roi_changed(self):
        current_positions = [tuple(handle["item"].pos()) for handle in self.handles]
        if set(current_positions) <= set(self.previous_positions):
            return
        else:
            self.previous_positions = current_positions
            self.roi_changed.emit()


class UpperView(W.QWidget):
    new_bmu_selection = Signal(np.ndarray)

    cmap_list = pg.colormap.listMaps(source="matplotlib")
    cmaps = {}
    for cmap in cmap_list:
        cmaps[cmap] = pg.colormap.get(cmap, source="matplotlib")

    cmaps["Earth"] = EarthColorMap
    cmaps["Cyclic Green"] = CyclicGreen

    def __init__(
        self,
        umap: UMap,
        bmu_map: BMUMap,
        bmu_colors: BMUColors,
        bmu_filter: BMUFilter,
        data_columns: dict[str, ColumnProperties],
        base_model: CommonDataModel,
        parent=None,
    ):
        super().__init__(parent=parent)

        self.umap = umap
        self.bmu_map = bmu_map
        self.bmu_filter = (
            bmu_filter  # keep alive: only referenced via signal connections otherwise
        )
        self.source_model = base_model
        self.bmu_pen = mkPen("k", width=1.5)
        self._master_visible: bool = True
        # None means "no filter active" (every datapoint passes)
        self._datapoint_filter_mask: Optional[NDArray] = None
        self.bmus_points = pg.ScatterPlotItem(
            x=bmu_map.bmu_map_coordinates[:, 1],
            y=bmu_map.bmu_map_coordinates[:, 0],
            pxMode=True,
            size=10,
            brush=bmu_colors.current_colors,
            pen=self.bmu_pen,
            hoverable=True,
            tip=None,  # disable pyqtgraph's own hover tooltip; we show our own popup
        )

        bmu_colors.colors_updated.connect(self.set_bmu_colors)

        self._hover_popup: Optional[BmuHoverPopup] = None
        self._popup_scatter_indices: Optional[frozenset[int]] = None
        self._pending_scatter_indices: Optional[frozenset[int]] = None
        self._hover_screen_pos: Optional[QPoint] = None
        self._hover_dwell_timer = QTimer(self)
        self._hover_dwell_timer.setSingleShot(True)
        self._hover_dwell_timer.setInterval(1000)
        self._hover_dwell_timer.timeout.connect(self._open_hover_popup)
        self.bmus_points.sigHovered.connect(self.handle_bmu_hover)

        # Initialize ROI variables
        self.roi = Roi(
            closed=True,
            pen={"color": "k", "width": 4},
            movable=False,
            resizable=False,
        )
        self.roi.roi_changed.connect(self.get_roi)

        self.map_view = pg.ViewBox(invertY=True, lockAspect=True)
        self.map_view.addItem(self.umap.ImageItem)
        self.map_view.addItem(self.bmus_points)
        self.map_view.addItem(self.roi)

        self.map_colorbar = pg.ColorBarItem(
            values=(0, 1),
            limits=(0, 1),
            orientation="vertical",
            interactive=False,
            label="U-height",
        )
        self.map_colorbar.setImageItem(self.umap.ImageItem)

        self.bmu_colorbar = pg.ColorBarItem(
            orientation="vertical",
            interactive=False,
        )
        self.bmu_colorbar.setVisible(False)
        bmu_colors.cmap_updated.connect(self.change_bmu_colorbar)

        self.graphic_layout = pg.GraphicsLayoutWidget(parent=self)
        self.graphic_layout.addItem(self.map_view)
        self.graphic_layout.addItem(self.bmu_colorbar)
        self.graphic_layout.addItem(self.map_colorbar)

        self.map_view.scene().sigMouseClicked.connect(self.handle_click)

        self.control = ControlWidget(
            cmaps=self.cmaps,
            data_columns=data_columns,
            bmu_colors=bmu_colors,
            parent=self,
        )
        self.control.colormap_changed.connect(self.change_map_colormap)
        self.control.bmus_resized.connect(self.bmus_points.setSize)
        self.control.bmus_toggled.connect(self.set_master_visibility)
        self.control.colorbar_visible.connect(self.bmu_colorbar.setVisible)
        self.control.continuous_color_selected.connect(
            bmu_colors.update_bmu_colors_gradient
        )
        self.control.categorical_color_selected.connect(
            bmu_colors.update_bmu_colors_categorical
        )
        self.control.continuous_filter_selected.connect(
            bmu_filter.update_filter_continuous
        )
        self.control.categorical_filter_selected.connect(
            bmu_filter.update_filter_categorical
        )
        self.control.filter_cleared.connect(bmu_filter.clear_filter)
        bmu_filter.datapoint_mask_updated.connect(self.set_datapoint_filter_mask)
        bmu_filter.datapoint_mask_updated.connect(bmu_colors.set_datapoint_filter_mask)

        # Set the initial colormap to Earth
        self.control.cmap_selector.setCurrentText("Earth")
        self.change_map_colormap("Earth")

        layout = W.QHBoxLayout(self)
        layout.addWidget(self.graphic_layout)
        layout.addWidget(self.control)
        self.setLayout(layout)

    @Slot(list)
    def set_bmus(self, bmu_values: np.ndarray):
        x = bmu_values[:, 1]
        y = bmu_values[:, 0]
        self.bmus_points.setData(x=x, y=y)

    @Slot(bool)
    def set_master_visibility(self, visible: bool):
        self._master_visible = visible
        self._apply_visibility()

    @Slot(np.ndarray)
    def set_datapoint_filter_mask(self, mask: NDArray):
        self._datapoint_filter_mask = mask
        self._apply_visibility()

    def _bmu_pass_mask(self) -> NDArray:
        """Per-unique-BMU boolean: True if that BMU has at least one datapoint passing the active filter."""
        if self._datapoint_filter_mask is None:
            return np.ones(len(self.bmu_map), dtype=bool)
        bmu_ids = self.bmu_map.index_to_unique_mapping
        counts = np.bincount(
            bmu_ids, weights=self._datapoint_filter_mask, minlength=len(self.bmu_map)
        )
        return counts > 0

    def visibility_mask(self) -> NDArray:
        """Per-unique-BMU boolean visibility, combining the master show/hide toggle with the active filter."""
        if not self._master_visible:
            return np.zeros(len(self.bmu_map), dtype=bool)
        return self._bmu_pass_mask()

    def datapoint_selection_mask(self) -> NDArray:
        """Per-datapoint boolean: True if that datapoint is eligible for ROI selection right now."""
        num_datapoints = len(self.bmu_map.index_to_unique_mapping)
        if not self._master_visible:
            return np.zeros(num_datapoints, dtype=bool)
        if self._datapoint_filter_mask is None:
            return np.ones(num_datapoints, dtype=bool)
        return self._datapoint_filter_mask

    def _apply_visibility(self):
        self.bmus_points.setPointsVisible(self.visibility_mask())

    @Slot(str)
    def change_map_colormap(self, cmap: str):
        """Change the colormap of the map and colorbar."""
        if cmap in self.cmaps:
            self.map_colorbar.setColorMap(self.cmaps[cmap])
        else:
            print(f"Colormap {cmap} not found.")

    @Slot(list)
    def change_bmu_colorbar(self, info: list):
        property_cmap, minimum, maximum, property_name = info
        self.bmu_colorbar.setColorMap(property_cmap)
        self.bmu_colorbar.setLevels(low=minimum, high=maximum)
        self.bmu_colorbar.setLabel(axis="left", text=property_name)

    @Slot(list)
    def set_bmu_colors(self, colors):
        self.bmus_points.setBrush(colors)
        self.bmus_points.setPen(self.bmu_pen)

    def handle_click(self, event):
        # Get click position in the view coordinates
        pos = self.map_view.mapSceneToView(event.scenePos())
        modifier = event.modifiers()

        if modifier & Qt.KeyboardModifier.ControlModifier:
            self.roi.addPoint((pos.x(), pos.y()))
        else:
            self.roi.clear()

    @Slot()
    def get_roi(self):
        data, roi_coords = self.roi.getArrayRegion(
            self.umap.ImageItem.image,
            self.umap.ImageItem,
            returnMappedCoords=True,
        )

        roi_coords = np.transpose(roi_coords, (1, 2, 0))
        roi_coords = np.floor(roi_coords)  # round to nearest full pixel
        mask = np.where(data > 0)
        roi_coords = roi_coords[mask[0], mask[1]]
        if len(roi_coords) > 0:
            self.new_bmu_selection.emit(roi_coords)

    def handle_bmu_hover(self, plot, points, ev):
        if len(points) == 0:
            self._hover_dwell_timer.stop()
            self._pending_scatter_indices = None
            if self._hover_popup is not None:
                self._hover_popup.bmu_left()
            return

        # At low zoom levels multiple BMU dots can occupy the same screen
        # pixels, so more than one point may be hovered at once.
        scatter_indices = frozenset(point.index() for point in points)

        if (
            self._hover_popup is not None
            and scatter_indices == self._popup_scatter_indices
        ):
            # Still hovering the point(s) the open popup belongs to: keep it open.
            self._hover_popup.bmu_still_hovered()
            self._hover_dwell_timer.stop()
            self._pending_scatter_indices = scatter_indices
            return

        if scatter_indices == self._pending_scatter_indices:
            return

        if self._hover_popup is not None:
            # Hovering different point(s) than the ones the open popup belongs to.
            self._hover_popup.bmu_left()

        self._pending_scatter_indices = scatter_indices
        screen_pos = ev.screenPos()
        self._hover_screen_pos = QPoint(int(screen_pos.x()), int(screen_pos.y()))
        self._hover_dwell_timer.start()

    def _open_hover_popup(self):
        if not self._pending_scatter_indices or self._hover_screen_pos is None:
            return

        data_rows = np.concatenate(
            [
                self.bmu_map.get_dataset_rows_for_unique_index(scatter_index)
                for scatter_index in self._pending_scatter_indices
            ]
        )
        if len(data_rows) == 0:
            return

        molecule_rows = self._build_molecule_rows(data_rows)

        if self._hover_popup is not None:
            self._hover_popup.close()

        self._popup_scatter_indices = self._pending_scatter_indices
        self._hover_popup = BmuHoverPopup(parent=self.window())
        self._hover_popup.destroyed.connect(self._clear_hover_popup_ref)
        self._hover_popup.set_molecules(molecule_rows)
        self._hover_popup.show_near(self._hover_screen_pos)

    def _clear_hover_popup_ref(self, *_):
        self._hover_popup = None
        self._popup_scatter_indices = None

    def _build_molecule_rows(
        self, data_rows: np.ndarray
    ) -> list[tuple[Optional[str], list[tuple[str, str]]]]:
        model = self.source_model
        smiles_col = model.structure_info_column_id

        rows: list[tuple[Optional[str], list[tuple[str, str]]]] = []
        for row in data_rows:
            row = int(row)
            smiles = (
                model.data_source.get_value(row, smiles_col)
                if smiles_col is not None
                else None
            )
            properties = []
            for col_idx, name in enumerate(model.columns):
                data_col = model.column_back_map[col_idx]
                if smiles_col is not None and data_col == smiles_col:
                    continue  # excludes both the raw SMILES column and the "Structure" column
                value = model.data_source.get_value(row, data_col)
                properties.append((name, str(value)))
            rows.append((smiles, properties))
        return rows


class SelectionView(W.QWidget):
    def __init__(self, data_model: FilterModel, parent=None):
        super().__init__(parent=parent)
        self.installEventFilter(self)
        # self.setOrientation(Qt.Orientation.Horizontal)

        self.table = CompoundTable(parent=self)
        self.table.setModel(data_model)
        data_model.selection_changed.connect(self.table.resize_to_contents)
        layout = W.QHBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)


class MainView(W.QSplitter):
    def __init__(
        self,
        umap: UMap,
        data_model: FilterModel,
        base_model: CommonDataModel,
        bmu_map: BMUMap,
        bmu_colors: BMUColors,
        bmu_filter: BMUFilter,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.installEventFilter(self)
        self.setOrientation(Qt.Orientation.Vertical)

        self.data_model = data_model
        self.bmu_map = bmu_map

        self.upper_view = UpperView(
            umap=umap,
            bmu_map=self.bmu_map,
            bmu_colors=bmu_colors,
            bmu_filter=bmu_filter,
            data_columns=data_model.columns_with_properties,
            base_model=base_model,
            parent=self,
        )
        self.data_view = SelectionView(data_model=self.data_model, parent=self)

        self.addWidget(self.upper_view)

        self.addWidget(self.data_view)

        self.upper_view.new_bmu_selection.connect(self.new_bmu_selection)

    @Slot(np.ndarray)
    def new_bmu_selection(self, selection_coords: NDArray):
        # Get the BMU indices and data indices from the map coordinates
        scatter_indices, data_indices = self.bmu_map.get_bmu_info_from_map_coordinates(
            selection_coords
        )
        # Exclude datapoints that don't individually pass the active filter (or are
        # globally hidden), even if other datapoints on the same BMU do pass it
        selectable = self.upper_view.datapoint_selection_mask()
        data_indices = data_indices[selectable[data_indices.flatten()]]
        self.data_model.set_selected_rows(data_indices)
        # self.upper_view.bmus_points.setSelected(scatter_indices)


class DataLoadOptionsDialog(W.QDialog):
    """
    Asks for the dataset options that cannot be inferred from the file itself:
    which column holds the SMILES, and — for HDF5 stores — which groups to load.
    """

    NO_STRUCTURE_COLUMN = "<none>"

    def __init__(
        self,
        column_names: list[str],
        groups: Optional[list[str]] = None,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.setWindowTitle("Dataset options")

        layout = W.QVBoxLayout(self)

        form_layout = W.QFormLayout()
        self.structure_selector = W.QComboBox()
        self.structure_selector.addItem(self.NO_STRUCTURE_COLUMN)
        self.structure_selector.addItems(column_names)
        self.structure_selector.setCurrentText(
            self._guess_structure_column(column_names)
        )
        form_layout.addRow("Structure column:", self.structure_selector)
        layout.addLayout(form_layout)

        self.group_list: Optional[W.QListWidget] = None
        if groups:
            layout.addWidget(W.QLabel("Groups to load:"))
            self.group_list = W.QListWidget()
            for group in groups:
                item = W.QListWidgetItem(group)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                self.group_list.addItem(item)
            self.group_list.itemChanged.connect(self._update_accept_enabled)
            layout.addWidget(self.group_list)

        self.buttons = W.QDialogButtonBox(
            W.QDialogButtonBox.StandardButton.Ok
            | W.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def structure_info_column(self) -> Optional[str]:
        """The selected structure column, or None if the user chose not to use one."""
        selected = self.structure_selector.currentText()
        return None if selected == self.NO_STRUCTURE_COLUMN else selected

    def group_subset(self) -> Optional[list[str]]:
        """The checked groups, or None if there is no group choice or all are checked."""
        if self.group_list is None:
            return None
        checked = self._checked_groups()
        return None if len(checked) == self.group_list.count() else checked

    def _checked_groups(self) -> list[str]:
        if self.group_list is None:
            return []
        checked = []
        for i in range(self.group_list.count()):
            item = self.group_list.item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                checked.append(item.text())
        return checked

    @Slot()
    def _update_accept_enabled(self):
        # Loading zero groups would yield an empty dataset, so don't allow it
        ok_button = self.buttons.button(W.QDialogButtonBox.StandardButton.Ok)
        ok_button.setEnabled(bool(self._checked_groups()))

    @classmethod
    def _guess_structure_column(cls, column_names: list[str]) -> str:
        for name in column_names:
            if name.lower() == "smiles":
                return name
        for name in column_names:
            if "smiles" in name.lower():
                return name
        return cls.NO_STRUCTURE_COLUMN


class MainSomWindow(W.QMainWindow):
    ARRAY_FILTER = "NumPy array (*.npy);;All files (*)"
    DATA_FILTER = (
        "Datasets (*.h5 *.hdf5 *.csv *.tsv *.txt *.parquet *.pq);;"
        "HDF5 (*.h5 *.hdf5);;"
        "CSV/TSV (*.csv *.tsv *.txt);;"
        "Parquet (*.parquet *.pq);;"
        "All files (*)"
    )
    # Stand-in for "nothing loaded yet", so the map view stays valid while empty
    PLACEHOLDER_UMATRIX = np.zeros((1, 8, 8), dtype=np.float32)
    DEFAULT_BMU_SIZE = 10

    def __init__(
        self,
        umatrix: Optional[NDArray] = None,
        bmu_coordinates: Optional[NDArray] = None,
        data: Optional[DatasetBase | DataFrame] = None,
        structure_info_column: Optional[str] = None,
        scaling_factor: int = 3,
    ):
        super().__init__()
        self.setMinimumSize(QSize(800, 600))
        self.setWindowTitle("ChI-SOM")

        self._umatrix = self._as_layered(umatrix)
        self._bmu_coordinates = bmu_coordinates
        self._data = data
        self._structure_info_column = structure_info_column
        self._scaling_factor = scaling_factor
        # Datasets opened through the File menu are ours to close again; one passed
        # in by a caller belongs to that caller
        self._owns_data = False
        self._source_names: dict[str, str] = {}

        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        file_menu.addAction("Load U-matrix", self.load_umatrix)
        file_menu.addAction("Load BMUs", self.load_bmus)
        file_menu.addAction("Load data", self.load_data)

        self._rebuild_views()

    def _rebuild_views(self):
        """
        (Re)build the model graph and the central widget from the currently loaded
        artefacts. Everything from the table model to the control combo boxes is
        derived from them at construction time, so a change to any artefact is
        applied by building the whole graph anew.
        """
        view_state = self._capture_view_state()
        previous_models = [
            getattr(self, "base_model", None),
            getattr(self, "data_model", None),
        ]

        umatrix = (
            self._umatrix if self._umatrix is not None else self.PLACEHOLDER_UMATRIX
        )

        self.base_model = CommonDataModel(
            self._data, structure_info_column=self._structure_info_column, parent=self
        )
        self.data_model = FilterModel(self.base_model, parent=self)
        self.umap = UMap(umatrix, scaling_factor=self._scaling_factor, parent=self)
        self.bmu_map = BMUMap(
            bmu_raw_coordinates=self._bmu_coordinates,
            scaling_factor=self._scaling_factor,
        )
        self.bmu_colors = BMUColors(self.data_model, self.bmu_map)
        self.bmu_filter = BMUFilter(self.data_model, self.bmu_map)

        self.main_view = MainView(
            umap=self.umap,
            data_model=self.data_model,
            base_model=self.base_model,
            bmu_map=self.bmu_map,
            bmu_colors=self.bmu_colors,
            bmu_filter=self.bmu_filter,
            parent=self,
        )

        # Disposes of the previous central widget, and with it the previous views
        self.setCentralWidget(self.main_view)

        # Set BMUs and trigger for later chages in scaling
        self.bmu_map.map_bmu_coordinates_changed.connect(
            self.main_view.upper_view.set_bmus
        )
        self.main_view.upper_view.control.set_bmu_state(
            self.bmu_map.bmu_state, view_state["bmu_size"]
        )

        self._apply_view_state(view_state)

        for model in previous_models:
            if model is not None:
                model.deleteLater()

    def _capture_view_state(self) -> dict:
        """Collect the view settings that should survive a rebuild."""
        state = {
            "colormap": "Earth",
            "bmu_size": self.DEFAULT_BMU_SIZE,
            "splitter_sizes": None,
        }
        main_view = getattr(self, "main_view", None)
        if main_view is None:
            return state

        control = main_view.upper_view.control
        state["colormap"] = control.cmap_selector.currentText()
        state["bmu_size"] = control.bmu_size_selector.value()
        state["splitter_sizes"] = main_view.sizes()
        return state

    def _apply_view_state(self, state: dict):
        self.main_view.upper_view.control.change_colormap(state["colormap"])
        if state["splitter_sizes"]:
            self.main_view.setSizes(state["splitter_sizes"])

    def _update_window_title(self):
        if not self._source_names:
            self.setWindowTitle("ChI-SOM")
            return
        loaded = ", ".join(
            f"{label}: {name}" for label, name in self._source_names.items()
        )
        self.setWindowTitle(f"ChI-SOM - {loaded}")

    @staticmethod
    def _as_layered(umatrix: Optional[NDArray]) -> Optional[NDArray]:
        if umatrix is None:
            return None
        if umatrix.ndim == 2:
            return umatrix[np.newaxis, :, :]
        if umatrix.ndim != 3:
            raise ValueError("U-matrix must be 2D or 3D")
        return umatrix

    @staticmethod
    def _validation_error(
        umatrix: Optional[NDArray],
        bmu_coordinates: Optional[NDArray],
        data: Optional[DatasetBase | DataFrame],
    ) -> Optional[str]:
        """
        Check whether the given artefacts describe the same SOM, so that loading
        them in any order can be rejected before anything is swapped in.
        """
        if bmu_coordinates is not None and data is not None:
            if len(bmu_coordinates) != len(data):
                return (
                    f"The BMU coordinates describe {len(bmu_coordinates)} datapoints, "
                    f"but the dataset holds {len(data)}. They do not belong to the "
                    "same SOM run, or the dataset is restricted to a different subset."
                )

        if bmu_coordinates is not None and umatrix is not None and len(bmu_coordinates):
            rows, columns = umatrix.shape[1], umatrix.shape[2]
            max_row, max_column = bmu_coordinates.max(axis=0)
            if max_row >= rows or max_column >= columns:
                return (
                    f"The BMU coordinates reach up to ({max_row}, {max_column}), which "
                    f"does not fit the U-matrix lattice of {rows} x {columns} units."
                )

        return None

    @staticmethod
    def _close_dataset(data: Optional[DatasetBase | DataFrame]):
        if isinstance(data, DatasetBase):
            data.close()

    def _show_error(self, title: str, message: str):
        W.QMessageBox.critical(self, title, message)

    def _show_warning(self, title: str, message: str):
        W.QMessageBox.warning(self, title, message)

    @Slot()
    def load_umatrix(self):
        file_path, _ = W.QFileDialog.getOpenFileName(
            self, "Load U-Matrix", "", self.ARRAY_FILTER
        )
        if not file_path:
            return

        try:
            umatrix = loading.load_umatrix(file_path)
        except Exception as exc:
            self._show_error("Could not load the U-matrix", str(exc))
            return

        error = self._validation_error(umatrix, self._bmu_coordinates, self._data)
        if error is not None:
            self._show_warning("U-matrix does not match the loaded data", error)
            return

        self._umatrix = umatrix
        self._source_names["U-matrix"] = Path(file_path).name
        self._rebuild_views()
        self._update_window_title()

    @Slot()
    def load_bmus(self):
        file_path, _ = W.QFileDialog.getOpenFileName(
            self, "Load BMUs", "", self.ARRAY_FILTER
        )
        if not file_path:
            return

        try:
            bmu_coordinates = loading.load_bmu_coordinates(file_path)
        except Exception as exc:
            self._show_error("Could not load the BMU coordinates", str(exc))
            return

        error = self._validation_error(self._umatrix, bmu_coordinates, self._data)
        if error is not None:
            self._show_warning("BMUs do not match the loaded data", error)
            return

        self._bmu_coordinates = bmu_coordinates
        self._source_names["BMUs"] = Path(file_path).name
        self._rebuild_views()
        self._update_window_title()

    @Slot()
    def load_data(self):
        file_path, _ = W.QFileDialog.getOpenFileName(
            self, "Load data", "", self.DATA_FILTER
        )
        if not file_path:
            return

        try:
            data, structure_info_column = self._load_data_with_options(file_path)
        except Exception as exc:
            self._show_error("Could not load the dataset", str(exc))
            return

        if data is None:  # options dialog was cancelled
            return

        error = self._validation_error(self._umatrix, self._bmu_coordinates, data)
        if error is not None:
            self._show_warning("Dataset does not match the loaded BMUs", error)
            self._close_dataset(data)
            return

        previous_data = self._data if self._owns_data else None
        self._data = data
        self._structure_info_column = structure_info_column
        self._owns_data = True
        self._source_names["data"] = Path(file_path).name
        self._rebuild_views()
        self._update_window_title()
        self._close_dataset(previous_data)

    def _load_data_with_options(
        self, file_path: str
    ) -> tuple[Optional[DatasetBase | DataFrame], Optional[str]]:
        """
        Load a dataset and ask for the options that the file itself doesn't carry.
        Returns (None, None) if the user cancelled the options dialog.
        """
        groups = (
            loading.inspect_hdf5_groups(file_path)
            if Path(file_path).suffix.lower() in loading.HDF5_SUFFIXES
            else None
        )
        # Loaded with all groups first, so the dialog can list the actual columns
        data = loading.load_dataset(file_path)
        column_names = loading.dataset_column_names(data)

        dialog = DataLoadOptionsDialog(column_names, groups=groups, parent=self)
        if dialog.exec() != W.QDialog.DialogCode.Accepted:
            self._close_dataset(data)
            return None, None

        group_subset = dialog.group_subset()
        if group_subset is not None:
            self._close_dataset(data)
            data = loading.load_dataset(file_path, group_subset=group_subset)

        return data, dialog.structure_info_column()

    def closeEvent(self, event):
        if self._owns_data:
            self._close_dataset(self._data)
        super().closeEvent(event)


def start_chisom_viewer(
    umatrix: Optional[NDArray] = None,
    bmu_coordinates: Optional[NDArray] = None,
    data: Optional[DatasetBase | DataFrame] = None,
    structure_info_column: Optional[str] = None,
    scaling_factor: int = 3,
):
    """Start the GUI interface

    All arguments are optional; anything not passed here can be loaded from the
    viewer's File menu instead.

    Parameters
    ----------
    umatrix
        U-Matrix of the SOM.
    bmu_coordinates
        Coordinates of the BMU to the data points.
    data
        Additional data to the data points. Will be renderd in the tabel view and used for coloring of BMUs.
    structure_info_column
        With chemical dataset the column with this index supplies the SMILES to render the molecule, by default None.
    scaling_factor
        Will scale the U-Matrix by this factor ands interpolation for an anti-aliased view, by default 3.
    """

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError(
            "The ChI-SOM viewer needs a graphical display, but none was found "
            "(no $DISPLAY / $WAYLAND_DISPLAY). This happens on headless servers and "
            "remote shells. Run the viewer on a machine with a display, or forward the "
            "display over SSH with `ssh -X` (or `ssh -Y`). Keep in mind the latter comes "
            "with a heavy performance penalty. We would advice to only train large SOM "
            "remotly and perform analysis on a headful system. To train remotely and "
            "non-interactively plot results, use the headless `plot_som` path instead."
        )

    app = pg.mkQApp("ChI-SOM Viewer")

    window = MainSomWindow(
        umatrix, bmu_coordinates, data, structure_info_column, scaling_factor
    )
    window.show()

    app.exec()
