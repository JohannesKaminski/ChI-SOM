#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 2024

@author: j.kaminski@uni-muenster.de for agkoch
"""

import warnings
from abc import ABC, abstractmethod
from cmath import e
from typing import Callable, Collection, Dict, List, Optional, Union

import numpy as np
import tables
from numpy.typing import NDArray

from chisom._interface._types import ColumnProperties

# Fitler out repetitive warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The codec `vlen-utf8`")


def binary_search(prefix_sums, indexes):
    out_group, out_localindex = (
        np.empty(len(indexes), dtype=np.int32),
        np.empty(len(indexes), dtype=np.int32),
    )
    for i in range(len(indexes)):
        left, right = 0, len(prefix_sums) - 1
        while left < right:
            mid = (left + right) // 2
            if prefix_sums[mid] <= indexes[i]:
                left = mid + 1
            else:
                right = mid
        out_group[i] = left - 1
        out_localindex[i] = indexes[i] - prefix_sums[left - 1]
    return out_group, out_localindex


class DatasetBase(ABC):
    def __init__(self, filepath: str) -> None:
        self.fingerprint_length: int  # Indicate the length of the fingerprint stored
        self.packed: bool  # Indicate if fingerprint is stored in compressed form
        self.fingerprint_min: float
        self.fingerprint_max: float

        self.columns_with_properties: Dict[str, ColumnProperties]
        # Storing information on the stored properties in the format: dict(column_name: (dtype, (value_type, [categories/ranges])))

        self.filepath = filepath

        self.prefix_sums = np.array(
            [0], dtype=np.int32
        )  # Used to track where each subgroup beginns in the overall index
        self.total_items = 0  # Used to track total number of items for overall index

    def __len__(self) -> int:
        return self.total_items

    def _build_unpack(self, packed_flag) -> Callable:
        if packed_flag:

            def unpack(array):
                return np.unpackbits(array, axis=-1)

            return unpack
        else:

            def unpack(array):
                return array

            return unpack

    @property
    def column_names(self) -> List[str]:
        return list(self.columns_with_properties.keys())

    @property
    def index(self) -> NDArray[np.int32]:
        return np.arange(self.total_items, dtype=np.int32)

    @abstractmethod
    def __getitem__(self, row_idx: int) -> NDArray: ...

    @abstractmethod
    def get_value(
        self, row_idc: Union[int, Collection[int]], column_idx: int
    ) -> NDArray: ...

    @abstractmethod
    def _get_values(self, row_idc: Collection[int], column_idx: int) -> NDArray: ...

    @abstractmethod
    def get_values_for_column(self, column_name: str) -> NDArray: ...

    @abstractmethod
    def close(self) -> None: ...


class HDF5Dataset(DatasetBase):
    """
    Loads HDF5 Datasets for large cheminformatic datasets created with the supplied script.
    Adheres to the PyTorch Dataset interface for the use with the PyTorch DataLoader for milisecond on-disc access data access.


    """

    def __init__(
        self,
        filepath: str,
        group_subset: Optional[List[str]] = None,
    ) -> None:
        """
        Initializes the Dataset

        Parameters
        ----------
        filepath
            Path to HDF5 file
        group_subset
            Subsets to use, by default None
        """
        super().__init__(filepath=filepath)
        self.file = tables.open_file(filepath, mode="r")
        self.fingerprint_length = self.file.root._v_attrs.fingerprint_length
        self.packed = self.file.root._v_attrs.packed_flag

        self._initialize_groups(group_subset)
        self.fingerprint_min = float(
            self.columns_with_properties["fingerprint"].value_range[0]
        )
        self.fingerprint_max = float(
            self.columns_with_properties["fingerprint"].value_range[1]
        )

    def _initialize_groups(self, group_subset):
        self.columns_with_properties = {}

        # If no subset specified use all available groups
        all_groups = sorted([x._v_name for x in self.file.list_nodes("/")])
        if group_subset:
            self.group_subset = sorted(group_subset)
        else:
            self.group_subset = all_groups

        # Iterate over groups to initialize prefixes, datatypes and value range annotation
        for group_num, group_name in enumerate(self.group_subset):
            current_group = self.file.root[group_name]

            for leaf_name, leaf in current_group._v_leaves.items():
                data_type = leaf.atom.type
                if data_type == "vlunicode":
                    data_type = np.dtypes.StringDType()
                else:
                    data_type = np.dtype(data_type)

                value_type = str(leaf.attrs["type"])
                value_type = (
                    "na"
                    if value_type not in ["categorical", "continuous"]
                    else value_type
                )
                value_range = list(leaf.attrs["value_range"])

                # Construct on first encounter of property, update on all further encouters
                if group_num == 0:
                    self.columns_with_properties[leaf_name] = ColumnProperties(
                        data_type, value_type, value_range
                    )
                else:
                    current_column_state = self.columns_with_properties[leaf_name]
                    assert current_column_state.dtype == data_type
                    assert current_column_state.value_type == value_type

                    current_value_range = current_column_state.value_range[1]
                    if value_type == "categorical":
                        self.columns_with_properties[leaf_name].value_range[1] = list(
                            set.union(set(current_value_range), set(value_range))
                        )
                    elif value_type == "continous":
                        self.columns_with_properties[leaf_name].value_range[0] = min(
                            current_value_range[0], value_range[0]
                        )
                        self.columns_with_properties[leaf_name].value_range[1] = max(
                            current_value_range[1], value_range[1]
                        )
                    else:
                        pass

            # Add column_id entry for group membership
            self.columns_with_properties["group"] = ColumnProperties(
                np.dtypes.StringDType(), "categorical", self.group_subset
            )
            # Update total_items with length of fingerprint array
            self.total_items += current_group._f_get_child("fingerprint").shape[0]
            self.prefix_sums = np.append(self.prefix_sums, self.total_items)

        self.fingerprint_column_number = self.column_names.index("fingerprint")

        self.unpack = self._build_unpack(self.packed)

    def __getitem__(self, row_idx: int):
        return self.get_value(row_idx, self.fingerprint_column_number)

    def get_value(self, row_idc: Union[int, Collection[int]], column_idx: int):
        if isinstance(row_idc, Collection):
            return self._get_values(row_idc, column_idx)
        else:
            if row_idc < 0 or row_idc >= self.total_items:
                raise IndexError("Index out of range")

            # Find the group using binary search
            left, right = 0, len(self.prefix_sums) - 1
            while left < right:
                mid = (left + right) // 2
                if self.prefix_sums[mid] <= row_idc:
                    left = mid + 1
                else:
                    right = mid

            group_index = left - 1  # Adjust to get correct group index
            group_name = self.group_subset[group_index]

            local_index = row_idc - self.prefix_sums[group_index]

            group = self.file.root[group_name]

            leaves = group._v_leaves
            column_name = self.column_names[column_idx]

            if column_name == "group":
                return group_name
            elif column_name == "fingerprint":
                packed_fingerprint = leaves["fingerprint"][local_index]
                return self.unpack(packed_fingerprint)
            else:
                return leaves[column_name][local_index]

    def _get_values(self, row_idc: Collection[int], column_idx: int) -> NDArray:
        group_indices, local_indices = binary_search(self.prefix_sums, row_idc)

        groups_in_rows = set(group_indices)

        column_name = self.column_names[column_idx]
        column_dtype = self.columns_with_properties[column_name].dtype

        if column_name != "fingerprint":
            out = np.empty((len(row_idc)), dtype=column_dtype)
        else:
            out = np.empty((len(row_idc), self.fingerprint_length), dtype=column_dtype)

        if self.column_names[column_idx] == "group":
            for group_idx in groups_in_rows:
                original_position = np.argwhere(group_indices == group_idx).flatten()
                out[original_position] = self.group_subset[group_idx]
            return out
        elif self.column_names[column_idx] == "fingerprint":
            for group_idx in groups_in_rows:
                original_position = np.argwhere(group_indices == group_idx).flatten()
                mol_indices = local_indices[original_position]
                group_name = self.group_subset[group_idx]
                leaves = self.file.root[group_name]._v_leaves
                packed_fingerprints = np.array(
                    [leaves["fingerprint"][i] for i in mol_indices]
                )
                out[original_position] = self.unpack(packed_fingerprints)
            return out
        else:
            for group_idx in groups_in_rows:
                original_position = np.argwhere(group_indices == group_idx).flatten()
                mol_indices = local_indices[original_position]
                group_name = self.group_subset[group_idx]
                leaves = self.file.root[group_name]._v_leaves
                out[original_position] = leaves[self.column_names[column_idx]][
                    mol_indices
                ]
            return out

    def get_values_for_column(self, column_name: str) -> NDArray:
        values = []

        if column_name == "group":
            # self.prefix_sums has 0 as first element, length of first group is second element
            for i, prefix in enumerate(self.prefix_sums[1:]):
                num_elements = self.prefix_sums[i + 1] - self.prefix_sums[i]

                values.extend([self.group_subset[i]] * num_elements)

            values_out = np.array(values)

        else:
            for group_name in self.group_subset:
                group = self.file.root[group_name]
                if column in group._v_leaves.keys():
                    values.append(group._v_leaves[column][:])

            values_out = np.concat(values)

        return values_out

    def close(self):
        try:
            self.file.close()
        except AttributeError:
            pass

    def __del__(self):
        self.close()

    def __exit__(self):
        self.close()
