from typing import (
    Any,
    Collection,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

bmu_type = np.dtype([("row", np.uint16), ("column", np.uint16)])


class ColumnProperties(NamedTuple):
    dtype: Optional[np.dtype]
    value_type: Literal["categorical", "continuous", "na"]
    value_range: list


class BMUCompositionCategorical(NamedTuple):
    category_ratio: NDArray[np.float16]
    alphas: dict[str, NDArray[np.uint8]]


class BMUCompositionContinuous(NamedTuple):
    average: NDArray[np.float16]
    minimum: float
    maximum: float


@runtime_checkable
class ColorDataSource(Protocol):
    columns_with_properties: dict[str, ColumnProperties]

    def get_values_for_column(self, column_name: str) -> Union[Collection, NDArray]: ...


@runtime_checkable
class TabularDatasource(Protocol):
    columns_with_properties: dict[str, ColumnProperties]

    @property
    def column_names(self) -> list[str]: ...

    def __len__(self) -> int: ...
    def get_values_for_column(self, column_name: str) -> Union[Collection, NDArray]: ...
    def get_value(self, row_idc: Union[int, list[int]], column_idx: int) -> Any: ...
