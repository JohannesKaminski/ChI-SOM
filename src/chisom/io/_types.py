from collections.abc import Callable
from typing import Protocol, Set, TypeVar, runtime_checkable

import numpy as np
import tables
from numpy.typing import NDArray
from rdkit.Chem.rdchem import Mol

Smiles = TypeVar("Smiles", bound=str)
Atom_Type = TypeVar("Atom_Type", str, int, float, np.dtype)
MolGenerator = Callable[[str], Mol]
FileList = dict[str, list[str]]
ExtraColumns = dict[str, tuple[int, str]]
InputLine = list[str]
OutputLine = dict[str, list[str] | NDArray]
LeafMap = dict[str, tuple[int, Atom_Type, str] | tuple[int, Atom_Type]]
Range = TypeVar("Range", list[float], Set)
RangesDict = dict[str, dict[str, str | Range]]
FileRoot = type(tables.Group)
FingerprintStack = TypeVar("FingerprintStack", NDArray, list[NDArray])
Packer = Callable[[FingerprintStack], NDArray]


class rdFingerprintGenerator:
    def __init__(self): ...
    def GetFingerprintAsNumPy(self, mol: Mol) -> NDArray[np.uint8]: ...


@runtime_checkable
class DataLoader(Protocol):
    collate_fn: Callable

    def __init__(
        self, dataset, batch_size, shuffel, num_workers, collate_fn, pin_memory
    ): ...


Timeout = TypeVar("Timeout", int, float)
Message = TypeVar("Message", InputLine, OutputLine)
