# --8<-- [start:imports]
from numpy import float32
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator

from chisom.io.datastore_creation import HDF5Creator
from chisom.io.datastore_factories import rdStyleFactory

generator = rdFingerprintGenerator.GetMorganGenerator
# --8<-- [end:imports]
# --8<-- [start:fingerprint_kwargs]
fingerprint_kwargs = {"fpSize": 1024, "radius": 2}
# --8<-- [end:fingerprint_kwargs]

# --8<-- [start:file_dict]
file_dict = {
    "active": [
        "tests/VDR/actives.smi",
    ],
    "inactive": [
        "tests/VDR/inactives.smi",
    ],
}
# --8<-- [end:file_dict]

# --8<-- [start:factory]
molgen = rdStyleFactory(
    MolFromSmiles,
    generator,
    generator_kwargs=fingerprint_kwargs,
    count_fingerprint=True,
)
file_creator = HDF5Creator(fingerprint_generator_factory=molgen)
# --8<-- [end:factory]

# --8<-- [start:leaf_map]
leaf_map = {
    "primary": (0, str),
    "ID": (1, str, "na"),
    "Activity": (2, int, "categorical"),
    "MolWt": (3, float32, "continuous"),
    "MolLogP": (4, float32, "continuous"),
    "TPSA": (5, float32, "continuous"),
}
# --8<-- [end:leaf_map]

# --8<-- [start:create]
file_creator.create(
    file_dict,
    "tests/VDR.h5",
    leaf_map,
    skip_lines=1,
    sep="\t",
)
# --8<-- [end:create]
