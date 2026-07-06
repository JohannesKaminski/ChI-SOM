import numpy as np

from chisom import start_chisom_viewer
from chisom.io import HDF5Dataset

if __name__ == "__main__":
    umatrix = np.load("tests/umx.npy")
    bmus = np.load("tests/bmus.npy")
    data = HDF5Dataset("tests/VDR.h5", ["active"])

    start_chisom_viewer(umatrix, bmus, data, structure_info_column="smiles")
