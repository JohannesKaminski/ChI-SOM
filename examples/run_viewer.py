import numpy as np

from chisom import start_chisom_viewer
from chisom.io import HDF5Dataset
from chisom.io.plotting import plot_som

if __name__ == "__main__":
    umatrix = np.load("tests/umx.npy")
    bmus = np.load("tests/bmus.npy")
    data = HDF5Dataset("tests/VDR.h5", ["active"])

    fig = plot_som(
        umatrix,
        bmu_coordinates=bmus,
        data=data,
        marker_size=5,
        color_by="Activity",
        save_as="som_plot.png",
    )
    start_chisom_viewer(umatrix, bmus, data, structure_info_column="smiles")
