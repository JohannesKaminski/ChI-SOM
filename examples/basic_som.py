import numpy as np
from torch.utils.data import DataLoader

from chisom import Som, start_chisom_viewer
from chisom.io import HDF5Dataset
from chisom.utils import lattice_size

EPOCHS = 5
ALPHA = 0.5
BATCHSIZE = 100

# Create a ChemDataset object that from a HDF5 file, that is compatible with the pytorch DataLoader
ds = HDF5Dataset("tests/VDR.h5", ["active"])

# Create a DataLoader object that will be used to train the SOM
dl = DataLoader(
    ds,
    batch_size=BATCHSIZE,
    shuffle=True,
    num_workers=4,
)

rows, columns = lattice_size(len(ds))

# Create a SOM object
# The high and low parameters should be chosen according to the dataset values, to decrease the training time
som = Som(
    rows,
    columns,
    ds.fingerprint_length,
    "cosine",
    use_cuda=True,
    low=ds.fingerprint_min,
    high=ds.fingerprint_max,
)


# Train the SOM for all epochs in a single call
som.train(dl, EPOCHS, ALPHA, BATCHSIZE)

# Create a DataLoader object for prediction (no shuffling)
dl = DataLoader(
    ds,
    batch_size=1000,
    shuffle=False,
    num_workers=10,
)

# Calculate the U-map for the current state of the notebook
umx = som.get_umatrix()
np.save("tests/umx.npy", umx)

# Predict the best matching units and quantization errors for all data points
bmus, qe = som.predict(dl)
np.save("tests/bmus.npy", bmus)

start_chisom_viewer(umx, bmus, ds, structure_info_column="smiles")
