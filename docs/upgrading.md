# Upgrading to 1.1

Version 1.1 carries three changes that will break existing setups, plus a data-correctness fix worth acting on. Everything else is additive.

## Breaking changes

### The GUI dependencies are now optional

`pip install chi-som` no longer installs PySide6 and pyqtgraph. A plain install gives you the library only, and importing the viewer raises an `ImportError`.

```sh
pip install 'chi-som[gui]'
```

This keeps the library installable on headless compute nodes, where a Qt stack is dead weight. `import chisom` and `chisom.plot_som` work without the extra; `chisom.start_chisom_viewer` is resolved lazily and only then requires it.

### The CUDA backend changed

The `cu12` and `cu13` extras now install **`numba-cuda-mlir`** instead of `numba-cuda`.

Upgrading in place can leave both backends present in the same environment. Recreate the environment rather than upgrading over the top:

```sh
uv sync --extra cu12 --extra gui
```

`use_cuda=True` on `Som` is unchanged — no code changes are needed.

### Training is a single call

Manual epoch loops are replaced by one `train` call that runs the whole schedule internally, including per-step decay of `alpha` and `sigma`:

```python
# Before (1.0)
for epoch in range(EPOCHS):
    for batch in batches:
        som.train_batch(batch, sigma, alpha)
        # ... decay sigma and alpha yourself

# Now (1.1)
som.train(data, EPOCHS, ALPHA, BATCHSIZE)
```

`train` accepts either a NumPy array or a PyTorch `DataLoader`. Decay schedules are configurable via `alpha_decay`, `sigma_decay`, `alpha_end` and `sigma_end`.

`train_batch` still exists for manual stepping, but it is no longer the expected way to train a map.

!!! warning "`sigma` and `alpha` bounds are validated"
    `train` raises `ValueError` if `sigma <= sigma_end` or `alpha <= alpha_end`. If you previously passed a small `sigma`, check it against the `sigma_end` default of `1`.

## Deprecations

### `get_umatrix()` → the `umatrix` property

```python
umx = som.get_umatrix()   # deprecated
umx = som.umatrix         # use this
```

The U-Matrix is now always **3D**, of shape `(n_layers, rows, columns)`. With `save_progress` set, one layer is stored per epoch, giving you the training history; otherwise there is a single layer. Code that indexed a 2D array needs `umx[-1]` to get the final layer.

`plot_som` and the viewer both accept the 3D form directly.

## New in 1.1

- **A command line interface.** `chisom view` opens the viewer without writing a script. See [The Viewer](gui.md#from-the-command-line).
- **U-Distance queries.** `Som.u_graph` exposes the map's weighted neighbour graph, and `chisom.u_distance` computes shortest-path distances over it. See [Measuring distances on a trained SOM](how-to-guides.md#measuring-distances-on-a-trained-som).
- **Filtering by data property in the viewer.** Narrow the map and the table to a subset of your datapoints. See [Filtering by data property](gui.md#filtering-by-data-property).
- **Loading artefacts from the viewer.** `start_chisom_viewer` now takes no required arguments; U-Matrix, BMUs and dataset can be loaded from the *File* menu.
- **More dataset formats.** CSV, TSV and Parquet alongside the HDF5 store, via the new `chisom.io.loading` helpers.
- **Molecule popups.** Hovering a BMU shows the structures and properties of the molecules mapped to it.

## Regenerate HDF5 stores written by 1.0

Two bugs in the 1.0 data store path affected the recorded value range of continuous columns:

- The upper bound was written from the wrong index during store creation, so continuous columns could carry a wrong maximum.
- Merging value ranges across multiple groups did not widen them correctly, so multi-group stores reported wrong ranges.

Both are fixed. The stored fingerprints and properties were never wrong — only the cached per-column ranges — but those ranges drive the viewer's continuous colour scales and the default bounds of the new range filters.

If colour scales on continuous properties ever looked wrong, or if you use multi-group stores, **regenerate the store** with `HDF5Creator` under 1.1. Otherwise no action is needed.
