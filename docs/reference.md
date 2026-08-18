# Library Reference

This page documents the public Python API. Everything under `chisom._core` and
`chisom._interface`, and any module whose name begins with an underscore, is internal and
may change at any release — see [Limitations](limitations.md).

The `chisom` command line interface is documented in [The Viewer](gui.md#from-the-command-line)
rather than here, since its surface is its arguments rather than its functions.

::: chisom
    options:
        members:
        - Som
        show_docstring_attributes: true

::: chisom._interface.gui.start_chisom_viewer
    options:
        show_root_full_path: false

::: chisom.utils

::: chisom.analysis

Datasets are loaded by suffix. `chisom.io.load_dataset` (and the viewer's *File → Load data*
dialog) accept `.h5`/`.hdf5` for HDF5 stores, `.csv` for comma-separated text, `.tsv`/`.txt`
for tab-separated text, and `.parquet`/`.pq` for Parquet. The full tuple is available as
`chisom.io.loading.DATASET_SUFFIXES`.

::: chisom.io
    options:
        members_order: source
        summary:
            modules: false
