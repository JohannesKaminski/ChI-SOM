# Limitations

This page collects the constraints worth knowing before committing to &#7521;-SOM for a project. None of them are bugs — they are deliberate scope decisions or consequences of the design.

## Project maturity

This software may be considered to be in beta stage. While the user-facing API is expected to remain stable up to a 2.0 release, the internal API might change at any release and can not be considered stable.

"Internal" means everything under `chisom._core` and `chisom._interface`, and any module whose name starts with an underscore. The documented surface in the [Reference](reference.md) is what carries the stability promise.

## Platform and environment

- **Linux only**, or Windows through WSL2. macOS and native Windows are not supported, and this is enforced by the packaging metadata.
- **Python 3.13 or newer** (`>=3.13`). 3.13 is a hard floor: the data store uses `queue.ShutDown`, which does not exist in earlier versions.
- **`cu12` and `cu13` are mutually exclusive.** One environment cannot serve both CUDA 12 and CUDA 13.
- **PyTorch is not a dependency**, but the `DataLoader` training workflow requires it. Install it separately.

## The viewer needs a display

The _Viewer_ will only work on a system with a display attached. When running the application on a server via a remote shell and calling `start_chisom_viewer`, this will usually lead to errors (`"This application failed to start because no Qt platform plugin could be initialized"`). As solutions to this are very setup dependent, the recommended approach for very large SOMs is to only train the SOM on a powerful remote machine and analyse the trained SOM with the GUI locally.

For headless figure generation, [`plot_som`](how-to-guides.md#rendering-a-figure-without-a-display) is the supported path and has no display requirement.

## Viewer feature scope

- **No training.** The viewer explores trained SOMs; it cannot train them.
- **No session save/restore.** Colour assignments, filters and selections are lost when the window closes.
- **Only the last U-Matrix layer is shown.** For a SOM trained with `save_progress`, the training history is stored but the viewer always displays the final layer. `plot_som` accepts a `layer` argument if you need an earlier one.
- **The interpolation scaling factor is launch-time only** — it cannot be changed from within the window.

## Data type inference

How a column is treated for colouring and filtering depends on where it came from:

- **HDF5 stores** record the value type explicitly, from the `leaf_map` you supply at creation time. Anything not marked `categorical` or `continuous` is treated as `na`: visible in the table, unavailable for colouring and filtering.
- **CSV, TSV and Parquet files** have no such metadata, so the type is inferred with a heuristic: a column is treated as **categorical if its first 100 rows contain 10 or fewer distinct values**, and continuous otherwise.

That heuristic can misjudge a column whose variety only appears later in the file, or a numeric code column with few distinct values. If it matters, build an HDF5 store and declare the types explicitly.

## Map topology

The U-Matrix and the U-Distance graph both assume a **toroidal** map: the top edge is adjacent to the bottom, and the left edge to the right. Distances and neighbourhoods wrap accordingly. This matches the default `euclidean_toroid` map distance, but be aware of it when interpreting map edges — there are, in effect, no edges.

## Data store compatibility

HDF5 stores written by &#7521;-SOM 1.0 may carry incorrect value ranges for continuous columns. See [Upgrading to 1.1](upgrading.md) for what to do about it.
