# Installation

## Requirements

| | |
|---|---|
| Operating system | Linux, or Windows through [WSL2](https://learn.microsoft.com/windows/wsl/install) |
| Python | 3.13 or newer (`>=3.13`) |
| GPU (optional) | NVIDIA GPU with a recent driver, for the CUDA compute backend |
| Display (optional) | Required for the interactive viewer — see [Limitations](limitations.md) |

macOS and native Windows are not supported.

## Installing from PyPI

The base install gives you the library: SOM training on CPU, the HDF5 data store, and headless plotting with `plot_som`.

```sh
pip install chi-som
```

That base install does **not** include the interactive viewer. Almost everyone wants it:

```sh
pip install 'chi-som[gui]'
```

!!! note "Quote the brackets"
    In `zsh` (the default shell on many systems) unquoted square brackets are glob syntax and the command will fail with `no matches found`. Quote the whole argument, as above.

### Optional extras

| Extra | Installs | Gives you |
|---|---|---|
| `gui` | `pyside6`, `pyqtgraph`, `pyarrow` | The interactive viewer and the `chisom view` command; Parquet dataset support |
| `cu12` | `numba-cuda-mlir[cu12]` | CUDA 12 compute backend |
| `cu13` | `numba-cuda-mlir[cu13]` | CUDA 13 compute backend |

`cu12` and `cu13` are mutually exclusive — pick the one matching your driver.

Extras combine as usual:

```sh
pip install 'chi-som[gui]'          # viewer, CPU training
pip install 'chi-som[cu12]'         # CUDA 12, no viewer
pip install 'chi-som[cu12,gui]'     # CUDA 12 and the viewer
```

!!! tip "No system CUDA toolkit needed"
    The `cu12`/`cu13` extras bundle the toolkit components they need (`cuda-bindings`, `cuda-toolkit` with `cccl`, `cudart`, `nvcc` and `nvrtc`, plus `nvidia-nvjitlink`). A sufficiently recent NVIDIA **driver** is all your host has to provide — there is no need to install CUDA system-wide.

    For more involved setups, see the [numba-cuda-mlir](https://nvidia.github.io/numba-cuda-mlir/latest/) documentation.

### Things that are deliberately not dependencies

- **PyTorch.** The [DataLoader training workflow](how-to-guides.md#training-an-esom-on-data-in-a-hdf5dataset-using-cuda) needs `torch`, but it is not pulled in by any extra — it is a large install and only one workflow needs it. Add it yourself with `pip install torch`. Training from a plain NumPy array needs nothing extra.
- **PyArrow**, if you skip the `gui` extra. Reading `.parquet` files through `chisom.io.load_dataset` needs it; `[gui]` already includes it, so this only affects library-only installs. `pip install pyarrow` covers it.

## Verifying the install

```sh
chisom --version
chisom view --help
```

If `chisom view --help` prints its options, the console script and the GUI extra are both in place. To check the viewer actually opens, run `chisom view` with no arguments — you should get an empty window with a *File* menu.

## Development setup

&#7521;-SOM is developed, built, and packaged with [Astral uv](https://docs.astral.sh/uv/).

```sh
# Base development environment, including the viewer
uv sync --group dev --extra gui

# With a CUDA backend (choose one)
uv sync --group dev --extra cu12 --extra gui
uv sync --group dev --extra cu13 --extra gui
```

The `dev` group also carries the documentation toolchain and `torch`, so no extra install is needed to run the examples or build the docs.

Common tasks:

```sh
uv run ty check                 # type checking
uv run ruff check src/          # linting
uv run ruff format src/         # formatting
uv run pytest                   # tests
uv run mkdocs serve             # docs preview at http://127.0.0.1:8000
uv build                        # build wheel + sdist
```

## Troubleshooting

??? failure "`This application failed to start because no Qt platform plugin could be initialized`"
    The viewer needs a graphical display and could not find one. &#7521;-SOM checks for `$DISPLAY`/`$WAYLAND_DISPLAY` before starting Qt and raises a `RuntimeError` explaining this.

    Options, roughly in order of preference:

    1. Train on the remote machine, copy the U-Matrix and BMU `.npy` files locally, and run the viewer on your own workstation. This is the recommended workflow for large SOMs.
    2. Render a static figure instead with [`plot_som`](how-to-guides.md#rendering-a-figure-without-a-display) — it works headlessly.
    3. Forward X11 with `ssh -X` (or `-Y`). This works but is noticeably sluggish for large maps.

    Fixes are very setup-dependent, which is why option 1 is recommended.

??? failure "`ImportError: The interactive viewer requires the optional GUI dependencies`"
    You installed `chi-som` without the `gui` extra. Install it with:

    ```sh
    pip install 'chi-som[gui]'
    ```

    Note the distribution name is `chi-som` (with a hyphen), even though the import name is `chisom`.

??? failure "`chisom: command not found`"
    The `chisom` console script ships with the package itself, so this usually means the install went to a different environment than the one on your `PATH`. Check with `python -c "import chisom; print(chisom.__version__)"`. Inside a `uv` project, prefix commands with `uv run`.

??? failure "CUDA kernels fail to compile, or the GPU is not found"
    Confirm the driver is visible with `nvidia-smi`, then check that the extra you installed matches its CUDA major version — `cu12` for CUDA 12, `cu13` for CUDA 13. Installing both is not supported.

    If you switched extras in an existing environment, recreate it rather than upgrading in place; see [Upgrading to 1.1](upgrading.md).

    For deeper configuration, refer to the [numba-cuda-mlir](https://nvidia.github.io/numba-cuda-mlir/latest/) documentation.

??? failure "`ValueError` when loading a `.parquet` dataset"
    Parquet support comes from `pyarrow`. It ships with the `gui` extra; on a library-only install add it with `pip install pyarrow`.
