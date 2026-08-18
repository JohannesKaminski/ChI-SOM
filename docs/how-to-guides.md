# How-To Guides
As large datasets of enumerated fingerprints and other information for cheminformatics might not fit into working memory, &#7521;-SOM supplies a dedicated HDF5 file layout and _HDF5Dataset_ class. This _HDF5Dataset_ class is compatible with the [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html) for random, millisecond latency, access into this on-disk storage.

## Creating a HDF5Dataset file
&#7521;-SOM supplies a tool to generate the HDF5 files containing the fingerprints directly from text files of molecular data in different line notations, e.g. SMILES, INCHI, etc., or text files already containing enumerated fingerprint data. Other properties of the molecules can also be recorded for examination with the GUI.  

We import _HDF5Creator_ and either _rdStyleFactory_ or _CSVStyleFactory_. Both need an _rdMolGenerator_ to build the internal representation from line notation. _rdFingerprintGenerator_ is specific to the direct generation of enumerated fingerprints.  
```python
--8<-- "examples/datafile_creation.py:imports"
```  
Next, we need to supply arguments to the fingerprint generator as a dictionary
```python
--8<-- "examples/datafile_creation.py:fingerprint_kwargs"
```  
The paths and files considered for the HDF5 file creation must be supplied as a dictionary. Keys are distinct groups that can later be accessed individually when using the _Dataset_. Each item contains a list of files and paths that should be included in the respective group. Paths are walked recursively, and all files are included that match the file extensions later supplied to the _HDF5Creator_.  
```python
--8<-- "examples/datafile_creation.py:file_dict"
```  
Next, we initialize the factory that will be used to supply individual generators to the _HDF5Creator_ tool, with the generators and variables we defined previously, and pass it to the _HDF5Creator_.
```python
--8<-- "examples/datafile_creation.py:factory"
```  
The file creation routine further needs a *leaf_map*, indicating the columns of the data to consider, their data type, and value type. The only required key is the 'primary' key, indicating in what column the molecules line notation is stored. The data type can be any Numpy or standard Python type. The value type is later used for the GUI to infer colour-coding behavior. Possible values are 'continuous', 'categorical' or 'na', indicating that the value should only be displayed by the table view, but not used for colour-coding the BMUs.  
```python
--8<-- "examples/datafile_creation.py:leaf_map"
```  
To finally create the file, we call the _HDF5Creator.create()_ method, with the desired output filepath. We can further skip lines, e.g. in case of a header, and change the separation character. Both are optional — `skip_lines` defaults to `0` and `sep` to a tab.
```python
--8<-- "examples/datafile_creation.py:create"
```  
A full working example can be found in the [Examples]({{ config.repo_url.rstrip('/') }}/tree/main/examples)


## Training an ESOM on data in a HDF5Dataset using CUDA
To use the generated `HDF5Dataset` [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html) needs to be set up accoringly.

!!! note "PyTorch is not installed with &#7521;-SOM"
    `torch` is not a runtime dependency of &#7521;-SOM — only this DataLoader workflow needs it. Install it separately with `pip install torch`. Training from a plain NumPy array works without it.

The whole guide needs these imports:
```python
--8<-- "examples/basic_som.py:imports"
```  
First we load the HDF5 file using the _HDF5Dataset_ class. In this case we only make the `"active"` subset defined previously available for training. If omitted, the whole dataset is used for training.
```python
--8<-- "examples/basic_som.py:dataset"
```  
We create the _DataLoader_ with the dataset instance as the input.
```python
--8<-- "examples/basic_som.py:dataloader"
```  
The lattice dimensions follow the ESOM heuristic and are derived from the number of datapoints.
```python
--8<-- "examples/basic_som.py:lattice"
```  
During initialization of the SOM, we can get the necessary data features from the _HDF5Dataset_, e.g. *fingerprint_length*. We set the *use_cuda* variable to use the CUDA compute backend.  
```python
--8<-- "examples/basic_som.py:som"
```  
The DataLoader is then passed to the _train_ method.
```python
--8<-- "examples/basic_som.py:train"
```  
The U-Matrix is available from the `umatrix` property once training has finished. Saving it to disk lets you reopen the map later with [`chisom view`](gui.md#from-the-command-line).
```python
--8<-- "examples/basic_som.py:umatrix"
```  
Should shuffling be used during training, a new instance of the _DataLoader_ must be created before prediction of BMUs and QE to keep the correct association between the datapoints indices and prediction.
```python
--8<-- "examples/basic_som.py:predict_loader"
--8<-- "examples/basic_som.py:predict"
```  
A full working example can be found in the [Examples]({{ config.repo_url.rstrip('/') }}/tree/main/examples)


## Measuring distances on a trained SOM
The U-Matrix shows where the map is stretched, but it does not by itself tell you how far apart two regions are *along* the map. &#7521;-SOM exposes the underlying **U-Distance graph** for that: an undirected, edge-weighted graph whose nodes are grid positions and whose edges connect toroidal grid neighbours, weighted by the high-dimensional distance between the corresponding codebook vectors.

Build the graph once from the trained SOM and reuse it — it is cached on the `Som`, but binding it to a local name keeps repeated queries obvious and cheap.

```python
from chisom import u_distance

graph = som.u_graph
```

`u_distance` then computes shortest-path lengths through that graph. The `target` is always a single `(row, column)` position; the `source` argument accepts three forms.

```python
# 1. Distance from every unit on the map to a reference position
all_distances = u_distance(graph, None, (12, 30))

# 2. Distance from one position
single = u_distance(graph, (4, 4), (12, 30))

# 3. Distance from a collection of positions, e.g. the BMUs of a set of hits
hits = bmus[labels == "active"]
hit_distances = u_distance(graph, hits, (12, 30))
```

Each call returns a `dict` mapping the queried source node — as an `(int, int)` tuple — to its u-distance. Passing `None` yields one entry per node in the graph, which is the form to use when you want a full distance field to plot over the map.

!!! note "Results are keyed by position, so duplicates collapse"
    Because the result is a dictionary keyed by grid position, passing a collection of BMUs returns one entry per *distinct* position. Several datapoints sharing a BMU yield a single entry, so the result can be shorter than the collection you passed in.

!!! tip "Build the graph once"
    Every call to `u_distance` runs a shortest-path search over the graph you hand it. Constructing the graph is the expensive part, so build it once and pass the same object to all your queries rather than accessing `som.u_graph` inside a loop over BMU pairs.


## Rendering a figure without a display
The interactive viewer needs a graphical display. On a headless machine — a compute node, a CI job, an SSH session without X forwarding — use `plot_som` instead, which renders the same style of map through matplotlib and can write straight to a file. It is also the way to generate figures in bulk or from a script; if you already have the viewer open, it can [export the map directly](gui.md#exporting-the-map).

```python
from chisom import plot_som

fig = plot_som(
    som.umatrix,
    bmu_coordinates=bmus,
    data=ds,
    color_by="Activity",
    save_as="som_plot.png",
)
```

`data` accepts either a `HDF5Dataset` or a plain `pandas.DataFrame`, and `color_by` names one of its columns. Whether that column is treated as categorical or continuous is inferred, but you can force it with `categorical=True`/`False`. For a SOM trained with `save_progress`, `layer` selects which epoch's U-Matrix to draw (default `-1`, the last).

The figure sizes itself from the lattice, so non-square and very large maps come out with sensible proportions instead of being squeezed into a default canvas. The relevant knobs:

| Parameter | Default | Effect |
|---|---|---|
| `cell_size_in` | `0.2` | Inches per grid cell — the main size control |
| `chrome_scale` | `1.0` | Scales legend, colorbar and label sizes together |
| `marker_cell_fraction` | `0.7` | BMU marker diameter as a fraction of one grid cell |
| `legend_ncol` | auto | Columns in the categorical legend; derived from the category count if omitted |
| `legend_label_maxlen` | `24` | Long category labels are elided to this length |

Passing `figsize` or `marker_size` explicitly opts back out of the grid-derived sizing. Passing `ax` draws into an existing axes instead of creating a figure.


## Using less cores than available
When &#7521;-SOM should use less core than are currently available on the machine, the desired number of cores to use (`n_cores`) can be set via `numba`, either programmatically
```python
from numba import set_num_threads

set_num_threads(n_cores)
```

or via an environment variable
```bash
export NUMBA_NUM_THREADS=n_cores
```
