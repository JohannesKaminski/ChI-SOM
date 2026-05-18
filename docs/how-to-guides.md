# How-To Guides
As large datasets of enumerated fingerprints and other information for cheminformatics might not fit into working memory, &#7521;-SOM supplies a dedicated HDF5 file layout and _HDF5Dataset_ class. This _HDF5Dataset_ class is compatible with the [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html) for random, millisecond latency, access into this on-disk storage.

## Creating a HDF5Dataset file
&#7521;-SOM supplies a tool to generate the HDF5 files containing the fingerprints directly from text files of molecular data in different line notations, e.g. SMILES, INCHI, etc., or text files already containing enumerated fingerprint data. Other properties of the molecules can also be recorded for examination with the GUI.  

We import _HDF5Creator_ and either _rdStyleFactory_ or _CSVStyleFactory_. Both need an _rdMolGenerator_ to build the internal representation from line notation. _rdFingerprintGenerator_ is specific to the direct generation of enumerated fingerprints.  
```python
--8<-- "examples/datafile_creation.py::7"
```  
Next, we need to supply arguments to the fingerprint generator as a dictionary
```python
--8<-- "examples/datafile_creation.py:8:8"
```  
The paths and files considered for the HDF5 file creation must be supplied as a dictionary. Keys are distinct groups that can later be accessed individually when using the _Dataset_. Each item contains a list of files and paths that should be included in the respective group. Paths are walked recursively, and all files are included that match the file extensions later supplied to the _HDF5Creator_.  
```python
--8<-- "examples/datafile_creation.py:10:17"
```  
Next, we initialize the factory that will be used to supply individual generators to the _HDF5Creator_ tool, with the generators and variables we defined previously, and pass it to the _HDF5Creator_.
```python
--8<-- "examples/datafile_creation.py:19:25"
```  
The file creation routine further needs a *leaf_map*, indicating the columns of the data to consider, their data type, and value type. The only required key is the 'primary' key, indicating in what column the molecules line notation is stored. The data type can be any Numpy or standard Python type. The value type is later used for the GUI to infer colour-coding behavior. Possible values are 'continouos', 'categorical' or 'na', indicating that the value should only be displayed by the table view, but not used for colour-coding the BMUs.  
```python
--8<-- "examples/datafile_creation.py:27:34"
```  
To finally create the file, we call the _HDF5Creator.create()_ method, with the desired output filepath. We can further skip lines, e.g. in case of a header, and change the separation character.
```python
--8<-- "examples/datafile_creation.py:36:42"
```  
A full working example can be found in the [Examples]({{ config.repo_url.rstrip('/') }}/tree/main/examples)


## Training an ESOM on data in a HDF5Dataset using CUDA
To use the generated `HDF5Dataset` [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html) needs to be set up accoringly.

First we load the HDF5 file using the _HDF5Dataset_ class. In this case we only make the `"active"` subset defined previously available for training. If omitted, the whole dataset is used for training.
```python
--8<-- "examples/basic_som.py:5:5"
```  
```python
--8<-- "examples/basic_som.py:11:12"
```  
We create the _DataLoader_ with the dataset instance as the input.
```python
--8<-- "examples/basic_som.py:2:2"
```  
```python
--8<-- "examples/basic_som.py:14:20"
```  
During initialization of the SOM, we can get the necessary data features from the _HDF5Dataset_, e.g. *fingerprint_length*. We set the *use_cuda* variable to use the CUDA compute backend.  
```python
--8<-- "examples/basic_som.py:4:4"
```  
```python
--8<-- "examples/basic_som.py:25:35"
```  
The DataLoader is then passed to the _train_ method.
```python
--8<-- "examples/basic_som.py:38:39"
--8<-- "examples/basic_som.py:44:44"
```  
Should shuffling be used during training, a new instance of the _DataLoader_ must be created before prediction of BMUs and QE to keep the correct association between the datapoints indices and prediction.
```python
--8<-- "examples/basic_som.py:46:52"
--8<-- "examples/basic_som.py:58:59"
```  
A full working example can be found in the [Examples]({{ config.repo_url.rstrip('/') }}/tree/main/examples)


## Using less cores than available
When &#7521;-SOM should use less core than are currently available on the machine, the desired number of cores to use (`n_cores`) can be set via `numba`, either programmatically
```python
from numba import set_num_threads

set_num_treads(n_cores)
```

or via an environment variable
```bash
export NUMBA_NUM_THREADS=n_cores
```