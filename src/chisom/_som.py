#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial
from math import log
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm, trange

from chisom._core.cpu import trainer as cpu_trainer
from chisom._core.cpu.umatrix import make_umatrix_calculation
from chisom._core.types import Codebook, UMatrix
from chisom._core.utils import _decay_exponential, _decay_linear
from chisom.io._types import DataLoader
from chisom.io._utils import numpy_collate

try:
    from chisom._core.cuda import trainer as gpu_trainer

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class Som:
    """Main Class to create and train a Self-Organizing Map"""

    vector_distances = ["manhattan", "euclidean", "cosine"]

    map_distances = [
        "manhattan",
        "euclidean",
        "manhattan_toroid",
        "euclidean_toroid",
    ]

    neighborhood_kernels = ["gaussian", "mexican", "cone"]

    def __init__(
        self,
        rows: int,
        columns: int,
        features: int,
        vector_distance: str = "euclidean",
        map_distance: str = "euclidean_toroid",
        neighborhood_kernel: str = "gaussian",
        use_cuda: bool = False,
        use_local_neighborhood: bool = False,
        use_fastmath: bool = True,
        save_progress: Optional[str] = None,
        low: float = 0.0,
        high: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initializes the Som Object.

        Parameters
        ----------
        rows
            Number of rows of neurons.
        columns
            Number of columns of neurons.
        features
            Numbers of features to the data / weights of each neuron.
        vector_distance
            Distance used in original data space, by default "euclidean".
            Possible values: "euclidean", "manhattan", "cosine"
        map_distance
            Distance used in map space, by default "euclidean_toroid".
            Possbile values: "euclidean", "manhattan", "euclidean_toroid", "manhattan_toroid"
        neighborhood_kernel
            Shape of the neighborhood kernel, by default "gaussian".
        use_cuda
            If True, CUDA accelleration is used. Needs numba-cuda. By default False.
        use_local_neighborhood
            Sets a hard neighborhood cutoff, by default False.
            Only used on CPU. Significantly increases performance at cost of numerical accuracy.
        use_fastmath
            Slightly decrease numerical accuracy to increase performance, by default True.
        save_progress
            Saves codebook and U-Matrix to the given location if set, by default None.
            Usefull if long running computations crash / time out.
        low
            Lower bound for codebook initialization, by default 0.0.
        high
            Upper bound for codebook initialization, by default 1.0.
        seed
            Randomness seed for replicability, by default None.

        Raises
        ------
        ValueError
            If the map dimensions are less than 1x1.
        ValueError
            If the number of features is less than 2.
        ImportError
            If CUDA is requested but not available.
        ValueError
            If the vector distance norm is not one of the supported norms.
        ValueError
            If the map distance norm is not one of the supported norms.
        ValueError
            If the neighborhood kernel is not one of the supported kernels.
        """

        # Sanity checks and definitions
        if rows <= 0 or columns <= 0:
            raise ValueError("Map dimension must be at least 1x1")
        else:
            self.rows = rows
            self.columns = columns
        if features <= 1:
            raise ValueError("Needs at least 2 features")
        else:
            self.features = features

        if use_cuda and not CUDA_AVAILABLE:
            raise ImportError(
                "CUDA is not available. Please install the CUDA version of chi-som."
            )

        self.use_cuda = use_cuda and CUDA_AVAILABLE
        self.use_local_neighborhood = use_local_neighborhood
        self.fastmath = use_fastmath

        if save_progress is not None:
            self.outpath: Path | None = Path(save_progress)
            self.outpath.mkdir(parents=False, exist_ok=False)
        else:
            self.outpath = None

        self.save_progress = save_progress

        if vector_distance not in Som.vector_distances:
            raise ValueError(f"Unknown vector distance norm: {vector_distance}")
        else:
            self.vector_distance_norm = vector_distance

        if map_distance not in Som.map_distances:
            raise ValueError(f"Unknown map distance norm: {map_distance}")
        else:
            self.map_distance_norm = map_distance

        if neighborhood_kernel not in Som.neighborhood_kernels:
            raise ValueError(f"Unknown neighborhood kernel: {neighborhood_kernel}")
        else:
            self.neighborhood_kernel = neighborhood_kernel

        # Initial setup of codebook
        self.dimensions = np.array((self.rows, self.columns), dtype=np.int32)
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        codebook = self.rng.uniform(low, high, (self.rows, self.columns, self.features))
        codebook = np.asarray(codebook, dtype=np.float32, order="C")
        trainer_type: type

        if self.use_cuda:
            trainer_type = gpu_trainer.CudaTrainer
        else:
            if self.use_local_neighborhood:
                trainer_type = cpu_trainer.CPUTrainerLocal
            else:
                trainer_type = cpu_trainer.CPUTrainer

        self.trainer_instance = trainer_type(
            codebook,
            self.vector_distance_norm,
            self.map_distance_norm,
            self.neighborhood_kernel,
            self.fastmath,
        )

        self.umatrix: UMatrix

    def train(
        self,
        data: NDArray | DataLoader,
        epochs: int,
        alpha: float,
        batchsize: int = 1,
        shuffle: bool = True,
        sigma: Optional[int] = None,
        alpha_decay: str = "linear",
        sigma_decay: str = "exponential",
        alpha_end: float = 0.01,
        sigma_end: int = 1,
    ) -> None:
        """
        Train the SOM with the given data for a number of epochs.

        Runs the full training loop internally, including per-step decay of
        ``alpha`` and ``sigma`` and (if ``save_progress`` was set on
        initialization) periodic U-Matrix/codebook checkpointing once per
        epoch. Progress is reported via tqdm progress bars for both the
        epoch loop and the per-epoch batch loop.

        Parameters
        ----------
        data
            The data to train the SOM with. If a DataLoader is used, it is
            iterated batch by batch as configured on the DataLoader itself.
            If a numpy array is used, it is split into batches of size
            `batchsize`.
        epochs
            Number of epochs to train for.
        alpha
            The initial learning rate. Decays to `alpha_end` over the
            course of training according to `alpha_decay`.
        batchsize
            Number of data points per batch. Only used when `data` is a
            numpy array (ignored for DataLoader input, which defines its
            own batching), by default 1.
        shuffle
            If True, the order of batches is reshuffled at the start of
            each epoch. Only applies when `data` is a numpy array;
            DataLoader shuffling is controlled by the DataLoader itself.
            By default True.
        sigma
            The initial neighborhood radius. Must be greater than 0 if
            given. Defaults to None, in which case it is set to half the
            smaller of the map's row/column dimensions.
        alpha_decay
            The decay schedule for `alpha`, one of "linear" or
            "exponential", by default "linear".
        sigma_decay
            The decay schedule for `sigma`, one of "linear" or
            "exponential", by default "exponential".
        alpha_end
            The value `alpha` decays towards by the end of training, by
            default 0.01.
        sigma_end
            The value `sigma` decays towards by the end of training, by
            default 1.

        Raises
        ------
        ValueError
            If sigma is given and is less than or equal to 0.
        ValueError
            If `alpha_decay` or `sigma_decay` is not "linear" or
            "exponential".
        """

        if sigma is not None and sigma <= 0:
            raise ValueError("Sigma can not be zero or smaller")

        if sigma is None:
            sigma = min(self.rows, self.columns) // 2

        # Transform the input data to adhere to batching and CPU training if necessary
        batches = self._transform_in_data(data, batchsize)
        steps = epochs * len(batches)

        if alpha_decay == "linear":
            alpha_decay_func = partial(_decay_linear, init=alpha, decay=steps)
        elif alpha_decay == "exponential":
            alpha_decay_func = partial(
                _decay_exponential, init=alpha, decay=steps / -log(alpha_end / alpha)
            )
        else:
            raise ValueError(f"Unknown alpha decay: {alpha_decay}")

        if sigma_decay == "linear":
            sigma_decay_func = partial(_decay_linear, init=sigma, decay=steps)
        elif sigma_decay == "exponential":
            sigma_decay_func = partial(
                _decay_exponential, init=sigma, decay=steps / -log(sigma_end / sigma)
            )
        else:
            raise ValueError(f"Unknown sigma decay: {sigma_decay}")

        step = 0
        for epoch in trange(epochs, desc="Epoch: "):
            if shuffle and isinstance(batches, list):
                epoch_data = [batches[i] for i in self.rng.permutation(len(batches))]
            else:
                epoch_data = batches

            for batch in tqdm(epoch_data, leave=False, desc="Batch: "):
                # Use the update coefficients function to set the trainers parameter for the step, including factors for map distance
                self.trainer_instance.update_coefficients(
                    alpha=np.float32(alpha_decay_func(step)),
                    sigma=np.float32(sigma_decay_func(step)),
                )
                self.trainer_instance.train(batch)
                step += 1

            # Save the umatrix and codebook if save_progress is set
            if self.outpath is not None:
                self.umatrix = np.concat(
                    (
                        self.umatrix,
                        self.get_umatrix()[np.newaxis, :, :],
                    ),
                    axis=0,
                )
                np.save(self.outpath / "umatrix", self.umatrix)
                np.save(self.outpath / "codebook", self.trainer_instance.codebook)

    def train_batch(self, data: NDArray, sigma: int, alpha: float) -> None:
        """
        Manually train the SOM with a single batch of data.

        Provides fine-grained control over training by allowing the caller
        to set `alpha` and `sigma` explicitly for a single batch, bypassing
        the automatic per-epoch decay scheduling used by `train`. Useful for
        custom training loops or schedules not covered by `train`.

        Parameters
        ----------
        data
            A single batch of vectors to train the SOM on.
        sigma
            The neighborhood radius to use for this batch.
        alpha
            The learning rate to use for this batch.
        """
        self.trainer_instance.update_coefficients(
            alpha=np.float32(alpha),
            sigma=np.float32(sigma),
        )
        self.trainer_instance.train(data)

    @property
    def codebook(self) -> Codebook:
        return self.trainer_instance.codebook

    @codebook.setter
    def codebook(self, codebook: Codebook) -> None:
        self.trainer_instance.codebook = codebook

    def get_umatrix(self) -> UMatrix:
        """
        Calculate the UMatrix for the current codebook

        Returns
        -------
        UMatrix
            The UMatrix for the current codebook.
        """
        # TODO move to trainer factory, support more distances
        umatrix_func = make_umatrix_calculation(self.vector_distance_norm)
        umatrix = umatrix_func(self.codebook)

        return umatrix

    def predict(
        self, data: NDArray | DataLoader
    ) -> Tuple[NDArray[np.uint16], NDArray[np.float32]]:
        """
        Return the positions of the BMU for a dataset

        Parameters
        ----------
        data
            Dataset to find the BMUs for.

        Returns
        -------
        NDArray[np.uint16]
            The BMUs for the data.
        NDArray[np.float32]
            The Quantization Error

        Raises
        ------
        TypeError
            Error if the data format is not known
        """

        batchsize = (
            (data.batch_size or 1) if isinstance(data, DataLoader) else len(data)
        )
        batches = self._transform_in_data(data, batchsize)
        if isinstance(batches, DataLoader):
            batches.shuffle = False

        bmu_batches, qe_batches = [], []
        for batch in batches:
            bmu_batch, qe_batch = self.trainer_instance.predict(batch)
            bmu_batches.append(bmu_batch)
            qe_batches.append(qe_batch.flatten())
        bmu = np.vstack(bmu_batches)
        qe = np.concat(qe_batches)

        return bmu.astype(np.uint16), qe

    def _transform_in_data(
        self, data: NDArray | DataLoader, batchsize: int
    ) -> List[NDArray[np.float32]] | DataLoader:
        # return_fp_from_dict is necessary to select to correct column from the Dataloader
        if isinstance(data, DataLoader):
            # Collate to numpy arrays if using CPU calculation
            if not self.use_cuda:
                data.collate_fn = numpy_collate
            return data

        # Split a numpy array input into a list of batches of size `batchsize`,
        # to conform to batched data iterating
        data = np.astype(data, np.float32)
        n_batches = -(-len(data) // batchsize)  # ceil division
        return np.array_split(data, n_batches)
