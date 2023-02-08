# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Type, List
import numpy as np
from pathlib import Path
from random import shuffle
import os
import cv2

from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements
from kenning.datasets.helpers.detection_and_segmentation import DectObject


class RandomizedClassificationDataset(Dataset):
    """
    Creates a sample randomized classification dataset.

    It is a mock dataset with randomized inputs and outputs.

    It can be used only for speed and utilization metrics, no quality metrics.
    """

    arguments_structure = {
        'samplescount': {
            'argparse_name': '--num-samples',
            'description': 'Number of samples to process',
            'type': int,
            'default': 1000
        },
        'numclasses': {
            'argparse_name': '--num-classes',
            'description': 'Number of classes in inputs',
            'type': int,
            'default': 3
        },
        'inputdims': {
            'argparse_name': '--input-dims',
            'description': 'Dimensionality of the inputs',
            'type': int,
            'default': [224, 224, 3],
            'is_list': True
        },
        'outputdims': {
            'argparse_name': '--output-dims',
            'description': 'Dimensionality of the outputs',
            'type': int,
            'default': [1000, ],
            'is_list': True
        }
    }

    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            download_dataset: bool = False,
            samplescount: int = 1000,
            numclasses: int = 3,
            integer_classes: bool = False,
            inputdims: List = [224, 224, 3],
            dtype: Type = np.float32):
        """
        Creates randomized dataset.

        Parameters
        ----------
        root : Path
            Deprecated argument, not used in this dataset
        batch_size : int
            The size of batches of data delivered during inference
        download_dataset : bool
            True if dataset should be downloaded first
        samplescount : int
            The number of samples in the dataset
        numclasses : int
            The number of classes in the dataset
        integer_classes : bool
            Indicates if classes should be represented by single integer
            instead of one-hot encoding
        inputdims : List
            The dimensionality of the inputs
        dtype : Type
            Type of the data
        """
        self.samplescount = samplescount
        self.inputdims = inputdims
        self.numclasses = numclasses
        self.integer_classes = integer_classes
        self.dtype = dtype
        self.classnames = self.get_class_names()

        super().__init__(root, batch_size, download_dataset)

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.dataset_root,
            args.inference_batch_size,
            False,
            args.num_samples,
            args.num_classes,
            args.input_dims
        )

    def get_class_names(self):
        return [str(i) for i in range(self.numclasses)]

    def get_input_mean_std(self):
        return (0.0, 1.0)

    def prepare(self):
        self.dataX = [
            f'{self.root}/images/{i}.jpg' for i in range(self.samplescount)
        ]
        self.dataY = [j % self.numclasses for j in range(self.samplescount)]
        shuffle(self.dataY)

        if not os.path.isdir(self.root):
            os.mkdir(self.root)
        if not os.path.isdir(self.root / 'images'):
            os.mkdir(self.root / 'images')
        samples = self.prepare_input_samples(self.dataX)
        for img_path, img_data in zip(self.dataX, samples):
            cv2.imwrite(img_path, img_data)

    def download_dataset_fun(self):
        pass

    def prepare_input_samples(self, samples):
        result = []
        for sample in samples:
            result.append(np.random.randn(*self.inputdims).astype(self.dtype))
        return result

    def prepare_output_samples(self, samples):
        if self.integer_classes:
            return samples
        return list(np.eye(self.numclasses)[samples])

    def evaluate(self, predictions, truth):
        return Measurements()

    def calibration_dataset_generator(
            self,
            percentage: float = 0.25,
            seed: int = 12345):
        for _ in range(int(self.samplescount * percentage)):
            yield [np.random.randint(0, 255, size=self.inputdims)]


class RandomizedDetectionDataset(Dataset):
    """
    Creates a sample randomized detection dataset.

    It is a mock dataset with randomized inputs and outputs.

    It can be used only for speed and utilization metrics, no quality metrics.
    """

    arguments_structure = {
        'samplescount': {
            'argparse_name': '--num-samples',
            'description': 'Number of samples to process',
            'type': int,
            'default': 1000
        },
        'numclasses': {
            'argparse_name': '--num-classes',
            'description': 'Number of classes in inputs',
            'type': int,
            'default': 3
        },
        'inputdims': {
            'argparse_name': '--input-dims',
            'description': 'Dimensionality of the inputs',
            'type': int,
            'default': [224, 224, 3],
            'is_list': True
        },
        'outputdims': {
            'argparse_name': '--output-dims',
            'description': 'Dimensionality of the outputs',
            'type': int,
            'default': [1000, ],
            'is_list': True
        }
    }

    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            download_dataset: bool = False,
            samplescount: int = 1000,
            numclasses: int = 3,
            inputdims: List = [224, 224, 3],
            dtype: Type = np.float32):
        """
        Creates randomized dataset.

        Parameters
        ----------
        root : Path
            Deprecated argument, not used in this dataset
        batch_size : int
            The size of batches of data delivered during inference
        download_dataset : bool
            True if dataset should be downloaded first
        samplescount : int
            The number of samples in the dataset
        numclasses : int
            The number of classes in the dataset
        inputdims : List
            The dimensionality of the inputs
        dtype : Type
            Type of the data
        """
        self.samplescount = samplescount
        self.inputdims = inputdims
        self.numclasses = numclasses
        self.dtype = dtype
        self.classnames = self.get_class_names()
        super().__init__(root, batch_size, download_dataset)

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.dataset_root,
            args.inference_batch_size,
            False,
            args.num_samples,
            args.num_classes,
            args.input_dims
        )

    def get_class_names(self):
        return [str(i) for i in range(self.numclasses)]

    def prepare(self):
        self.dataX = list(range(self.samplescount))
        self.dataY = []

        classes = [i % self.numclasses for i in range(self.samplescount)]
        shuffle(classes)
        for i in range(self.samplescount):
            x_rand = np.random.random((2,))
            y_rand = np.random.random((2,))
            self.dataY.append(DectObject(
                clsname=str(classes[i]),
                xmin=x_rand.min(),
                ymin=y_rand.min(),
                xmax=x_rand.min(),
                ymax=y_rand.max(),
                score=1.0,
                iscrowd=(np.random.randint(0, 1) == 1)
            ))

    def download_dataset_fun(self):
        pass

    def prepare_input_samples(self, samples):
        result = []
        for sample in samples:
            np.random.seed(sample)
            result.append(np.random.randn(*self.inputdims).astype(self.dtype))
        return result

    def prepare_output_samples(self, samples):
        return samples

    def evaluate(self, predictions, truth):
        return Measurements()
