"""
Data from our chosen experimental data in a readily consumed format for machine learning applications
"""
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data import experiment


TVT_SPLITS = (0.7, 0.2)
BATCH_SIZE = 64


class Dataset:
    def __init__(self, standardize=True, dimensions: str = "3d", scale=1.0, limit=3, splits=TVT_SPLITS):
        self._d = dimensions

        X, y = self._get_experimental_data(limit=limit)
        dataset = self._transform(X, y, standardize=standardize, scale=scale)
        splits = self._split(dataset, splits)
        train, val, test = self._clean(*splits, dimensions=dimensions)

        self._train = train
        self._val = val
        self._test = test

    def _get_split(self, split, numpy=False):
        data, labels = split.tensors

        if self._d == "1d":
            data = data.view(data.shape[0], -1)

        if numpy:
            data = data.numpy()
            labels = labels.numpy()

        return data, labels

    def get_train(self, *args, **kwargs):
        return self._get_split(self._train, *args, **kwargs)
    train = property(get_train)

    def get_val(self, *args, **kwargs):
        return self._get_split(self._val, *args, **kwargs)
    val = property(get_val)

    def get_test(self, *args, **kwargs):
        return self._get_split(self._test, *args, **kwargs)
    test = property(get_test)

    @property
    def dimensions(self):
        return np.prod(list(self.train[0].shape)[1:])

    def __repr__(self):
        s_train = list(self.train[0].shape)
        s_val = list(self.val[0].shape)
        s_test = list(self.test[0].shape)
        return "{} | Features: {}  | Train: {} | Validation: {} | Test: {}".format(
            Dataset.__name__,
            s_train[1:],
            s_train[0],
            s_val[0],
            s_test[0],
        )

    @staticmethod
    def _get_experimental_data(limit: int):
        """
        Get brain scans as labeled data, partitioned into Train, Validate, and Test

        :param layout: optional, supply if you already have a layout on hand
        :param limit: TODO I'm figuring out some memory constraints, the data if mishandled can exceed memory

        TODO how am I going to cope with the memory issue?
        64 * 64 * 30 = Per TimeStep (122,880)

        dtype is <f4, for little endian, 4 bit float (Did I see it was float32 at one point?)
        TS * 4 = 491,520 bits per ts or 61.44 kB

        we have ~364 * 30 TS which is about 670.9248 mB
        """
        np.random.seed(2019)

        layout = experiment.get_food_temptation_data()
        subject_data = experiment.get_all_subject_data(layout)

        data = []
        for s, _t1, b, e in subject_data[:limit]:
            print("Assigning data from subject {}".format(s))
            image_data = b.get_data()
            num_scans = image_data.shape[-1]

            # print("Total number of scans: ", num_scans)
            scan_assignments = experiment.get_scan_assignments(num_scans, e)
            # print(list("{}: {}".format(k, len(v)) for k, v in scan_assignments.items()))

            # Gather our classification examples
            # Ignore break / unassigned for now
            # Use a numeric 1 (food image) and 0 (nonfood image) for our task
            for timestep in scan_assignments['food']:
                data.append((image_data[:, :, :, timestep], 1))
            for timestep in scan_assignments['nonfood']:
                data.append((image_data[:, :, :, timestep], 0))

        # Stack our information in numpy arrays
        scans, labels = zip(*data)
        scans = np.stack(scans, axis=-1)
        labels = np.array(labels)

        return scans.T, labels.T

    @staticmethod
    def _transform(X, y, standardize=True, scale=3./4):

        # Cut down the data size, there's a lot of empty space in the scans
        # this may be clumsy but really helps keep the size of data down
        # retain all samples, vertical dimensions, cut the front to back & side to side
        print("Shaving bulk of empty space out of scans")
        print("Before shave: ", X.shape)
        X = X[:, :, 10:54, 12:52]
        print("After shave: ", X.shape)

        # Load in data, add a fake channel dimension to allow interpolation
        data = torch.from_numpy(X.reshape(X.shape[0], 1, *X.shape[1:]))
        labels = torch.from_numpy(y)

        if standardize:
            # Standardize data, 0 - 1
            data = (data - data.mean()) / (data.max() - data.min())

        # Data is quite large, downsampling is fairly important to survive
        #  mini-batch x channels x [optional depth] x [optional height] x width.
        #  orientation doesn't matter
        print("Before downsampling: shape={}, features={}".format(
            data.shape,
            np.prod(data.shape[1:])))
        data = F.interpolate(
            data,
            scale_factor=scale,
        )
        data = data.squeeze()
        print("After downsampling: shape={}, features={}".format(
            data.shape,
            np.prod(data.shape[1:])))

        return TensorDataset(data, labels)

    @staticmethod
    def _split(dataset, splits):
        train_size = int(splits[0] * len(dataset))
        val_size = int(splits[1] * len(dataset))
        test_size = len(dataset) - (train_size + val_size)
        return torch.utils.data.random_split(
            dataset,
            [train_size, val_size, test_size])

    @staticmethod
    def _clean(train, val, test, dimensions):
        X_train, y_train = train.dataset[train.indices]
        X_val, y_val = val.dataset[val.indices]
        X_test, y_test = test.dataset[test.indices]

        # Take out the mean training image from all splits
        mean_image = X_train.mean(axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

        # Add an image channel for non 1d formats
        if dimensions in ["2d", "3d"]:
            print("Adding a channel dimension for convolutional networks")
            X_train = X_train.view(X_train.shape[0], 1, *X_train.shape[1:])
            X_val = X_val.view(X_val.shape[0], 1, *X_val.shape[1:])
            X_test = X_test.view(X_test.shape[0], 1, *X_test.shape[1:])

        # Flatten axially for 2d, creating the average top-down view
        if dimensions == "2d":
            print("Flattening axially for 2d view")
            X_train = X_train.mean(dim=2)
            X_val = X_val.mean(dim=2)
            X_test = X_test.mean(dim=2)

        return (
            TensorDataset(X_train, y_train),
            TensorDataset(X_val, y_val),
            TensorDataset(X_test, y_test),
        )

    def get_loaders(self) -> (DataLoader, DataLoader, DataLoader):
        return [
            DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
            ) for dataset in [
                self._train,
                self._val,
                self._test,
            ]
        ]
