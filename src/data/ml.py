"""
Data from our chosen experimental data in a readily consumed format for machine learning applications
"""
import numpy as np
import pandas as pd

from bids import BIDSLayout
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T

from src.data import experiment


TVT_SPLITS = (0.7, 0.9)


class MLDataset:
    def __init__(self,
             X_train, y_train,
             X_val, y_val,
             X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        self.mean_image = np.mean(self.X_train, axis=0)

        self._normalized = False
        self._flattened = False

    def __repr__(self):
        return "{} | Features: {}  | Train: {} | Validation: {} | Test: {}".format(
            MLDataset.__name__,
            self.X_train.shape[1:],
            self.X_train.shape[0],
            self.X_val.shape[0],
            self.X_test.shape[0],
        )

    def normalize(self):
        if self._normalized:
            return

        # Subtract the mean training image
        self.X_train -= self.mean_image
        self.X_val -= self.mean_image
        self.X_test -= self.mean_image

        self._normalized = True

    def flatten(self):
        if self._flattened:
            return

        features = np.prod(self.X_train.shape[1:])
        self.X_train = self.X_train.reshape(self.X_train.shape[0], features)
        self.X_val = self.X_val.reshape(self.X_val.shape[0], features)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], features)
        # self.mean_image = self.mean_image.reshape(self.mean_image.shape[0], features)

        self._flattened = True


def get_dset(layout: BIDSLayout = None, limit: int = 3, splits=TVT_SPLITS) -> MLDataset:
    """
    Get brain scans as labeled data, partitioned into Train, Validate, and Test

    :param layout: optional, supply if you already have a layout on hand
    :param limit: TODO I'm figuring out some memory constraints, the data if mishandled can exceed memory
    :param splits: The proportions of train and validation (test is inferred) to split up
    :return: MLDataset class wrapping the data splits

    TODO how am I going to cope with the memory issue?
    64 * 64 * 30 = Per TimeStep (122,880)

    dtype is <f4, for little endian, 4 bit float (Did I see it was float32 at one point?)
    TS * 4 = 491,520 bits per ts or 61.44 kB

    we have ~364 * 30 TS which is about 670.9248 mB
    """
    np.random.seed(2019)

    if layout is None:
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

    # Shuffle and partition our options
    index_df = pd.DataFrame(data=np.arange(len(data)), columns=["data_index"])
    train_ix, validate_ix, test_ix = np.split(
        index_df.sample(frac=1),
        [int(splits[0] * len(index_df)),
         int(splits[1] * len(index_df))])

    # TODO Is it concerning there's a stray newaxis at the end of the indexing?
    X_train = scans[:, :, :, train_ix].squeeze().T
    y_train = labels[train_ix].squeeze()
    X_val = scans[:, :, :, validate_ix].squeeze().T
    y_val = labels[validate_ix].squeeze()
    X_test = scans[:, :, :, test_ix].squeeze().T
    y_test = labels[test_ix].squeeze()

    return MLDataset(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )


def get_loaders(layout: BIDSLayout = None, limit: int = 3, splits=TVT_SPLITS) -> (DataLoader, DataLoader, DataLoader):
    # TODO definitely going to want to bake in the pytorch operations
    ml_dataset = get_dset(layout=layout, limit=limit, splits=splits)
    # ml_dataset.normalize()
    # ml_dataset.flatten()
    print(ml_dataset)

    # inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
    # tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)

    # Adding np.newaxis for "channels" which this dataset doesn't seem to have
    X_train = torch.from_numpy(ml_dataset.X_train[:,np.newaxis,:,:,:])
    y_train = torch.from_numpy(ml_dataset.y_train)
    X_val = torch.from_numpy(ml_dataset.X_val[:,np.newaxis,:,:,:])
    y_val = torch.from_numpy(ml_dataset.y_val)
    X_test = torch.from_numpy(ml_dataset.X_test[:,np.newaxis,:,:,:])
    y_test = torch.from_numpy(ml_dataset.y_test)

    # TODO this data is way too goddamn big, I need to downsample to survive
    #  mini-batch x channels x [optional depth] x [optional height] x width.
    #  orientation doesn't matter
    X_train = F.interpolate(
        X_train,
        scale_factor=3./4.,
    )
    X_val = F.interpolate(
        X_val,
        scale_factor=3./4,
    )
    X_test = F.interpolate(
        X_test,
        scale_factor=3./4,
    )
    print(X_train.shape, X_train.squeeze().shape)
    train = TensorDataset(X_train, y_train)
    val = TensorDataset(X_val, y_val)
    test = TensorDataset(X_test, y_test)

    features = np.prod(X_train.shape[2:])
    print("Before downsampling: ", np.prod(ml_dataset.X_train.shape[1:]))
    print("After: ", features)

    # The torchvision.transforms package provides tools for preprocessing data
    # and for performing data augmentation; here we set up a transform to
    # preprocess the data by subtracting the mean RGB value and dividing by the
    # standard deviation of each RGB value; we've hardcoded the mean and std.
    data_transforms = T.Compose([
        T.ToTensor(),
    #     transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    loader_train = DataLoader(
        train,
        batch_size=64,
    #     transformer=data_transforms, TODO something like that
    #     sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)),
    )

    loader_val = DataLoader(
        val,
        batch_size=64,
    #     sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)),
    )

    loader_test = DataLoader(
        test,
        batch_size=64,
    )

    return loader_train, loader_val, loader_test
