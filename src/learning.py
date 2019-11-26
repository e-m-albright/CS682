import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision import models

from src import data, env


# TODO definitely going to want to bake in the pytorch operations
ml_dataset = data.get_ml_dataset()
# ml_dataset.normalize()
# ml_dataset.flatten()
print(ml_dataset)


transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

"""
ALRIGHT 
a looot of TODO here

working through them, first don't flatten, second can we default to using
pytorch as the data handler from the get go? transforms seem useful
"""


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


"""DATASET LOADER FOR PYTORCH"""

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
