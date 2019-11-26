"""
A Fully Connected Neural Network

Meant to be more of a proof of concept for faster
development and debugging of other more promising models
"""
import torch.nn as nn
import torch.optim as optim

from src.data.ml import get_loaders
from src.defs.layers import Flatten
from src.utils.training import train


hidden_layer_size = 4000
learning_rate = 1e-2


l_train, l_validate, l_test = get_loaders()


model = nn.Sequential(
    Flatten(),
    nn.Linear(3 * 32 * 32, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size, 10),
)


optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=0.9,
    nesterov=True)


train(model, optimizer, l_train, l_validate, epochs=1)
