"""
Definitions of layers for use in neural network model layouts
"""
from torch import nn


def flatten(x):
    return x.view(x.shape[0], -1)


class Flatten(nn.Module):
    """Wrap `flatten` function in a module in order to stack it in nn.Sequential"""
    def forward(self, x):
        return flatten(x)
