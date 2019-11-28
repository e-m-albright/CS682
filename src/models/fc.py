"""
A Fully Connected Neural Network

Meant to be more of a proof of concept for faster
development and debugging of other more promising models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.defs.layers import flatten, Flatten


def model(input_dimension, hidden_layer_size: int = 6000):
    return nn.Sequential(

        Flatten(),

        nn.Linear(input_dimension, hidden_layer_size),

        # nn.ReLU(),
        nn.Tanh(),

        # nn.Dropout(0.2),
        #
        # nn.Linear(hidden_layer_size, hidden_layer_size),
        #
        # nn.Tanh(),

        nn.Linear(hidden_layer_size, 1),

        nn.Sigmoid(),

    )


def optimizer(model, learning_rate: float = 1e-2):
    return optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        # nesterov=True,
    )

    # return optim.Adam(
    #     model.parameters(),
    #     lr=learning_rate)


# # TODO net or model
# class Net(nn.Module):
#
#     def __init__(self, idim: int, hdim: int = 6000):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(idim, hdim)
#         self.fc2 = nn.Linear(hdim, 2)
#
#     def forward(self, x):
#         x = flatten(x)
#         x = self.fc1(x)
#         x = F.tanh(x)
#         x = self.fc2(x)
#
#         return F.log_softmax(x)
#
#     def predict(self, x):
#
#         pred = F.softmax(self.forward(x))
#         ans = []
#
#         # Pick the class with maximum weight
#         for t in pred:
#             if t[0] > t[1]:
#                 ans.append(0)
#             else:
#                 ans.append(1)
#
#         return torch.tensor(ans)
