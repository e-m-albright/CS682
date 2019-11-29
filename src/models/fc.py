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


def model(idim, hdim: int = 300):
    return nn.Sequential(

        Flatten(),

        nn.Linear(idim, hdim),
        nn.ReLU(),
        # nn.Dropout(0.2),

        nn.Linear(hdim, hdim),
        nn.ReLU(),

        nn.Linear(hdim, 2),

        # nn.Sigmoid(),
        # nn.LogSigmoid(),
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


criterion = nn.NLLLoss()


class Net(nn.Module):

    def __init__(self, idim: int, hdim: int = 300):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(idim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.fc3 = nn.Linear(hdim, 2)

    def forward(self, x):
        x = flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x)

    def predict(self, x):

        pred = F.softmax(self.forward(x))
        ans = []

        # Pick the class with maximum weight
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)

        return torch.tensor(ans)
