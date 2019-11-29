"""
Neural Network training and evaluation
"""
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from src.env import device, dtype
from src.data.ml import Dataset


def accuracy(l, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in l:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print("{:d} / {:d} correct (%{:.2f})".format(num_correct, num_samples, 100 * acc))


def train(model, optimizer, dataset: Dataset, epochs=1):
    model = model.to(device=device)

    l_train, l_validate, _ = dataset.get_loaders()

    for e in range(epochs):
        print("\nEpoch: {}, with {} batches".format(e, len(l_train)))

        for t, (x, y) in enumerate(l_train):
            model.train()

            x = Variable(x, requires_grad=True)

            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)  #dtype=torch.long)

            scores = model(x).squeeze()
            # scores = model.forward(x)
            # print("ALL ONES: ", torch.all(scores.eq(1.0)))

            # loss = F.cross_entropy(scores, y)
            loss = F.binary_cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Loss at {:.4f}".format(t, loss.item()))
        # accuracy(l_train, model)
        accuracy(l_validate, model)
    print()
