"""
Neural Network training and evaluation
"""
from sklearn import metrics

import torch
from torch.nn import functional as F

from src.env import device, dtype
from src.data.ml import Dataset


def evaluate(l, model):
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

    return acc
    # roc_auc = metrics.roc_auc_score
    # roc_auc(y, pred)


def train(
        model,
        optimizer,
        criterion,
        dataset: Dataset,
        epochs=1,
        print_frequency: int = 1):

    model = model.to(device=device)

    l_train, l_validate, _ = dataset.get_loaders()
    num_batches = len(l_train)

    print("Training with {} batches".format(num_batches))
    losses, accuracies = [], []

    for e in range(epochs):
        print("\nEpoch: {}".format(e + 1))

        round_loss = 0.0
        for i, (x, y) in enumerate(l_train):

            # Prepare for training
            model.train()
            # x = x.requires_grad_()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            # Forward pass
            scores = model(x)
            # print(scores, y)
            # print("ALL ONES: ", torch.all(scores.eq(1.0)))

            loss = criterion(scores, y)

            # Save info for plotting
            round_loss += loss.item()
            losses.append(loss.item())

            # Backward pass
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log performance
            if False and i % print_frequency == 0:
                print("[{:>3d}/{:>3d}] Loss: {:.4f}".format(
                    i + 1, num_batches,
                    loss.item(),
                ))

            # Try to help memory constraints out
            del x, y, scores
            torch.cuda.empty_cache()

        accuracy = evaluate(l_validate, model)

        losses.append(round_loss / num_batches)
        accuracies.append(accuracy)

    print("\nDone training!\n")
    return losses, accuracies
