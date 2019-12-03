"""
Neural Network training and evaluation
"""
import pandas as pd
from sklearn import metrics

import torch
from torch.nn import functional as F

from src.env import device, dtype
from src.data.ml import Dataset


def evaluate_accuracy(l, model):
    """Intermediate training monitor metric"""
    num_correct = 0
    num_samples = 0

    model.eval()
    with torch.no_grad():
        for x, y in l:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print("{:d} / {:d} correct (%{:.2f})".format(num_correct, num_samples, 100 * acc))

    return acc


def evaluate_final(d: Dataset, model):
    """Final evaluation using the val/test looking at performance of model more in depth"""
    X, y = d.val

    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        x = X.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        scores = model(x)

        # For AUC
        probs = F.softmax(scores)
        # probs = probs.gather(1, y.view(-1, 1)).squeeze()
        probs = probs[:, 1]  # roc_auc works oddly

        # For F1
        _, preds = scores.max(1)

    y = y.cpu()
    preds = preds.cpu()
    probs = probs.cpu()

    results = metrics.precision_recall_fscore_support(y, preds)
    df = pd.DataFrame(
        data=results,
        columns=['nonfood', 'food'],
        index=['precision', 'recall', 'f1', 'support'])

    auc = metrics.roc_auc_score(y, probs)

    return df, auc


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

        accuracy = evaluate_accuracy(l_validate, model)

        losses.append(round_loss / num_batches)
        accuracies.append(accuracy)

    f1, auc = evaluate_final(dataset, model)

    print(f1)
    print("Final AUC: {:.4f}".format(auc))

    print("\nDone training!\n")
    return losses, accuracies
