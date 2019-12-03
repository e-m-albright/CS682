"""
TODO
    Saliency isn't looking that great, 1 channel makes it hard to see much I suppose
    Base code from HW3, might want to tinker more to get more sensible saliency maps
"""

from matplotlib import pyplot as plt

import torch

from src.data.ml import Dataset
from src.models import conv2d
from src.utils import reload


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    model.eval()
    X.requires_grad_()

    scores = model(X)
    correct_scores = scores.gather(1, y.view(-1, 1)).squeeze()
    correct_scores.backward(torch.ones(y.shape))

    abs_grad = X.grad.abs()
    max_values, _argmax_values = abs_grad.max(dim=1)  # dim 1 is channels
    saliency = max_values

    return saliency


def show_saliency_maps():
    d = Dataset(dimensions="2d", limit=1)
    X, y = d.train

    model = conv2d.model()
    reload.load(model, "conv_2d_s6_lr0.001_e500")

    saliency = compute_saliency_maps(X, y, model)

    Xnp = X.detach().numpy()
    snp = saliency.numpy()
    # N = X.shape[0]
    N = 6

    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(Xnp[i].squeeze(), cmap="gray")
        plt.axis('off')
        # plt.title(class_names[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(snp[i].squeeze(), cmap="gray")
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()

