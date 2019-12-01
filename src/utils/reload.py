import os

import torch

from src import MODELS_DIR


PYTORCH_STATE_EXT = ".pt"


def save(model, name):
    os.makedirs(MODELS_DIR, exist_ok=True)

    path = os.path.join(MODELS_DIR, name + PYTORCH_STATE_EXT)
    print("Saving model state to {}".format(path))

    torch.save(model.state_dict(), path)


def load(model, name):
    torch.load()

    path = os.path.join(MODELS_DIR, name + PYTORCH_STATE_EXT)
    print("Saving model state to {}".format(path))

    model.load_state_dict(torch.load(path))
    # model.eval
