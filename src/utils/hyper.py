"""
Neural Network training and evaluation
"""
from src.data.ml import Dataset
from src.utils.training import train as vanilla_train


LEARNING_RATES = [1e-1]


class Hyperparameters:

    def __init__(
            self,
            epochs: int = 2,
            learning_rates: list = LEARNING_RATES,
    ):
        self.learning_rates = learning_rates
        self.epochs = epochs


def train(model_fn, optimizer_fn, criterion_fn, dataset: Dataset, h: Hyperparameters):

    for lr in h.learning_rates:
        print("\nLR: {}".format(lr))

        model = model_fn(dataset.dimensions)
        optimizer = optimizer_fn(model, learning_rate=lr)
        criterion = criterion_fn()

        vanilla_train(
            model,
            optimizer,
            criterion,
            dataset,
            epochs=h.epochs,
        )
