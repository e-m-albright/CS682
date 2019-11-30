"""
Assess the data readiness by using a simple SKLearn model to demonstrate learning
"""
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier

from src.data.ml import Dataset


def test_mlp(limit: int = 1000):
    dataset = Dataset(dimensions="2d", limit=3, splits=(0.8, 0.2))
    print(dataset)

    X_train, y_train = dataset.get_train(numpy=True)
    X_val, y_val = dataset.get_val(numpy=True)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)

    classifier = MLPClassifier(
        solver='adam',
        hidden_layer_sizes=(200, 200, 200),
        random_state=1,
    )

    print("Training MLPClassifier model on {} scans".format(limit))
    classifier.fit(X_train[:limit], y_train[:limit])

    print("Predicting using MLPClassifier model")
    metrics = precision_recall_fscore_support(y_val, classifier.predict(X_val))
    results = pd.DataFrame(
        data=metrics,
        columns=['food', 'nonfood'],
        index=['precision', 'recall', 'fscore', 'support']
    )

    return results
