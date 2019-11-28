"""
Assess the data readiness by using a simple SKLearn model to demonstrate learning
"""
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier

from bids import BIDSLayout

from src.data.ml import get_dset


def test_mlp(layout: BIDSLayout = None, limit: int = 300):

    ml_dataset = get_dset(layout=layout)
    print("Data gathered")
    print(ml_dataset)
    ml_dataset.normalize()
    ml_dataset.flatten()
    print(ml_dataset)

    classifier = MLPClassifier(
        solver='adam',
        hidden_layer_sizes=(100, 100),
        random_state=1,
    )
    print("Training MLPClassifier model on {} scans".format(limit))
    classifier.fit(ml_dataset.X_train[:limit], ml_dataset.y_train[:limit])

    print("Predicting using MLPClassifier model")
    metrics = precision_recall_fscore_support(ml_dataset.y_val, classifier.predict(ml_dataset.X_val))
    results = pd.DataFrame(
        data=metrics,
        columns=['food', 'nonfood'],
        index=['precision', 'recall', 'fscore', 'support']
    )

    return results
