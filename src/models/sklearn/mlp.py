"""
Assess the data readiness by using a simple SKLearn model to demonstrate learning
"""
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier

from src.data.ml import Dataset


def test_mlp(limit: int = 1000):
    dataset = Dataset(limit=3, splits=(0.8, 0.2))
    print(dataset)

    X_train, y_train = dataset.get_train(flat=True, numpy=True)
    X_val, y_val = dataset.get_val(flat=True, numpy=True)

    # Normalize
    # TODO normalize per image? or accross everything?
    # X = (X - X.mean()) / X.std()

    import numpy as np
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image

    from sklearn.preprocessing import normalize
    X_train = normalize(X_train)
    X_val = normalize(X_val)

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
