"""
Assess the data readiness by using a simple SKLearn model to demonstrate learning
"""
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC

from src.data.ml import Dataset


def test_svm(limit: int = 300):
    dataset = Dataset(standardize=False, dimensions="1d", limit=5, splits=(0.8, 0.2))
    print(dataset)

    X_train, y_train = dataset.get_train(numpy=True)
    X_val, y_val = dataset.get_val(numpy=True)

    svc = SVC(
        kernel='linear',
        C=1.,
    )

    print("Training SVC model on {} scans".format(limit))
    svc.fit(X_train[:limit], y_train[:limit])

    print("Predicting using SVC model")
    metrics = precision_recall_fscore_support(y_val, svc.predict(X_val))
    results = pd.DataFrame(
        data=metrics,
        columns=['food', 'nonfood'],
        index=['precision', 'recall', 'fscore', 'support']
    )

    return results
