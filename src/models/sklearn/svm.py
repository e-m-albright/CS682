"""
Assess the data readiness by using a simple SKLearn model to demonstrate learning
"""
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC

from src.data.ml import Dataset


def test_svm(limit: int = 400):
    # 1d is the standard way, seems a little better than the single slice 2d method
    # but this is not a bad approach
    dataset = Dataset(standardize=False, dimensions="2d", limit=5, splits=(0.7, 0.2))
    print(dataset)

    X_train, y_train = dataset.get_train(numpy=True)
    X_val, y_val = dataset.get_val(numpy=True)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)

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
