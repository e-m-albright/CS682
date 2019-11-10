"""
Okay let's get into the machine learning of it, using an access method from data
"""
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC

from src import data


def test_ml_data(limit: int = 300):

    X_train, y_train, \
    X_val, y_val, \
    X_test, y_test = data.get_machine_learning_data(flat=True)

    print("Data gathered")
    svc = SVC(kernel='linear', C=1.)
    print("Training SVC model on {} scans".format(limit))
    svc.fit(X_train[:limit], y_train[:limit])

    print("Predicting using SVC model")
    metrics = precision_recall_fscore_support(y_val, svc.predict(X_val))
    results = pd.DataFrame(
        data=metrics,
        columns=['food', 'nonfood'],
        index=['precision', 'recall', 'fscore', 'support']
    )

    # TODO bother with test? meh

    return results
