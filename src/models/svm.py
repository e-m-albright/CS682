"""
Okay let's get into the machine learning of it, using an access method from data
"""
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC

from bids import BIDSLayout

from src.data.ml import get_dset


def test_ml_data(layout: BIDSLayout = None, limit: int = 300):

    ml_dataset = get_dset(layout=layout)
    print("Data gathered")
    print(ml_dataset)
    ml_dataset.normalize()
    ml_dataset.flatten()
    print(ml_dataset)

    svc = SVC(kernel='linear', C=1.)
    print("Training SVC model on {} scans".format(limit))
    svc.fit(ml_dataset.X_train[:limit], ml_dataset.y_train[:limit])

    print("Predicting using SVC model")
    metrics = precision_recall_fscore_support(ml_dataset.y_val, svc.predict(ml_dataset.X_val))
    results = pd.DataFrame(
        data=metrics,
        columns=['food', 'nonfood'],
        index=['precision', 'recall', 'fscore', 'support']
    )

    return results
