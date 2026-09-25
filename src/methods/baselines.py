"""
Naive reference models — the floor every real model has to beat.

Without a floor, a score means nothing. On a dataset where ~70% of the samples
are `Low`, a classifier that always answers `Low` already reaches ~70% accuracy.
That is exactly the trap this project sets out to expose.

These references share the real models' interface (fit / predict), so they go
through exactly the same evaluation loop.
"""

import numpy as np

from ..utils import get_n_classes


class MajorityClassBaseline:
    """
    Always predicts the most frequent training class.

    High accuracy under imbalance, very low macro-F1: the shortest possible
    demonstration of the gap between the two metrics.
    """

    def __init__(self, **kwargs):
        self.majority_class = None

    def fit(self, training_data, training_labels):
        counts = np.bincount(np.asarray(training_labels).astype(int))
        self.majority_class = int(np.argmax(counts))
        return self.predict(training_data)

    def predict(self, test_data):
        return np.full(test_data.shape[0], self.majority_class, dtype=int)


class RandomBaseline:
    """
    Predicts a random class, drawn with the class proportions seen in training.
    A floor for macro-F1.
    """

    def __init__(self, seed=0, **kwargs):
        self.proportions = None
        self.rng = np.random.default_rng(seed)

    def fit(self, training_data, training_labels):
        labels = np.asarray(training_labels).astype(int)
        counts = np.bincount(labels, minlength=get_n_classes(labels))
        self.proportions = counts / counts.sum()
        return self.predict(training_data)

    def predict(self, test_data):
        return self.rng.choice(
            len(self.proportions), size=test_data.shape[0], p=self.proportions
        )


class MeanRegressor:
    """
    For regression: always predicts the mean training target. Its MSE equals the
    variance of the targets, which sets the scale any model has to beat.
    """

    def __init__(self, **kwargs):
        self.mean = None

    def fit(self, training_data, training_labels):
        self.mean = float(np.mean(training_labels))
        return self.predict(training_data)

    def predict(self, test_data):
        return np.full(test_data.shape[0], self.mean)
