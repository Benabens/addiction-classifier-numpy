import numpy as np


class KNN:
    """
    k-nearest neighbours, for classification or regression.

    KNN has no training phase: `fit` simply memorises the training set, and all
    the work happens at prediction time. For each query point we compute the
    Euclidean distance to every stored point, keep the k closest, and either take
    a majority vote (classification) or average their targets (regression).

    Prediction therefore costs O(N·D) per query. That is fine at this dataset's
    scale (~1,600 points) and is the honest trade-off of a lazy learner: zero
    training cost, full cost at inference.
    """

    def __init__(self, k=1, task_kind="classification"):
        """
        Arguments:
            k (int): number of neighbours consulted for each prediction
            task_kind (str): "classification" (majority vote) or "regression" (mean)
        """
        self.k = k
        self.task_kind = task_kind
        self.training_data = None
        self.training_labels = None

    def fit(self, training_data, training_labels):
        """
        Memorise the training set; return the predictions on it.

        Arguments:
            training_data (np.array): shape (N, D)
            training_labels (np.array): shape (N,)
        Returns:
            np.array: predictions on the training set, shape (N,)
        """
        self.training_data = training_data
        self.training_labels = training_labels
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Arguments:
            test_data (np.array): shape (N, D)
        Returns:
            np.array: predicted labels (or values), shape (N,)
        """
        predictions = []

        for x in test_data:
            distances = np.linalg.norm(self.training_data - x, axis=1)
            nearest = np.argsort(distances)[:self.k]
            nearest_labels = self.training_labels[nearest]

            if self.task_kind == "classification":
                values, counts = np.unique(nearest_labels, return_counts=True)
                predictions.append(values[np.argmax(counts)])
            else:
                predictions.append(np.mean(nearest_labels))

        return np.array(predictions)
