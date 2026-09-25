import numpy as np


class LinearRegression:
    """
    Ordinary least squares, solved in closed form.

    With the bias absorbed as a constant first feature, the optimal weights are

        w* = X⁺ y

    where X⁺ is the Moore–Penrose pseudo-inverse. Using the pseudo-inverse rather
    than (XᵀX)⁻¹Xᵀ keeps the solution well defined even when features are
    collinear and XᵀX is singular.

    This closed-form model is the reference the MLP is compared against on the
    regression task — and the MLP does not beat it (see the README).
    """

    def __init__(self):
        self.weights = None

    @staticmethod
    def _with_bias(data):
        return np.concatenate([np.ones((data.shape[0], 1)), data], axis=1)

    def fit(self, training_data, training_labels):
        """
        Arguments:
            training_data (np.array): shape (N, D)
            training_labels (np.array): regression targets, shape (N,) or (N, 1)
        Returns:
            np.array: fitted values on the training set
        """
        x = self._with_bias(training_data)
        self.weights = np.linalg.pinv(x) @ training_labels
        return x @ self.weights

    def predict(self, test_data):
        """
        Arguments:
            test_data (np.array): shape (N, D)
        Returns:
            np.array: predicted values
        """
        return self._with_bias(test_data) @ self.weights
