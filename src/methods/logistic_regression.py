import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression:
    """
    Multi-class (softmax) logistic regression, trained by full-batch gradient descent.

    One weight column per class; the bias is absorbed by prepending a constant
    feature. The loss is the multi-class cross-entropy, whose gradient with respect
    to the weights has the closed form

        ∇W = Xᵀ (softmax(XW) − Y) / N

    where Y is the one-hot label matrix. Logits are shifted by their row maximum
    before exponentiating, so large scores never overflow.
    """

    def __init__(self, lr, max_iters=500):
        """
        Arguments:
            lr (float): gradient-descent learning rate
            max_iters (int): number of full-batch gradient steps
        """
        self.lr = lr
        self.max_iters = max_iters
        self.W = None

    @staticmethod
    def _softmax(scores):
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def fit(self, training_data, training_labels):
        """
        Arguments:
            training_data (np.array): shape (N, D)
            training_labels (np.array): class indices, shape (N,)
        Returns:
            np.array: predictions on the training set, shape (N,)
        """
        N = training_data.shape[0]
        x = np.concatenate([np.ones((N, 1)), training_data], axis=1)

        C = get_n_classes(training_labels)
        y_onehot = label_to_onehot(training_labels, C)
        self.W = np.zeros((x.shape[1], C))

        probs = None
        for _ in range(self.max_iters):
            probs = self._softmax(x @ self.W)
            grad = (x.T @ (probs - y_onehot)) / N
            self.W = self.W - self.lr * grad

        return onehot_to_label(probs)

    def predict(self, test_data):
        """
        Arguments:
            test_data (np.array): shape (N, D)
        Returns:
            np.array: predicted class indices, shape (N,)
        """
        N = test_data.shape[0]
        x = np.concatenate([np.ones((N, 1)), test_data], axis=1)
        return onehot_to_label(self._softmax(x @ self.W))
