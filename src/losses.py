import numpy as np


class MSE:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return float(np.mean((y_pred - y_true) ** 2))

    @staticmethod
    def gradient(y_true, y_pred):
        # 2*(y_pred - y_true); the 1/(N*C) factor is handled by the averaging in back_prop
        return 2.0 * (y_pred - y_true)


# cross-entropy for classification, meant to be used with a Softmax output
class CrossEntropy:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        eps = 1e-12  # avoid log(0)
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))

    @staticmethod
    def gradient(y_true, y_pred):
        # combined softmax+CE gradient w.r.t. the logits: y_pred - y_true
        return y_pred - y_true


# additional L1 / Manhattan loss, alternative to MSE (more robust to outliers)
class MAE:
    @staticmethod
    def loss(y_true, y_pred):
        return float(np.mean(np.abs(y_pred - y_true)))

    @staticmethod
    def gradient(y_true, y_pred):
        return np.sign(y_pred - y_true)


# same as CrossEntropy but with a per-class weight (larger for rare classes),
# to fight class imbalance. Also meant to be used with a Softmax output.
# Instantiate with the class weights: loss = WeightedCrossEntropy(weights)
class WeightedCrossEntropy:
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)  # shape (C,)

    def loss(self, y_true, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.mean(np.sum(self.weights * y_true * np.log(y_pred), axis=1)))

    def gradient(self, y_true, y_pred):
        # combined softmax+CE gradient, weighted per class
        return self.weights * (y_pred - y_true)
