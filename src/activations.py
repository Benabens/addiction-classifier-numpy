import numpy as np

class Sigmoid:
    @staticmethod
    def forward(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def gradient(z):
        s = Sigmoid.forward(z)
        return s * (1.0 - s)

class ReLU:
    @staticmethod
    def forward(z):
        return np.maximum(0.0, z)

    @staticmethod
    def gradient(z):
        return (z >= 0).astype(float)

# extra activation, used on the output layer for regression
class Identity:
    @staticmethod
    def forward(z):
        return z

    @staticmethod
    def gradient(z):
        return np.ones_like(z)

# softmax output, to pair with (weighted) cross-entropy for classification
class Softmax:
    @staticmethod
    def forward(z):
        z = z - np.max(z, axis=1, keepdims=True)  # stability
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    @staticmethod
    def gradient(z):
        # the softmax jacobian is handled jointly with the cross-entropy:
        # d(CE o softmax)/da = y_pred - y_true, so we let the loss return that
        # combined term and keep this gradient at 1.
        return np.ones_like(z)
