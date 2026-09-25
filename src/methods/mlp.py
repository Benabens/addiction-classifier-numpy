import numpy as np

class MLP:
    def __init__(self, dimensions, activations):
        """
        :param dimensions: list of dimensions of the neural net. (input, hidden layer, ... ,hidden layer, output)
        :param activations: list of activation functions. Must contain N-1 activation function, where N = len(dimensions).
        Example of one hidden layer with
        - 2 inputs
        - 10 hidden nodes
        - 5 outputs
        layers -->    [0,        1,          2]
        ----------------------------------------
        dimensions =  (2,     10,          5)
        activations = (      Sigmoid,      Sigmoid)
        """
        self.dimensions = dimensions
        self.activations = activations
        self.n_layers = len(dimensions) - 1

        # weights and biases stored per layer (1..n_layers)
        self.weights = {}
        self.biases = {}
        for i in range(1, len(dimensions)):
            fan_in = dimensions[i - 1]
            fan_out = dimensions[i]
            # W ~ N(0,1)/sqrt(fan_in), b = 0  (keeps activation variance stable)
            self.weights[i] = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)
            self.biases[i] = np.zeros((1, fan_out))

        self.learning_rate = None  # set in fit()

    def feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """
        z = {0: x}
        a = {}
        for i in range(1, self.n_layers + 1):
            a[i] = z[i - 1] @ self.weights[i] + self.biases[i]
            z[i] = self.activations[i - 1].forward(a[i])
        return z, a

    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        z, _ = self.feed_forward(x)
        return z[self.n_layers]

    def back_prop(self, z, a, y_true, loss):
        """
        The input dicts keys represent the layers of the net.
        a = { 0: x,
              1: f(w1(x) + b1)
              2: f(w2(a2) + b2)
              }
        :param a: (dict) w^T@x + b
        :param z: (dict) f(a)
        :param y_true: (array) One hot encoded truth vector.
        :param loss: Loss class with a static .gradient(y_true, y_pred) method.
        :return:
        """
        L = self.n_layers
        N = y_true.shape[0]
        dw = {}
        delta = {}

        # error at the output layer: dL/dy * f'(a_L)
        delta[L] = loss.gradient(y_true, z[L]) * self.activations[L - 1].gradient(a[L])

        for i in range(L, 0, -1):
            dw[i] = (z[i - 1].T @ delta[i]) / N    # average over the batch
            if i > 1:
                # propagate the error one layer back
                delta[i - 1] = (delta[i] @ self.weights[i].T) * self.activations[i - 2].gradient(a[i - 1])

        return dw, delta

    def update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """
        self.weights[index] = self.weights[index] - self.learning_rate * dw
        # bias gradient = dL/da, so we average delta over the batch
        self.biases[index] = self.biases[index] - self.learning_rate * np.mean(delta, axis=0, keepdims=True)

    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs.
        :param batch_size: (int)
        :param learning_rate: (flt)
        """
        self.learning_rate = learning_rate
        N = x.shape[0]

        for epoch in range(epochs):
            # shuffle the data each epoch (the stochastic part of SGD)
            idx = np.random.permutation(N)
            x_sh = x[idx]
            y_sh = y_true[idx]

            for start in range(0, N, batch_size):
                xb = x_sh[start:start + batch_size]
                yb = y_sh[start:start + batch_size]
                z, a = self.feed_forward(xb)
                dw, delta = self.back_prop(z, a, yb, loss)
                for i in range(1, self.n_layers + 1):
                    self.update_w_b(i, dw[i], delta[i])

        # return predictions on the training set, like the other methods
        pred = self.predict(x)
        if y_true.ndim == 2 and y_true.shape[1] > 1:
            return np.argmax(pred, axis=1)   # classification
        return pred.reshape(-1)              # regression
