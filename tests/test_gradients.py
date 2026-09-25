"""
Finite-difference gradient checks, plus metric tests.

This is the one test that really matters when backpropagation is written by
hand. A sign error or a transposed matrix in a gradient does not crash anything:
the network simply learns badly, and it can take hours to find out why. Comparing
each analytic gradient with a numerical one catches that class of bug at once.

For every parameter we estimate

    dL/dw  ~=  (L(w + h) - L(w - h)) / (2h)

and require the gap with the analytic gradient to stay under a tight tolerance.

Run:  python tests/test_gradients.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.activations import ReLU, Sigmoid, Identity, Softmax          # noqa: E402
from src.losses import MSE, CrossEntropy, MAE, WeightedCrossEntropy   # noqa: E402
from src.utils import (                                               # noqa: E402
    label_to_onehot, onehot_to_label, accuracy_fn, macrof1_fn,
    normalize_fn, append_bias_term, confusion_matrix, get_n_classes,
)

failures = 0
checks = 0


def check(condition, message):
    global failures, checks
    checks += 1
    if not condition:
        failures += 1
        print(f"FAIL: {message}")


# ---------------------------------------------------------------------------
# 1. Activations: analytic derivative vs. finite differences
# ---------------------------------------------------------------------------

def test_activation_derivatives():
    h = 1e-6
    z = np.array([[-2.0, -0.5, 0.3, 1.7, 4.0]])

    for activation in (Sigmoid, Identity):
        name = activation.__name__
        analytic = activation.gradient(z)
        numeric = (activation.forward(z + h) - activation.forward(z - h)) / (2 * h)
        gap = float(np.max(np.abs(analytic - numeric)))
        check(gap < 1e-5, f"{name} derivative: max gap {gap:.2e}")

    # ReLU is checked away from its kink at z = 0, where it is not differentiable.
    z_smooth = np.array([[-2.0, -0.5, 0.3, 1.7]])
    analytic = ReLU.gradient(z_smooth)
    numeric = (ReLU.forward(z_smooth + h) - ReLU.forward(z_smooth - h)) / (2 * h)
    gap = float(np.max(np.abs(analytic - numeric)))
    check(gap < 1e-5, f"ReLU derivative: max gap {gap:.2e}")


def test_softmax():
    z = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    p = Softmax.forward(z)
    check(np.allclose(p.sum(axis=1), 1.0), "softmax rows sum to 1")
    check(np.all(p > 0), "softmax probabilities are strictly positive")

    # Numerical stability: large logits must not produce NaN.
    large = np.array([[1000.0, 1001.0, 1002.0]])
    check(np.all(np.isfinite(Softmax.forward(large))), "softmax stays finite on large logits")

    # Shift invariance: softmax(z + c) == softmax(z)
    check(np.allclose(Softmax.forward(z), Softmax.forward(z + 7.0)), "softmax is shift-invariant")


# ---------------------------------------------------------------------------
# 2. Losses: analytic gradient vs. finite differences
# ---------------------------------------------------------------------------

def numerical_gradient(loss_fn, y_true, y_pred, h=1e-6):
    """Gradient of the loss with respect to y_pred, one coordinate at a time."""
    grad = np.zeros_like(y_pred)
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            plus = y_pred.copy()
            minus = y_pred.copy()
            plus[i, j] += h
            minus[i, j] -= h
            grad[i, j] = (loss_fn(y_true, plus) - loss_fn(y_true, minus)) / (2 * h)
    return grad


def test_mse_gradient():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(4, 3))
    y_pred = rng.normal(size=(4, 3))

    # MSE.gradient returns 2*(y_pred - y_true); the 1/(N*C) averaging factor is
    # applied later, in backprop. So we compare against the numerical gradient
    # rescaled to the same convention.
    analytic = MSE.gradient(y_true, y_pred)
    numeric = numerical_gradient(MSE.loss, y_true, y_pred) * y_pred.size
    gap = float(np.max(np.abs(analytic - numeric)))
    check(gap < 1e-4, f"MSE gradient: max gap {gap:.2e}")


def test_mae_gradient():
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=(4, 3))
    y_pred = y_true + rng.normal(size=(4, 3)) * 0.5   # never exactly equal
    analytic = MAE.gradient(y_true, y_pred)
    numeric = numerical_gradient(MAE.loss, y_true, y_pred) * y_pred.size
    gap = float(np.max(np.abs(analytic - numeric)))
    check(gap < 1e-3, f"MAE gradient: max gap {gap:.2e}")


def test_softmax_cross_entropy_gradient():
    """
    The joint softmax + cross-entropy gradient must equal y_pred - y_true.

    That is the classic simplification: the softmax Jacobian is never formed on
    its own, the two terms telescope. We verify it numerically by differentiating
    loss(softmax(logits)) with respect to the LOGITS.
    """
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(5, 3))
    y_true = label_to_onehot(np.array([0, 1, 2, 1, 0]), C=3)

    def composed_loss(yt, z):
        return CrossEntropy.loss(yt, Softmax.forward(z))

    numeric = numerical_gradient(composed_loss, y_true, logits) * logits.shape[0]
    analytic = CrossEntropy.gradient(y_true, Softmax.forward(logits))
    gap = float(np.max(np.abs(analytic - numeric)))
    check(gap < 1e-4, f"softmax + cross-entropy gradient: max gap {gap:.2e}")


def test_weighted_cross_entropy():
    """
    The weighted loss is the core of the project: with all weights at 1 it must
    collapse exactly onto the plain cross-entropy.
    """
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(6, 3))
    y_pred = Softmax.forward(logits)
    y_true = label_to_onehot(np.array([0, 0, 1, 2, 0, 1]), C=3)

    neutral = WeightedCrossEntropy(np.ones(3))
    check(abs(neutral.loss(y_true, y_pred) - CrossEntropy.loss(y_true, y_pred)) < 1e-10,
          "unit weights: weighted loss equals cross-entropy")
    check(np.allclose(neutral.gradient(y_true, y_pred), CrossEntropy.gradient(y_true, y_pred)),
          "unit weights: weighted gradient equals cross-entropy gradient")

    # A larger weight on the rare class must amplify its gradient.
    weighted = WeightedCrossEntropy(np.array([1.0, 1.0, 10.0]))
    g_neutral = neutral.gradient(y_true, y_pred)
    g_weighted = weighted.gradient(y_true, y_pred)
    check(np.all(np.abs(g_weighted[:, 2]) >= np.abs(g_neutral[:, 2]) - 1e-12),
          "a weight of 10 on the rare class amplifies its gradient")


# ---------------------------------------------------------------------------
# 3. Metrics: the imbalance trap, checked explicitly
# ---------------------------------------------------------------------------

def test_metrics():
    y = np.array([0, 1, 2, 1, 0])
    check(accuracy_fn(y, y) == 100.0, "accuracy of a perfect prediction")
    check(abs(macrof1_fn(y, y) - 1.0) < 1e-12, "macro-F1 of a perfect prediction")

    # 9 samples of class 0, a single one of class 2. Always answering 0 gives 90%
    # accuracy while macro-F1 collapses — exactly the effect this project fixes.
    truth = np.array([0] * 9 + [2])
    always_zero = np.zeros(10, dtype=int)
    acc = accuracy_fn(always_zero, truth)
    f1 = macrof1_fn(always_zero, truth)
    check(abs(acc - 90.0) < 1e-9, f"expected 90% accuracy, got {acc}")
    check(f1 < 0.5, f"macro-F1 should collapse, got {f1:.3f}")


def test_label_encoding():
    labels = np.array([0, 2, 1, 2])
    onehot = label_to_onehot(labels, C=3)
    check(onehot.shape == (4, 3), "one-hot shape")
    check(np.all(onehot.sum(axis=1) == 1), "exactly one 1 per row")
    check(np.array_equal(onehot_to_label(onehot), labels), "one-hot round trip")
    check(get_n_classes(labels) == 3, "number of classes inferred")


def test_data_preparation():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=2.0, size=(200, 4))

    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)
    normalized = normalize_fn(data, means, stds)
    check(np.max(np.abs(normalized.mean(axis=0))) < 1e-10, "zero mean after standardisation")
    check(np.max(np.abs(normalized.std(axis=0) - 1)) < 1e-10, "unit standard deviation")

    constant = np.ones((10, 2))
    out = normalize_fn(constant, constant.mean(axis=0, keepdims=True),
                       constant.std(axis=0, keepdims=True))
    check(np.all(np.isfinite(out)), "constant column produces no NaN or inf")

    with_bias = append_bias_term(data)
    check(with_bias.shape == (200, 5), "shape after adding the bias column")
    check(np.all(with_bias[:, 0] == 1.0), "the bias column is all ones")


def test_confusion_matrix():
    truth = np.array([0, 0, 1, 1, 2])
    pred = np.array([0, 1, 1, 1, 0])
    m = confusion_matrix(pred, truth, n_classes=3)
    check(m.sum() == 5, "the matrix counts every sample")
    check(m[0, 0] == 1 and m[0, 1] == 1, "row of class 0")
    check(m[1, 1] == 2, "row of class 1")
    check(m[2, 0] == 1, "class 2 predicted as 0")
    check(m[:, 2].sum() == 0, "class 2 is never predicted: empty column")


if __name__ == "__main__":
    test_activation_derivatives()
    test_softmax()
    test_mse_gradient()
    test_mae_gradient()
    test_softmax_cross_entropy_gradient()
    test_weighted_cross_entropy()
    test_metrics()
    test_label_encoding()
    test_data_preparation()
    test_confusion_matrix()

    if failures == 0:
        print(f"PASS: {checks} checks, 0 failures")
        sys.exit(0)
    print(f"FAIL: {failures} failure(s) out of {checks} checks")
    sys.exit(1)
