"""
Label encoding, data preparation and metrics.

Pure NumPy, including the metrics. Macro-F1 in particular is computed by hand
because it carries the whole argument of this project: on a dataset where the
`High` class is 3% of the samples, accuracy can reach 85% while ignoring that
class entirely, whereas macro-F1 — which averages per-class F1 without weighting
by class size — collapses.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def get_n_classes(labels):
    """Number of classes, inferred from the largest label (indices 0..C-1)."""
    return int(np.max(labels)) + 1


def label_to_onehot(labels, C=None):
    """
    Class indices (N,) -> one-hot encoding (N, C).

    Example: [0, 2, 1] with C=3 gives [[1,0,0], [0,0,1], [0,1,0]].
    """
    labels = np.asarray(labels).astype(int)
    if C is None:
        C = get_n_classes(labels)
    onehot = np.zeros((labels.shape[0], C))
    onehot[np.arange(labels.shape[0]), labels] = 1.0
    return onehot


def onehot_to_label(onehot):
    """One-hot or score matrix (N, C) -> index of the most likely class (N,)."""
    return np.argmax(onehot, axis=1)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def append_bias_term(data):
    """
    Prepend a constant column of ones: (N, D) -> (N, D+1).

    Lets the bias live inside the weight matrix: instead of `xW + b` we write
    `[1, x] W'`, which keeps the gradient computation uniform.
    """
    data = np.asarray(data)
    return np.concatenate([np.ones((data.shape[0], 1)), data], axis=1)


def normalize_fn(data, means, stds):
    """
    Standardise with `(x - mu) / sigma`, using statistics that are PASSED IN.

    Means and standard deviations are never recomputed here: they must come
    from the training split alone and be reused unchanged on validation and
    test. Recomputing them over the whole dataset would leak test information
    into training and inflate every score.

    Constant columns (sigma = 0) are guarded against division by zero and are
    left at 0 after centring.
    """
    stds = np.where(np.asarray(stds) == 0, 1.0, stds)
    return (data - means) / stds


def train_stats(data):
    """Per-column means and standard deviations, keeping shape (1, D)."""
    return data.mean(axis=0, keepdims=True), data.std(axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy_fn(pred_labels, gt_labels):
    """Percentage of correct predictions."""
    return float(np.mean(np.asarray(pred_labels) == np.asarray(gt_labels)) * 100.0)


def macrof1_fn(pred_labels, gt_labels):
    """
    Macro-F1: the UNWEIGHTED mean of per-class F1 scores.

    Every class counts the same regardless of its frequency, which is what makes
    this metric harsh under imbalance — and therefore the right one here. A class
    that is never predicted contributes 0 and drags the mean down, instead of
    going unnoticed as it does with accuracy.
    """
    pred_labels = np.asarray(pred_labels)
    gt_labels = np.asarray(gt_labels)

    total = 0.0
    classes = np.unique(gt_labels)
    for c in classes:
        pred_c = (pred_labels == c)
        gt_c = (gt_labels == c)

        tp = np.sum(pred_c & gt_c)
        fp = np.sum(pred_c & ~gt_c)
        fn = np.sum(~pred_c & gt_c)

        if tp == 0:
            continue  # F1 is 0 for this class
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        total += 2.0 * precision * recall / (precision + recall)

    return float(total / len(classes))


def mse_fn(pred, gt):
    """Mean squared error, for the regression task."""
    return float(np.mean((np.asarray(pred) - np.asarray(gt)) ** 2))


def confusion_matrix(pred_labels, gt_labels, n_classes=None):
    """
    Confusion matrix (C, C): row = ground truth, column = prediction.

    This is the central figure of the project: it shows at a glance what
    accuracy hides — an entirely empty column means the model never predicted
    that class.
    """
    gt_labels = np.asarray(gt_labels).astype(int)
    pred_labels = np.asarray(pred_labels).astype(int)
    if n_classes is None:
        n_classes = max(get_n_classes(gt_labels), get_n_classes(pred_labels))

    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for truth, pred in zip(gt_labels, pred_labels):
        matrix[truth, pred] += 1
    return matrix
