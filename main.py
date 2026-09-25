import argparse
import numpy as np
import os
import time

from src.methods.baselines import MajorityClassBaseline
from src.methods.knn import KNN
from src.methods.linear_regression import LinearRegression
from src.methods.logistic_regression import LogisticRegression
from src.methods.mlp import MLP
from src.losses import MSE, WeightedCrossEntropy
from src.activations import Sigmoid, ReLU, Identity, Softmax
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn, \
    label_to_onehot, onehot_to_label, get_n_classes

np.random.seed(100)


def main(args):
    """Load the data, split and standardise it, train the chosen model, report metrics."""

    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # --- Load ------------------------------------------------------------------
    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    # --- Split -----------------------------------------------------------------
    # Unless --test is given, the test set is left untouched and a validation
    # split is carved out of the training data for model selection.
    eval_name = "Test set"
    if not args.test:
        n = train_features.shape[0]
        idx = np.random.permutation(n)
        n_val = int(args.val_ratio * n)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        test_features = train_features[val_idx]
        test_labels_reg = train_labels_reg[val_idx]
        test_labels_classif = train_labels_classif[val_idx]
        train_features = train_features[train_idx]
        train_labels_reg = train_labels_reg[train_idx]
        train_labels_classif = train_labels_classif[train_idx]
        eval_name = "Validation set"

    # --- Standardise with TRAINING statistics only (no leakage) ------------------
    means = train_features.mean(axis=0, keepdims=True)
    stds = train_features.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    train_features = normalize_fn(train_features, means, stds)
    test_features = normalize_fn(test_features, means, stds)

    # --- Model -----------------------------------------------------------------
    # Majority-class baseline: the floor every real model has to beat.
    if args.method == "baseline":
        method_obj = MajorityClassBaseline()

    elif args.method == "knn":
        method_obj = KNN(k=args.K, task_kind=args.task)

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)

    elif args.method == "linear_regression":
        method_obj = LinearRegression()

    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)

    elif args.method == "mlp":
        input_dim = train_features.shape[1]
        if args.task == "classification":
            n_classes = get_n_classes(train_labels_classif)
            # --weighted: softmax output + weighted cross-entropy (handles class imbalance)
            out_act = Softmax if args.weighted else Sigmoid
            method_obj = MLP(dimensions=(input_dim, args.hidden_nodes, n_classes),
                             activations=(ReLU, out_act))
        else:  # regression -> linear output
            method_obj = MLP(dimensions=(input_dim, args.hidden_nodes, 1),
                             activations=(ReLU, Identity))
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # --- Train and evaluate ----------------------------------------------------
    if args.task == "classification":

        if args.method == "mlp":
            n_classes = get_n_classes(train_labels_classif)
            y_train = label_to_onehot(train_labels_classif, n_classes)
            if args.weighted:
                # inverse-frequency class weights for the weighted cross-entropy
                counts = np.bincount(train_labels_classif.astype(int), minlength=n_classes)
                weights = len(train_labels_classif) / (n_classes * counts)
                loss_fn = WeightedCrossEntropy(weights)
            else:
                loss_fn = MSE
            s1 = time.time()
            method_obj.fit(train_features, y_train, loss=loss_fn,
                           epochs=args.max_iters, batch_size=args.batch_size, learning_rate=args.lr)
            s2 = time.time()
            preds_train = onehot_to_label(method_obj.predict(train_features))
            preds = onehot_to_label(method_obj.predict(test_features))
            s3 = time.time()
        else:
            s1 = time.time()
            preds_train = method_obj.fit(train_features, train_labels_classif)
            s2 = time.time()
            preds = method_obj.predict(test_features)
            s3 = time.time()

        acc = accuracy_fn(preds_train, train_labels_classif)
        macrof1 = macrof1_fn(preds_train, train_labels_classif)
        print(f"\nTrain set: {acc:.3f}% accuracy - {macrof1:.6f} macro-F1")
        acc = accuracy_fn(preds, test_labels_classif)
        macrof1 = macrof1_fn(preds, test_labels_classif)
        print(f"{eval_name}: {acc:.3f}% accuracy - {macrof1:.6f} macro-F1")
        print(f"Training time: {s2 - s1:.6f} s")
        print(f"Prediction time: {s3 - s2:.6f} s")

    elif args.task == "regression":
        assert args.method not in ("kmeans", "logistic_regression"), \
            f"{args.method} is a classification method"

        if args.method == "mlp":
            y_train = train_labels_reg.reshape(-1, 1).astype(float)
            s1 = time.time()
            method_obj.fit(train_features, y_train, loss=MSE,
                           epochs=args.max_iters, batch_size=args.batch_size, learning_rate=args.lr)
            s2 = time.time()
            preds_train = method_obj.predict(train_features).reshape(-1)
            preds = method_obj.predict(test_features).reshape(-1)
            s3 = time.time()
        else:
            s1 = time.time()
            preds_train = method_obj.fit(train_features, train_labels_reg)
            s2 = time.time()
            preds = method_obj.predict(test_features)
            s3 = time.time()

        train_mse = mse_fn(preds_train, train_labels_reg)
        print(f"\nTrain set: MSE = {train_mse:.6f}")
        test_mse = mse_fn(preds, test_labels_reg)
        print(f"{eval_name}: MSE = {test_mse:.6f}")
        print(f"Training time: {s2 - s1:.6f} s")
        print(f"Prediction time: {s3 - s2:.6f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the from-scratch models on the gaming / mental-health data."
    )
    parser.add_argument("--task", default="classification", type=str,
                        help="classification | regression")
    parser.add_argument("--method", default="mlp", type=str,
                        help="baseline | knn | logistic_regression | linear_regression | kmeans | mlp")
    parser.add_argument("--data_path", default="data/features.npz", type=str,
                        help="path to the .npz feature file")
    parser.add_argument("--K", type=int, default=10,
                        help="number of neighbours (knn) or of clusters (kmeans)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate (logistic regression, MLP)")
    parser.add_argument("--max_iters", type=int, default=100,
                        help="gradient steps (logistic regression), epochs (MLP) or iterations (kmeans)")
    parser.add_argument("--test", action="store_true",
                        help="train on the full training set and evaluate on the held-out test set; "
                             "otherwise evaluate on a validation split")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="fraction of the training data used for validation")
    parser.add_argument("--batch_size", type=int, default=32, help="mini-batch size for the MLP")
    parser.add_argument("--hidden_nodes", type=int, default=64, help="hidden layer size for the MLP")
    parser.add_argument("--weighted", action="store_true",
                        help="MLP classification with softmax + weighted cross-entropy (for class imbalance)")

    args = parser.parse_args()
    main(args)
