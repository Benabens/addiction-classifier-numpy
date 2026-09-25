"""Hyperparameter search, weighted cross-entropy comparison and figure generation.

Produces every figure and number quoted in the README and in
reports/milestone2_report.pdf: the MLP learning-rate / hidden-size sweeps, the
K-Means accuracy-vs-K curve, and the before/after confusion matrices showing the
rare class being recovered by the weighted loss. Run from the repository root:

    python experiments.py
"""

import os
import time
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.methods.mlp import MLP
from src.methods.kmeans import KMeans
from src.losses import MSE, WeightedCrossEntropy
from src.activations import ReLU, Sigmoid, Identity, Softmax
from src.utils import (
    accuracy_fn, macrof1_fn, mse_fn,
    label_to_onehot, onehot_to_label, get_n_classes,
)

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
CLASS_NAMES = ["Low", "Medium", "High"]


def load():
    f = np.load("data/features.npz", allow_pickle=True)
    return (f["xtrain"], f["xtest"], f["ytrainreg"], f["ytestreg"],
            f["ytrainclassif"], f["ytestclassif"])


def standardize(train, test):
    m = train.mean(0, keepdims=True); s = train.std(0, keepdims=True); s[s == 0] = 1.0
    return (train - m) / s, (test - m) / s


def split(x, yr, yc, ratio, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x)); nv = int(ratio * len(x))
    v, t = perm[:nv], perm[nv:]
    return x[t], x[v], yr[t], yr[v], yc[t], yc[v]


def confusion(yt, yp, C):
    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(yt, yp):
        cm[int(t), int(p)] += 1
    return cm


def per_class_recall(yt, yp, C):
    return [np.sum((yp == c) & (yt == c)) / max(1, np.sum(yt == c)) for c in range(C)]


def class_weights(y, C):
    counts = np.bincount(y.astype(int), minlength=C)
    return len(y) / (C * counts)


# --- MLP classif baseline (sigmoid + MSE) ---
def mlp_classif(xtr, ytr, xv, h, lr, epochs, batch, C, seed=0):
    np.random.seed(seed)
    mlp = MLP((xtr.shape[1], h, C), (ReLU, Sigmoid))
    mlp.fit(xtr, label_to_onehot(ytr, C), loss=MSE, epochs=epochs, batch_size=batch, learning_rate=lr)
    return onehot_to_label(mlp.predict(xv))


# --- MLP classif weighted (softmax + weighted CE) ---
def mlp_classif_weighted(xtr, ytr, xv, h, lr, epochs, batch, C, seed=0):
    np.random.seed(seed)
    mlp = MLP((xtr.shape[1], h, C), (ReLU, Softmax))
    wce = WeightedCrossEntropy(class_weights(ytr, C))
    mlp.fit(xtr, label_to_onehot(ytr, C), loss=wce, epochs=epochs, batch_size=batch, learning_rate=lr)
    return onehot_to_label(mlp.predict(xv))


def mlp_reg(xtr, ytr, xv, h, lr, epochs, batch, seed=0):
    np.random.seed(seed)
    mlp = MLP((xtr.shape[1], h, 1), (ReLU, Identity))
    mlp.fit(xtr, ytr.reshape(-1, 1).astype(float), loss=MSE, epochs=epochs, batch_size=batch, learning_rate=lr)
    return mlp.predict(xv).reshape(-1)


def sweep_classif(xtr, ytr, xv, yv, C):
    hiddens, lrs = [16, 32, 64, 128], [0.01, 0.05, 0.1, 0.2]
    res, best = {}, {"acc": -1}
    for h in hiddens:
        for lr in lrs:
            p = mlp_classif(xtr, ytr, xv, h, lr, 80, 32, C)
            acc = accuracy_fn(p, yv); res[(h, lr)] = acc
            if acc > best["acc"]:
                best = {"acc": acc, "f1": macrof1_fn(p, yv), "hidden": h, "lr": lr, "epochs": 80, "batch": 32}
    plt.figure(figsize=(5, 3.3))
    for h in hiddens:
        plt.plot(lrs, [res[(h, lr)] for lr in lrs], marker="o", label=f"hidden={h}")
    plt.xscale("log"); plt.xlabel("learning rate"); plt.ylabel("val accuracy [%]")
    plt.title("MLP classification sweep"); plt.legend(fontsize=7); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/mlp_classif_sweep.png", dpi=140); plt.close()
    return best


def sweep_reg(xtr, ytr, xv, yv):
    hiddens, lrs = [16, 32, 64, 128], [0.0005, 0.001, 0.005, 0.01]
    res, best = {}, {"mse": 1e9}
    for h in hiddens:
        for lr in lrs:
            p = mlp_reg(xtr, ytr, xv, h, lr, 100, 32)
            mse = mse_fn(p, yv); res[(h, lr)] = mse
            if mse < best["mse"]:
                best = {"mse": mse, "hidden": h, "lr": lr, "epochs": 100, "batch": 32}
    plt.figure(figsize=(5, 3.3))
    for h in hiddens:
        plt.plot(lrs, [res[(h, lr)] for lr in lrs], marker="o", label=f"hidden={h}")
    plt.xscale("log"); plt.xlabel("learning rate"); plt.ylabel("val MSE")
    plt.title("MLP regression sweep"); plt.legend(fontsize=7); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/mlp_reg_sweep.png", dpi=140); plt.close()
    return best


def sweep_kmeans(xtr, ytr, xv, yv):
    Ks = [3, 5, 10, 20, 30, 50, 75, 100]
    accs, f1s, best = [], [], {"acc": -1}
    for K in Ks:
        a, f = [], []
        for seed in (0, 1, 2):
            np.random.seed(seed)
            km = KMeans(K=K, max_iters=200); km.fit(xtr, ytr); p = km.predict(xv)
            a.append(accuracy_fn(p, yv)); f.append(macrof1_fn(p, yv))
        accs.append(np.mean(a)); f1s.append(np.mean(f))
        if np.mean(a) > best["acc"]:
            best = {"acc": np.mean(a), "f1": np.mean(f), "K": K}
    fig, ax1 = plt.subplots(figsize=(5, 3.3))
    ax1.plot(Ks, accs, "o-", color="tab:blue"); ax1.set_xlabel("K"); ax1.set_ylabel("val acc [%]", color="tab:blue")
    ax2 = ax1.twinx(); ax2.plot(Ks, f1s, "s-", color="tab:red"); ax2.set_ylabel("val macro-F1", color="tab:red")
    ax1.set_title("K-Means vs K"); ax1.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/kmeans_K.png", dpi=140); plt.close()
    return best


def loss_curve(xtr, ytr, C, best):
    np.random.seed(0)
    mlp = MLP((xtr.shape[1], best["hidden"], C), (ReLU, Sigmoid))
    y = label_to_onehot(ytr, C); mlp.learning_rate = best["lr"]; losses = []
    for _ in range(best["epochs"]):
        idx = np.random.permutation(len(xtr))
        xs, ys = xtr[idx], y[idx]
        for s in range(0, len(xtr), best["batch"]):
            z, a = mlp.feed_forward(xs[s:s+best["batch"]])
            dw, d = mlp.back_prop(z, a, ys[s:s+best["batch"]], MSE)
            for i in range(1, mlp.n_layers+1):
                mlp.update_w_b(i, dw[i], d[i])
        losses.append(MSE.loss(y, mlp.predict(xtr)))
    plt.figure(figsize=(5, 3.3)); plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel("epoch"); plt.ylabel("train MSE"); plt.title("MLP training loss"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/mlp_loss_curve.png", dpi=140); plt.close()


def plot_confusion(cm, title, fname):
    C = cm.shape[0]
    plt.figure(figsize=(3.3, 3.0)); plt.imshow(cm, cmap="Blues")
    plt.xticks(range(C), CLASS_NAMES); plt.yticks(range(C), CLASS_NAMES)
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title(title)
    for i in range(C):
        for j in range(C):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/{fname}", dpi=140); plt.close()


def kfold(x, y, C, best, k=5):
    rng = np.random.default_rng(0); perm = rng.permutation(len(x)); fs = len(x)//k
    accs, f1s = [], []
    for i in range(k):
        v = perm[i*fs:(i+1)*fs]; t = np.concatenate([perm[:i*fs], perm[(i+1)*fs:]])
        xtr, xv = standardize(x[t], x[v])
        p = mlp_classif(xtr, y[t], xv, best["hidden"], best["lr"], best["epochs"], best["batch"], C)
        accs.append(accuracy_fn(p, y[v])); f1s.append(macrof1_fn(p, y[v]))
    return {"acc": (float(np.mean(accs)), float(np.std(accs))),
            "f1": (float(np.mean(f1s)), float(np.std(f1s)))}


if __name__ == "__main__":
    np.random.seed(100)
    xtr_all, xte_all, ytr_r, yte_r, ytr_c, yte_c = load()
    xtr, xv, ytr_rt, yv_r, ytr_ct, yv_c = split(xtr_all, ytr_r, ytr_c, 0.2, 100)
    xtr_z, xv_z = standardize(xtr, xv)
    C = get_n_classes(ytr_ct)
    print("train class counts:", np.bincount(ytr_ct.astype(int)))

    print("\n[MLP classif sweep]"); best_c = sweep_classif(xtr_z, ytr_ct, xv_z, yv_c, C); print(best_c)
    print("\n[MLP reg sweep]"); best_r = sweep_reg(xtr_z, ytr_rt, xv_z, yv_r); print(best_r)
    print("\n[KMeans sweep]"); best_k = sweep_kmeans(xtr_z, ytr_ct, xv_z, yv_c); print(best_k)
    loss_curve(xtr_z, ytr_ct, C, best_c)

    # --- baseline vs weighted CE on validation ---
    print("\n[baseline vs weighted CE]")
    pb = mlp_classif(xtr_z, ytr_ct, xv_z, best_c["hidden"], best_c["lr"], best_c["epochs"], best_c["batch"], C)
    pw = mlp_classif_weighted(xtr_z, ytr_ct, xv_z, best_c["hidden"], 0.2, 80, 32, C)
    comp = {
        "baseline": {"acc": accuracy_fn(pb, yv_c), "f1": macrof1_fn(pb, yv_c),
                     "recall": [round(r, 3) for r in per_class_recall(yv_c, pb, C)]},
        "weighted": {"acc": accuracy_fn(pw, yv_c), "f1": macrof1_fn(pw, yv_c),
                     "recall": [round(r, 3) for r in per_class_recall(yv_c, pw, C)]},
    }
    print(json.dumps(comp, indent=2, default=float))
    plot_confusion(confusion(yv_c, pb, C), "Baseline (sigmoid+MSE)", "confusion_baseline.png")
    plot_confusion(confusion(yv_c, pw, C), "Softmax + weighted CE", "confusion_weighted.png")

    print("\n[5-fold CV baseline]"); kf = kfold(xtr_all, ytr_c, C, best_c); print(kf)

    # --- final test-set evaluation, train on full training set ---
    print("\n[final test evaluation]")
    xtr_f, xte_f = standardize(xtr_all, xte_all)
    t0 = time.time(); pb_te = mlp_classif(xtr_f, ytr_c, xte_f, best_c["hidden"], best_c["lr"], best_c["epochs"], best_c["batch"], C); tb = time.time()-t0
    pw_te = mlp_classif_weighted(xtr_f, ytr_c, xte_f, best_c["hidden"], 0.2, 80, 32, C)
    pr_te = mlp_reg(xtr_f, ytr_r, xte_f, best_r["hidden"], best_r["lr"], best_r["epochs"], best_r["batch"])
    np.random.seed(0); km = KMeans(K=best_k["K"], max_iters=200); km.fit(xtr_f, ytr_c); pk_te = km.predict(xte_f)

    test = {
        "mlp_baseline": {"acc": accuracy_fn(pb_te, yte_c), "f1": macrof1_fn(pb_te, yte_c), "fit_time": tb},
        "mlp_weighted": {"acc": accuracy_fn(pw_te, yte_c), "f1": macrof1_fn(pw_te, yte_c),
                         "recall": [round(r, 3) for r in per_class_recall(yte_c, pw_te, C)]},
        "kmeans": {"acc": accuracy_fn(pk_te, yte_c), "f1": macrof1_fn(pk_te, yte_c)},
        "mlp_reg": {"mse": mse_fn(pr_te, yte_r)},
    }
    plot_confusion(confusion(yte_c, pb_te, C), "Test: baseline (sigmoid+MSE)", "confusion_matrix_test.png")
    plot_confusion(confusion(yte_c, pw_te, C), "Test: softmax + weighted CE", "confusion_weighted_test.png")
    print(json.dumps(test, indent=2, default=float))

    json.dump({"best_classif": best_c, "best_reg": best_r, "best_kmeans": best_k,
               "val_comparison": comp, "kfold": kf, "test": test},
              open(f"{OUT}/results.json", "w"), indent=2, default=float)
    print(f"\nsaved to ./{OUT}/")
