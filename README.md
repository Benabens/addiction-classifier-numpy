# Machine learning from scratch — beating class imbalance with a weighted loss

Five classical models implemented **entirely in NumPy** — k-nearest neighbours,
linear regression, softmax logistic regression, K-Means, and a multi-layer perceptron
with hand-written backpropagation. Activations, losses, metrics and the training loops
are all written by hand, with no machine-learning library.

The point of this repository is not that the models work. It is **what accuracy hides**:
on a dataset where one class accounts for 3% of the samples, a model can reach 85%
accuracy while never once predicting that class. The fix — a class-weighted
cross-entropy — trades **0.75 points of accuracy for a 0.21 jump in macro-F1**.

Course project for **CS-233 Introduction to Machine Learning** (EPFL, BA4), done in a
team of 3 over two milestones: the classical supervised models first (KNN, linear and
logistic regression), then the neural network and K-Means.

---

## The result in one table

Predicting addiction level (`Low` / `Medium` / `High`) from gaming-habit features.
Training split: **896 Low, 340 Medium, 44 High** — the rare class is ~3%.

| Model | Accuracy | Macro-F1 | Per-class recall `[Low, Med, High]` |
|---|---|---|---|
| Majority-class baseline | 69.1% | 0.272 | — |
| Softmax logistic regression | 85.75% | 0.597 | — |
| MLP, sigmoid + MSE | **85.25%** | 0.546 | `[0.95, 0.70, `**`0.00`**`]` |
| **MLP, softmax + weighted CE** | 84.50% | **0.757** | `[0.92, 0.66, `**`0.69`**`]` |
| K-Means classifier (K=100) | 78.25% | 0.585 | — |

*(test-set figures from the milestone reports; the baseline row is on the validation split)*

Read the two middle rows carefully. The baseline MLP looks like the better model —
it wins on accuracy. But its recall on the `High` class is **exactly zero**: it never
predicts that class, not once, on the entire test set. Accuracy does not notice,
because 85% of the answer is carried by the majority class. Macro-F1 does notice,
because it averages per-class F1 without weighting by class size, so an empty class
drags the whole score down.

Switching the output layer to softmax and the loss to a cross-entropy weighted by
inverse class frequency (`w = [0.48, 1.21, 10.4]`) recovers the rare class: recall on
`High` goes from 0.00 to 0.69, macro-F1 from 0.546 to 0.757 — and the cost is 0.75
points of accuracy. On an imbalanced problem, that is an obviously good trade, and
the table above is the argument for it.

## What is implemented by hand

**k-nearest neighbours** — lazy learner: `fit` memorises, `predict` computes Euclidean
distances to every stored point and takes a majority vote (classification) or a mean
(regression) over the k closest.

**Linear regression** — closed-form least squares `w* = X⁺y` through the Moore–Penrose
pseudo-inverse, which stays well defined when features are collinear.

**Softmax logistic regression** — one weight column per class, full-batch gradient
descent on the multi-class cross-entropy, `∇W = Xᵀ(softmax(XW) − Y)/N`, with logits
shifted by their row maximum so large scores never overflow.

**Multi-layer perceptron** — forward pass `a_l = z_{l-1} W_l + b_l`, `z_l = f_l(a_l)`;
backpropagation via the chain rule `δ_{l-1} = (δ_l W_lᵀ) ⊙ f'(a_{l-1})`; mini-batch SGD;
`N(0,1)/√fan_in` weight initialisation. Used for both classification and regression.

**Activations** — Sigmoid, ReLU, Identity, and a numerically stable Softmax
(max-subtraction before exponentiating).

**Losses** — MSE, cross-entropy, MAE, and the weighted cross-entropy. Softmax and
cross-entropy are deliberately differentiated *jointly*: the Jacobians telescope into
the clean `y_pred − y_true`, which is both faster and numerically better behaved than
composing them separately.

**K-Means classifier** — random init, assignment to nearest centre, recomputation of
centres, until convergence; each cluster then labelled by majority vote of its
training points, and unseen points take the label of their nearest centre. Empty
clusters keep their previous centre.

**Metrics** — accuracy, macro-F1 and confusion matrices, all recomputed from scratch.

## Correctness

Hand-written backpropagation fails silently: a sign error or a transposed matrix does
not crash anything, the network just learns badly. So every gradient is checked against
finite differences,

```
dL/dw  ≈  (L(w + h) − L(w − h)) / 2h
```

including the joint softmax + cross-entropy gradient, differentiated numerically with
respect to the *logits* to confirm the telescoping is right. The weighted loss is also
checked to collapse exactly onto the plain cross-entropy when all weights are 1.

The models are tested against problems whose answer is known in advance: linear
regression must recover the exact coefficients (bias included) of a noiseless linear
relation, KNN with k=1 must reclassify its own training points perfectly, logistic
regression must separate two linearly separable clouds, K-Means must find three
well-separated groups.

```bash
python tests/test_gradients.py     # 31 checks — gradients, softmax, metrics
python tests/test_models.py        # 8 checks — models on known-answer problems
```

## Methodology notes

Standardisation statistics come from the **training split only** and are reused
unchanged on validation and test — recomputing them over the whole dataset would leak
test information and inflate the scores. Constant columns are guarded against division
by zero.

Hyperparameters were selected on a 20% validation split, never on test: hidden size
∈ {16, 32, 64, 128} and learning rate ∈ {0.01, 0.05, 0.1, 0.2}, retaining hidden=64,
η=0.01 (80 epochs, batch 32). The test set was touched once, for the final table.

Validation and test accuracy land at 85.6% and 85.25% — nearly identical. To confirm
that is not luck, a 5-fold cross-validation of the chosen architecture gives
**84.2 ± 0.7%** accuracy and **0.541 ± 0.006** macro-F1. The tiny standard deviation
says the residual val/test gap is split variance (320 validation / 400 test points),
not a generalisation problem.

An honest negative result worth keeping: on the **regression** task the MLP
(MSE 1.037) does *not* beat a closed-form linear regression (0.994). That strongly
suggests the addiction score is essentially a linear function of these features, and
that the extra capacity of a neural network buys nothing here.

## Running it

```bash
pip install numpy matplotlib

python main.py --method mlp --weighted          # MLP, softmax + weighted cross-entropy
python main.py --method mlp                     # MLP, plain loss (the "before")
python main.py --method logistic_regression --lr 0.1 --max_iters 500
python main.py --method knn --K 10
python main.py --method kmeans --K 100
python main.py --method baseline                # majority-class floor

python main.py --task regression --method linear_regression
python main.py --task regression --method mlp

python main.py --method mlp --weighted --test   # final evaluation on the held-out test set

python experiments.py                            # hyperparameter sweeps + report figures
```

Only NumPy is required; matplotlib is used solely to draw the figures.

## Layout

```
main.py             CLI: load, train, evaluate
experiments.py      hyperparameter sweeps, weighted-CE comparison, figures
src/activations.py  Sigmoid, ReLU, Identity, Softmax (+ their gradients)
src/losses.py       MSE, CrossEntropy, MAE, WeightedCrossEntropy
src/utils.py        label encoding, standardisation, accuracy / macro-F1 / confusion matrix
src/methods/knn.py                  k-nearest neighbours
src/methods/linear_regression.py    closed-form least squares
src/methods/logistic_regression.py  softmax regression, gradient descent
src/methods/mlp.py                  the perceptron: forward, backprop, SGD
src/methods/kmeans.py               K-Means + majority-vote labelling
src/methods/baselines.py            majority-class, random and mean-predictor floors
tests/              gradient checks and known-answer model tests
reports/            the two written milestone reports
```

`reports/milestone2_report.pdf` holds the before/after confusion matrices (figures 2a
and 2b) that the table at the top summarises; `reports/milestone1_report.pdf` covers the
classical models.
