# Deep Learning HW1 — Four New PyTorch Tasks

## Assignment Instructions

> *Apply Deep Learning Training and Evaluation to Four New Tasks (New Datasets / New Features)*
>
> Using the provided task-driven JSON file as a reference, add four new tasks that follow the same `pytorch_task_v1` protocol (e.g., introduce a new dataset, a different optimization method, or additional training features). All code must be implemented in PyTorch and should be self-verifiable by returning an appropriate exit status via `sys.exit(exit_code)`.

---

## How This Repository Satisfies the Assignment

### Protocol Compliance — `pytorch_task_v1`

Every task is implemented as a **single self-contained `task.py`** file located at `tasks/<task_id>/task.py`. Each file implements every required function from the protocol:

| Required Function | Purpose |
|---|---|
| `get_task_metadata()` | Returns task ID, series, level, algorithm name |
| `set_seed(seed)` | Reproducibility — seeds `torch.manual_seed` |
| `get_device()` | Returns `torch.device` (cpu/cuda agnostic) |
| `make_dataloaders(cfg)` | Generates/loads dataset, returns train/val splits |
| `build_model(...)` | Initialises model parameters |
| `train(model, data, cfg)` | Full training loop, returns updated model + history |
| `evaluate(model, data)` | Computes MSE, R², accuracy, and task-specific metrics on validation split |
| `predict(model, X)` | Runs inference |
| `save_artifacts(model, metrics)` | Saves metrics JSON to `outputs/` |

Each `if __name__ == '__main__':` block:
1. Trains the model
2. Evaluates on **both** train and validation splits
3. Prints all metrics clearly
4. **Asserts quality thresholds** — exits `0` on success, non-zero on failure

---

## Task 1 — Univariate Linear Regression (Raw Tensors)

**Path:** `tasks/linreg_lvl1_raw_tensors/task.py`

### What It Does
Implements univariate linear regression using **only raw PyTorch tensors** — no `torch.nn`, no `torch.optim`, no autograd at any point.

### Dataset
Synthetic: `y = 2x + 3 + noise` with 500 samples, 80/20 train/val split.

### Algorithm
```
h(x) = theta_0 + theta_1 * x

J(theta) = (1/2m) * sum( (h(x_i) - y_i)^2 )   ← MSE cost

grad_0 = (1/m) * sum(h - y)
grad_1 = (1/m) * sum((h - y) * x)

theta_0 = theta_0 - lr * grad_0
theta_1 = theta_1 - lr * grad_1
```

### Results Achieved
| Split | MSE | R² |
|---|---|---|
| Train | 0.2455 | 0.9417 |
| Val | 0.1838 | 0.9521 |

Learned parameters: `theta_0 = 2.978` (true: 3.0), `theta_1 = 2.030` (true: 2.0)

### Assertions Verified
- Validation R² > 0.90 ✅
- `|theta_0 - 3.0|` < 1.0 ✅
- `|theta_1 - 2.0|` < 1.0 ✅
- Exit code: **0** ✅

---

## Task 2 — Binary Logistic Regression (Raw Tensors)

**Path:** `tasks/logreg_lvl1_binary_raw/task.py`

### What It Does
Implements binary logistic regression with a **manually coded sigmoid function and hand-computed gradients** — no autograd, no `torch.nn`.

### Dataset
Two isotropic Gaussian clusters (300 samples each), centred at `(-2, -2)` and `(+2, +2)`. Features are standardised before training.

### Algorithm
```
sigma(z) = 1 / (1 + exp(-z))

J(w, b) = -(1/m) * sum( y*log(h) + (1-y)*log(1-h) )   ← Binary Cross-Entropy

dJ/dw = (1/m) * X^T @ (h - y)
dJ/db = (1/m) * sum(h - y)

w = w - lr * dJ/dw
b = b - lr * dJ/db
```

### Results Achieved
| Split | Loss | Accuracy | F1 |
|---|---|---|---|
| Train | 0.0192 | 99.58% | 0.9958 |
| Val | 0.0188 | 99.17% | 0.9917 |

Val Precision: 0.9836 — Val Recall: 1.0000

### Assertions Verified
- Validation Accuracy > 0.90 ✅
- Exit code: **0** ✅

---

## Task 3 — MLP with Manual Backpropagation (XOR Problem)

**Path:** `tasks/mlp_lvl1_numpy_to_torch/task.py`

### What It Does
Implements a **2-layer MLP (2→4→1) with fully manual backpropagation** using the chain rule — no `torch.autograd`, no `torch.nn`, no optimizer. Solves the classic XOR classification problem.

### Dataset
XOR: four input corners `{0,1}²` with their XOR labels, augmented with 500 noisy copies (σ=0.1) for robustness. 80/20 train/val split.

### Architecture & Chain Rule Derivation
```
Forward pass:
    z1 = X @ W1 + b1        ← linear transform
    a1 = sigmoid(z1)        ← hidden activations
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)        ← output probability

Loss: Binary Cross-Entropy
    L = -(y * log(a2) + (1-y) * log(1-a2))

Backprop (chain rule):
    delta2 = (a2 - y) * a2 * (1 - a2)        ← output delta (sigmoid + BCE)
    dW2    = a1.T @ delta2 / m
    db2    = mean(delta2)

    delta1 = (delta2 @ W2.T) * a1 * (1 - a1) ← hidden delta
    dW1    = X.T @ delta1 / m
    db1    = mean(delta1, axis=0)
```

### Results Achieved
| Split | Loss | Accuracy |
|---|---|---|
| Train | 0.0023 | 100% |
| Val | 0.0022 | 100% |

XOR corners accuracy: **100%** (all four corners correctly classified)

### Assertions Verified
- Validation Accuracy > 0.95 ✅
- All 4 XOR corners correctly classified ✅
- Exit code: **0** ✅

---

## Task 4 — k-Nearest Neighbours Classifier (Brute Force, Pure Tensors)

**Path:** `tasks/knn_lvl1_bruteforce/task.py`

### What It Does
Implements a kNN classifier using **fully vectorised L2 distance computation** — no Python loops over samples, no sklearn for classification. Validated against sklearn's `KNeighborsClassifier`.

### Dataset
5-class blobs generated via `sklearn.make_blobs` (600 samples, 4 features), converted to PyTorch tensors and standardised. 80/20 split.

### Algorithm
```
L2 distance (vectorised):
    ||a - b||² = ||a||² + ||b||² - 2(a · b)

Prediction (majority vote, vectorised):
    For each query point, find k nearest neighbours in training set.
    Assign the most frequent label among those k neighbours.
```

The identity `||a-b||² = ||a||² + ||b||² - 2·aᵀb` allows the entire distance matrix `[n_query × n_train]` to be computed in a single matrix operation with no sample-level loops.

### Results Achieved (k=5)
| Split | Accuracy | MSE |
|---|---|---|
| Train | 100% | 0.0000 |
| Val | 100% | 0.0000 |

Difference vs sklearn `KNeighborsClassifier`: **0.0000** (threshold ≤ 0.02)

### Assertions Verified
- Validation Accuracy > 0.85 ✅
- Accuracy within 2% of sklearn kNN ✅
- Exit code: **0** ✅

---

## Repository Structure

```
Deeplearning_HW1/
├── readme.md
├── ml_tasks.json                          ← reference task protocol file
└── tasks/
    ├── linreg_lvl1_raw_tensors/
    │   └── task.py                        ← Task 1
    ├── logreg_lvl1_binary_raw/
    │   └── task.py                        ← Task 2
    ├── mlp_lvl1_numpy_to_torch/
    │   └── task.py                        ← Task 3
    └── knn_lvl1_bruteforce/
        └── task.py                        ← Task 4
```

---

## How to Run

Each task is a standalone script. Run any of them with:

```bash
python tasks/linreg_lvl1_raw_tensors/task.py
python tasks/logreg_lvl1_binary_raw/task.py
python tasks/mlp_lvl1_numpy_to_torch/task.py
python tasks/knn_lvl1_bruteforce/task.py
```

**Dependencies:**
```bash
pip install torch scikit-learn numpy
```

All scripts exit with code `0` on success and `1` on failure (threshold not met), satisfying the `sys.exit(exit_code)` requirement.

---

## Summary

| Task | Algorithm | New Dataset | Optimization | Val Metric | Passes |
|---|---|---|---|---|---|
| `linreg_lvl1_raw_tensors` | Linear Regression | Synthetic `y=2x+3` | Manual GD (raw tensors) | R²=0.952 | ✅ |
| `logreg_lvl1_binary_raw` | Logistic Regression | 2-Gaussian clusters | Manual GD (no autograd) | Acc=99.2% | ✅ |
| `mlp_lvl1_numpy_to_torch` | 2-layer MLP | XOR (augmented) | Manual Backprop | Acc=100% | ✅ |
| `knn_lvl1_bruteforce` | k-Nearest Neighbours | 5-class blobs | No training (lazy) | Acc=100% | ✅ |
