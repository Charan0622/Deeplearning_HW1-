"""
k-Nearest Neighbours Classifier — pure PyTorch tensor implementation.

Distance:    d(x, q) = sqrt( sum_j (x_j - q_j)^2 )   (Euclidean / L2)

Prediction:  label(q) = argmax over classes of count of class c
             among the k training points with smallest d(x_i, q).

All distance computations are fully vectorised (no Python loops over samples).

Validation: accuracy must be within 2 percentage points of sklearn
            KNeighborsClassifier on the same data split.
"""

import sys
import os
import json
import torch


# ---------------------------------------------------------------------------
# Protocol functions
# ---------------------------------------------------------------------------

def get_task_metadata():
    return {
        "id": "knn_lvl1_bruteforce",
        "series": "k-Nearest Neighbors",
        "level": 1,
        "algorithm": "kNN (Brute Force)",
        "description": (
            "kNN classifier with vectorised L2 distances (pure PyTorch tensors). "
            "No sklearn used for the core classifier."
        ),
        "interface_protocol": "pytorch_task_v1",
    }


def set_seed(seed: int = 42):
    torch.manual_seed(seed)


def get_device():
    return torch.device("cpu")


def make_dataloaders(cfg: dict = None):
    """
    Multi-class blobs dataset (generated with sklearn, then converted to tensors).
    Returns (X_train, y_train), (X_val, y_val).
    """
    from sklearn.datasets import make_blobs
    set_seed(42)

    X_np, y_np = make_blobs(
        n_samples=600,
        n_features=4,
        centers=5,
        cluster_std=1.2,
        random_state=42,
    )

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)

    # Standardise features
    mu  = X.mean(dim=0)
    std = X.std(dim=0)
    X   = (X - mu) / std

    # Shuffle
    perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(42))
    X, y = X[perm], y[perm]

    split = int(0.8 * X.shape[0])
    return (X[:split], y[:split]), (X[split:], y[split:])


def build_model(k: int = 5):
    """
    kNN 'model' is just the stored training data + hyper-parameter k.
    Actual storage happens inside train().
    """
    return {"k": k, "X_train": None, "y_train": None}


def train(model, train_data, cfg: dict = None):
    """
    kNN has no parameter fitting; training simply stores the data.
    """
    cfg = cfg or {}
    model = dict(model)              # shallow copy so we don't mutate the original
    model["k"] = cfg.get("k", model.get("k", 5))
    model["X_train"], model["y_train"] = train_data
    return model, {}


def _pairwise_l2(X_query, X_ref):
    """
    Compute all pairwise L2 distances in a vectorised manner.
    Uses the identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b

    Returns: dist  [n_query, n_ref]
    """
    dot = X_query @ X_ref.T                             # [nq, nr]
    sq_q = (X_query ** 2).sum(dim=1, keepdim=True)      # [nq, 1]
    sq_r = (X_ref   ** 2).sum(dim=1, keepdim=True).T    # [1,  nr]
    dist_sq = sq_q + sq_r - 2 * dot
    # Clamp to avoid tiny negatives from floating-point errors
    return dist_sq.clamp(min=0).sqrt()


def evaluate(model, data):
    """
    Compute accuracy (and MSE / R2 as protocol requires).
    """
    X_query, y_true = data
    X_train = model["X_train"]
    y_train = model["y_train"]
    k       = model["k"]
    n_classes = int(y_train.max().item()) + 1

    dist = _pairwise_l2(X_query, X_train)              # [nq, n_train]
    topk_idx = dist.topk(k, dim=1, largest=False).indices   # [nq, k]
    topk_labels = y_train[topk_idx]                    # [nq, k]

    # Majority vote via one-hot accumulation (fully vectorised)
    one_hot = torch.zeros(X_query.shape[0], n_classes)
    one_hot.scatter_add_(1, topk_labels, torch.ones_like(topk_labels, dtype=torch.float))
    preds = one_hot.argmax(dim=1)                      # [nq]

    accuracy = (preds == y_true).float().mean().item()

    # MSE / R2 on class indices (protocol compatibility)
    pf = preds.float()
    yf = y_true.float()
    mse = ((pf - yf) ** 2).mean().item()
    ss_tot = ((yf - yf.mean()) ** 2).sum().item() + 1e-8
    ss_res = ((pf - yf) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot

    return {"accuracy": accuracy, "mse": mse, "r2": r2}


def predict(model, X_query):
    """Return predicted class labels for X_query."""
    X_train = model["X_train"]
    y_train = model["y_train"]
    k       = model["k"]
    n_classes = int(y_train.max().item()) + 1

    dist = _pairwise_l2(X_query, X_train)
    topk_idx    = dist.topk(k, dim=1, largest=False).indices
    topk_labels = y_train[topk_idx]
    one_hot = torch.zeros(X_query.shape[0], n_classes)
    one_hot.scatter_add_(1, topk_labels, torch.ones_like(topk_labels, dtype=torch.float))
    return one_hot.argmax(dim=1)


def save_artifacts(model, metrics, output_dir: str = "outputs/knn_lvl1_bruteforce"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metrics.json")
    serialisable = {
        split: {k: float(v) for k, v in m.items()}
        for split, m in metrics.items()
    }
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Artifacts saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(42)

    train_data, val_data = make_dataloaders()
    model = build_model(k=5)

    model, _ = train(model, train_data, cfg={"k": 5})

    train_metrics = evaluate(model, train_data)
    val_metrics   = evaluate(model, val_data)

    print("=" * 55)
    print("  kNN Brute-Force Classifier (k=5)")
    print("=" * 55)
    print(f"  Train  acc={train_metrics['accuracy']:.4f}  MSE={train_metrics['mse']:.4f}")
    print(f"  Val    acc={val_metrics['accuracy']:.4f}  MSE={val_metrics['mse']:.4f}")

    # --- Compare with sklearn for the ±2% threshold ---
    try:
        from sklearn.neighbors import KNeighborsClassifier
        import numpy as np

        X_tr_np = train_data[0].numpy()
        y_tr_np = train_data[1].numpy()
        X_val_np = val_data[0].numpy()
        y_val_np = val_data[1].numpy()

        sk_knn = KNeighborsClassifier(n_neighbors=5, algorithm="brute", metric="euclidean")
        sk_knn.fit(X_tr_np, y_tr_np)
        sk_acc = sk_knn.score(X_val_np, y_val_np)
        diff = abs(val_metrics["accuracy"] - sk_acc)
        print(f"  sklearn kNN val acc : {sk_acc:.4f}")
        print(f"  Difference          : {diff:.4f} (threshold 0.02)")
        sklearn_check = diff <= 0.02
    except ImportError:
        print("  sklearn not available — skipping comparison.")
        sklearn_check = True
        diff = 0.0

    print("=" * 55)

    try:
        assert val_metrics["accuracy"] > 0.85, \
            f"Val accuracy {val_metrics['accuracy']:.4f} < 0.85"
        assert sklearn_check, \
            f"Diff vs sklearn {diff:.4f} > 0.02"
        print("  All assertions PASSED.")
        save_artifacts(model, {"train": train_metrics, "val": val_metrics})
        sys.exit(0)
    except AssertionError as e:
        print(f"  FAILED: {e}")
        sys.exit(1)
