"""
Binary Logistic Regression — manual sigmoid and manual gradients (no autograd).

Sigmoid:
    sigma(z) = 1 / (1 + exp(-z))

Log-loss (binary cross-entropy):
    J(w, b) = -(1/m) * sum_i [ y_i * log(h_i) + (1 - y_i) * log(1 - h_i) ]

Gradients:
    dJ/dw = (1/m) * X^T @ (h - y)
    dJ/db = (1/m) * sum(h - y)

No torch.nn, torch.optim, or autograd used.
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
        "id": "logreg_lvl1_binary_raw",
        "series": "Logistic Regression",
        "level": 1,
        "algorithm": "Logistic Regression (Binary, Raw Tensors)",
        "description": (
            "Binary logistic regression with manual sigmoid and manual gradients. "
            "No autograd used."
        ),
        "interface_protocol": "pytorch_task_v1",
    }


def set_seed(seed: int = 42):
    torch.manual_seed(seed)


def get_device():
    return torch.device("cpu")


def make_dataloaders(cfg: dict = None):
    """
    Two-Gaussian synthetic dataset (linearly separable).
    Standardises features before returning.
    Returns (X_train, y_train), (X_val, y_val) — both torch.Tensor.
    """
    set_seed(42)
    n_per_class = 300

    # Class 0: centred at (-2, -2)
    X0 = torch.randn(n_per_class, 2) + torch.tensor([-2.0, -2.0])
    y0 = torch.zeros(n_per_class)

    # Class 1: centred at (+2, +2)
    X1 = torch.randn(n_per_class, 2) + torch.tensor([2.0, 2.0])
    y1 = torch.ones(n_per_class)

    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([y0, y1], dim=0)

    # Shuffle
    idx = torch.randperm(X.shape[0])
    X, y = X[idx], y[idx]

    # Standardise
    mu = X.mean(dim=0)
    std = X.std(dim=0)
    X = (X - mu) / std

    split = int(0.8 * X.shape[0])
    return (X[:split], y[:split]), (X[split:], y[split:])


def build_model(n_features: int = 2):
    """Return (w, b) initialised to zero."""
    w = torch.zeros(n_features)
    b = torch.tensor(0.0)
    return w, b


def _sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))


def train(model, train_data, cfg: dict = None):
    """Manual gradient descent for logistic regression."""
    cfg = cfg or {}
    lr = cfg.get("lr", 0.5)
    epochs = cfg.get("epochs", 500)

    w, b = model
    X, y = train_data
    m = float(X.shape[0])

    loss_history = []

    for _ in range(epochs):
        z = X @ w + b
        h = _sigmoid(z)

        # Log-loss (clip for numerical stability)
        h_clipped = h.clamp(1e-7, 1 - 1e-7)
        loss = -(y * torch.log(h_clipped) + (1 - y) * torch.log(1 - h_clipped)).mean()
        loss_history.append(loss.item())

        delta = h - y                       # shape [m]
        grad_w = (X.T @ delta) / m          # shape [n_features]
        grad_b = delta.mean()

        w = w - lr * grad_w
        b = b - lr * grad_b

    return (w, b), {"loss_history": loss_history}


def evaluate(model, data):
    """
    Compute loss, accuracy, precision, recall, F1 on a data split.
    Returns a metrics dict.
    """
    w, b = model
    X, y = data
    m = float(X.shape[0])

    z = X @ w + b
    h = _sigmoid(z)

    h_clipped = h.clamp(1e-7, 1 - 1e-7)
    loss = -(y * torch.log(h_clipped) + (1 - y) * torch.log(1 - h_clipped)).mean().item()

    preds = (h >= 0.5).float()
    accuracy = (preds == y).float().mean().item()

    tp = ((preds == 1) & (y == 1)).sum().item()
    fp = ((preds == 1) & (y == 0)).sum().item()
    fn = ((preds == 0) & (y == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def predict(model, X):
    w, b = model
    z = X @ w + b
    h = _sigmoid(z)
    return (h >= 0.5).float()


def save_artifacts(model, metrics, output_dir: str = "outputs/logreg_lvl1_binary_raw"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Artifacts saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(42)

    train_data, val_data = make_dataloaders()
    n_features = train_data[0].shape[1]
    model = build_model(n_features=n_features)

    model, info = train(model, train_data, cfg={"lr": 0.5, "epochs": 500})

    train_metrics = evaluate(model, train_data)
    val_metrics = evaluate(model, val_data)

    print("=" * 55)
    print("  Binary Logistic Regression — Raw Tensors")
    print("=" * 55)
    print(f"  Train  loss={train_metrics['loss']:.4f}  acc={train_metrics['accuracy']:.4f}  "
          f"F1={train_metrics['f1']:.4f}")
    print(f"  Val    loss={val_metrics['loss']:.4f}  acc={val_metrics['accuracy']:.4f}  "
          f"F1={val_metrics['f1']:.4f}")
    print(f"  Val    precision={val_metrics['precision']:.4f}  "
          f"recall={val_metrics['recall']:.4f}")
    print("=" * 55)

    try:
        assert val_metrics["accuracy"] > 0.90, \
            f"Validation accuracy too low: {val_metrics['accuracy']:.4f} (required > 0.90)"
        print("  All assertions PASSED.")
        save_artifacts(model, {"train": train_metrics, "val": val_metrics})
        sys.exit(0)
    except AssertionError as e:
        print(f"  FAILED: {e}")
        sys.exit(1)
