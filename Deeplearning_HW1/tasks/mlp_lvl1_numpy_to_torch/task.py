"""
2-layer MLP with fully manual backpropagation (no autograd).

Architecture:  Input(2) -> Hidden(4, sigmoid) -> Output(1, sigmoid)

Forward pass:
    z1 = X @ W1 + b1          [batch, 4]
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2         [batch, 1]
    a2 = sigmoid(z2)          -- predicted probability

Loss (binary cross-entropy):
    L = -(y * log(a2) + (1 - y) * log(1 - a2))

Chain rule backprop (one hidden layer):
    delta2 = (a2 - y) * a2 * (1 - a2)       [batch, 1]
    dW2    = a1.T @ delta2 / m
    db2    = delta2.mean()

    delta1 = (delta2 @ W2.T) * a1 * (1 - a1) [batch, 4]
    dW1    = X.T @ delta1 / m
    db1    = delta1.mean(dim=0)

Target dataset: XOR (all four input corners with their XOR labels).
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
        "id": "mlp_lvl1_numpy_to_torch",
        "series": "Neural Networks (MLP)",
        "level": 1,
        "algorithm": "MLP (Manual Backprop)",
        "description": (
            "2-layer MLP with fully manual backpropagation (no autograd). "
            "Solves the XOR problem."
        ),
        "interface_protocol": "pytorch_task_v1",
    }


def set_seed(seed: int = 42):
    torch.manual_seed(seed)


def get_device():
    return torch.device("cpu")


def make_dataloaders(cfg: dict = None):
    """
    XOR dataset: 4 pure corners + augmented noisy copies.
    Returns (X_train, y_train), (X_val, y_val).
    """
    set_seed(42)
    # Pure XOR corners
    corners = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    labels  = torch.tensor([[0.], [1.], [1.], [0.]])

    # Augment with noise for a more robust test
    n_aug = 500
    idx = torch.randint(0, 4, (n_aug,))
    X_aug = corners[idx] + 0.1 * torch.randn(n_aug, 2)
    y_aug = labels[idx]

    X = torch.cat([corners, X_aug], dim=0)
    y = torch.cat([labels, y_aug], dim=0)

    perm = torch.randperm(X.shape[0])
    X, y = X[perm], y[perm]

    split = int(0.8 * X.shape[0])
    return (X[:split], y[:split]), (X[split:], y[split:])


def build_model(n_hidden: int = 4):
    """
    Xavier-initialised weights for a 2-4-1 network.
    Returns a dict of tensors: W1, b1, W2, b2.
    """
    set_seed(42)
    W1 = torch.randn(2, n_hidden) * (2.0 / 2) ** 0.5
    b1 = torch.zeros(n_hidden)
    W2 = torch.randn(n_hidden, 1) * (2.0 / n_hidden) ** 0.5
    b2 = torch.zeros(1)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def _sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))


def _forward(model, X):
    W1, b1 = model["W1"], model["b1"]
    W2, b2 = model["W2"], model["b2"]
    z1 = X @ W1 + b1
    a1 = _sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = _sigmoid(z2)
    return a1, a2


def train(model, train_data, cfg: dict = None):
    """Manual backprop training loop."""
    cfg = cfg or {}
    lr     = cfg.get("lr", 1.0)
    epochs = cfg.get("epochs", 5000)

    X, y = train_data

    loss_history = []

    for _ in range(epochs):
        # ---- forward ----
        a1, a2 = _forward(model, X)

        a2_clipped = a2.clamp(1e-7, 1 - 1e-7)
        loss = -(y * torch.log(a2_clipped) + (1 - y) * torch.log(1 - a2_clipped)).mean()
        loss_history.append(loss.item())

        # ---- backward ----
        m = float(X.shape[0])

        # Output layer delta: dL/dz2 = a2 - y  (for sigmoid + BCE)
        delta2 = a2 - y                         # [m, 1]
        dW2 = (a1.T @ delta2) / m
        db2 = delta2.mean(dim=0)

        # Hidden layer delta
        delta1 = (delta2 @ model["W2"].T) * a1 * (1 - a1)   # [m, 4]
        dW1 = (X.T @ delta1) / m
        db1 = delta1.mean(dim=0)

        # ---- update ----
        model["W1"] = model["W1"] - lr * dW1
        model["b1"] = model["b1"] - lr * db1
        model["W2"] = model["W2"] - lr * dW2
        model["b2"] = model["b2"] - lr * db2

    return model, {"loss_history": loss_history}


def evaluate(model, data):
    """Compute loss and accuracy on a data split."""
    X, y = data
    _, a2 = _forward(model, X)

    a2_clipped = a2.clamp(1e-7, 1 - 1e-7)
    loss = -(y * torch.log(a2_clipped) + (1 - y) * torch.log(1 - a2_clipped)).mean().item()

    preds = (a2 >= 0.5).float()
    accuracy = (preds == y).float().mean().item()

    # MSE and R2 (for protocol compatibility)
    mse = ((a2 - y) ** 2).mean().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item()
    ss_res = ((a2 - y) ** 2).sum().item()
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    return {"loss": loss, "accuracy": accuracy, "mse": mse, "r2": r2}


def predict(model, X):
    _, a2 = _forward(model, X)
    return (a2 >= 0.5).float()


def save_artifacts(model, metrics, output_dir: str = "outputs/mlp_lvl1_numpy_to_torch"):
    os.makedirs(output_dir, exist_ok=True)
    # Save metrics
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
    model = build_model(n_hidden=4)

    model, info = train(model, train_data, cfg={"lr": 1.0, "epochs": 5000})

    train_metrics = evaluate(model, train_data)
    val_metrics   = evaluate(model, val_data)

    print("=" * 55)
    print("  MLP — Manual Backprop  (XOR task)")
    print("=" * 55)
    print(f"  Train  loss={train_metrics['loss']:.4f}  acc={train_metrics['accuracy']:.4f}")
    print(f"  Val    loss={val_metrics['loss']:.4f}  acc={val_metrics['accuracy']:.4f}")
    print("=" * 55)

    # Verify on the four pure XOR corners
    corners = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    corner_labels = torch.tensor([[0.], [1.], [1.], [0.]])
    corner_preds = predict(model, corners)
    corner_acc = (corner_preds == corner_labels).float().mean().item()
    print(f"  XOR corners accuracy: {corner_acc:.4f}")

    try:
        assert val_metrics["accuracy"] > 0.95, \
            f"Val accuracy {val_metrics['accuracy']:.4f} < 0.95"
        assert corner_acc == 1.0, \
            f"Model fails on at least one XOR corner (corner_acc={corner_acc:.2f})"
        print("  All assertions PASSED.")
        save_artifacts(model, {"train": train_metrics, "val": val_metrics})
        sys.exit(0)
    except AssertionError as e:
        print(f"  FAILED: {e}")
        sys.exit(1)
