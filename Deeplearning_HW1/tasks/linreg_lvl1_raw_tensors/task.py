"""
Univariate Linear Regression using ONLY PyTorch raw tensors.
No torch.nn, torch.optim, or autograd used anywhere.

Model:   h_theta(x) = theta_0 + theta_1 * x

Cost:    J(theta) = (1 / 2m) * sum_i (h_theta(x_i) - y_i)^2   [MSE]

Manual gradient descent:
    grad_0 = (1/m) * sum(h - y)
    grad_1 = (1/m) * sum((h - y) * x)
    theta_0 <- theta_0 - lr * grad_0
    theta_1 <- theta_1 - lr * grad_1
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
        "id": "linreg_lvl1_raw_tensors",
        "series": "Linear Regression",
        "level": 1,
        "algorithm": "Linear Regression (Raw Tensors)",
        "description": (
            "Univariate Linear Regression using ONLY PyTorch tensors. "
            "No torch.nn, torch.optim, or autograd."
        ),
        "interface_protocol": "pytorch_task_v1",
    }


def set_seed(seed: int = 42):
    torch.manual_seed(seed)


def get_device():
    return torch.device("cpu")


def make_dataloaders(cfg: dict = None):
    """
    Generate synthetic data: y = 2x + 3 + noise
    Returns (x_train, y_train), (x_val, y_val) tensors.
    """
    set_seed(42)
    n = 500
    x = torch.randn(n)
    y = 2.0 * x + 3.0 + 0.5 * torch.randn(n)

    split = int(0.8 * n)   # 80 / 20
    return (x[:split], y[:split]), (x[split:], y[split:])


def build_model():
    """Return (theta_0, theta_1) initialised to zero."""
    theta_0 = torch.tensor(0.0)
    theta_1 = torch.tensor(0.0)
    return theta_0, theta_1


def train(model, train_data, cfg: dict = None):
    """
    Gradient descent update loop (raw tensors, no autograd).
    Returns updated model and a dict with loss_history.
    """
    cfg = cfg or {}
    lr = cfg.get("lr", 0.1)
    epochs = cfg.get("epochs", 1000)

    theta_0, theta_1 = model
    x_train, y_train = train_data
    m = float(len(x_train))

    loss_history = []
    val_loss_history = []

    for _ in range(epochs):
        h = theta_0 + theta_1 * x_train
        residuals = h - y_train
        loss = (residuals ** 2).mean().item()
        loss_history.append(loss)

        grad_0 = residuals.mean()
        grad_1 = (residuals * x_train).mean()

        theta_0 = theta_0 - lr * grad_0
        theta_1 = theta_1 - lr * grad_1

    return (theta_0, theta_1), {
        "loss_history": loss_history,
    }


def evaluate(model, data):
    """
    Compute MSE, R2, and parameter accuracy on a given data split.
    Parameter accuracy = how close theta_0 is to 3.0 and theta_1 is to 2.0.
    """
    theta_0, theta_1 = model
    x, y = data

    h = theta_0 + theta_1 * x
    residuals = h - y
    mse = (residuals ** 2).mean().item()

    ss_tot = ((y - y.mean()) ** 2).sum().item()
    ss_res = (residuals ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot

    param_error_0 = abs(theta_0.item() - 3.0)
    param_error_1 = abs(theta_1.item() - 2.0)

    return {
        "mse": mse,
        "r2": r2,
        "theta_0": theta_0.item(),
        "theta_1": theta_1.item(),
        "param_error_0": param_error_0,
        "param_error_1": param_error_1,
    }


def predict(model, x):
    theta_0, theta_1 = model
    return theta_0 + theta_1 * x


def save_artifacts(model, metrics, output_dir: str = "outputs/linreg_lvl1_raw_tensors"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Artifacts saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main — train, evaluate, assert, exit
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(42)

    train_data, val_data = make_dataloaders()
    model = build_model()

    model, info = train(model, train_data, cfg={"lr": 0.1, "epochs": 1000})

    train_metrics = evaluate(model, train_data)
    val_metrics = evaluate(model, val_data)

    print("=" * 55)
    print("  Univariate Linear Regression — Raw Tensors")
    print("=" * 55)
    print(f"  Train  MSE = {train_metrics['mse']:.4f}   R2 = {train_metrics['r2']:.4f}")
    print(f"  Val    MSE = {val_metrics['mse']:.4f}   R2 = {val_metrics['r2']:.4f}")
    print(f"  Learned:  theta_0 = {val_metrics['theta_0']:.4f}  (true 3.0)  "
          f"theta_1 = {val_metrics['theta_1']:.4f}  (true 2.0)")
    print(f"  Param errors:  |theta_0 - 3| = {val_metrics['param_error_0']:.4f}  "
          f"|theta_1 - 2| = {val_metrics['param_error_1']:.4f}")
    print("=" * 55)

    try:
        assert val_metrics["r2"] > 0.9, \
            f"Validation R2 too low: {val_metrics['r2']:.4f} (required > 0.90)"
        assert val_metrics["param_error_0"] < 1.0, \
            f"|theta_0 - 3.0| = {val_metrics['param_error_0']:.4f} (required < 1.0)"
        assert val_metrics["param_error_1"] < 1.0, \
            f"|theta_1 - 2.0| = {val_metrics['param_error_1']:.4f} (required < 1.0)"
        print("  All assertions PASSED.")
        save_artifacts(model, {"train": train_metrics, "val": val_metrics})
        sys.exit(0)
    except AssertionError as e:
        print(f"  FAILED: {e}")
        sys.exit(1)
