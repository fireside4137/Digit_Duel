"""
Training pipeline for Digit Duel models (CNN and RNN).
Supports CLI arguments, MPS/CUDA/CPU auto-detection, checkpointing, and curve plotting.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    CHECKPOINTS_DIR,
    REPORTS_DIR,
    CNN_CONFIG,
    RNN_CONFIG,
)
from src.dataset import get_dataloaders
from src.models import CNN, RNN
from src.utils.device import get_device
from src.utils.metrics import plot_learning_curves


def build_model(model_name: str) -> nn.Module:
    """Instantiates model by name."""
    name = model_name.lower()
    if name == "cnn":
        return CNN(**CNN_CONFIG)
    elif name == "rnn":
        return RNN(**RNN_CONFIG)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}. Choose 'cnn' or 'rnn'.")


def train_model(
    model_name: str = "cnn",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    device_pref: str = "auto"
):
    """Executes model training, evaluation, checkpointing, and reporting."""
    device = get_device(device_pref)
    print(f"[*] Training {model_name.upper()} on device: {device}")

    # Data loaders
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # Initialize model, loss, optimizer
    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

        # Quick validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_acc = (correct / total) * 100.0
        print(
            f"Epoch [{epoch+1:2d}/{epochs:2d}] - Loss: {epoch_loss:.5f} | Test Acc: {val_acc:.2f}%"
        )

    # Save checkpoint
    checkpoint_path = CHECKPOINTS_DIR / f"{model_name.lower()}_mnist.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[✓] Checkpoint saved: {checkpoint_path}")

    # Plot learning curve
    curve_path = REPORTS_DIR / f"learning_curves_{model_name.lower()}.png"
    plot_learning_curves(
        loss_history,
        save_path=curve_path,
        title=f"{model_name.upper()} Training Loss"
    )
    print(f"[✓] Loss curve saved: {curve_path}")

    return model, val_acc


def main():
    parser = argparse.ArgumentParser(description="Train CNN or RNN on MNIST dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "rnn"],
        help="Model architecture: 'cnn' or 'rnn'",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'mps', 'cuda', or 'cpu'")

    args = parser.parse_args()
    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_pref=args.device,
    )


if __name__ == "__main__":
    main()
