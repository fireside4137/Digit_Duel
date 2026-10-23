"""
Evaluation metrics and plotting utilities.
Generates confusion matrices and loss curves.
"""

from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    save_path: Path,
    title: str = "Confusion Matrix"
) -> None:
    """Computes and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))

    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)

    for i in range(10):
        for j in range(10):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), va="center", ha="center", color=color, fontsize=8)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([str(i) for i in range(10)])
    ax.set_yticklabels([str(i) for i in range(10)])
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=13, pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_learning_curves(
    train_losses: List[float],
    save_path: Path,
    title: str = "Training Loss Curve"
) -> None:
    """Plots and saves the training loss progression across epochs."""
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, marker="o", color="#2563eb", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Cross Entropy Loss", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def print_classification_summary(y_true: List[int], y_pred: List[int]) -> str:
    """Returns formatted precision, recall, and f1 classification report."""
    target_names = [f"Digit {i}" for i in range(10)]
    return classification_report(y_true, y_pred, target_names=target_names, digits=4)
