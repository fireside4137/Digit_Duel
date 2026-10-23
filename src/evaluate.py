"""
Evaluation pipeline for Digit Duel models.
Computes test accuracy, per-class metrics, and generates confusion matrices.
"""

import argparse
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHECKPOINTS_DIR, REPORTS_DIR, CNN_CONFIG, RNN_CONFIG
from src.dataset import get_dataloaders
from src.models import CNN, RNN
from src.utils.device import get_device
from src.utils.metrics import plot_confusion_matrix, print_classification_summary


def evaluate_model(model_name: str = "cnn", device_pref: str = "auto"):
    """Evaluates the model checkpoint on the test set."""
    device = get_device(device_pref)
    name = model_name.lower()
    checkpoint_path = CHECKPOINTS_DIR / f"{name}_mnist.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train the model first using train.py."
        )

    # Initialize model and load weights
    if name == "cnn":
        model = CNN(**CNN_CONFIG)
    else:
        model = RNN(**RNN_CONFIG)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    _, test_loader = get_dataloaders()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    # Summary report
    print(f"\n=================== {name.upper()} EVALUATION REPORT ===================")
    print(print_classification_summary(all_targets, all_preds))

    # Confusion matrix
    cm_path = REPORTS_DIR / f"confusion_matrix_{name}.png"
    plot_confusion_matrix(
        all_targets,
        all_preds,
        save_path=cm_path,
        title=f"{name.upper()} Confusion Matrix (MNIST Test Set)"
    )
    print(f"[✓] Confusion matrix saved: {cm_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained CNN or RNN on MNIST.")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "rnn"],
        help="Model architecture: 'cnn' or 'rnn'",
    )
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'mps', 'cuda', or 'cpu'")
    args = parser.parse_args()
    evaluate_model(model_name=args.model, device_pref=args.device)


if __name__ == "__main__":
    main()
