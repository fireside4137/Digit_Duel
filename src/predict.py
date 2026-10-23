"""
Command-line inference script for single images.
Preprocesses arbitrary handwritten images and outputs predicted digit with probabilities.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHECKPOINTS_DIR, CNN_CONFIG, RNN_CONFIG
from src.models import CNN, RNN
from src.utils.device import get_device
from src.utils.preprocessor import preprocess_image


def predict_digit(image_path: str, model_name: str = "cnn"):
    """Performs inference on an image file."""
    device = get_device("auto")
    name = model_name.lower()
    checkpoint_path = CHECKPOINTS_DIR / f"{name}_mnist.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load model
    if name == "cnn":
        model = CNN(**CNN_CONFIG)
    else:
        model = RNN(**RNN_CONFIG)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocess
    tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_digit = int(probs.argmax())
        confidence = float(probs[pred_digit]) * 100.0

    print(f"\nModel: {name.upper()}")
    print(f"Predicted Digit: {pred_digit} (Confidence: {confidence:.2f}%)")
    print("Class Probabilities:")
    for digit, prob in enumerate(probs):
        bar = "█" * int(prob * 30)
        print(f"  Digit {digit}: {prob*100:5.1f}% | {bar}")


def main():
    parser = argparse.ArgumentParser(description="Predict digit from an image file.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "rnn"],
        help="Model architecture: 'cnn' or 'rnn'",
    )
    args = parser.parse_args()
    predict_digit(args.image, args.model)


if __name__ == "__main__":
    main()
