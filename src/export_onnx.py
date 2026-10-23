"""
ONNX export pipeline for Digit Duel models.
Converts PyTorch checkpoints into optimized ONNX files for zero-sleep browser inference.
"""

import argparse
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHECKPOINTS_DIR, WEB_MODELS_DIR, CNN_CONFIG, RNN_CONFIG
from src.models import CNN, RNN


def export_to_onnx(model_name: str = "cnn"):
    """Exports model weights to ONNX format."""
    name = model_name.lower()
    checkpoint_path = CHECKPOINTS_DIR / f"{name}_mnist.pt"
    onnx_dest = CHECKPOINTS_DIR / f"{name}_mnist.onnx"
    web_dest = WEB_MODELS_DIR / f"{name}_mnist.onnx"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load model on CPU for universal export
    if name == "cnn":
        model = CNN(**CNN_CONFIG)
    else:
        model = RNN(**RNN_CONFIG)

    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28, dtype=torch.float32)

    print(f"[*] Exporting {name.upper()} to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_dest),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=18,
    )

    # Ensure all weights are embedded into a single standalone ONNX file
    import onnx
    onnx_model = onnx.load(str(onnx_dest), load_external_data=True)
    onnx.save(onnx_model, str(onnx_dest), save_as_external_data=False)

    # Clean up separate external data file if created
    ext_data = CHECKPOINTS_DIR / f"{name}_mnist.onnx.data"
    if ext_data.exists():
        ext_data.unlink()

    # Copy standalone model to web folder
    import shutil
    shutil.copyfile(onnx_dest, web_dest)

    web_ext_data = WEB_MODELS_DIR / f"{name}_mnist.onnx.data"
    if web_ext_data.exists():
        web_ext_data.unlink()

    file_size_mb = onnx_dest.stat().st_size / (1024 * 1024)
    print(f"[✓] Standalone ONNX exported: {onnx_dest} ({file_size_mb:.2f} MB)")
    print(f"[✓] Synced to web directory: {web_dest}")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX.")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "rnn"],
        help="Model architecture: 'cnn' or 'rnn'",
    )
    args = parser.parse_args()
    export_to_onnx(args.model)


if __name__ == "__main__":
    main()
