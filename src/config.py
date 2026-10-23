"""
Configuration module for Digit Duel.
Defines paths, hyperparameters, and model configurations.
"""

from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"
WEB_DIR = PROJECT_ROOT / "web"
WEB_MODELS_DIR = WEB_DIR / "models"

# Ensure runtime directories exist
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
WEB_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = (28, 28)
NUM_CLASSES = 10

# Normalization parameters (MNIST standard)
NORMALIZE_MEAN = (0.5,)
NORMALIZE_STD = (0.5,)

# Model architectures
CNN_CONFIG = {
    "in_channels": 1,
    "num_classes": NUM_CLASSES,
}

RNN_CONFIG = {
    "input_size": 28,
    "hidden_size": 128,
    "num_layers": 2,
    "num_classes": NUM_CLASSES,
}
