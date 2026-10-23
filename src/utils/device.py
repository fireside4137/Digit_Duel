"""
Device detection utility for PyTorch.
Selects Apple Silicon MPS, NVIDIA CUDA, or CPU automatically.
"""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """
    Returns the most suitable torch.device.

    Args:
        preference: 'auto', 'mps', 'cuda', or 'cpu'.
    """
    pref = preference.lower()

    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")

    # Auto selection
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
