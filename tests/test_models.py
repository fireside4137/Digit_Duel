"""
Unit tests for Digit Duel models and preprocessor.
"""

import numpy as np
import pytest
import torch
from src.models import CNN, RNN
from src.utils.preprocessor import preprocess_image


def test_cnn_output_shape():
    model = CNN()
    model.eval()
    dummy_input = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (4, 10), f"Expected (4, 10), got {output.shape}"


def test_rnn_output_shape_4d():
    model = RNN()
    model.eval()
    dummy_input = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (4, 10), f"Expected (4, 10), got {output.shape}"


def test_rnn_output_shape_3d():
    model = RNN()
    model.eval()
    dummy_input = torch.randn(4, 28, 28)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (4, 10), f"Expected (4, 10), got {output.shape}"


def test_preprocessor_from_array():
    arr = np.zeros((100, 100), dtype=np.uint8)
    # Draw simple cross
    arr[40:60, 48:52] = 255
    arr[48:52, 40:60] = 255

    tensor = preprocess_image(arr)
    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0
