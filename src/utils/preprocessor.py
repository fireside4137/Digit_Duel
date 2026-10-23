"""
Image preprocessor for MNIST digit recognition.
Replicates Yann LeCun's center-of-mass centering and bounding-box scaling.
Crucial for user-drawn digits and real-world uploads to achieve 99% accuracy.
"""

import numpy as np
from PIL import Image
import torch


def preprocess_image(image_input) -> torch.Tensor:
    """
    Takes a PIL Image, filepath, or NumPy array, normalizes it according to MNIST standards:
      1. Converts to grayscale.
      2. Inverts colors if background is light (MNIST is white digit on black background).
      3. Crops to bounding box of the digit.
      4. Resizes to 20x20 maintaining aspect ratio.
      5. Centers inside 28x28 canvas using center of mass.
      6. Normalizes to [-1, 1] range.

    Returns:
        torch.Tensor of shape (1, 1, 28, 28)
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("L")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input).convert("L")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    arr = np.array(img, dtype=np.float32)

    # Check if background is bright (mean > 127); if so, invert to white digit on black
    if np.mean(arr) > 127:
        arr = 255.0 - arr

    # Threshold noise
    arr[arr < 30] = 0

    # If completely empty, return zeros
    if np.max(arr) == 0:
        norm = (np.zeros((1, 1, 28, 28), dtype=np.float32) - 0.5) / 0.5
        return torch.tensor(norm, dtype=torch.float32)

    # Find bounding box
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop
    cropped = arr[rmin : rmax + 1, cmin : cmax + 1]

    # Resize bounding box to fit inside a 20x20 box
    crop_h, crop_w = cropped.shape
    if crop_h > crop_w:
        factor = 20.0 / crop_h
        new_h = 20
        new_w = max(1, int(round(crop_w * factor)))
    else:
        factor = 20.0 / crop_w
        new_w = 20
        new_h = max(1, int(round(crop_h * factor)))

    cropped_img = Image.fromarray(cropped.astype(np.uint8)).resize(
        (new_w, new_h), Image.Resampling.BILINEAR
    )
    cropped_arr = np.array(cropped_img, dtype=np.float32)

    # Create empty 28x28 canvas and compute center of mass
    canvas = np.zeros((28, 28), dtype=np.float32)

    # Paste in center first
    start_r = (28 - new_h) // 2
    start_c = (28 - new_w) // 2
    canvas[start_r : start_r + new_h, start_c : start_c + new_w] = cropped_arr

    # Compute center of mass
    total_mass = np.sum(canvas)
    if total_mass > 0:
        y_indices, x_indices = np.indices((28, 28))
        cy = np.sum(y_indices * canvas) / total_mass
        cx = np.sum(x_indices * canvas) / total_mass

        # Shift to center (13.5, 13.5)
        shift_y = int(round(13.5 - cy))
        shift_x = int(round(13.5 - cx))

        shifted = np.zeros_like(canvas)
        src_r_start = max(0, -shift_y)
        src_r_end = min(28, 28 - shift_y)
        dst_r_start = max(0, shift_y)
        dst_r_end = min(28, 28 + shift_y)

        src_c_start = max(0, -shift_x)
        src_c_end = min(28, 28 - shift_x)
        dst_c_start = max(0, shift_x)
        dst_c_end = min(28, 28 + shift_x)

        r_len = min(src_r_end - src_r_start, dst_r_end - dst_r_start)
        c_len = min(src_c_end - src_c_start, dst_c_end - dst_c_start)

        if r_len > 0 and c_len > 0:
            shifted[
                dst_r_start : dst_r_start + r_len, dst_c_start : dst_c_start + c_len
            ] = canvas[
                src_r_start : src_r_start + r_len, src_c_start : src_c_start + c_len
            ]
            canvas = shifted

    # Scale to [0, 1] then normalize to [-1, 1] with mean=0.5, std=0.5
    canvas_norm = (canvas / 255.0 - 0.5) / 0.5

    # Shape: (1, 1, 28, 28)
    tensor = torch.tensor(canvas_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor
