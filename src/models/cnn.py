"""
Convolutional Neural Network (CNN) architecture for MNIST digit classification.
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    3-Stage Convolutional Neural Network.
    
    Spatial progression:
      Input: (B, 1, 28, 28)
      Conv1 + MaxPool: (B, 32, 14, 14)
      Conv2 + MaxPool: (B, 64, 7, 7)
      Conv3 + MaxPool: (B, 128, 3, 3)
      Flatten: (B, 1152)
      Linear1: (B, 256)
      Linear2 (Logits): (B, 10)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns unnormalized logits of shape (B, num_classes).
        """
        features = self.conv_layers(x)
        flattened = features.view(features.size(0), -1)
        logits = self.fc_layers(flattened)
        return logits
