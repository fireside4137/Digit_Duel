"""
Recurrent Neural Network (RNN) architecture for sequential MNIST classification.
Treats each 28x28 image as a sequence of 28 time-steps of 28 pixels each.
"""

import torch
import torch.nn as nn


class RNN(nn.Module):
    """
    Sequential Recurrent Neural Network.
    
    Processes the image row-by-row:
      Input: (B, 1, 28, 28) or (B, 28, 28)
      Squeeze channel: (B, 28, 28) -> seq_len=28, input_size=28
      RNN: hidden_size=128, num_layers=2
      Final time-step hidden state: (B, 128)
      Linear Head (Logits): (B, 10)
    """

    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 10
    ):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns unnormalized logits of shape (B, num_classes).
        """
        # Accommodate 4D tensor (B, 1, 28, 28) or 3D tensor (B, 28, 28)
        if x.dim() == 4:
            x = x.squeeze(1)

        # out shape: (B, seq_len=28, hidden_size=128)
        out, _ = self.rnn(x)

        # Extract representation from the last time step
        last_time_step = out[:, -1, :]

        # Output raw logits
        logits = self.fc(last_time_step)
        return logits
