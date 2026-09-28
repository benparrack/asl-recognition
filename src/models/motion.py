"""
Motion-letter model (J, Z): a small GRU over a landmark sequence.

Deliberately tiny, matching the lesson from LandmarkMLP -- this is an easy
binary problem (J's hook and Z's zigzag are very different trajectories) on a
very small dataset (a few dozen clips per class), so a big model would just
overfit. One GRU layer over the (T, 63) sequence, take the final hidden state,
classify with a linear head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config


class MotionGRU(nn.Module):
    def __init__(self, num_classes: int = len(config.MOTION_LETTERS),
                 input_size: int = config.LANDMARK_FEATURE_SIZE,
                 hidden_size: int = 32, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 63) -> (B, num_classes) logits."""
        _, h_n = self.gru(x)     # h_n: (1, B, hidden_size), one layer
        return self.head(h_n[-1])
