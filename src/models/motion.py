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


class MotionGRUv2(nn.Module):
    """
    J / Z / NONE classifier over v2 motion features (src/data/motion_features.py).

    Differences from v1, each for a reason:
      - 3 classes: NONE is what lets the gate say "that movement wasn't J or Z"
        instead of forcing a choice.
      - Bidirectional GRU + attention pooling instead of "last hidden state":
        v2 clips are resampled to a fixed length with no padding, so every
        timestep is real data; attention lets the readout weight the part of
        the stroke that matters rather than whatever happened last.
      - Feature normalisation (per-channel mean/std from the TRAINING split)
        lives inside the model as buffers. It's saved in model_state, so the
        demo can't forget to apply it or apply different numbers.

    Still tiny on purpose: a few hundred clips at most.
    """

    def __init__(self, num_classes: int = len(config.MOTION_V2_CLASSES),
                 input_size: int = 77, hidden_size: int = 32, dropout: float = 0.3):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(input_size))
        self.register_buffer("feat_std", torch.ones(input_size))
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden_size, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, num_classes),
        )

    def set_normalisation(self, mean, std) -> None:
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, input_size) raw v2 features -> (B, num_classes) logits."""
        x = (x - self.feat_mean) / self.feat_std
        h, _ = self.gru(x)                                  # (B, L, 2H)
        w = torch.softmax(self.attn(h).squeeze(-1), dim=1)  # (B, L)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled)
