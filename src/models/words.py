"""
Isolated-sign (word) classifiers over holistic_features sequences.

Both models take (x (B, L, D), mask (B, L) bool, True = real frame) and return
logits (B, num_classes). Padding never reaches the output: padded steps are
zeroed after normalisation, excluded from attention, and excluded from pooling.

WordBiGRU            the baseline: a small bidirectional GRU plus masked mean
                     pooling, the same family as the J/Z motion model
WordConvTransformer  the stronger model. Depthwise temporal convolutions for
                     local motion (a few frames), then a small transformer
                     encoder for whole-sign context, then masked mean pooling.
                     This is a scaled-down version of the conv+transformer
                     designs that did well on this dataset family.

Input standardisation (per-channel mean/std from the TRAIN split) is stored in
the model as buffers, as in MotionGRUv2, so the live demo can't skip it or
apply it differently.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _Standardised(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(input_dim))
        self.register_buffer("feat_std", torch.ones(input_dim))

    def set_normalisation(self, mean, std):
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-3)
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std)

    def _standardise(self, x, mask):
        x = (x - self.feat_mean) / self.feat_std
        return x * mask.unsqueeze(-1).to(x.dtype)


def masked_mean(h, mask):
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


class WordBiGRU(_Standardised):
    def __init__(self, input_dim: int, num_classes: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__(input_dim)
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.LayerNorm(hidden_size),
                                  nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden_size, num_classes))

    def forward(self, x, mask):
        h = self.proj(self._standardise(x, mask))
        lengths = mask.sum(1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True,
                                                  total_length=x.shape[1])
        return self.head(masked_mean(out, mask))


class ConvBlock(nn.Module):
    """Pointwise expand -> depthwise temporal conv -> BatchNorm -> pointwise
    project, residual. Padded steps are zeroed before the conv so they can't
    leak into real frames at the sequence's end."""

    def __init__(self, dim: int, kernel_size: int = 15, expand: int = 2, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, dim * expand)
        self.act = nn.SiLU()
        self.dw = nn.Conv1d(dim * expand, dim * expand, kernel_size, padding=kernel_size // 2,
                            groups=dim * expand)
        self.bn = nn.BatchNorm1d(dim * expand)
        self.project = nn.Linear(dim * expand, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        m = mask.unsqueeze(-1).to(x.dtype)
        h = self.act(self.expand(self.norm(x))) * m
        h = self.act(self.bn(self.dw(h.transpose(1, 2)))).transpose(1, 2)
        return x + self.drop(self.project(h)) * m


class WordConvTransformer(_Standardised):
    def __init__(self, input_dim: int, num_classes: int, dim: int = 192, max_len: int = 64,
                 n_conv: int = 3, n_transformer: int = 2, heads: int = 4,
                 kernel_size: int = 15, dropout: float = 0.2):
        super().__init__(input_dim)
        self.embed = nn.Sequential(nn.Linear(input_dim, dim), nn.LayerNorm(dim))
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        self.convs = nn.ModuleList(ConvBlock(dim, kernel_size, dropout=dropout)
                                   for _ in range(n_conv))
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_transformer,
                                                 enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, num_classes))

    def forward(self, x, mask):
        h = self.embed(self._standardise(x, mask)) + self.pos[:, :x.shape[1]]
        for block in self.convs:
            h = block(h, mask)
        h = self.transformer(h, src_key_padding_mask=~mask)
        return self.head(masked_mean(self.norm(h), mask))


def build_word_model(name: str, input_dim: int, num_classes: int, max_len: int, **kw):
    if name == "gru":
        return WordBiGRU(input_dim, num_classes, **kw)
    if name == "transformer":
        return WordConvTransformer(input_dim, num_classes, max_len=max_len, **kw)
    raise ValueError(f"unknown word model {name!r}")
