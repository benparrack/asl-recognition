"""
Model 1 of 3: a convolutional network built from scratch.

THIS FILE IS THE CENTREPIECE OF YOUR PROJECT. It is the part your professor
wants to see you write, so the architecture below is left deliberately as a
skeleton with guidance rather than working code. Build it yourself; you will
need to defend every choice in it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config


class ASLNet(nn.Module):
    """
    A CNN mapping a (B, 3, 128, 128) batch of hand images to NUM_CLASSES logits.

    ------------------------------------------------------------------
    DESIGN NOTES -- read before writing the layers
    ------------------------------------------------------------------

    A conventional starting architecture is 3-4 convolutional blocks, each:

        Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d

    with channel counts roughly doubling per block (32 -> 64 -> 128 -> 256)
    while spatial dimensions halve (128 -> 64 -> 32 -> 16 -> 8). The intuition:
    early layers detect edges and gradients, later layers compose them into
    finger and knuckle configurations, and you trade spatial resolution for
    representational depth as you go.

    Then flatten (or better, AdaptiveAvgPool2d(1) -- far fewer parameters and
    less prone to overfitting than a big flatten) and finish with a small
    classifier head, with Dropout before the final Linear.

    Things you should be able to justify in your report:
      - Why BatchNorm? (stabilises training, permits higher learning rates,
        acts as mild regularisation)
      - Why 3x3 kernels? (two stacked 3x3s have the receptive field of one 5x5
        with fewer parameters and an extra nonlinearity)
      - Where does dropout belong, and why not between convolutional layers?
      - What is your parameter count, and is it plausible given your dataset
        size? A 20M-parameter model on 5,000 images will memorise them.

    START SMALL. Get an overfitting model working on a tiny subset first (see
    the sanity check at the bottom of this file), then add capacity. A model
    that cannot overfit 20 images has a bug, not a capacity problem, and no
    amount of extra layers will fix it.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, dropout: float = 0.5):
        super().__init__()

        # TODO: define self.features -- your convolutional blocks
        # TODO: define self.classifier -- pooling, dropout, final Linear
        raise NotImplementedError("Build your architecture here.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: pass through features, flatten, pass through classifier
        #
        # Return raw LOGITS, not softmax probabilities. nn.CrossEntropyLoss
        # applies log-softmax internally -- applying softmax yourself first is a
        # common bug that silently weakens your gradients.
        raise NotImplementedError


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count. Quote this in your report for each model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sanity_check():
    """
    Two checks that catch most architecture bugs before you waste a training run.

      1. Shape check -- does a dummy batch flow through and come out with the
         right shape?
      2. Overfit check -- can the model drive the loss to ~0 on a handful of
         samples? If not, something is broken in the model, the loss, or the
         optimiser. Diagnose it here, where iteration takes seconds, rather than
         after a 40-minute training run.
    """
    model = ASLNet()
    dummy = torch.randn(4, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
    out = model(dummy)

    assert out.shape == (4, config.NUM_CLASSES), f"bad output shape: {out.shape}"
    print(f"[ok] output shape {tuple(out.shape)}")
    print(f"[ok] {count_parameters(model):,} trainable parameters")


if __name__ == "__main__":
    sanity_check()
