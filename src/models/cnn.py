"""
Model 1 of 3: a convolutional network built from scratch.

Architecture:

    input                        (B,   3, 128, 128)
    block 1  conv 3->32,  pool   (B,  32,  64,  64)
    block 2  conv 32->64, pool   (B,  64,  32,  32)
    block 3  conv 64->128, pool  (B, 128,  16,  16)
    global average pool          (B, 128,   1,   1)
    flatten                      (B, 128)
    dropout + linear             (B, num_classes)

Each block is Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d. Channels double as
spatial dimensions halve: the network trades resolution for representational
depth as it goes deeper.

Run this file directly to check it works:

    python src/models/cnn.py            # shape check + overfit test
    python src/models/cnn.py --wide     # a larger variant, for comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402


class ASLNet(nn.Module):
    """
    A CNN mapping (B, 3, H, W) hand images to num_classes logits.

    Parameters
    ----------
    num_classes : int
        Size of the output layer. Comes from config so that flipping between the
        3-class smoke test and the 24-letter task needs no edit here.
    channels : tuple[int, ...]
        Output channels per block. Length determines depth. (32, 64, 128) is a
        reasonable default; try (16, 32) for something smaller or
        (32, 64, 128, 256) for something deeper and compare.
    dropout : float
        Dropout probability before the final linear layer.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.5,
    ):
        super().__init__()

        blocks: list[nn.Module] = []
        in_ch = config.IMAGE_CHANNELS

        for out_ch in channels:
            blocks += [
                # bias=False because BatchNorm immediately subtracts the mean,
                # which cancels any constant the conv bias would add. Keeping it
                # would be harmless but would waste out_ch parameters per layer.
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            ]
            in_ch = out_ch

        self.features = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            # Global average pooling: collapse each channel to one number by
            # averaging over its whole spatial map. Makes the network
            # resolution-independent and keeps the final Linear tiny.
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits -- CrossEntropyLoss applies log-softmax itself."""
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count. Quote this per model in your report."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def shape_check(model: nn.Module) -> None:
    """Does a dummy batch flow through and come out the right shape?"""
    batch = 4
    dummy = torch.randn(batch, config.IMAGE_CHANNELS,
                        config.IMAGE_SIZE, config.IMAGE_SIZE)
    out = model(dummy)

    assert out.shape == (batch, config.NUM_CLASSES), \
        f"expected {(batch, config.NUM_CLASSES)}, got {tuple(out.shape)}"

    print(f"[ok] forward pass: {tuple(dummy.shape)} -> {tuple(out.shape)}")
    print(f"[ok] {count_parameters(model):,} trainable parameters")


def overfit_check(model: nn.Module, steps: int = 200) -> None:
    """
    Can the model memorise a handful of fixed examples?

    THIS IS THE MOST USEFUL DEBUGGING STEP IN THE PROJECT. A network that cannot
    drive the loss to near zero on 8 samples has a bug -- gradients not flowing,
    wrong loss function, a broken learning rate -- and no amount of extra data or
    training time will fix it.

    Random noise as input is deliberate: there is no real pattern to learn, so
    success here means "the optimisation machinery works", nothing more. It costs
    seconds instead of the 40 minutes a real training run would take to tell you
    the same thing.
    """
    torch.manual_seed(config.SEED)

    n = 8
    x = torch.randn(n, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
    y = torch.randint(0, config.NUM_CLASSES, (n,))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    first_loss = None

    for step in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        if first_loss is None:
            first_loss = loss.item()
        if (step + 1) % 50 == 0:
            print(f"     step {step + 1:>3}  loss {loss.item():.4f}")

    final = loss.item()
    print(f"[..] loss {first_loss:.4f} -> {final:.4f}")

    if final < 0.05:
        print("[ok] model can overfit -- optimisation works")
    else:
        print("[!!] model did NOT converge on 8 samples. Something is wrong:")
        print("     - is the learning rate sane?")
        print("     - does forward() return logits (not softmax)?")
        print("     - are all layers registered as attributes of the module?")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide", action="store_true",
                        help="use a deeper/wider variant for comparison")
    parser.add_argument("--skip-overfit", action="store_true")
    args = parser.parse_args()

    channels = (64, 128, 256, 512) if args.wide else (32, 64, 128)
    model = ASLNet(channels=channels)

    print(f"ASLNet(channels={channels}, num_classes={config.NUM_CLASSES})")
    print(f"classes: {config.CLASSES}\n")

    shape_check(model)

    if not args.skip_overfit:
        print()
        overfit_check(model)


if __name__ == "__main__":
    main()
