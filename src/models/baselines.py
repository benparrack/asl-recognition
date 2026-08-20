"""
Models 2 and 3: the comparison baselines.

Model 1 (src/models/cnn.py) is your own CNN. These two exist so that the number
it produces means something. An accuracy figure with nothing to compare it
against is not a result.

ASK YOUR PROFESSOR whether a pretrained backbone is permitted. Some courses
forbid it on the grounds that the point is to train something yourself. If it is
disallowed, drop TransferNet and keep LandmarkMLP -- the interesting comparison
survives either way.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config


class TransferNet(nn.Module):
    """
    Model 2: a pretrained backbone with a replaced classification head.

    The argument for including it: ImageNet features (edges, textures, shapes)
    transfer well to hand images, so this will probably beat your scratch CNN
    when data is limited. Quantifying that gap tells you how much of your error
    is architecture and how much is simply not enough data.

    TODO:
      1. Load a pretrained backbone:
             from torchvision.models import resnet18, ResNet18_Weights
             backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
      2. Replace the final layer:
             backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
      3. Decide on a freezing strategy and JUSTIFY IT:
           - freeze everything but the head: fastest, least prone to overfitting,
             but cannot adapt low-level filters to hand imagery
           - fine-tune everything at a low LR (1e-4 or below): usually best when
             you have enough data
           - freeze early blocks, fine-tune later ones: the usual compromise
         Trying two and reporting the difference is cheap and looks thorough.

    NOTE: pretrained models expect 224x224 ImageNet-normalised input. If you keep
    IMAGE_SIZE at 128 the model still runs but performs below its potential --
    mention the resolution you used.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, freeze_backbone: bool = True):
        super().__init__()
        raise NotImplementedError("See the TODO above.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LandmarkMLP(nn.Module):
    """
    Model 3: a small MLP over normalised MediaPipe landmarks (63 inputs).

    This one is deliberately tiny -- a few thousand parameters against millions
    in the CNNs -- and it will train in seconds. Watch what happens on the
    signer-disjoint test set: it frequently *wins*, because the landmark
    representation has already discarded background, lighting, and skin tone,
    the exact nuisance variables a pixel model overfits to.

    If that is what you observe, it is the most interesting sentence in your
    report. Explicit feature engineering beat end-to-end learning at this data
    scale, and you can say precisely why.

    A reasonable shape: 63 -> 128 -> 64 -> num_classes, with ReLU, BatchNorm1d,
    and dropout between layers.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES,
                 hidden: tuple[int, ...] = (128, 64), dropout: float = 0.3):
        super().__init__()

        # TODO: build the stack. A loop over `hidden` keeps this clean and makes
        # the layer sizes easy to sweep as a hyperparameter.
        raise NotImplementedError("Build the MLP here.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# A fourth, non-neural baseline worth 20 minutes of your time
# ---------------------------------------------------------------------------
"""
Fit sklearn's RandomForestClassifier or SVC on the same 63-dim landmark vectors.

Two reasons this is worth doing:

  1. It is the honest floor. If a random forest gets within a couple of points of
     your neural networks, that is worth knowing and worth saying -- classical ML
     on good features is a strong baseline, which is exactly the lesson your
     course textbook opens with.
  2. It costs almost nothing:

         from sklearn.ensemble import RandomForestClassifier
         clf = RandomForestClassifier(n_estimators=300, random_state=config.SEED)
         clf.fit(X_train, y_train)
         print(clf.score(X_test, y_test))

Use the same signer-disjoint split so the comparison is fair.
"""
