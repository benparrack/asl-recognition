"""
Final evaluation. Run this ONCE per model, at the end.

Run:  python src/evaluate.py --checkpoint checkpoints/cnn_best.pt

Everything here is about producing the figures and tables that go in your report.
A single accuracy number is a weak result; a confusion matrix showing that your
model confuses exactly the handshapes a human learner confuses is a strong one.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from data.dataset import build_dataloaders

@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Run the model over a loader and return (y_true, y_pred, y_prob).

    Keeping the probabilities lets you analyse CONFIDENCE, not just correctness.
    A model that is wrong while confident is a different and more worrying
    failure than one that is wrong while uncertain -- worth a paragraph.

    TODO: implement. model.eval(), iterate, softmax the logits, argmax for the
    prediction, and accumulate everything into numpy arrays.
    """
    raise NotImplementedError


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """
    Rows are true classes, columns predicted. cm[i][j] = times class i was called j.

    Write this yourself in three lines rather than importing it -- it is a good
    check that you actually understand what the matrix represents:

        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t][p] += 1
    """
    raise NotImplementedError


def per_class_metrics(cm: np.ndarray):
    """
    Precision, recall, and F1 per class, from the confusion matrix.

        precision_i = cm[i][i] / column_i.sum()   (of everything called i, how much was i)
        recall_i    = cm[i][i] / row_i.sum()      (of all true i, how much did we catch)
        f1_i        = harmonic mean of the two

    Report per-class figures, not just the overall average. Overall accuracy hides
    the interesting structure: a model at 92% overall might be at 40% on M/N/S/T
    and near-perfect everywhere else, and that is the finding.
    """
    raise NotImplementedError


def analyse_confusable_groups(cm: np.ndarray):
    """
    Measure how much of the total error falls inside the visually-similar groups
    listed in config.KNOWN_CONFUSABLE_GROUPS.

    This turns a vague claim into a number. Instead of "the model struggles with
    similar handshapes", you can write: "63% of all errors occur within the
    {M, N, S, T} fist family, which differ only in thumb position -- the same
    distinction human ASL learners find hardest."

    That sentence is worth more than another point of accuracy.

    TODO: for each group, sum the off-diagonal entries where both true and
    predicted labels are in the group; express as a fraction of total errors.
    """
    raise NotImplementedError


def plot_confusion_matrix(cm, classes, out_path):
    """
    TODO: matplotlib imshow, tick labels from `classes`, colourbar, tight_layout,
    savefig at dpi>=150.

    Two tips that make it readable with 24 classes:
      - Normalise each row to sum to 1 so classes with more samples do not
        dominate the colour scale.
      - Annotate cells only where the value exceeds a threshold; 576 numbers on
        one figure is unreadable.
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    device = config.DEVICE
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Guard against a subtle and painful bug: if you changed CLASSES between
    # training and evaluation, every label silently shifts and your results are
    # nonsense with no error raised.
    if ckpt.get("classes") != config.CLASSES:
        raise RuntimeError(
            "Class list in the checkpoint does not match config.CLASSES. "
            "The model was trained on a different label set."
        )

    print(f"[eval] checkpoint from epoch {ckpt['epoch']}, val acc {ckpt['val_acc']:.4f}")

    # TODO: rebuild the right model type from ckpt["args"]["model"],
    # load_state_dict, move to device, then run the analysis functions above
    # and write results to config.RESULTS_DIR.

    raise NotImplementedError("Wire up the evaluation pipeline.")


if __name__ == "__main__":
    main()
