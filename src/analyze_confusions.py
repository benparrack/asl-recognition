"""
Dig past the summary "X% of errors fall in known confusable groups" number into
which SPECIFIC pairs actually drive each model's errors, and whether the biggest
ones are already accounted for by config.KNOWN_CONFUSABLE_GROUPS or not.

Run:  python src/analyze_confusions.py
Reads the *_confusion_matrix.npy files evaluate.py already wrote to results/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

RUNS = ["cnn_pugeault_best", "transfer_pugeault_best", "mlp_pugeault_best",
        "random_forest_pugeault", "cnn_pugeault_bs128_best", "cnn_pugeault_dropout03_best"]
TOP_N = 10


def known_group_for(a: str, b: str) -> tuple | None:
    for group in config.KNOWN_CONFUSABLE_GROUPS:
        if a in group and b in group:
            return tuple(group)
    return None


def analyse(run_name: str) -> None:
    path = config.RESULTS_DIR / f"{run_name}_val_confusion_matrix.npy"
    if not path.exists():
        print(f"[skip] {path} not found")
        return

    cm = np.load(path)
    classes = config.CLASSES
    total_errors = int(cm.sum() - np.trace(cm))

    # (true, pred, count) for every off-diagonal cell, sorted by count desc.
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((classes[i], classes[j], int(cm[i, j])))
    pairs.sort(key=lambda p: -p[2])

    print(f"\n=== {run_name} ===  total errors: {total_errors}")

    top = pairs[:TOP_N]
    top_sum = sum(p[2] for p in top)
    print(f"top {TOP_N} pairs account for {top_sum}/{total_errors} "
          f"({top_sum / total_errors:.1%}) of all errors")

    for true_c, pred_c, count in top:
        group = known_group_for(true_c, pred_c)
        # look up the reverse direction too, to flag genuinely mutual confusion
        # vs. one-way "everything gets called X"
        reverse = cm[classes.index(pred_c), classes.index(true_c)]
        tag = f"IN known group {group}" if group else "NOT in any known group"
        print(f"  {true_c} -> {pred_c}: {count:>4}  (reverse {pred_c}->{true_c}: {reverse:>4})  {tag}")

    # Which classes get over-predicted overall (candidates for "the model's
    # default guess when unsure"), vs which just get missed a lot.
    predicted_totals = cm.sum(axis=0)
    true_totals = cm.sum(axis=1)
    over_predicted = predicted_totals - np.diag(cm)   # times X was guessed wrongly
    worst_over = np.argsort(-over_predicted)[:3]
    print("  most over-predicted (wrong guesses landing here):",
          ", ".join(f"{classes[i]} ({int(over_predicted[i])})" for i in worst_over))


def main() -> None:
    for run in RUNS:
        analyse(run)


if __name__ == "__main__":
    main()
