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
import pandas as pd
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from data.dataset import build_dataloaders
from train import build_model


@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Run the model over a loader and return (y_true, y_pred, y_prob).

    Keeping the probabilities lets you analyse CONFIDENCE, not just correctness.
    A model that is wrong while confident is a different and more worrying
    failure than one that is wrong while uncertain -- worth a paragraph.
    """
    model.eval()

    all_true = []
    all_pred = []
    all_prob = []

    for inputs, targets in loader:
        inputs = inputs.to(device)

        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_true.append(targets.numpy())
        all_pred.append(preds.cpu().numpy())
        all_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    return y_true, y_pred, y_prob


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """
    Rows are true classes, columns predicted. cm[i][j] = times class i was called j.
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def per_class_metrics(cm: np.ndarray) -> pd.DataFrame:
    """
    Precision, recall, and F1 per class, from the confusion matrix.

        precision_i = cm[i][i] / column_i.sum()   (of everything called i, how much was i)
        recall_i    = cm[i][i] / row_i.sum()      (of all true i, how much did we catch)
        f1_i        = harmonic mean of the two

    Report per-class figures, not just the overall average. Overall accuracy hides
    the interesting structure: a model at 92% overall might be at 40% on M/N/S/T
    and near-perfect everywhere else, and that is the finding.
    """
    num_classes = cm.shape[0]
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)      # true count per class (row sum)
    predicted = cm.sum(axis=0)    # predicted count per class (column sum)

    # np.divide's `where` guards against 0/0 for a class with no support or no
    # predictions -- those cells stay 0 instead of raising or becoming NaN.
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    return pd.DataFrame({
        "class": config.CLASSES[:num_classes],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    })


def analyse_confusable_groups(cm: np.ndarray) -> dict:
    """
    Measure how much of the total error falls inside the visually-similar groups
    listed in config.KNOWN_CONFUSABLE_GROUPS.

    This turns a vague claim into a number. Instead of "the model struggles with
    similar handshapes", you can write: "63% of all errors occur within the
    {M, N, S, T} fist family, which differ only in thumb position -- the same
    distinction human ASL learners find hardest."
    """
    total_errors = int(cm.sum() - np.trace(cm))

    per_group = []
    # Some classes appear in more than one group (S is in both {M,N,S,T} and
    # {A,S,T}), so summing the per-group counts would double-count those
    # errors. Track a combined "in any group" mask separately for the honest
    # headline number.
    in_any_group = np.zeros_like(cm, dtype=bool)

    for group in config.KNOWN_CONFUSABLE_GROUPS:
        idxs = [config.CLASS_TO_IDX[c] for c in group if c in config.CLASS_TO_IDX]
        if len(idxs) < 2:
            continue

        sub = cm[np.ix_(idxs, idxs)]
        group_errors = int(sub.sum() - np.trace(sub))
        fraction = group_errors / total_errors if total_errors else 0.0
        per_group.append({
            "group": group,
            "errors": group_errors,
            "fraction_of_total_errors": fraction,
        })

        for i in idxs:
            for j in idxs:
                in_any_group[i, j] = True

    np.fill_diagonal(in_any_group, False)  # correct predictions aren't errors
    combined_errors = int(cm[in_any_group].sum())
    combined_fraction = combined_errors / total_errors if total_errors else 0.0

    return {
        "total_errors": total_errors,
        "groups": per_group,
        "combined_errors": combined_errors,
        "combined_fraction": combined_fraction,
    }


def plot_confusion_matrix(cm, classes, out_path, annotate_threshold: float = 0.05):
    """
    Row-normalised confusion matrix, annotated only where it matters.

    Normalising each row to sum to 1 keeps classes with more samples from
    dominating the colour scale. Annotating every cell would put 576 numbers on
    one 24x24 figure, which is unreadable -- only cells above the threshold get
    a text label.
    """
    import matplotlib
    matplotlib.use("Agg")  # no display available when this runs headless / over SSH
    import matplotlib.pyplot as plt

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.4), max(8, n * 0.4)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="fraction of true class")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)")

    for i in range(n):
        for j in range(n):
            if cm_norm[i, j] > annotate_threshold:
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                         fontsize=6, color="white" if cm_norm[i, j] > 0.5 else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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

    model_name = ckpt["args"]["model"]
    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state"])

    kind = "landmark" if model_name == "mlp" else "image"
    # Always the honest signer-disjoint split here -- "random" only exists for
    # the leakage-comparison experiment, never for a number that goes in a table.
    train_loader, val_loader, test_loader = build_dataloaders(kind=kind, split="signer")
    loader = test_loader if args.split == "test" else val_loader

    if args.split == "test":
        print("[eval] WARNING: evaluating on the TEST set. This should happen exactly "
              "once, at the very end. If you're still iterating, use --split val.")

    y_true, y_pred, y_prob = collect_predictions(model, loader, device)

    overall_acc = float((y_true == y_pred).mean())
    print(f"[eval] {args.split} accuracy: {overall_acc:.4f}")

    cm = confusion_matrix(y_true, y_pred, config.NUM_CLASSES)
    metrics_df = per_class_metrics(cm)
    confusable = analyse_confusable_groups(cm)

    run_name = Path(args.checkpoint).stem  # e.g. "cnn_pugeault_best"
    out_prefix = config.RESULTS_DIR / f"{run_name}_{args.split}"

    np.save(f"{out_prefix}_confusion_matrix.npy", cm)
    metrics_df.to_csv(f"{out_prefix}_per_class_metrics.csv", index=False)
    plot_confusion_matrix(cm, config.CLASSES, f"{out_prefix}_confusion_matrix.png")

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "overall_accuracy": overall_acc,
        "confusable_groups": confusable,
    }
    Path(f"{out_prefix}_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n[eval] worst 5 classes by recall:")
    print(metrics_df.sort_values("recall").head(5).to_string(index=False))

    print(f"\n[eval] {confusable['combined_fraction']:.1%} of all errors fall inside "
          f"known confusable groups ({confusable['combined_errors']}/{confusable['total_errors']})")

    print(f"\n[eval] results written to {config.RESULTS_DIR} (prefix: {run_name}_{args.split})")


if __name__ == "__main__":
    main()
