"""
Evaluate a word checkpoint. Validation during development, test ONCE at the end.

Run:
    python src/evaluate_words.py --checkpoint checkpoints/words_transformer_full_best.pt
    python src/evaluate_words.py --checkpoint ... --split test     # once, final model only
    python src/evaluate_words.py --checkpoint ... --extra-cache-dir data/processed/asl_citizen \
        --dataset asl_citizen                                      # score the other dataset

Reports, beyond top-1:
  - top-5 accuracy
  - per-class precision / recall / F1 (evaluate.per_class_metrics)
  - the most-confused sign pairs
  - "default-guess sinks": classes predicted far more often than they occur.
    The letters study found the model dumping uncertain inputs into one or two
    classes, so this looks for the same pattern
  - one-handed vs two-handed signs, and static / moderate / dynamic signs
    (data/word_dataset.sign_properties, computed from TRAIN sequences)
  - a row-normalised confusion-matrix figure

--split test is refused once results/words_test_used.json exists
(words_test_used_<dataset>.json for a dataset other than GISLR).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from data import holistic_features as hf  # noqa: E402
from data import holistic_landmarks as hl  # noqa: E402
from data.word_dataset import (  # noqa: E402
    WordDataset, load_combined_index, load_or_compute_sign_properties, load_word_index,
)
from evaluate import confusion_matrix, per_class_metrics, plot_confusion_matrix  # noqa: E402
from models.words import build_word_model  # noqa: E402

TEST_MARKER = config.RESULTS_DIR / "words_test_used.json"


def load_word_checkpoint(path, device):
    """Rebuild a word model from its checkpoint, refusing any spec mismatch
    with the feature code it would be fed by."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt["landmark_spec_hash"] != hl.LANDMARK_SPEC_HASH:
        raise RuntimeError(f"{path}: landmark selection changed since training "
                           f"({ckpt['landmark_spec_hash']} != {hl.LANDMARK_SPEC_HASH})")
    if ckpt["feature_version"] != hf.FEATURE_VERSION:
        raise RuntimeError(f"{path}: feature_version {ckpt['feature_version']} != "
                           f"holistic_features {hf.FEATURE_VERSION}")
    if ckpt["n_channels"] != hf.FEATURE_SETS[ckpt["feature_set"]]:
        raise RuntimeError(f"{path}: feature set {ckpt['feature_set']} size changed")
    model = build_word_model(ckpt["model"], ckpt["n_channels"], len(ckpt["classes"]),
                             ckpt["max_len"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def collect(model, loader, device):
    ys, probs = [], []
    for x, m, y in loader:
        logits = model(x.to(device), m.to(device))
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def topk_accuracy(y, probs, k):
    top = np.argsort(-probs, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def most_confused(cm, classes, n=20):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None)[:n]
    rows = []
    for f in flat:
        i, j = np.unravel_index(f, off.shape)
        if off[i, j] == 0:
            break
        rows.append(dict(true=classes[i], pred=classes[j], count=int(off[i, j]),
                         frac_of_true=float(off[i, j] / max(cm[i].sum(), 1))))
    return rows


def default_guess_sinks(cm, classes, ratio=2.0, min_extra=5):
    """Classes predicted at least `ratio`x as often as they truly occur."""
    support, predicted = cm.sum(1), cm.sum(0)
    rows = []
    for i in np.argsort(-(predicted - support)):
        if predicted[i] >= ratio * max(support[i], 1) and predicted[i] - support[i] >= min_extra:
            rows.append(dict(sign=classes[i], predicted=int(predicted[i]),
                             support=int(support[i]),
                             absorbed_from=int(predicted[i] - cm[i, i])))
    return rows


def breakdowns(y, y_pred, props: pd.DataFrame) -> dict:
    p = props.set_index("label")
    correct = y_pred == y
    out = {}
    for col in ("two_handed", "motion_bucket"):
        groups = p[col].reindex(y).to_numpy()
        out[col] = {str(g): {"acc": float(correct[groups == g].mean()),
                             "n": int((groups == g).sum()),
                             "n_signs": int((p[col] == g).sum())}
                    for g in pd.unique(groups)}
    return out


def evaluate_split(model, ckpt, index, split, cache_dir, device, props, batch_size=256,
                   workers=8, fig_path=None) -> dict:
    classes = ckpt["classes"]
    df = index[index["split"] == split]
    ds = WordDataset(df, cache_dir, ckpt["feature_set"], ckpt["max_len"], augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    y, probs = collect(model, loader, device)
    y_pred = probs.argmax(1)
    cm = confusion_matrix(y, y_pred, len(classes))
    pcm = per_class_metrics(cm, classes)
    report = {
        "split": split, "n": int(len(y)),
        "participants": sorted(int(p) for p in df["participant_id"].unique()),
        "acc": float((y_pred == y).mean()),
        "top5": topk_accuracy(y, probs, 5),
        "mean_confidence_correct": float(probs.max(1)[y_pred == y].mean()) if (y_pred == y).any() else None,
        "mean_confidence_wrong": float(probs.max(1)[y_pred != y].mean()) if (y_pred != y).any() else None,
        "worst_classes": pcm.sort_values("f1").head(15).to_dict(orient="records"),
        "most_confused": most_confused(cm, classes),
        "default_guess_sinks": default_guess_sinks(cm, classes),
        "breakdowns": breakdowns(y, y_pred, props),
    }
    if fig_path is not None:
        plot_confusion_matrix(cm, classes, fig_path, annotate_threshold=1.1)
        report["confusion_figure"] = str(fig_path)
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate a word checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--cache-dir", type=Path, default=config.GISLR_CACHE_DIR)
    parser.add_argument("--extra-cache-dir", type=Path, default=None)
    parser.add_argument("--dataset", default="gislr",
                        help="with --extra-cache-dir: 'gislr' or the extra cache's folder name")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    suffix = "" if args.dataset == "gislr" else f"_{args.dataset}"
    marker = TEST_MARKER.with_name(f"{TEST_MARKER.stem}{suffix}.json")
    if args.split == "test" and marker.exists():
        sys.exit(f"Refusing: the word test split was already used "
                 f"({marker.read_text().strip()}). Report that result.")

    device = config.DEVICE
    model, ckpt = load_word_checkpoint(args.checkpoint, device)
    if args.extra_cache_dir:
        index, classes = load_combined_index(args.cache_dir, args.extra_cache_dir)
        index = index[index["dataset"] == args.dataset]
        if index.empty:
            sys.exit(f"no rows for dataset {args.dataset!r}")
    else:
        index, classes = load_word_index(args.cache_dir)
    if classes != ckpt["classes"]:
        raise RuntimeError("class list in the cache differs from the checkpoint's")
    props = load_or_compute_sign_properties(
        index, classes, args.cache_dir, config.RESULTS_DIR / f"words_sign_properties{suffix}.csv")
    stem = Path(args.checkpoint).stem.replace("_best", "")
    report = evaluate_split(model, ckpt, index, args.split, args.cache_dir, device, props,
                            workers=args.workers,
                            fig_path=config.RESULTS_DIR / f"{stem}_{args.split}{suffix}_confusion.png")
    report["dataset"] = args.dataset
    out = config.RESULTS_DIR / f"{stem}_{args.split}{suffix}_eval.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"{args.split}: acc {report['acc']:.4f}  top5 {report['top5']:.4f}  (n={report['n']})")
    print(f"breakdowns: {json.dumps(report['breakdowns'], indent=1)}")
    print(f"sinks: {report['default_guess_sinks'][:5]}")
    print(f"[done] -> {out}")
    if args.split == "test":
        marker.write_text(json.dumps({"checkpoint": args.checkpoint, "result": str(out),
                                           "acc": report["acc"], "top5": report["top5"]}) + "\n")


if __name__ == "__main__":
    main()
