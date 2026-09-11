"""
Model 4 (non-neural): RandomForestClassifier on the same landmark vectors
LandmarkMLP uses.

Run:  python src/models/random_forest_baseline.py

Worth doing because it's the honest floor. If a random forest gets within a
couple of points of the neural models, that's worth knowing and worth saying --
classical ML on good features is a strong baseline. Costs almost nothing since
landmark extraction is already cached in data/processed/landmarks.npz.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import config  # noqa: E402
from data.dataset import build_dataloaders  # noqa: E402
from evaluate import (  # noqa: E402
    confusion_matrix, per_class_metrics, analyse_confusable_groups, plot_confusion_matrix,
)


def dataset_to_numpy(loader):
    """Pull the underlying arrays straight out of ASLLandmarkDataset -- no need
    to iterate batches for something this small."""
    ds = loader.dataset
    X = ds.landmarks
    y = np.array([config.CLASS_TO_IDX[label] for label in ds.df["label"]], dtype=np.int64)
    return X, y


def main() -> None:
    train_loader, val_loader, _ = build_dataloaders(kind="landmark", split="signer")

    X_train, y_train = dataset_to_numpy(train_loader)
    X_val, y_val = dataset_to_numpy(val_loader)

    clf = RandomForestClassifier(n_estimators=300, random_state=config.SEED, n_jobs=-1)

    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_pred = clf.predict(X_val)
    val_acc = float((y_pred == y_val).mean())
    print(f"[rf] fit time: {fit_time:.1f}s")
    print(f"[rf] val accuracy: {val_acc:.4f}")

    cm = confusion_matrix(y_val, y_pred, config.NUM_CLASSES)
    metrics_df = per_class_metrics(cm)
    confusable = analyse_confusable_groups(cm)

    run_name = "random_forest_pugeault"
    out_prefix = config.RESULTS_DIR / f"{run_name}_val"

    np.save(f"{out_prefix}_confusion_matrix.npy", cm)
    metrics_df.to_csv(f"{out_prefix}_per_class_metrics.csv", index=False)
    plot_confusion_matrix(cm, config.CLASSES, f"{out_prefix}_confusion_matrix.png")

    print("\n[rf] worst 5 classes by recall:")
    print(metrics_df.sort_values("recall").head(5).to_string(index=False))
    print(f"\n[rf] {confusable['combined_fraction']:.1%} of all errors fall inside "
          f"known confusable groups ({confusable['combined_errors']}/{confusable['total_errors']})")

    summary = {
        "model": "random_forest",
        "n_estimators": 300,
        "val_accuracy": val_acc,
        "fit_time_seconds": fit_time,
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "confusable_groups": confusable,
    }
    Path(f"{out_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[rf] results written to {config.RESULTS_DIR} (prefix: {run_name}_val)")


if __name__ == "__main__":
    main()
