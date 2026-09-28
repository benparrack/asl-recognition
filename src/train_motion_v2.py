"""
Train the v2 motion model (J / Z / NONE) on raw-landmark clips.

Run:  python src/train_motion_v2.py --epochs 80

Differences from train_motion.py (v1), which stays as-is for the old checkpoint:
  - v2 dataset: raw clips -> cut by the live gate's own code -> shared
    motion_features.clip_to_features
  - train-time augmentation on the raw clip (never mirrored)
  - normalisation stats computed on the TRAIN split only, stored in the model
  - checkpoint carries everything the demo needs to rebuild the exact input:
    classes, feature_version, resample length, feature dim, normalisation spec
  - selection on val accuracy, ties broken by lower val loss (val is tiny, so
    exact ties are common)

The test split is not touched here. evaluate it once, at the very end, with
diagnose_motion.py --split test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from data import motion_features as mf  # noqa: E402
from data.motion_dataset_v2 import (  # noqa: E402
    MotionClipDatasetV2, load_motion_v2_metadata, split_motion_v2,
)
from models.motion import MotionGRUv2  # noqa: E402
from train import evaluate, set_seed, train_one_epoch  # noqa: E402


def build_checkpoint(model, epoch, val_acc, val_loss, args, split_signers) -> dict:
    return {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "classes": config.MOTION_V2_CLASSES,
        "feature_version": mf.FEATURE_VERSION,
        "resample_len": mf.RESAMPLE_LEN,
        "feature_dim": mf.FEATURE_DIM,
        "normalisation": {
            "kind": "per_channel_standardise_train_split",
            "mean": model.feat_mean.cpu().tolist(),
            "std": model.feat_std.cpu().tolist(),
        },
        # Training clips were cut by the live gate at its default threshold
        # (motion_dataset_v2.load_model_clip); recorded so a later change is visible.
        "train_segmentation": {"kind": "gate_segment", "threshold": 1.0},
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "split_signers": split_signers,
        "args": vars(args),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the J/Z/NONE motion model (v2)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--tag", default="", help="suffix for checkpoint/result files")
    args = parser.parse_args()

    set_seed()
    device = config.DEVICE
    print(f"[init] device: {device}")

    df = load_motion_v2_metadata()
    train_df, val_df, test_df = split_motion_v2(df)
    split_signers = {name: sorted(d["signer_id"].unique().tolist())
                     for name, d in (("train", train_df), ("val", val_df), ("test", test_df))}
    print(f"[motion-v2] train {len(train_df)}  val {len(val_df)}  test {len(test_df)}")
    print(f"[motion-v2] signers {split_signers}")
    print(f"[motion-v2] train label counts {train_df['label'].value_counts().to_dict()}")

    train_ds = MotionClipDatasetV2(train_df, augment=not args.no_augment)
    val_ds = MotionClipDatasetV2(val_df, augment=False)

    model = MotionGRUv2(input_size=mf.FEATURE_DIM, hidden_size=args.hidden_size,
                        dropout=args.dropout)
    model.set_normalisation(*train_ds.feature_stats())
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = f"motion_v2{('_' + args.tag) if args.tag else ''}"
    ckpt_path = config.CHECKPOINT_DIR / f"{run_name}_best.pt"
    history_path = config.RESULTS_DIR / f"{run_name}_history.json"

    history = []
    best = (-1.0, float("inf"))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc,
                            lr=scheduler.get_last_lr()[0]))
        print(f"epoch {epoch:>3}/{args.epochs}  train {train_loss:.4f}/{train_acc:.3f}  "
              f"val {val_loss:.4f}/{val_acc:.3f}  ({time.time() - t0:.2f}s)")

        if (val_acc, -val_loss) > (best[0], -best[1]):
            best = (val_acc, val_loss)
            torch.save(build_checkpoint(model, epoch, val_acc, val_loss, args, split_signers),
                       ckpt_path)
            print(f"          saved -> {ckpt_path.name}")

    history_path.write_text(json.dumps(history, indent=2))
    print(f"\n[done] best val accuracy {best[0]:.4f} (val loss {best[1]:.4f})")
    print(f"[done] history -> {history_path}")
    print(f"[next] python src/diagnose_motion.py --checkpoint {ckpt_path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
