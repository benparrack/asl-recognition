"""
Train the motion-letter (J, Z) GRU.

Run:  python src/train_motion.py --epochs 30

Reuses train_one_epoch/evaluate/set_seed from train.py unchanged -- they only
operate on (inputs, targets) batches and a model/criterion/optimizer, so
they're generic across the image/landmark-frame and landmark-sequence cases.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from data.motion_dataset import build_motion_dataloaders
from models.motion import MotionGRU
from train import set_seed, train_one_epoch, evaluate


def main():
    parser = argparse.ArgumentParser(description="Train the J/Z motion GRU")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--tag", default="", help="suffix for checkpoint/result files")
    args = parser.parse_args()

    set_seed()
    device = config.DEVICE
    print(f"[init] device: {device}")

    train_loader, val_loader, test_loader, max_len = build_motion_dataloaders(
        batch_size=args.batch_size)

    model = MotionGRU(hidden_size=args.hidden_size, dropout=args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = f"motion_gru{('_' + args.tag) if args.tag else ''}"
    ckpt_path = config.CHECKPOINT_DIR / f"{run_name}_best.pt"
    history_path = config.RESULTS_DIR / f"{run_name}_history.json"

    history = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc,
                            lr=scheduler.get_last_lr()[0]))

        print(f"epoch {epoch:>3}/{args.epochs}  "
              f"train {train_loss:.4f}/{train_acc:.3f}  "
              f"val {val_loss:.4f}/{val_acc:.3f}  "
              f"({time.time() - t0:.2f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "classes": config.MOTION_LETTERS,
                        "max_len": max_len,
                        "args": vars(args)}, ckpt_path)
            print(f"          saved -> {ckpt_path.name}")

    history_path.write_text(json.dumps(history, indent=2))
    print(f"\n[done] best val accuracy {best_val_acc:.4f}")
    print(f"[done] history -> {history_path}")


if __name__ == "__main__":
    main()
