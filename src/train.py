"""
Training loop.

Run:  python src/train.py --model cnn --epochs 30

The plumbing (argument parsing, seeding, checkpointing, logging) is written for
you. The training step itself is left as a TODO because writing it once by hand
is how the optimisation loop stops being magic.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config
from data.dataset import build_dataloaders


# ---------------------------------------------------------------------------
def set_seed(seed: int = config.SEED):
    """
    Fix every source of randomness.

    Without this, two runs of the same code give different numbers and you cannot
    tell whether a change helped or you got lucky. Set it once, at the top of
    every entry point.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name: str) -> nn.Module:
    if name == "cnn":
        from models.cnn import ASLNet
        return ASLNet()
    if name == "transfer":
        from models.baselines import TransferNet
        return TransferNet()
    if name == "mlp":
        from models.baselines import LandmarkMLP
        return LandmarkMLP()
    raise ValueError(f"unknown model: {name}")


# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    """
    One pass over the training set. Returns (mean_loss, accuracy).

    TODO -- the five-step loop, which you should be able to write from memory
    by the end of this project:

        model.train()
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()        # 1. clear old gradients
            outputs = model(inputs)      # 2. forward pass
            loss = criterion(outputs, targets)   # 3. compute loss
            loss.backward()              # 4. backpropagate
            optimizer.step()             # 5. update weights

            # accumulate loss * batch_size and correct predictions

    Two traps:
      - Forgetting zero_grad() makes gradients accumulate across batches. Training
        will not crash; it will just quietly fail to converge.
      - Accumulate `loss.item()`, never the tensor itself. Keeping the tensor
        retains the whole computation graph and you will exhaust your 8 GB of
        VRAM within a few dozen batches.
    """
    raise NotImplementedError("Write the training loop here.")


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """
    Evaluate without updating weights. Returns (mean_loss, accuracy).

    model.eval() matters and is not optional: it switches BatchNorm to using its
    running statistics and disables Dropout. Forget it and your validation numbers
    will be wrong in a way that is hard to spot.

    The @torch.no_grad() decorator stops autograd building a graph, which roughly
    halves memory use and speeds evaluation up substantially.

    TODO: mirror the training loop without the backward pass.
    """
    raise NotImplementedError("Write the evaluation loop here.")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train an ASL fingerspelling classifier")
    parser.add_argument("--model", default="cnn", choices=["cnn", "transfer", "mlp"])
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--tag", default="", help="suffix for checkpoint/result files")
    args = parser.parse_args()

    set_seed()
    device = config.DEVICE
    print(f"[init] device: {device}")
    if device.type == "cuda":
        print(f"[init] gpu: {torch.cuda.get_device_name(0)}")

    kind = "landmark" if args.model == "mlp" else "image"
    train_loader, val_loader, test_loader = build_dataloaders(kind=kind)

    model = build_model(args.model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=config.WEIGHT_DECAY)

    # Cosine annealing: high LR early to explore, low LR late to settle. A
    # reasonable default; worth an ablation if you have time.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = f"{args.model}{('_' + args.tag) if args.tag else ''}"
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
              f"({time.time() - t0:.1f}s)")

        # Select the checkpoint on VALIDATION accuracy, never on test. The test
        # set is touched exactly once, in evaluate.py, at the very end.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "classes": config.CLASSES,
                        "args": vars(args)}, ckpt_path)
            print(f"          saved -> {ckpt_path.name}")

    history_path.write_text(json.dumps(history, indent=2))
    print(f"\n[done] best val accuracy {best_val_acc:.4f}")
    print(f"[done] history -> {history_path}")
    print("[next] run evaluate.py for the single, final test-set number.")


if __name__ == "__main__":
    main()
