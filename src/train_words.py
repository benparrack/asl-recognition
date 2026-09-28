"""
Train an isolated-sign (word) model on the GISLR cache.

Run:
    python src/train_words.py --model transformer --features full
    python src/train_words.py --model gru --features full --tag baseline

Follows train.py / train_motion_v2.py: best-on-val checkpoint in checkpoints/,
per-epoch history JSON in results/, and the test split is not touched (that's
evaluate_words.py --split test, once).

The checkpoint records everything the demo needs to rebuild the exact input:
class list, feature_version, landmark_spec_hash (the selected landmark index
lists), feature set and channel count, max_len, the normalisation (also inside
the model as buffers), and the participant split.

Training details (all standard for this dataset family): AdamW, one-epoch
linear warm-up into cosine decay, label smoothing 0.1, mixed precision on
CUDA, augmentation inside holistic_features (affine jitter, time-stretch,
frame drop, non-dominant hand drop). Selection is on val top-1, ties broken
by lower val loss.

After training, the best checkpoint is evaluated on val with the full
breakdown (evaluate_words.evaluate_split), written to
results/<run>_val_eval.json. The ablation script reads these files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from data import holistic_features as hf  # noqa: E402
from data import holistic_landmarks as hl  # noqa: E402
from data.word_dataset import (  # noqa: E402
    WordDataset, load_or_compute_sign_properties, load_word_index,
)
from evaluate_words import evaluate_split, load_word_checkpoint  # noqa: E402
from models.words import build_word_model  # noqa: E402
from train import set_seed  # noqa: E402


def model_kwargs(args) -> dict:
    if args.model == "gru":
        return dict(hidden_size=args.dim, dropout=args.dropout)
    return dict(dim=args.dim, dropout=args.dropout)


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None,
              amp=False):
    train = optimizer is not None
    model.train(train)
    total, correct, top5, loss_sum = 0, 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, m, y in loader:
            x, m, y = x.to(device, non_blocking=True), m.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x, m)
                loss = criterion(logits.float(), y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            n = len(y)
            total += n
            loss_sum += loss.item() * n
            correct += (logits.argmax(1) == y).sum().item()
            k = min(5, logits.shape[1])
            top5 += (logits.topk(k, 1).indices == y[:, None]).any(1).sum().item()
    return loss_sum / total, correct / total, top5 / total


def main():
    parser = argparse.ArgumentParser(description="Train a GISLR word model")
    parser.add_argument("--model", choices=["gru", "transformer"], default="transformer")
    parser.add_argument("--features", choices=list(hf.FEATURE_SETS), default="full")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=None,
                        help="model width (default 192 transformer, 128 gru)")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=hf.MAX_LEN)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--cache-dir", type=Path, default=config.GISLR_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="write checkpoint and results here instead of checkpoints/ "
                             "and results/ (smoke tests)")
    args = parser.parse_args()
    if args.dim is None:
        args.dim = 192 if args.model == "transformer" else 128

    set_seed()
    device = config.DEVICE
    amp = device.type == "cuda" and not args.no_amp
    ckpt_dir = args.out_dir or config.CHECKPOINT_DIR
    res_dir = args.out_dir or config.RESULTS_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    index, classes = load_word_index(args.cache_dir)
    tr_idx, va_idx = index[index["split"] == "train"], index[index["split"] == "val"]
    split_participants = {s: sorted(int(p) for p in g["participant_id"].unique())
                          for s, g in index.groupby("split")}
    print(f"[init] device {device}, amp {amp}")
    print(f"[words] {len(classes)} signs; train {len(tr_idx)}  val {len(va_idx)} sequences; "
          f"participants {({k: len(v) for k, v in split_participants.items()})}")

    train_ds = WordDataset(tr_idx, args.cache_dir, args.features, args.max_len, augment=True)
    val_ds = WordDataset(va_idx, args.cache_dir, args.features, args.max_len, augment=False)
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=len(train_ds) > args.batch_size, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, **loader_kw)

    model = build_word_model(args.model, train_ds.n_channels, len(classes), args.max_len,
                             **model_kwargs(args))
    t0 = time.time()
    model.set_normalisation(*train_ds.feature_stats())
    print(f"[words] normalisation stats in {time.time() - t0:.1f}s; "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(train_loader)
    warm = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    run = f"words_{args.model}_{args.features}{('_' + args.tag) if args.tag else ''}"
    ckpt_path = ckpt_dir / f"{run}_best.pt"
    history, best = [], (-1.0, float("inf"))
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer, scheduler, scaler, amp)
        va = run_epoch(model, val_loader, criterion, device, amp=amp)
        history.append(dict(epoch=epoch, train_loss=tr[0], train_acc=tr[1], val_loss=va[0],
                            val_acc=va[1], val_top5=va[2], lr=scheduler.get_last_lr()[0]))
        print(f"epoch {epoch:>3}/{args.epochs}  train {tr[0]:.3f}/{tr[1]:.3f}  "
              f"val {va[0]:.3f}/{va[1]:.3f} top5 {va[2]:.3f}  ({time.time() - t0:.0f}s)",
              flush=True)
        if (va[1], -va[0]) > (best[0], -best[1]):
            best = (va[1], va[0])
            torch.save({
                "model_state": model.state_dict(), "epoch": epoch,
                "val_acc": va[1], "val_top5": va[2], "val_loss": va[0],
                "classes": classes, "model": args.model, "model_kwargs": model_kwargs(args),
                "feature_version": hf.FEATURE_VERSION, "feature_set": args.features,
                "n_channels": train_ds.n_channels, "max_len": args.max_len,
                "landmark_spec_hash": hl.LANDMARK_SPEC_HASH,
                "normalisation": {"kind": "per_channel_standardise_train_split",
                                  "mean": model.feat_mean.cpu().tolist(),
                                  "std": model.feat_std.cpu().tolist()},
                "split_participants": split_participants, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            }, ckpt_path)
            print(f"          saved -> {ckpt_path.name}", flush=True)
        (res_dir / f"{run}_history.json").write_text(json.dumps(history, indent=2))

    print(f"\n[done] best val acc {best[0]:.4f} (loss {best[1]:.4f}) -> {ckpt_path}")
    best_model, ckpt = load_word_checkpoint(ckpt_path, device)
    props = load_or_compute_sign_properties(index, classes, args.cache_dir,
                                            res_dir / "words_sign_properties.csv")
    report = evaluate_split(best_model, ckpt, index, "val", args.cache_dir, device, props,
                            workers=args.workers,
                            fig_path=res_dir / f"{run}_val_confusion.png")
    report.update(run=run, epoch=ckpt["epoch"], model=args.model, features=args.features,
                  n_channels=train_ds.n_channels)
    (res_dir / f"{run}_val_eval.json").write_text(json.dumps(report, indent=2))
    print(f"[done] val acc {report['acc']:.4f} top5 {report['top5']:.4f}; "
          f"breakdowns {json.dumps(report['breakdowns'])}")
    print(f"[done] -> {res_dir / f'{run}_val_eval.json'}")


if __name__ == "__main__":
    main()
