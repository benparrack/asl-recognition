"""
Fine-tune a GISLR word checkpoint on your own recorded clips
(data/word_videos/<sign>/<session>_<nn>.mp4, from capture_words.py).

Run:
    python src/finetune_words.py --checkpoint checkpoints/words_transformer_full_best.pt
    python src/finetune_words.py --checkpoint ... --holdout none     # final model, all sessions

GISLR almost never has the second hand (NOTES.md), so two-handed signs need
data that does. Keeps the full 250-way head so the demo and every other sign
keep working; each epoch mixes your clips (repeated, augmented) with a random
sample of GISLR train sequences so the model doesn't forget the rest.

Split is by SESSION (the file-name prefix), never by clip: --holdout (default:
the last session alphabetically) is scored before and after fine-tuning and
never trained on. There's no separate val session, so the epoch count is fixed
rather than selected; don't tune it on the holdout and then quote the holdout.
GISLR val accuracy is reported before and after to show what forgetting costs.

Landmarks go through the live path (eval_word_videos.video_landmarks: Holistic
on the un-flipped frame, frame_from_holistic), resampled to REFERENCE_FPS, and
are cached as <clip>.npy next to each video, so only the first run is slow.

Writes checkpoints/<run>_ft.pt (same format as train_words.py, loadable by the
demo) and results/<run>_ft.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from data import holistic_features as hf  # noqa: E402
from data.word_dataset import WordDataset, load_word_index  # noqa: E402
from eval_word_videos import VIDEO_EXTS, video_landmarks  # noqa: E402
from evaluate_words import load_word_checkpoint  # noqa: E402
from train import set_seed  # noqa: E402
from train_words import run_epoch  # noqa: E402
from words_live import WordClassifier  # noqa: E402


def clip_landmarks(path: Path, holistic_factory) -> np.ndarray:
    """(T, 84, 3) at REFERENCE_FPS for one video, cached as <path>.npy."""
    cache = path.with_suffix(".npy")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return np.load(cache)
    with holistic_factory() as h:
        frames, times, _ = video_landmarks(path, h)
    seq = hf.resample_to_fps(np.stack(frames), times).astype(np.float32)
    np.save(cache, seq)
    return seq


def load_personal(root: Path, classes: list[str], holistic_complexity: int = 1) -> list[dict]:
    def factory():
        import mediapipe as mp
        return mp.solutions.holistic.Holistic(static_image_mode=False,
                                              model_complexity=holistic_complexity)

    items, skipped = [], []
    for sign_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if sign_dir.name not in classes:
            skipped.append(sign_dir.name)
            continue
        for path in sorted(p for p in sign_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS):
            items.append(dict(sign=sign_dir.name, label=classes.index(sign_dir.name),
                              session=path.stem.rsplit("_", 1)[0], file=str(path),
                              frames=clip_landmarks(path, factory)))
    if skipped:
        print(f"[skip] not in the checkpoint's classes: {skipped}")
    return items


class PersonalDataset(Dataset):
    """Personal clips in WordDataset's (features, mask, label) format."""

    def __init__(self, items, n_channels, max_len, augment=False, repeat=1):
        self.items, self.n_channels, self.max_len = items, n_channels, max_len
        self.augment, self.repeat = augment, repeat

    def __len__(self):
        return len(self.items) * self.repeat

    def __getitem__(self, idx):
        it = self.items[idx % len(self.items)]
        rng = np.random.default_rng() if self.augment else None
        feats, mask, _ = hf.sequence_to_features(it["frames"], self.max_len, rng)
        return torch.from_numpy(feats[:, :self.n_channels]), torch.from_numpy(mask), it["label"]


def score_personal(clf: WordClassifier, items: list[dict], signs: list[int]) -> dict:
    """Whole-clip top-1/top-3 over all classes, plus top-1 among only the
    recorded signs (how well it separates your 20 from each other)."""
    if not items:
        return {}
    probs = np.stack([clf.probs(it["frames"]) for it in items])
    y = np.array([it["label"] for it in items])
    top3 = np.argsort(-probs, 1)[:, :3]
    among = np.array(signs)[probs[:, signs].argmax(1)]
    return dict(n=len(items), top1=float((top3[:, 0] == y).mean()),
                top3=float((top3 == y[:, None]).any(1).mean()),
                top1_among_recorded=float((among == y).mean()),
                per_sign={clf.classes[s]: float((top3[y == s, 0] == s).mean())
                          for s in sorted(set(y))})


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a word model on personal clips")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dir", type=Path, default=config.DATA_DIR / "word_videos")
    parser.add_argument("--holdout", default=None,
                        help="session to hold out (default: last alphabetically); 'none' = train on all")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--repeat", type=int, default=20, help="personal clips per epoch = n x repeat")
    parser.add_argument("--replay", type=int, default=6000, help="GISLR train sequences per epoch")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--holistic-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=config.GISLR_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    set_seed()
    device = config.DEVICE
    amp = device.type == "cuda" and not args.no_amp
    ckpt_dir = args.out_dir or config.CHECKPOINT_DIR
    res_dir = args.out_dir or config.RESULTS_DIR

    model, ckpt = load_word_checkpoint(args.checkpoint, device)
    classes, max_len, n_ch = ckpt["classes"], ckpt["max_len"], ckpt["n_channels"]
    items = load_personal(args.dir, classes, args.holistic_complexity)
    sessions = sorted({it["session"] for it in items})
    if not items:
        sys.exit(f"no clips under {args.dir}/<sign>/")
    holdout = None if args.holdout == "none" else (args.holdout or sessions[-1])
    if holdout is not None and (holdout not in sessions or len(sessions) < 2):
        sys.exit(f"holdout {holdout!r} needs to be one of >= 2 sessions: {sessions}")
    train_items = [it for it in items if it["session"] != holdout]
    test_items = [it for it in items if it["session"] == holdout]
    signs = sorted({it["label"] for it in items})
    print(f"[ft] {len(items)} clips, {len(signs)} signs, sessions {sessions}; "
          f"holdout {holdout} ({len(test_items)} clips)")

    index, gislr_classes = load_word_index(args.cache_dir)
    if gislr_classes != classes:
        sys.exit("checkpoint classes differ from the GISLR cache")
    tr_idx, va_idx = index[index["split"] == "train"], index[index["split"] == "val"]
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(WordDataset(va_idx, args.cache_dir, ckpt["feature_set"], max_len),
                            batch_size=256, **loader_kw)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    clf = WordClassifier(model, ckpt, device)
    before = dict(holdout=score_personal(clf, test_items, signs),
                  gislr_val_acc=run_epoch(model, val_loader, criterion, device, amp=amp)[1])
    print(f"[ft] before: {json.dumps({k: v for k, v in before['holdout'].items() if k != 'per_sign'})} "
          f"gislr val {before['gislr_val_acc']:.4f}")

    personal_ds = PersonalDataset(train_items, n_ch, max_len, augment=True, repeat=args.repeat)
    rng = np.random.default_rng(config.SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_replay = min(args.replay, len(tr_idx))
    steps = args.epochs * math.ceil((len(personal_ds) + n_replay) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    history = []
    for epoch in range(1, args.epochs + 1):
        replay = WordDataset(tr_idx.iloc[rng.choice(len(tr_idx), n_replay, replace=False)],
                             args.cache_dir, ckpt["feature_set"], max_len, augment=True)
        loader = DataLoader(ConcatDataset([personal_ds, replay]), batch_size=args.batch_size,
                            shuffle=True, **loader_kw)
        tr = run_epoch(model, loader, criterion, device, optimizer, scheduler, scaler, amp)
        model.eval()
        ho = score_personal(clf, test_items, signs)
        history.append(dict(epoch=epoch, train_loss=tr[0], train_acc=tr[1],
                            holdout_top1=ho.get("top1")))
        print(f"epoch {epoch:>3}/{args.epochs}  train {tr[0]:.3f}/{tr[1]:.3f}  "
              f"holdout top1 {ho.get('top1', float('nan')):.3f}", flush=True)

    after = dict(holdout=score_personal(clf, test_items, signs),
                 gislr_val_acc=run_epoch(model, val_loader, criterion, device, amp=amp)[1])
    run = Path(args.checkpoint).stem.removesuffix("_best") + "_ft"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    out = dict(ckpt, model_state=model.state_dict(), val_acc=after["gislr_val_acc"],
               finetune=dict(base=str(args.checkpoint), holdout=holdout, sessions=sessions,
                             n_train_clips=len(train_items), epochs=args.epochs, lr=args.lr))
    torch.save(out, ckpt_dir / f"{run}.pt")
    report = dict(base=str(args.checkpoint), holdout=holdout, sessions=sessions,
                  train_sessions=sorted({it["session"] for it in train_items}),
                  n_train_clips=len(train_items), before=before, after=after, history=history,
                  args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (res_dir / f"{run}.json").write_text(json.dumps(report, indent=2))
    for name, r in (("before", before), ("after", after)):
        h = r["holdout"]
        print(f"[ft] {name:>6}: holdout top1 {h.get('top1', float('nan')):.3f} "
              f"top3 {h.get('top3', float('nan')):.3f} among-recorded "
              f"{h.get('top1_among_recorded', float('nan')):.3f}   gislr val {r['gislr_val_acc']:.4f}")
    print(f"[done] -> {ckpt_dir / f'{run}.pt'}, {res_dir / f'{run}.json'}")


if __name__ == "__main__":
    main()
