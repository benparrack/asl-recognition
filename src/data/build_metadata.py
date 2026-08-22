"""
Build raw_metadata.csv from a folder-structured image dataset.

Two layouts are supported:

  --layout flat      <source>/<LABEL>/image.jpg
                     e.g. Kaggle ASL Alphabet. One signer, unknown identity.

  --layout nested    <source>/<SIGNER>/<LABEL>/image.png
                     e.g. Pugeault & Bowden fingerspelling (dataset5/A/a/...).
                     Signer identity is REAL here, which is what makes an
                     honest signer-disjoint split possible.

Examples:
    # Pugeault-style, RGB frames only, 300 per class
    python src/data/build_metadata.py \
        --source data/raw/dataset5 --layout nested \
        --pattern 'color_*' --per-class 300 --force

    # Kaggle ASL Alphabet
    python src/data/build_metadata.py \
        --source data/raw/asl_alphabet_train/asl_alphabet_train \
        --layout flat --per-class 400 --force

WHY --pattern MATTERS
=====================
Depth-sensor datasets store colour and depth frames side by side in the same
folder. A depth map is a greyscale image of distance, not a photograph -- feed
those to a model trained on RGB and you are training on two different modalities
mixed together. MediaPipe will also usually fail to find a hand in them, which
would inflate your detection failure rate for no real reason.

'color_*' keeps only the RGB frames. Check the actual filenames in your dataset
before trusting any default.

ON --per-class WITH NESTED LAYOUTS
==================================
The cap applies per (signer, label) pair, not per label overall. With 5 signers
and --per-class 300 you get up to 1500 images per letter, balanced across
signers -- which is what you want, since a per-label cap could silently draw
almost everything from one person.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import config  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def collect(class_dir: Path, pattern: str) -> list[Path]:
    """Image files in one class folder, filtered by suffix and filename glob."""
    return sorted(
        p for p in class_dir.iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES and fnmatch.fnmatch(p.name, pattern)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build raw_metadata.csv from class folders")
    parser.add_argument("--source", required=True)
    parser.add_argument("--layout", default="flat", choices=["flat", "nested"],
                        help="flat: <source>/<LABEL>/  nested: <source>/<SIGNER>/<LABEL>/")
    parser.add_argument("--pattern", default="*",
                        help="filename glob, e.g. 'color_*' to exclude depth frames")
    parser.add_argument("--per-class", type=int, default=0,
                        help="max images per class (per signer when nested); 0 = all")
    parser.add_argument("--signer-id", default="unknown",
                        help="only used with --layout flat")
    parser.add_argument("--session-id", default="default")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.is_dir():
        raise SystemExit(f"not a directory: {source}")

    out_path = Path(args.output) if args.output else config.RAW_METADATA_CSV
    if out_path.exists() and not args.force:
        raise SystemExit(
            f"{out_path} exists. Back it up and pass --force, or use --output:\n"
            f"  cp {out_path} {out_path}.backup"
        )

    rows: list[dict] = []
    found_labels: set[str] = set()
    skipped: list[str] = []

    # ------------------------------------------------------------ walk
    if args.layout == "nested":
        signer_dirs = sorted(p for p in source.iterdir() if p.is_dir())
        if not signer_dirs:
            raise SystemExit(f"no subdirectories under {source}")

        for signer_dir in signer_dirs:
            for class_dir in sorted(p for p in signer_dir.iterdir() if p.is_dir()):
                label = class_dir.name.upper()
                if label not in config.CLASSES:
                    skipped.append(f"{signer_dir.name}/{class_dir.name}")
                    continue

                found_labels.add(label)
                for img in collect(class_dir, args.pattern):
                    rows.append({
                        "path": str(img.relative_to(config.RAW_DIR)),
                        "label": label,
                        "signer_id": signer_dir.name,
                        "session_id": f"{signer_dir.name}_{args.session_id}",
                    })
    else:
        for class_dir in sorted(p for p in source.iterdir() if p.is_dir()):
            label = class_dir.name.upper()
            if label not in config.CLASSES:
                skipped.append(class_dir.name)
                continue

            found_labels.add(label)
            for img in collect(class_dir, args.pattern):
                rows.append({
                    "path": str(img.relative_to(config.RAW_DIR)),
                    "label": label,
                    "signer_id": args.signer_id,
                    "session_id": args.session_id,
                })

    if not rows:
        raise SystemExit(
            f"No images matched under {source}.\n"
            f"  layout={args.layout} pattern={args.pattern!r}\n"
            f"  folders seen: {skipped[:20]}\n"
            f"  config.CLASSES = {config.CLASSES}"
        )

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------ subsample
    if args.per_class:
        keys = ["signer_id", "label"] if args.layout == "nested" else ["label"]
        chunks = [
            group.sample(min(len(group), args.per_class), random_state=args.seed)
            for _, group in df.groupby(keys, sort=False)
        ]
        df = pd.concat(chunks).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # ------------------------------------------------------------ report
    print(f"[metadata] {len(df)} rows -> {out_path}")
    print(f"[metadata] labels: {len(found_labels)}/{len(config.CLASSES)}")
    print(f"[metadata] signers: {sorted(df['signer_id'].unique())}")

    missing = sorted(set(config.CLASSES) - found_labels)
    if missing:
        print(f"[metadata] MISSING labels: {missing}")
    if skipped:
        print(f"[metadata] ignored {len(skipped)} folders, e.g. {skipped[:8]}")

    print("\n  images per signer:")
    print(df["signer_id"].value_counts().sort_index().to_string())

    print("\n  images per label:")
    print(df["label"].value_counts().sort_index().to_string())

    n_signers = df["signer_id"].nunique()
    if n_signers >= 3:
        print(f"\n  {n_signers} signers -> split_by_signer() will work. "
              f"Train with the default --split signer.")
    else:
        print(f"\n  WARNING: only {n_signers} signer(s). split_by_signer() cannot "
              f"give a meaningful held-out set on this data alone.")


if __name__ == "__main__":
    main()
