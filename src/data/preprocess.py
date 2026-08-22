"""
Turn raw captured frames into cropped, model-ready images.

Run:
    python src/data/preprocess.py                 # square-padded crops (default)
    python src/data/preprocess.py --crop clamp    # the clamping variant
    python src/data/preprocess.py --overwrite     # redo everything

Reads  data/raw/raw_metadata.csv
Writes data/processed/<same relative path>  and  data/processed/metadata.csv

WHY THIS SCRIPT EXISTS SEPARATELY FROM CAPTURE
==============================================
Capture is cheap to redo; cropping is not. Keeping raw frames untouched means
you can reprocess with different padding, a different crop function, or a newer
MediaPipe version without ever recapturing. Raw data is the ground truth you
never overwrite -- exactly the "keep the original, transform into a copy"
discipline the Chapter 2 workflow describes.

ON DETECTION FAILURES
=====================
MediaPipe will not find a hand in every frame. Those frames produce no crop and
are dropped from metadata.csv, which is correct -- but the RATE is a number you
should record and report. If detection fails on 6% of frames, your end-to-end
system has a 6% error floor no matter how good the classifier becomes. That is a
real property of your pipeline, not a bug to hide.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import config  # noqa: E402
from data.landmarks import build_detector, crop_hand, crop_hand_square  # noqa: E402


# ---------------------------------------------------------------- (1) helper
 
def letterbox_square(image, size: int, fill=(0, 0, 0)):
    """
    Resize to a square WITHOUT distorting the aspect ratio.
 
    The images in this dataset are already tight hand crops, at sizes like
    84x126 and 155x86. Two ways to make those square:
 
      - cv2.resize() straight to (size, size): STRETCHES. A 155x86 image
        squashed to square makes a wide hand look narrow, and a 64x146 one makes
        a narrow hand look wide. The model would then have to learn handshapes
        across arbitrary distortions -- and worse, the distortion correlates
        with how the original was framed, so it is a spurious signal.
 
      - letterboxing (this function): scale the longer side to `size`, keep the
        aspect ratio, and pad the shorter side with a constant colour. Geometry
        is preserved; the padding is a benign, consistent artifact.
 
    Letterboxing is the standard choice for object detection pipelines for
    exactly this reason.
    """
    h, w = image.shape[:2]
    scale = size / max(h, w)
 
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
 
    # INTER_AREA for shrinking (averages the source region, avoids aliasing),
    # INTER_LINEAR for enlarging.
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
 
    pad_w = size - new_w
    pad_h = size - new_h
    top = pad_h // 2
    left = pad_w // 2
 
    return cv2.copyMakeBorder(
        resized,
        top, pad_h - top,
        left, pad_w - left,
        cv2.BORDER_CONSTANT, value=fill,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop raw frames into processed images")
    parser.add_argument("--crop", default="square", choices=["square", "clamp"],
                        help="square = pad at frame edges (no distortion); "
                             "clamp = truncate at frame edges (may be non-square)")
    parser.add_argument("--padding", type=float, default=0.25,
                        help="fraction of hand size to add as margin")
    parser.add_argument("--resize", type=int, default=256,
                        help="store crops at this size; 0 keeps native size")
    parser.add_argument("--overwrite", action="store_true",
                        help="reprocess images that already exist")
    parser.add_argument("--no-detect", action="store_true",
                        help="images are ALREADY hand crops: skip MediaPipe and "
                             "just letterbox to square. Use for the Pugeault "
                             "fingerspelling dataset, where running a detector "
                             "on pre-cropped images fails ~30%% of the time.")
    args = parser.parse_args()

    crop_fn = crop_hand_square if args.crop == "square" else crop_hand

    raw = pd.read_csv(config.RAW_METADATA_CSV)
    print(f"[preprocess] {len(raw)} raw frames")
    print(f"[preprocess] crop={args.crop} padding={args.padding} resize={args.resize}")

    # Build the detector ONCE. Constructing it per image works but is roughly an
    # order of magnitude slower -- it reloads the model graph every time.
    detector = None if args.no_detect else build_detector(static_mode=True)

    kept: list[dict] = []
    failed: list[dict] = []
    skipped = 0
    t0 = time.time()

    for i, row in raw.iterrows():
        src = config.RAW_DIR / row["path"]
        dst = config.PROCESSED_DIR / row["path"]

        if dst.exists() and not args.overwrite:
            kept.append(dict(row))
            skipped += 1
            continue

        image = cv2.imread(str(src))
        if image is None:
            failed.append(dict(row, reason="unreadable"))
            continue

        if args.no_detect:
            # Already a hand crop: preserve geometry, pad to square.
            crop = letterbox_square(image, args.resize or 256)
        else:
            crop = crop_fn(image, detector, padding=args.padding)
            if crop is None:
                failed.append(dict(row, reason="no_hand_detected"))
                continue
            if args.resize:
                crop = cv2.resize(crop, (args.resize, args.resize),
                                  interpolation=cv2.INTER_AREA)

        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), crop)
        kept.append(dict(row))

        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(raw)}  ({rate:.1f} img/s)")

    # ---------------------------------------------------------------- outputs
    processed = pd.DataFrame(kept)
    processed.to_csv(config.METADATA_CSV, index=False)

    print(f"\n[preprocess] wrote {len(processed)} rows -> {config.METADATA_CSV}")
    if skipped:
        print(f"[preprocess] skipped {skipped} already-processed "
              f"(use --overwrite to redo)")

    if failed:
        fail_df = pd.DataFrame(failed)
        fail_path = config.PROCESSED_DIR / "failures.csv"
        fail_df.to_csv(fail_path, index=False)

        rate = len(failed) / len(raw)
        print(f"[preprocess] FAILED on {len(failed)} frames ({rate:.1%}) -> {fail_path}")
        print("\n  failures by class:")
        print(fail_df.groupby(["label", "reason"]).size().to_string())
        print(f"\n  RECORD THIS: {rate:.1%} detection failure is your pipeline's "
              f"error floor.\n  Write it in NOTES.md; it belongs in the report.")
    else:
        print("[preprocess] no detection failures")

    print("\n  images per class per session:")
    print(processed.groupby(["signer_id", "label"]).size().to_string())


if __name__ == "__main__":
    main()
