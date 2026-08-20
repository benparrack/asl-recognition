"""
Live webcam demo.

Run:
    python src/demo.py --checkpoint checkpoints/cnn_smoketest_best.pt

Controls:
    q       quit
    c       clear the sentence
    SPACE   append the current letter to the sentence
    d       toggle the debug view (shows the model's actual input)

WHY THE DEBUG VIEW MATTERS
==========================
Press 'd' and you see the exact 128x128 crop being fed to the network. If live
accuracy is worse than your test accuracy, look at that panel first. Nine times
out of ten the crop looks different from your training images -- different
framing, different colour, hand too small in frame -- and that mismatch is
train/serve skew, not a model failure.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from data.dataset import build_transforms  # noqa: E402
from data.landmarks import build_detector, detect_and_crop  # noqa: E402


class PredictionSmoother:
    """
    Majority vote over a sliding window of recent frames.

    Raw per-frame predictions flicker badly -- at 30 fps a bare argmax produces
    an unreadable strobe. Voting over ~15 frames (half a second) turns that into
    something a person can read.

    Low-confidence frames append a sentinel (-1) rather than being ignored, so
    that when your hand leaves the frame the buffer actually drains and the
    display clears instead of freezing on the last confident guess.
    """

    def __init__(self, window: int = 15, min_confidence: float = 0.6):
        self.buffer: deque[int] = deque(maxlen=window)
        self.min_confidence = min_confidence

    def update(self, class_idx: int | None, confidence: float = 0.0) -> str | None:
        if class_idx is None or confidence < self.min_confidence:
            self.buffer.append(-1)
        else:
            self.buffer.append(class_idx)

        if len(self.buffer) < self.buffer.maxlen // 2:
            return None

        values, counts = np.unique(self.buffer, return_counts=True)
        winner = int(values[counts.argmax()])

        return None if winner < 0 else config.IDX_TO_CLASS[winner]


def load_model(checkpoint_path: str, device: torch.device):
    """Rebuild the architecture recorded in the checkpoint and load its weights."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    if ckpt.get("classes") != config.CLASSES:
        raise RuntimeError(
            f"Checkpoint was trained on {ckpt.get('classes')}, but config.CLASSES "
            f"is {config.CLASSES}. Label indices would not line up."
        )

    name = ckpt["args"]["model"]
    if name == "cnn":
        from models.cnn import ASLNet
        model = ASLNet()
    elif name == "transfer":
        from models.baselines import TransferNet
        model = TransferNet()
    elif name == "mlp":
        raise ValueError("the landmark MLP needs a different demo path")
    else:
        raise ValueError(f"unknown model: {name}")

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()   # dropout off, BatchNorm on running stats -- essential

    print(f"[demo] {name} from epoch {ckpt['epoch']}, val acc {ckpt['val_acc']:.4f}")
    return model


def draw_overlay(frame, bbox, letter, confidence, sentence, fps):
    """All the cv2 drawing, kept out of the main loop for readability."""
    h, w = frame.shape[:2]

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        colour = (0, 200, 0) if letter else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        if letter:
            cv2.putText(frame, f"{letter}  {confidence:.0%}", (x1, max(28, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    else:
        cv2.putText(frame, "no hand detected", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)

    # Sentence bar along the bottom.
    cv2.rectangle(frame, (0, h - 56), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "".join(sentence) or "(empty)", (12, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(frame, f"{fps:.0f} fps", (w - 96, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window", type=int, default=15,
                        help="frames to vote over")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--padding", type=float, default=0.25,
                        help="MUST match what preprocess.py used")
    args = parser.parse_args()

    device = config.DEVICE
    model = load_model(args.checkpoint, device)

    # static_mode=False enables tracking between frames: faster and steadier on
    # video than treating every frame as an unrelated image.
    detector = build_detector(static_mode=False)

    # The SAME transforms validation used -- no augmentation, same resize, same
    # normalisation. Importing build_transforms rather than rewriting it here
    # guarantees they cannot drift apart.
    transform = build_transforms(train=False)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. Check `ls /dev/video*` and that "
            "your user is in the 'video' group."
        )

    smoother = PredictionSmoother(args.window, args.min_confidence)
    sentence: list[str] = []
    show_debug = False
    letter, confidence = None, 0.0
    fps, last_t = 0.0, time.time()

    print("[demo] q quit | c clear | SPACE append | d debug view")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror -- capture_data.py also flipped, so this keeps handedness
        # consistent with training. Remove it in one place and you must remove
        # it in both.
        frame = cv2.flip(frame, 1)

        crop, bbox = detect_and_crop(frame, detector, padding=args.padding)

        if crop is not None:
            # BGR (OpenCV) -> RGB PIL, matching how ASLImageDataset loaded
            # training images. Skip the colour conversion and every channel is
            # swapped relative to training.
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensor = transform(pil).unsqueeze(0).to(device)   # add batch dim

            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0]

            conf, idx = probs.max(dim=0)
            confidence = conf.item()
            letter = smoother.update(int(idx.item()), confidence)
        else:
            letter = smoother.update(None)
            confidence = 0.0

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-6)   # smoothed
        last_t = now

        frame = draw_overlay(frame, bbox, letter, confidence, sentence, fps)
        cv2.imshow("ASL Fingerspelling", frame)

        if show_debug and crop is not None:
            cv2.imshow("model input", cv2.resize(crop, (256, 256)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence.clear()
        elif key == ord(" ") and letter:
            sentence.append(letter[0] if len(letter) == 1 else letter + " ")
        elif key == ord("d"):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow("model input")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
