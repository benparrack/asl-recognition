"""
Live webcam demo -- the thing that makes people care about your project.

Run:  python src/demo.py --checkpoint checkpoints/cnn_best.pt

Build this EARLY, not the night before. It is the fastest way to discover that
your 97% test accuracy does not survive contact with your actual webcam, and
that discovery is much more useful in week 8 than in week 14.
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

# pip install opencv-python mediapipe
try:
    import cv2
except ImportError:
    cv2 = None

from data.landmarks import crop_hand

class PredictionSmoother:
    """
    Majority vote over a sliding window of recent frames.

    Raw per-frame predictions flicker badly -- at 30 fps a bare argmax produces
    an unreadable strobe of letters. Averaging over ~15 frames (half a second)
    turns that into something a person can actually read.

    This is also the seed of the Tier 2 extension: once you are reasoning over a
    window of frames rather than one, you are most of the way to handling J and Z,
    which need motion by definition.
    """

    def __init__(self, window: int = 15, min_confidence: float = 0.6):
        self.buffer: deque[int] = deque(maxlen=window)
        self.min_confidence = min_confidence

    def update(self, class_idx: int, confidence: float) -> str | None:
        if confidence < self.min_confidence:
            return None

        self.buffer.append(class_idx)
        if len(self.buffer) < self.buffer.maxlen // 2:
            return None

        values, counts = np.unique(self.buffer, return_counts=True)
        return config.IDX_TO_CLASS[int(values[counts.argmax()])]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window", type=int, default=15)
    args = parser.parse_args()

    if cv2 is None:
        raise ImportError("pip install opencv-python mediapipe")

    device = config.DEVICE
    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"[demo] loaded checkpoint (val acc {ckpt['val_acc']:.4f})")

    # TODO: rebuild the model, load_state_dict, model.eval()
    # TODO: build a MediaPipe detector with static_mode=False (tracking mode is
    #       faster and steadier on video than treating each frame independently)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. On Linux, check that your user "
            "is in the 'video' group and that /dev/video0 exists. `ls /dev/video*` "
            "and `v4l2-ctl --list-devices` are the tools for diagnosing this."
        )

    smoother = PredictionSmoother(window=args.window)
    sentence: list[str] = []

    print("[demo] q quits, c clears the sentence, space appends the current letter")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)   # mirror, so moving right moves right on screen

        # TODO:
        #   crop = crop_hand(frame, detector)
        #   if crop is not None:
        #       apply the SAME transforms as validation -- not the training
        #       augmentations, and do not forget the normalisation. Mismatched
        #       preprocessing between training and inference is the second most
        #       common cause of a demo that works on disk and fails live.
        #       run the model, softmax, take max -> (confidence, class_idx)
        #       letter = smoother.update(class_idx, confidence)
        #   draw the bounding box, current letter, confidence, and sentence

        cv2.imshow("ASL Fingerspelling", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            sentence.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
