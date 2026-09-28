"""
Live webcam demo.

Run:
    python src/demo.py --checkpoint checkpoints/cnn_smoketest_best.pt

Controls:
    q       quit
    c       clear the sentence
    SPACE   append the current letter to the sentence
    s       append a literal space (word separator) to the sentence
    d       toggle the debug view (shows the model's actual input)

MOTION LETTERS (J, Z), via --motion-checkpoint
================================================
Pass --motion-checkpoint checkpoints/motion_gru_best.pt (only meaningful when
the primary --checkpoint is the landmark MLP) to also recognise J and Z. This
is the gated two-stage design: the static classifier runs as normal every
frame, but a simple motion detector (frame-to-frame displacement of the hand's
landmark centroid) watches for an actual gesture in progress. When it fires,
static classification pauses, landmark vectors are buffered into a clip, and
once the motion settles the small motion GRU classifies the whole clip as J or
Z in one shot -- no majority-vote smoothing needed there, it's already one
decision per gesture, not a noisy per-frame stream.

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
from data.landmarks import (  # noqa: E402
    build_detector, detect_and_crop, letterbox_square,
    extract_landmarks, normalise_landmarks,
)


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
    """
    Rebuild the architecture recorded in the checkpoint and load its weights.

    Returns (model, kind), where kind is "image" (feed a cropped/transformed
    pixel tensor) or "landmark" (feed a normalised 63-vector) -- same split the
    training/eval code uses to pick a dataloader.
    """
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
        kind = "image"
    elif name == "transfer":
        from models.baselines import TransferNet
        model = TransferNet()
        kind = "image"
    elif name == "mlp":
        from models.baselines import LandmarkMLP
        model = LandmarkMLP()
        kind = "landmark"
    else:
        raise ValueError(f"unknown model: {name}")

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()   # dropout off, BatchNorm on running stats -- essential

    print(f"[demo] {name} ({kind}) from epoch {ckpt['epoch']}, val acc {ckpt['val_acc']:.4f}")
    return model, kind


def load_motion_model(checkpoint_path: str, device: torch.device):
    """Load the J/Z GRU. Returns (model, max_len) -- max_len must be used to
    pad/truncate live clips the exact same way training data was."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    if ckpt.get("classes") != config.MOTION_LETTERS:
        raise RuntimeError(
            f"Motion checkpoint was trained on {ckpt.get('classes')}, but "
            f"config.MOTION_LETTERS is {config.MOTION_LETTERS}."
        )

    from models.motion import MotionGRU
    model = MotionGRU(hidden_size=ckpt["args"]["hidden_size"], dropout=ckpt["args"]["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    max_len = ckpt["max_len"]
    print(f"[demo] motion gru from epoch {ckpt['epoch']}, val acc {ckpt['val_acc']:.4f}, "
          f"max_len={max_len}")
    return model, max_len


def classify_motion_clip(model, buffer: list, max_len: int, device: torch.device):
    """Pad/truncate a buffered list of normalised landmark vectors the same
    way MotionSequenceDataset does, then classify the whole clip in one shot."""
    seq = np.array(buffer, dtype=np.float32)   # (T, 63)
    T = seq.shape[0]
    if T >= max_len:
        seq = seq[:max_len]
    else:
        pad = np.zeros((max_len - T, seq.shape[1]), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)

    tensor = torch.from_numpy(seq).unsqueeze(0).to(device)   # (1, max_len, 63)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    conf, idx = probs.max(dim=0)
    return config.MOTION_IDX_TO_CLASS[int(idx.item())], conf.item()


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
    parser.add_argument("--motion-checkpoint", default=None,
                        help="optional J/Z GRU checkpoint; only meaningful when "
                             "--checkpoint is the landmark MLP")
    parser.add_argument("--motion-threshold", type=float, default=0.025,
                        help="frame-to-frame hand-centroid displacement that counts as motion")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window", type=int, default=15,
                        help="frames to vote over")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--padding", type=float, default=0.25,
                        help="MUST match what preprocess.py used")
    args = parser.parse_args()

    device = config.DEVICE
    model, kind = load_model(args.checkpoint, device)

    motion_model, motion_max_len = None, None
    if args.motion_checkpoint:
        if kind != "landmark":
            raise ValueError("--motion-checkpoint requires a landmark (--checkpoint mlp) primary model")
        motion_model, motion_max_len = load_motion_model(args.motion_checkpoint, device)

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

    # Motion-gate state (only used when motion_model is not None).
    MOTION_MIN_LOW_STREAK = 3     # consecutive low-displacement frames -> gesture ended
    MOTION_MIN_FRAMES = 8         # fewer detected frames than this -> not a real clip, discard
    MOTION_HOLD_SECONDS = 1.5     # how long a resolved J/Z stays on screen / append-able
    prev_centroid = None
    recording_motion = False
    motion_buffer: list = []
    low_motion_streak = 0
    motion_hold_until = 0.0

    print("[demo] q quit | c clear | SPACE append | s space | d debug view")
    if motion_model is not None:
        print("[demo] motion gate active -- move your hand through a J or Z to trigger it")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror -- capture_data.py also flipped, so this keeps handedness
        # consistent with training. Remove it in one place and you must remove
        # it in both.
        frame = cv2.flip(frame, 1)

        crop = None

        if kind == "image":
            crop, bbox = detect_and_crop(frame, detector, padding=args.padding, square=False)

            if crop is not None:
                # BGR (OpenCV) -> RGB PIL, matching how ASLImageDataset loaded
                # training images. Skip the colour conversion and every channel is
                # swapped relative to training.
                crop = letterbox_square(crop, 256)
                crop = cv2.resize(crop, (110, 110), interpolation=cv2.INTER_AREA)
                crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
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

        else:  # kind == "landmark"
            # MediaPipe finds the hand directly in the full frame -- no crop step
            # needed, unlike the pixel models. One detector.process() call per
            # frame gives us both the keypoints (for the model) and a bounding
            # box (for the overlay), from the same detection.
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw = extract_landmarks(detector, image_rgb)

            if raw is not None:
                h, w = frame.shape[:2]
                xs = raw[:, 0] * w
                ys = raw[:, 1] * h
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            else:
                bbox = None

            now_t = time.time()
            in_hold = motion_model is not None and now_t < motion_hold_until

            if motion_model is not None and not in_hold:
                # Is a J/Z gesture in progress? Measured as frame-to-frame
                # displacement of the hand landmarks' centroid -- translation
                # through space, which both letters' traced paths produce and
                # a held static letter mostly doesn't.
                centroid = raw[:, :2].mean(axis=0) if raw is not None else None
                displacement = (np.linalg.norm(centroid - prev_centroid)
                                if centroid is not None and prev_centroid is not None else 0.0)
                if centroid is not None:
                    prev_centroid = centroid

                if not recording_motion and raw is not None and displacement > args.motion_threshold:
                    recording_motion = True
                    motion_buffer = [normalise_landmarks(raw)]
                    low_motion_streak = 0
                    print(f"[motion] triggered, displacement {displacement:.4f}")

                elif recording_motion:
                    if raw is not None:
                        motion_buffer.append(normalise_landmarks(raw))
                        low_motion_streak = (0 if displacement > args.motion_threshold
                                             else low_motion_streak + 1)
                    else:
                        low_motion_streak += 1

                    # A brief natural pause mid-gesture can hit MOTION_MIN_LOW_STREAK
                    # before the clip has captured much of anything -- don't let the
                    # low-streak path end things until there's already a minimum
                    # viable clip; only the max-length cutoff can end a short one.
                    gesture_done = (
                        (low_motion_streak >= MOTION_MIN_LOW_STREAK
                         and len(motion_buffer) >= MOTION_MIN_FRAMES)
                        or len(motion_buffer) >= motion_max_len
                    )
                    if gesture_done:
                        if len(motion_buffer) >= MOTION_MIN_FRAMES:
                            pred, conf = classify_motion_clip(
                                motion_model, motion_buffer, motion_max_len, device)
                            if conf >= args.min_confidence:
                                letter, confidence = pred, conf
                                motion_hold_until = now_t + MOTION_HOLD_SECONDS
                                print(f"[motion] {pred} accepted, confidence {conf:.2f}, "
                                      f"{len(motion_buffer)} frames")
                            else:
                                print(f"[motion] {pred} REJECTED, confidence {conf:.2f} "
                                      f"< min-confidence {args.min_confidence}, "
                                      f"{len(motion_buffer)} frames")
                        else:
                            print(f"[motion] clip too short ({len(motion_buffer)} frames "
                                  f"< {MOTION_MIN_FRAMES}), discarded")
                        recording_motion = False
                        motion_buffer = []
                        low_motion_streak = 0

            skip_static = recording_motion or in_hold

            if not skip_static:
                if raw is not None:
                    vec = normalise_landmarks(raw)
                    tensor = torch.from_numpy(vec).unsqueeze(0).to(device)   # (1, 63)

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
        if recording_motion:
            cv2.putText(frame, f"recording J/Z... ({len(motion_buffer)})",
                        (12, frame.shape[0] - 68), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 165, 255), 2)
        cv2.imshow("ASL Fingerspelling", frame)

        if show_debug and crop is not None:
            cv2.imshow("model input", crop)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence.clear()
        elif key == ord(" ") and letter:
            sentence.append(letter[0] if len(letter) == 1 else letter + " ")
        elif key == ord("s"):
            sentence.append(" ")
        elif key == ord("d"):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow("model input")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
