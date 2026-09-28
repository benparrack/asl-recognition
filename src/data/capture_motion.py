"""
Capture short landmark-SEQUENCE clips for the motion letters (J, Z).

Run:
    python src/data/capture_motion.py --signer ben --session couch_evening

Controls:
    SPACE   start recording one clip (fixed duration, default 1.0s)
    r       delete the most recently saved clip for the current letter, redo it
    n       skip to the next letter (keeps whatever was captured so far)
    q       quit early (keeps everything captured so far)

WHY THIS IS A DIFFERENT SCRIPT FROM capture_alphabet.py
=========================================================
The static letters are one handshape, one frame -- capture_alphabet.py saves a
still image per sample. J and Z are defined by MOTION (a traced hook, a traced
zigzag), so a single frame can't represent them at all. This script instead
extracts a MediaPipe landmark vector on every frame for a short recording window
and saves the whole SEQUENCE as one clip -- shape (T, 63), T varying a little
clip to clip since camera frame timing isn't perfectly deterministic.

No raw images are saved here (unlike the static pipeline's raw/ -> processed/
split) -- there's no future need to re-derive pixel crops from these clips,
since the motion classifier this feeds is landmark-based from the start.

Each clip is its own .npz under data/motion/<signer>/<session>/<label>/, and
every clip gets a row in config.MOTION_METADATA_CSV.

FUTURE OPTION, not done here: capturing a third "NONE"/incidental-movement
class (hand moving but not actually signing J or Z) would make whatever
downstream classifier this feeds more robust against false triggers. Left out
for now to keep this first pass to exactly what was asked -- J and Z.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import config  # noqa: E402
from data.landmarks import build_detector, extract_landmarks, normalise_landmarks  # noqa: E402

try:
    import cv2
except ImportError:
    print("pip install opencv-python")
    raise


MOTION_METADATA_HEADER = ["path", "label", "signer_id", "session_id", "clip_id", "n_frames"]
MIN_FRAMES = 5   # clips with fewer detected frames than this are auto-discarded


def append_motion_metadata(rows: list[dict]) -> None:
    exists = config.MOTION_METADATA_CSV.exists()
    with open(config.MOTION_METADATA_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MOTION_METADATA_HEADER)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def draw_status(display, text_lines, colour=(255, 255, 255)):
    for i, line in enumerate(text_lines):
        cv2.putText(display, line, (12, 40 + i * 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, colour, 2)


def record_clip(cap, detector, duration: float, header_lines) -> np.ndarray:
    """Record for `duration` seconds, returning a (T, 63) array of every frame
    where a hand was actually detected. T can be 0 if nothing was detected."""
    landmarks_seq: list[np.ndarray] = []
    t_start = time.time()

    while time.time() - t_start < duration:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw = extract_landmarks(detector, image_rgb)
        if raw is not None:
            landmarks_seq.append(normalise_landmarks(raw))
            h, w = frame.shape[:2]
            xs, ys = raw[:, 0] * w, raw[:, 1] * h
            cv2.rectangle(display, (int(xs.min()), int(ys.min())),
                          (int(xs.max()), int(ys.max())), (0, 0, 220), 2)

        elapsed = time.time() - t_start
        draw_status(display, header_lines + [
            f"RECORDING  {elapsed:.1f}/{duration:.1f}s   frames: {len(landmarks_seq)}",
        ], colour=(0, 0, 220))
        cv2.imshow("capture", display)
        cv2.waitKey(1)

    return np.array(landmarks_seq, dtype=np.float32)


def wait_for_key(cap, header_lines) -> int:
    """Show a paused/ready frame until SPACE, n, r, or q is pressed."""
    while True:
        ok, frame = cap.read()
        if not ok:
            return ord("q")

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        draw_status(display, header_lines, colour=(0, 165, 255))
        cv2.imshow("capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord(" "), ord("n"), ord("r"), ord("q")):
            return key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture landmark-sequence clips for the motion letters (J, Z)")
    parser.add_argument("--signer", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--labels", nargs="+", default=config.MOTION_LETTERS,
                        help="letters to capture (default: config.MOTION_LETTERS, i.e. J Z)")
    parser.add_argument("--clips", type=int, default=15,
                        help="clips to record per letter")
    parser.add_argument("--duration", type=float, default=1.0,
                        help="seconds recorded per clip")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. On Linux check `ls /dev/video*` "
            "and that your user is in the 'video' group."
        )

    detector = build_detector(static_mode=False)   # tracking mode: steadier on video

    print(f"[capture] {len(args.labels)} letters x {args.clips} clips x "
          f"{args.duration}s, signer={args.signer} session={args.session}")
    print("[capture] SPACE record clip, r redo last clip, n next letter, q quit")

    all_rows: list[dict] = []
    quit_early = False

    for label_idx, label in enumerate(args.labels, start=1):
        out_dir = config.MOTION_DIR / args.signer / args.session / label
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_this_label: list[dict] = []
        clip_id = 0

        while len(saved_this_label) < args.clips:
            ready_lines = [
                f"[{label_idx}/{len(args.labels)}] motion letter: {label}",
                f"clip {len(saved_this_label)}/{args.clips}   SPACE to record",
            ]
            key = wait_for_key(cap, ready_lines)

            if key == ord("q"):
                quit_early = True
                break

            if key == ord("n"):
                break

            if key == ord("r"):
                if saved_this_label:
                    last = saved_this_label.pop()
                    (config.MOTION_DIR / last["path"]).unlink(missing_ok=True)
                    all_rows = [r for r in all_rows if r["path"] != last["path"]]
                    print(f"[capture] discarded {last['path']}")
                continue

            # SPACE
            rec_lines = [f"[{label_idx}/{len(args.labels)}] motion letter: {label}"]
            seq = record_clip(cap, detector, args.duration, rec_lines)

            if len(seq) < MIN_FRAMES:
                print(f"[capture] only {len(seq)} frames detected (need >= {MIN_FRAMES}) "
                      "-- discarded, try again")
                continue

            filename = f"{label}_{clip_id:04d}.npz"
            clip_path = out_dir / filename
            np.savez_compressed(clip_path, landmarks=seq)

            row = {
                "path": str(clip_path.relative_to(config.MOTION_DIR)),
                "label": label,
                "signer_id": args.signer,
                "session_id": args.session,
                "clip_id": clip_id,
                "n_frames": len(seq),
            }
            saved_this_label.append(row)
            all_rows.append(row)
            clip_id += 1
            print(f"[capture] {label} clip {len(saved_this_label)}/{args.clips}: "
                  f"{len(seq)} frames -> {clip_path.name}")

        if quit_early:
            break

    cap.release()
    cv2.destroyAllWindows()

    if all_rows:
        append_motion_metadata(all_rows)
        n_labels = len(set(r["label"] for r in all_rows))
        print(f"[capture] total saved: {len(all_rows)} clips across {n_labels} letters")
        print(f"[capture] metadata -> {config.MOTION_METADATA_CSV}")
    else:
        print("[capture] nothing saved")


if __name__ == "__main__":
    main()
