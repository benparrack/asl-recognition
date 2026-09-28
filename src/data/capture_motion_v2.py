"""
Capture motion-letter clips, v2: J, Z, and NONE, with RAW landmarks.

Run (one session; use a new --signer id per session so split_by_signer has
units to hold out -- ben_m1, ben_m2, ...):
    python src/data/capture_motion_v2.py --signer ben_m1 --session desk_day1

Controls:
    SPACE   start the countdown, then record one clip (default 2.0 s)
    r       delete the most recent clip for the current label, redo it
    n       skip to the next label (keeps what was captured)
    q       quit early (keeps what was captured)

WHAT CHANGED FROM v1, AND WHY
=============================
v1 (capture_motion.py) saved normalise_landmarks() output per frame: wrist at
the origin, unit scale. That erases the hand's path through space, so the J/Z
model could only learn handshape (NOTES.md, 2026-09-28). v2 saves:

    raw         (T, 21, 3) float32, MediaPipe image-normalised coordinates.
                Frames where no hand was detected are kept as NaN rows rather
                than dropped, so timing stays honest.
    timestamps  (T,) float64 seconds since the clip started
    detected    (T,) bool
    frame_w/h   pixel size of the frames (x and y are fractions of different
                lengths, so features need the aspect ratio)

Feature extraction happens later, in data/motion_features.py, so the features
can change without re-recording.

NONE
====
The third label is "not J or Z". It's what lets the live gate reject ordinary
movement. Its prompts cycle through config.MOTION_NONE_SUBTYPES:
    hold_I            hold the I handshape still (J's handshape, no hook)
    hold_D            hold D / index-up still (Z's handshape, no zigzag)
    hold_static       hold the static letter shown on screen
    between_letters   move from one shown letter to another, as when spelling
    wave_reposition   wave, or move the hand somewhere else in frame
The sub-type is saved per clip so diagnose_motion.py can report false
positives per sub-type (hold_I -> J and hold_D -> Z are the ones v1 would fail).

The frame is mirrored before detection, exactly as demo.py does. The direction
of the path is part of J and Z, so capture and demo must agree on this.

Clips go to data/motion_v2/<signer>/<session>/<label>/ and rows to
config.MOTION_V2_METADATA_CSV, a separate tree from v1 so they never mix.
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

FORMAT_VERSION = 2
MIN_DETECTED = 10          # clips with fewer detected frames are discarded on the spot
METADATA_HEADER = ["path", "label", "subtype", "prompt", "signer_id", "session_id", "clip_id",
                   "n_frames", "n_detected", "duration", "frame_w", "frame_h", "format_version"]

# Letters used in NONE prompts. I and D have their own sub-types.
_PROMPT_LETTERS = [c for c in config.STATIC_LETTERS if c not in ("I", "D")]


# ------------------------------------------------------------------ pure helpers
def none_prompt(clip_index: int, rng: np.random.Generator) -> tuple[str, str]:
    """(subtype, on-screen instruction) for the clip_index-th NONE clip."""
    subtype = config.MOTION_NONE_SUBTYPES[clip_index % len(config.MOTION_NONE_SUBTYPES)]
    if subtype == "hold_I":
        return subtype, "hold I still (pinky up) -- no hook"
    if subtype == "hold_D":
        return subtype, "hold D / index-up still -- no zigzag"
    if subtype == "hold_static":
        return subtype, f"hold the letter {rng.choice(_PROMPT_LETTERS)} still"
    if subtype == "between_letters":
        a, b = rng.choice(_PROMPT_LETTERS, size=2, replace=False)
        return subtype, f"sign {a}, then move to {b}"
    return subtype, "wave, or move your hand to a new spot"


def next_clip_id(out_dir: Path, label: str) -> int:
    """First unused clip id in out_dir, so re-running a session never overwrites."""
    ids = [int(p.stem.rsplit("_", 1)[1]) for p in out_dir.glob(f"{label}_*.npz")]
    return max(ids) + 1 if ids else 0


def save_clip(out_dir: Path, root: Path, label: str, clip_id: int, frames: list,
              timestamps: list, frame_w: int, frame_h: int, *, subtype: str = "",
              prompt: str = "", signer: str, session: str) -> dict:
    """
    frames: list of (21, 3) arrays or None (undetected). Writes one .npz and
    returns its metadata row (path relative to root).
    """
    raw = np.stack([f if f is not None else np.full((21, 3), np.nan) for f in frames]
                   ).astype(np.float32)
    t = np.asarray(timestamps, dtype=np.float64)
    t = t - t[0]
    detected = ~np.isnan(raw).any(axis=(1, 2))

    out_dir.mkdir(parents=True, exist_ok=True)
    clip_path = out_dir / f"{label}_{clip_id:04d}.npz"
    np.savez_compressed(clip_path, raw=raw, timestamps=t, detected=detected,
                        frame_w=frame_w, frame_h=frame_h, label=label, subtype=subtype,
                        format_version=FORMAT_VERSION)
    return {
        "path": str(clip_path.relative_to(root)), "label": label, "subtype": subtype,
        "prompt": prompt, "signer_id": signer, "session_id": session, "clip_id": clip_id,
        "n_frames": len(t), "n_detected": int(detected.sum()),
        "duration": float(t[-1]) if len(t) else 0.0,
        "frame_w": frame_w, "frame_h": frame_h, "format_version": FORMAT_VERSION,
    }


def append_metadata(rows: list[dict], csv_path: Path = config.MOTION_V2_METADATA_CSV) -> None:
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_HEADER)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------ camera loop
def _draw(cv2, display, lines, colour, big=None):
    for i, line in enumerate(lines):
        cv2.putText(display, line, (12, 36 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
    if big is not None:
        h, w = display.shape[:2]
        cv2.putText(display, big, (w // 2 - 40, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    3.0, colour, 6)


def _read_mirrored(cv2, cap):
    ok, frame = cap.read()
    return cv2.flip(frame, 1) if ok else None


def wait_for_key(cv2, cap, lines) -> int:
    while True:
        frame = _read_mirrored(cv2, cap)
        if frame is None:
            return ord("q")
        _draw(cv2, frame, lines, (0, 165, 255))
        cv2.imshow("capture v2", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord(" "), ord("n"), ord("r"), ord("q")):
            return key


def countdown(cv2, cap, lines, seconds: float) -> None:
    t0 = time.time()
    while (left := seconds - (time.time() - t0)) > 0:
        frame = _read_mirrored(cv2, cap)
        if frame is None:
            return
        _draw(cv2, frame, lines + ["get ready..."], (0, 200, 255), big=f"{left:.1f}")
        cv2.imshow("capture v2", frame)
        cv2.waitKey(1)


def record(cv2, cap, detector, duration: float, lines):
    """Record for `duration` s. Returns (frames, timestamps, frame_w, frame_h);
    frames holds a (21, 3) array or None per camera frame."""
    from data.landmarks import extract_landmarks

    frames, stamps = [], []
    fw = fh = 0
    t0 = time.time()
    while (now := time.time()) - t0 < duration:
        frame = _read_mirrored(cv2, cap)
        if frame is None:
            break
        fh, fw = frame.shape[:2]
        raw = extract_landmarks(detector, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(raw)
        stamps.append(now)

        display = frame.copy()
        if raw is not None:
            xs, ys = raw[:, 0] * fw, raw[:, 1] * fh
            cv2.rectangle(display, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())),
                          (0, 0, 220), 2)
        n_det = sum(f is not None for f in frames)
        bar = int((now - t0) / duration * (fw - 24))
        cv2.rectangle(display, (12, fh - 20), (12 + bar, fh - 8), (0, 0, 220), -1)
        _draw(cv2, display, lines + [f"RECORDING {now - t0:.1f}/{duration:.1f}s  "
                                     f"detected {n_det}/{len(frames)}"], (0, 0, 220))
        cv2.imshow("capture v2", display)
        cv2.waitKey(1)
    return frames, stamps, fw, fh


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture J/Z/NONE raw-landmark clips (v2)")
    parser.add_argument("--signer", required=True,
                        help="one id per SESSION for a single person (ben_m1, ben_m2, ...)")
    parser.add_argument("--session", required=True)
    parser.add_argument("--labels", nargs="+", default=config.MOTION_V2_CLASSES)
    parser.add_argument("--clips", type=int, default=20, help="clips per label")
    parser.add_argument("--duration", type=float, default=config.MOTION_V2_CLIP_SECONDS)
    parser.add_argument("--countdown", type=float, default=1.5)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for NONE prompts (default: time-based)")
    args = parser.parse_args()

    import cv2
    from data.landmarks import build_detector

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}. Check `ls /dev/video*`.")
    detector = build_detector(static_mode=False)
    rng = np.random.default_rng(args.seed)

    print(f"[capture-v2] labels {args.labels} x {args.clips} clips x {args.duration}s, "
          f"signer={args.signer} session={args.session}")
    print("[capture-v2] SPACE record, r redo last, n next label, q quit")

    all_rows: list[dict] = []
    try:
        for li, label in enumerate(args.labels, start=1):
            out_dir = config.MOTION_V2_DIR / args.signer / args.session / label
            out_dir.mkdir(parents=True, exist_ok=True)
            saved: list[dict] = []
            prompt_idx = 0
            subtype, prompt = ("", f"trace {label}")
            if label == config.MOTION_NONE:
                subtype, prompt = none_prompt(prompt_idx, rng)

            while len(saved) < args.clips:
                head = [f"[{li}/{len(args.labels)}] {label}: {prompt}",
                        f"clip {len(saved)}/{args.clips}   SPACE to record"]
                key = wait_for_key(cv2, cap, head)
                if key == ord("q"):
                    return
                if key == ord("n"):
                    break
                if key == ord("r"):
                    if saved:
                        last = saved.pop()
                        (config.MOTION_V2_DIR / last["path"]).unlink(missing_ok=True)
                        all_rows.remove(last)
                        print(f"[capture-v2] discarded {last['path']}")
                    continue

                countdown(cv2, cap, head[:1], args.countdown)
                frames, stamps, fw, fh = record(cv2, cap, detector, args.duration, head[:1])
                n_det = sum(f is not None for f in frames)
                if n_det < MIN_DETECTED:
                    print(f"[capture-v2] only {n_det} frames with a hand (need >= {MIN_DETECTED}) "
                          "-- discarded, try again")
                    continue

                row = save_clip(out_dir, config.MOTION_V2_DIR, label, next_clip_id(out_dir, label),
                                frames, stamps, fw, fh, subtype=subtype, prompt=prompt,
                                signer=args.signer, session=args.session)
                saved.append(row)
                all_rows.append(row)
                print(f"[capture-v2] {label} {len(saved)}/{args.clips} [{subtype or '-'}] "
                      f"{row['n_detected']}/{row['n_frames']} detected -> {row['path']}")

                if label == config.MOTION_NONE:
                    prompt_idx += 1
                    subtype, prompt = none_prompt(prompt_idx, rng)
    finally:
        # Written even on q / Ctrl-C / an exception, so nothing captured is lost.
        cap.release()
        cv2.destroyAllWindows()
        if all_rows:
            append_metadata(all_rows)
            print(f"[capture-v2] saved {len(all_rows)} clips -> {config.MOTION_V2_METADATA_CSV}")
        else:
            print("[capture-v2] nothing saved")


if __name__ == "__main__":
    main()
