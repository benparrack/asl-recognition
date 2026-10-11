"""
Record personal word-sign clips as .mp4, one sign per file, in the layout
eval_word_videos.py reads:

    data/word_videos/<sign>/<session>_<nn>.mp4

Run (a new --session per sitting: different day, lighting or seat), e.g.:
    python src/data/capture_words.py --session w1_desk_day

Controls: SPACE countdown then record, r delete the last clip of this sign and
redo, n next sign, q quit (everything already saved stays).

Frames are saved UN-mirrored, which is what Holistic sees live and in
eval_word_videos.py; only the preview is mirrored so it behaves like a mirror.
Start and end each clip with hands down. The file's fps is the measured capture
rate, so clip timing stays real.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import config  # noqa: E402

# 10 two-handed + 10 one-handed, chosen 2026-10-11 (NOTES.md). mom/dad differ
# only by location (chin vs forehead).
DEFAULT_SIGNS = ["book", "shoe", "car", "clean", "finish", "tree", "dance", "open", "wet", "bath",
                 "hello", "thankyou", "yes", "no", "please", "mom", "dad", "water", "drink", "where"]


def write_clip(path: Path, frames: list, fps: float) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def next_index(sign_dir: Path, session: str) -> int:
    ids = [int(p.stem.rsplit("_", 1)[1]) for p in sign_dir.glob(f"{session}_*.mp4")]
    return max(ids) + 1 if ids else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Record personal word-sign clips")
    parser.add_argument("--session", required=True)
    parser.add_argument("--signs", nargs="+", default=DEFAULT_SIGNS)
    parser.add_argument("--clips", type=int, default=5, help="clips per sign this session")
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--countdown", type=float, default=1.5)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--dir", type=Path, default=config.DATA_DIR / "word_videos")
    args = parser.parse_args()

    import cv2
    from data.capture_motion_v2 import _draw

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}. Check `ls /dev/video*`.")

    def show(frame, lines, colour, big=None):
        display = cv2.flip(frame, 1)
        _draw(cv2, display, lines, colour, big)
        cv2.imshow("capture words", display)
        return cv2.waitKey(1) & 0xFF

    n_saved = 0
    try:
        for si, sign in enumerate(args.signs, start=1):
            sign_dir = args.dir / sign
            saved: list[Path] = []
            while len(saved) < args.clips:
                head = [f"[{si}/{len(args.signs)}] sign: {sign.upper()}",
                        f"clip {len(saved)}/{args.clips}   SPACE record  r redo  n next  q quit"]
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        return
                    key = show(frame, head, (0, 165, 255))
                    if key in (ord(" "), ord("r"), ord("n"), ord("q")):
                        break
                if key == ord("q"):
                    return
                if key == ord("n"):
                    break
                if key == ord("r"):
                    if saved:
                        last = saved.pop()
                        last.unlink(missing_ok=True)
                        n_saved -= 1
                        print(f"[words] discarded {last}")
                    continue

                t0 = time.time()
                while (left := args.countdown - (time.time() - t0)) > 0:
                    ok, frame = cap.read()
                    if not ok:
                        return
                    show(frame, head[:1] + ["hands down, get ready..."], (0, 200, 255),
                         big=f"{left:.1f}")

                frames = []
                t0 = time.time()
                while (el := time.time() - t0) < args.duration:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
                    show(frame, head[:1] + [f"RECORDING {el:.1f}/{args.duration:.1f}s"],
                         (0, 0, 220))
                if len(frames) < 10:
                    print("[words] too few frames captured, try again")
                    continue

                path = sign_dir / f"{args.session}_{next_index(sign_dir, args.session):02d}.mp4"
                write_clip(path, frames, len(frames) / (time.time() - t0))
                saved.append(path)
                n_saved += 1
                print(f"[words] {sign} {len(saved)}/{args.clips} -> {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[words] saved {n_saved} clips under {args.dir}")


if __name__ == "__main__":
    main()
