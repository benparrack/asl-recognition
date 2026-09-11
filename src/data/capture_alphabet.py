"""
Capture webcam images for every static ASL letter in one continuous session.

Run:
    python src/data/capture_alphabet.py --signer ben --session couch_evening

Controls:
    SPACE   start/pause capturing the current letter
    n       skip to the next letter (keeps whatever was captured so far)
    r       restart the current letter (discards its frames from this run)
    q       quit early (keeps everything captured so far)

Same idea as capture_data.py's single-class bootstrap (FIST/PALM/PEACE), just looped
across config.STATIC_LETTERS so one run produces a full 24-letter dataset for one
signer/session instead of 24 separate command invocations. Each letter starts paused
so you have time to get your hand into position before it starts saving frames, and
advances to the next letter automatically once its frame count is reached.

Run this multiple times with a different --session (and vary lighting/background/
position each time) to build up multiple sessions, the same way the original
FIST/PALM/PEACE data has ben_s1/s2/s3.
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
from data.capture_data import append_metadata  # noqa: E402

try:
    import cv2
except ImportError:
    print("pip install opencv-python")
    raise


def capture_label(cap, label: str, signer: str, session: str, count: int,
                   delay: float, label_idx: int, total_labels: int):
    """Capture up to `count` frames for one label. Returns (rows, action)."""
    out_dir = config.RAW_DIR / signer / session / label
    out_dir.mkdir(parents=True, exist_ok=True)

    capturing = False
    saved = 0
    last_capture = 0.0
    rows: list[dict] = []

    while saved < count:
        ok, frame = cap.read()
        if not ok:
            return rows, "camera_error"

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        colour = (0, 200, 0) if capturing else (0, 0, 200)
        status = "CAPTURING" if capturing else "paused -- SPACE to start"
        cv2.putText(display, f"[{label_idx}/{total_labels}] sign: {label}",
                    (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        cv2.putText(display, f"{saved}/{count}  {status}",
                    (12, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        cv2.putText(display, "SPACE start/pause   n next letter   r restart   q quit",
                    (12, display.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)

        cv2.imshow("capture", display)

        now = time.time()
        if capturing and (now - last_capture) >= delay:
            filename = f"{label}_{saved:05d}.jpg"
            cv2.imwrite(str(out_dir / filename), frame)

            rows.append({
                "path": str((out_dir / filename).relative_to(config.RAW_DIR)),
                "label": label,
                "signer_id": signer,
                "session_id": session,
            })

            saved += 1
            last_capture = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return rows, "quit"
        if key == ord("n"):
            return rows, "next"
        if key == ord("r"):
            for row in rows:
                (config.RAW_DIR / row["path"]).unlink(missing_ok=True)
            rows = []
            saved = 0
            capturing = False
            continue
        if key == ord(" "):
            capturing = not capturing

    return rows, "done"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture webcam images for every static ASL letter in one session")
    parser.add_argument("--signer", required=True,
                        help="who is signing -- the unit the test split is built on")
    parser.add_argument("--session", required=True,
                        help="recording session: vary lighting/background/position between runs")
    parser.add_argument("--labels", nargs="+", default=config.STATIC_LETTERS,
                        help="letters to capture, in order "
                             "(default: config.STATIC_LETTERS -- all 24 static letters)")
    parser.add_argument("--count", type=int, default=80,
                        help="images per letter (80 matches the original 3-class test)")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="seconds between captures; keeps frames from being near-identical")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. On Linux check `ls /dev/video*` "
            "and that your user is in the 'video' group."
        )

    print(f"[capture] {len(args.labels)} letters x {args.count} frames, "
          f"signer={args.signer} session={args.session}")
    print("[capture] SPACE start/pause, n next letter, r restart letter, q quit")

    all_rows: list[dict] = []

    for i, label in enumerate(args.labels, start=1):
        rows, action = capture_label(cap, label, args.signer, args.session,
                                     args.count, args.delay, i, len(args.labels))
        all_rows.extend(rows)
        print(f"[capture] {label}: saved {len(rows)}/{args.count}")

        if action in ("quit", "camera_error"):
            print(f"[capture] stopped early ({action})")
            break

    cap.release()
    cv2.destroyAllWindows()

    if all_rows:
        append_metadata(all_rows)
        n_labels = len(set(r["label"] for r in all_rows))
        print(f"[capture] total saved: {len(all_rows)} images across {n_labels} letters")
        print(f"[capture] metadata -> {config.RAW_METADATA_CSV}")
    else:
        print("[capture] nothing saved")

    # MOVE YOUR HAND WHILE CAPTURING. Rotate it, shift it around the frame, vary
    # the distance from the camera -- same advice as capture_data.py. And for the
    # letters that look alike (M/N/S/T, U/V/R, K/P, G/H/Q -- see
    # config.KNOWN_CONFUSABLE_GROUPS), try to be consistent about orientation so
    # the labels stay visually distinguishable to a model that's never seen ASL.


if __name__ == "__main__":
    main()
