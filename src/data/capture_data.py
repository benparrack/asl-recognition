"""
Capture training images from your webcam and register them in metadata.csv.

Run:
    python src/data/capture_data.py --label A --signer ben --session desk_day
    python src/data/capture_data.py --label B --signer ben --session desk_day
    python src/data/capture_data.py --label C --signer ben --session desk_day

Controls:
    SPACE   start/stop capturing
    q       quit

WHY START HERE
==============
This gives you a real dataset in ten minutes with no downloads, no Kaggle
account, and no dependency on your model working yet. Use it to build a tiny
3-class set so you can get the whole pipeline running end to end before you
scale up.

The three classes do not need to be real ASL letters at first. Fist, open palm,
and peace sign are visually distinct and easy to hold steady -- perfect for
proving the plumbing works. Swap in real handshapes once training runs cleanly.

ON SESSIONS: capture each session under different conditions -- different room,
lighting, shirt, time of day. If every image comes from one session, your model
will learn your desk lamp rather than your hand, and split_by_signer() will have
nothing meaningful to hold out. Vary the conditions deliberately.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Make `import config` work when running this file directly from src/data/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402

try:
    import cv2
except ImportError:
    print("pip install opencv-python")
    raise


METADATA_HEADER = ["path", "label", "signer_id", "session_id"]


def append_metadata(rows: list[dict]) -> None:
    """Append rows to metadata.csv, writing the header if the file is new."""
    exists = config.METADATA_CSV.exists()

    with open(config.METADATA_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_HEADER)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture webcam images for one class")
    parser.add_argument("--label", required=True,
                        help="class label, e.g. A (or FIST/PALM/PEACE while testing)")
    parser.add_argument("--signer", required=True,
                        help="who is signing -- the unit the test split is built on")
    parser.add_argument("--session", required=True,
                        help="recording session: vary lighting/background between these")
    parser.add_argument("--count", type=int, default=200, help="images to capture")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="seconds between captures; keeps frames from being near-identical")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    out_dir = config.PROCESSED_DIR / args.signer / args.session / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. On Linux check `ls /dev/video*` "
            "and that your user is in the 'video' group."
        )

    capturing = False
    saved = 0
    last_capture = 0.0
    rows: list[dict] = []

    print(f"[capture] label={args.label} signer={args.signer} session={args.session}")
    print("[capture] SPACE toggles capture, q quits")

    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)          # mirror for natural movement
        display = frame.copy()              # draw on the copy, save the clean frame

        colour = (0, 200, 0) if capturing else (0, 0, 200)
        status = "CAPTURING" if capturing else "paused (SPACE)"
        cv2.putText(display, f"{args.label}  {saved}/{args.count}  {status}",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

        cv2.imshow("capture", display)

        now = time.time()
        if capturing and (now - last_capture) >= args.delay:
            filename = f"{args.label}_{saved:05d}.jpg"
            cv2.imwrite(str(out_dir / filename), frame)

            rows.append({
                "path": str((out_dir / filename).relative_to(config.PROCESSED_DIR)),
                "label": args.label,
                "signer_id": args.signer,
                "session_id": args.session,
            })

            saved += 1
            last_capture = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            capturing = not capturing

    cap.release()
    cv2.destroyAllWindows()

    if rows:
        append_metadata(rows)
        print(f"[capture] saved {len(rows)} images -> {out_dir}")
        print(f"[capture] metadata -> {config.METADATA_CSV}")
    else:
        print("[capture] nothing saved")

    # MOVE YOUR HAND WHILE CAPTURING. Rotate it, shift it around the frame, vary
    # the distance from the camera. 200 identical frames teach the model far less
    # than 50 varied ones, and they inflate your dataset size without adding
    # information.


if __name__ == "__main__":
    main()
