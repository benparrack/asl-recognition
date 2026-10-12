"""
Score a folder of your own recorded signs through the live words path.

Run:
    python src/eval_word_videos.py --checkpoint checkpoints/words_transformer_full_best.pt \
        --dir data/word_videos

Layout: data/word_videos/<sign>/<anything>.mp4, one sign per file, where
<sign> is a GISLR sign name exactly as in the checkpoint's class list (e.g.
`data/word_videos/apple/ben_1.mp4`). Folders that aren't in the class list are
reported and skipped.

Each file goes through the same functions demo.py uses in words mode:
Holistic on the un-flipped frame -> holistic_features.frame_from_holistic ->
words_live.WordRecogniser, on the file's own timestamps. Two readings per file:
  whole_clip  the entire file as one sign (manual segmentation), isolating
              the model from the segmenter
  auto        automatic segmentation; the file's first accepted-or-not
              classified event, plus how many events fired (1 is ideal)
The gap between the two is the segmenter's cost.

This is the live test PROMPT §E asks for. It's a personal, different-domain
set (your camera, your signing), so don't tune on it and then quote it as a
test result. Writes results/words_personal_eval.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from data import holistic_features as hf  # noqa: E402
from evaluate_words import load_word_checkpoint  # noqa: E402
from words_live import SegParams, WordClassifier, WordRecogniser  # noqa: E402

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def video_landmarks(path, holistic, max_fps=None):
    """(frames (T, 84, 3), timestamps (T,), (w, h)) for one file. max_fps skips
    frames (keeps every k-th) so Holistic runs at most that rate."""
    cap = cv2.VideoCapture(str(path))
    step = max(1, round(cap.get(cv2.CAP_PROP_FPS) / max_fps)) if max_fps else 1
    frames, times, wh = [], [], (1, 1)
    for i in itertools.count():
        if i % step and cap.grab():
            continue
        ok, frame = cap.read()
        if not ok:
            break
        wh = frame.shape[1::-1]
        times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        frames.append(hf.frame_from_holistic(
            holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    cap.release()
    return frames, np.asarray(times), wh


def score_file(frames, times, wh, clf, params) -> dict:
    out = {"n_frames": len(frames)}
    whole = WordRecogniser(clf, params, auto=False)
    whole.toggle(0.0)
    for f, t in zip(frames, times):
        whole.update(f, t, wh)
    out["whole_clip"] = whole.flush()
    auto = WordRecogniser(clf, params, auto=True)
    events = [st.event for st in (auto.update(f, t, wh) for f, t in zip(frames, times))
              if st.event]
    tail = auto.flush()
    events += [tail] if tail else []
    classified = [e for e in events if e["top"]]
    out["auto"] = classified[0] if classified else (events[0] if events else None)
    out["auto_n_events"] = len(events)
    return out


def hit(event, sign, k):
    return bool(event and event["top"] and sign in [w for w, _ in event["top"][:k]])


def main():
    parser = argparse.ArgumentParser(description="Score recorded sign videos")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dir", type=Path, default=config.DATA_DIR / "word_videos")
    parser.add_argument("--holistic-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "words_personal_eval.json")
    args = parser.parse_args()

    import mediapipe as mp
    model, ckpt = load_word_checkpoint(args.checkpoint, config.DEVICE)
    clf = WordClassifier(model, ckpt, config.DEVICE)
    params = SegParams(min_confidence=0.0)          # report every guess; scoring decides
    known = set(ckpt["classes"])

    rows, skipped = [], []
    for sign_dir in sorted(p for p in args.dir.iterdir() if p.is_dir()):
        if sign_dir.name not in known:
            skipped.append(sign_dir.name)
            continue
        for path in sorted(p for p in sign_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS):
            with mp.solutions.holistic.Holistic(static_image_mode=False,
                                                model_complexity=args.holistic_complexity) as h:
                frames, times, wh = video_landmarks(path, h)
            r = score_file(frames, times, wh, clf, params)
            r.update(file=str(path), sign=sign_dir.name)
            rows.append(r)
            w = r["whole_clip"]
            print(f"{sign_dir.name:>14} {path.name:<24} whole: "
                  + (", ".join(f"{s} {p:.0%}" for s, p in w["top"]) if w and w["top"]
                     else "(no hands)")
                  + f"   auto events: {r['auto_n_events']}")
    if skipped:
        print(f"[skip] not in the checkpoint's classes: {skipped}")
    if not rows:
        sys.exit(f"no videos found under {args.dir}/<sign>/")

    summary = {}
    for mode in ("whole_clip", "auto"):
        summary[mode] = {f"top{k}": float(np.mean([hit(r[mode], r["sign"], k) for r in rows]))
                         for k in (1, 3)}
    summary["auto_single_event_frac"] = float(np.mean([r["auto_n_events"] == 1 for r in rows]))
    summary.update(n_files=len(rows), n_signs=len({r["sign"] for r in rows}))
    args.out.write_text(json.dumps({"checkpoint": args.checkpoint, "summary": summary,
                                    "skipped_dirs": skipped, "files": rows}, indent=2))
    print(json.dumps(summary, indent=1))
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
