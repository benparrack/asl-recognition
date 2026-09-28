"""
Words-mode throughput: MediaPipe Holistic per frame, plus the word model per clip.

Run:
    python src/bench_words_fps.py --frames "data/raw/ben_s1/desk/PEACE/*.jpg"
    python src/bench_words_fps.py --video clip.mp4
    python src/bench_words_fps.py                      # blank frames (detector-only cost)

No webcam. Frames with a person in them matter: on a blank or noise frame
Holistic finds no pose and skips the hand and face stages, so it looks much
faster than it is live.

Reports ms/frame and FPS for each model_complexity, the fraction of frames with
a pose / hand detected (to show the frames exercised the full graph), and the
word model's latency per clip (paid once per sign, not per frame).
Writes results/words_fps.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from data import holistic_features as hf  # noqa: E402
from models.words import build_word_model  # noqa: E402


def load_frames(args) -> list[np.ndarray]:
    if args.video:
        cap = cv2.VideoCapture(args.video)
        frames = []
        while len(frames) < args.n:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        return frames
    if args.frames:
        paths = sorted(glob.glob(args.frames))[:args.n]
        return [cv2.imread(p) for p in paths]
    return [np.full((480, 640, 3), 128, np.uint8)] * args.n


def bench_holistic(frames, complexity: int) -> dict:
    import mediapipe as mp
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False,
                                              model_complexity=complexity)
    for f in frames[:5]:                                     # warm-up
        holistic.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    pose = hand = 0
    t0 = time.perf_counter()
    for f in frames:
        r = holistic.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        frame = hf.frame_from_holistic(r)                    # the adapter is part of the cost
        pose += r.pose_landmarks is not None
        hand += (r.left_hand_landmarks is not None) or (r.right_hand_landmarks is not None)
    dt = (time.perf_counter() - t0) / len(frames)
    holistic.close()
    return dict(model_complexity=complexity, ms_per_frame=1000 * dt, fps=1 / dt,
                pose_frac=pose / len(frames), hand_frac=hand / len(frames))


def bench_model(device, n_classes=250, reps=50) -> dict:
    out = {}
    for name in ("gru", "transformer"):
        model = build_word_model(name, hf.FEATURE_DIM, n_classes, hf.MAX_LEN).to(device).eval()
        x = torch.randn(1, hf.MAX_LEN, hf.FEATURE_DIM, device=device)
        m = torch.ones(1, hf.MAX_LEN, dtype=torch.bool, device=device)
        seq = np.random.default_rng(0).random((60, 84, 3)).astype(np.float32)
        with torch.no_grad():
            for _ in range(5):
                model(x, m)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                feats, mask, _ = hf.sequence_to_features(seq)
                model(torch.from_numpy(feats)[None].to(device),
                      torch.from_numpy(mask)[None].to(device)).cpu()
            dt = (time.perf_counter() - t0) / reps
        out[name] = {"ms_per_clip_incl_features": 1000 * dt}
    return out


def main():
    parser = argparse.ArgumentParser(description="Words-mode FPS benchmark (no webcam)")
    parser.add_argument("--frames", default=None, help="glob of images, in order")
    parser.add_argument("--video", default=None)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "words_fps.json")
    args = parser.parse_args()

    frames = load_frames(args)
    if not frames:
        sys.exit("no frames loaded")
    source = args.video or args.frames or "blank 640x480"
    print(f"[bench] {len(frames)} frames from {source}, {frames[0].shape[1]}x{frames[0].shape[0]}")
    rows = [bench_holistic(frames, c) for c in (0, 1, 2)]
    for r in rows:
        print(f"  model_complexity={r['model_complexity']}: {r['ms_per_frame']:.1f} ms/frame "
              f"= {r['fps']:.1f} FPS  (pose {r['pose_frac']:.0%}, hand {r['hand_frac']:.0%})")
    model = bench_model(config.DEVICE)
    for k, v in model.items():
        print(f"  {k} per clip on {config.DEVICE}: {v['ms_per_clip_incl_features']:.1f} ms")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"source": source, "n_frames": len(frames),
                                    "frame_size": list(frames[0].shape[1::-1]),
                                    "holistic": rows, "word_model": model,
                                    "device": str(config.DEVICE)}, indent=2))
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
