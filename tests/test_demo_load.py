"""The letters demo must keep working. No webcam here: import demo.py and load
the checkpoints the README tells people to run it with."""

import pytest
import torch

import config

CPU = torch.device("cpu")
MLP = config.CHECKPOINT_DIR / "mlp_pugeault_best.pt"
MOTION_V1 = config.CHECKPOINT_DIR / "motion_gru_best.pt"


@pytest.mark.skipif(not MLP.exists(), reason="letters checkpoint not present")
def test_demo_loads_letters_mlp():
    import demo
    model, kind = demo.load_model(str(MLP), CPU)
    assert kind == "landmark"
    with torch.no_grad():
        out = model(torch.zeros(1, config.LANDMARK_FEATURE_SIZE))
    assert out.shape == (1, len(config.CLASSES))


@pytest.mark.skipif(not MOTION_V1.exists(), reason="v1 motion checkpoint not present")
def test_v1_motion_checkpoint_still_loads_through_gate():
    from motion_gate import GateParams, MotionClassifierV1, load_motion_classifier
    clf = load_motion_classifier(MOTION_V1, CPU)
    assert isinstance(clf, MotionClassifierV1)
    params = GateParams.for_classifier(clf)
    assert params.threshold == 0.025 and params.max_frames == clf.max_len


def _blank_video(path, n=20):
    import cv2
    import numpy as np
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(n):
        w.write(np.full((240, 320, 3), 90, np.uint8))
    w.release()


@pytest.mark.skipif(not MLP.exists(), reason="letters checkpoint not present")
@pytest.mark.parametrize("words", [False, True])
def test_demo_runs_headless_on_a_video(words, synthetic_words, tmp_path):
    """The whole demo main loop (real MediaPipe, both modes) on a --video file
    with --no-display. A blank video finds nobody, so this checks the
    plumbing, not recognition."""
    import subprocess
    import sys
    from pathlib import Path
    video = tmp_path / "blank.mp4"
    _blank_video(video)
    cmd = [sys.executable, str(Path(__file__).parents[1] / "src/demo.py"),
           "--checkpoint", str(MLP), "--video", str(video), "--no-display"]
    if words:
        cmd += ["--words-checkpoint", str(synthetic_words[2]), "--start-in-words"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "[demo] sentence: ''" in out.stdout
    assert ("[demo] words: 8 signs" in out.stdout) == words
