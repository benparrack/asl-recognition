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
