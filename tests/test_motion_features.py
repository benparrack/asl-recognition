"""
Synthetic-data checks for the motion v2 feature pipeline.

These verify the PIPELINE (trajectory survives, invariances hold, a model can
learn from trajectory alone). They say nothing about accuracy on a real hand --
that needs Ben's recordings (HUMAN_TODO.md).
"""

import numpy as np
import pytest
import torch

import config
from data import motion_features as mf
from data.synthetic_motion import make_clip, random_clip


def feats(clip):
    raw, t, w, h = clip
    f = mf.clip_to_features(raw, t, w, h)
    assert f is not None
    return f


def test_feature_shape_and_constants():
    f = feats(make_clip("J", "I"))
    assert f.shape == (mf.RESAMPLE_LEN, mf.FEATURE_DIM)
    assert mf.FEATURE_DIM == 77
    assert np.isfinite(f).all()


def test_paths_differ_for_identical_handshape():
    """Same handshape, different path: handshape channels ~equal, motion channels not.
    Twist disabled so the handshape really is identical frame to frame."""
    fj = feats(make_clip("J", "D", twist_deg=0))
    fz = feats(make_clip("Z", "D", twist_deg=0))
    fs = feats(make_clip("still", "D", twist_deg=0))

    for a, b in [(fj, fz), (fj, fs), (fz, fs)]:
        assert np.allclose(a[:, mf.HANDSHAPE_SLICE], b[:, mf.HANDSHAPE_SLICE], atol=1e-5)
        assert np.abs(a[:, mf.TRAJECTORY_SLICE] - b[:, mf.TRAJECTORY_SLICE]).max() > 0.5

    # A held hand has (near) zero trajectory.
    assert np.abs(fs[:, mf.MOTION_SLICE]).max() < 1e-6


@pytest.mark.parametrize("path", ["J", "Z", "wave"])
def test_translation_and_scale_invariance(path):
    raw, t, w, h = make_clip(path, "I")
    base = mf.clip_to_features(raw, t, w, h)

    moved = raw.copy()
    moved[:, :, 0] += 0.13
    moved[:, :, 1] -= 0.07
    assert np.allclose(mf.clip_to_features(moved, t, w, h), base, atol=1e-4)

    # Uniform scale about an arbitrary point (hand closer to the camera).
    c = np.array([0.4, 0.6, 0.0])
    scaled = (raw - c) * 1.7 + c
    assert np.allclose(mf.clip_to_features(scaled, t, w, h), base, atol=1e-4)


def test_timestamps_not_frame_index():
    """Dropping frames (camera hiccup) must not change the resampled path much,
    because resampling is over timestamps, not frame indices."""
    raw, t, w, h = make_clip("Z", "D")
    base = mf.clip_to_features(raw, t, w, h)
    keep = np.ones(len(t), bool)
    keep[5:12] = False
    sparse = mf.clip_to_features(raw[keep], t[keep], w, h)
    assert np.abs(sparse[:, mf.TRAJECTORY_SLICE] - base[:, mf.TRAJECTORY_SLICE]).max() < 0.35


def test_nan_frames_are_interpolated_and_trimmed():
    raw, t, w, h = make_clip("J", "I", drop_frames=[0, 1, 10, 11, 12, 35])
    f = mf.clip_to_features(raw, t, w, h)
    assert f is not None and np.isfinite(f).all()


def test_too_few_detected_frames_returns_none():
    raw, t, w, h = make_clip("J", "I", duration=0.3)
    raw[3:] = np.nan
    assert mf.clip_to_features(raw, t, w, h) is None


def test_augmentation_never_mirrors_direction():
    """Z's first stroke goes +x (signer's view, mirrored frame). Every augmented
    copy must keep that direction -- a flip would turn it into a different path."""
    rng = np.random.default_rng(1)
    raw, t, w, h = make_clip("Z", "D", duration=1.5)
    for _ in range(50):
        a_raw, a_t = mf.augment_raw_clip(raw, t, w, h, rng)
        f = mf.clip_to_features(a_raw, a_t, w, h)
        if f is None:
            continue
        wrist_x = f[:, mf.TRAJECTORY_SLICE][:, 0]
        assert wrist_x[mf.RESAMPLE_LEN // 4] > 0.3


def test_time_shuffle_changes_features():
    """Sanity for diagnose_motion's shuffled control: permuting frames of the raw
    clip changes the trajectory channels (otherwise the control is meaningless)."""
    raw, t, w, h = make_clip("J", "I")
    perm = np.random.default_rng(0).permutation(len(t))
    base = mf.clip_to_features(raw, t, w, h)
    shuf = mf.clip_to_features(raw[perm], t, w, h)
    assert np.abs(base[:, mf.TRAJECTORY_SLICE] - shuf[:, mf.TRAJECTORY_SLICE]).max() > 0.5


def test_tiny_model_learns_from_trajectory_alone():
    """Every class uses the SAME handshape and handshape channels are zeroed, so
    the only signal is trajectory/velocity/rotation. Must clearly beat chance (1/3)."""
    from conftest import train_synthetic_motion_model

    model = train_synthetic_motion_model(n_per_class=50, epochs=25, zero_handshape=True,
                                         shape="B", seed=0)
    rng = np.random.default_rng(123)       # held-out clips, different seed
    correct = total = 0
    for label in config.MOTION_V2_CLASSES:
        for _ in range(30):
            f = feats(random_clip(label, rng, shape="B"))
            x = torch.from_numpy(f)[None]
            x[:, :, mf.HANDSHAPE_SLICE] = 0
            pred = model(x).argmax(1).item()
            correct += pred == config.MOTION_V2_CLASS_TO_IDX[label]
            total += 1
    acc = correct / total
    assert acc > 0.8, f"trajectory-only synthetic accuracy {acc:.2f}"
