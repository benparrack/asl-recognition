"""holistic_features: invariances, handedness canonicalisation, NaN safety, and
live-vs-offline parity (the demo adapter and the parquet loader must give
identical features for identical landmarks)."""

from types import SimpleNamespace

import numpy as np
import pytest

from data import holistic_features as hf
from data import holistic_landmarks as hl
from data.preprocess_gislr import parquet_to_raw, parquet_to_selected
from data.synthetic_gislr import make_sequence, selected_to_long_df, sign_table

SIGNS = sign_table(12, seed=0)
TWO_HANDED = next(s for s in SIGNS if s["two_handed"] and s["move"] != "still")


def _seq(sign=TWO_HANDED, seed=0, left_handed=False):
    return make_sequence(sign, np.random.default_rng(seed), left_handed=left_handed)


def _feats(seq):
    return hf.sequence_to_features(seq)[0]


def test_dims_and_feature_sets():
    f, mask, info = hf.sequence_to_features(_seq())
    assert f.shape == (hf.MAX_LEN, hf.FEATURE_DIM) and mask.shape == (hf.MAX_LEN,)
    assert np.isfinite(f).all() and mask.sum() == info["n_frames"]
    sizes = list(hf.FEATURE_SETS.values())
    assert sizes == sorted(sizes) and sizes[-1] == hf.FEATURE_DIM
    assert hf.VELOCITY.stop == hf.FEATURE_DIM


def test_translation_and_scale_invariance():
    seq = _seq()
    moved = seq.copy()
    moved[..., :2] = seq[..., :2] * 0.7 + np.array([0.12, -0.05])
    moved[..., 2] = seq[..., 2] * 0.7
    np.testing.assert_allclose(_feats(moved), _feats(seq), atol=1e-4)


def test_aspect_ratio_invariance():
    """Stretching x alone (a different camera aspect) changes nothing, because x
    and y are each normalised by a body measure on their own axis."""
    seq = _seq()
    seq[..., 2] = 0.0                                    # z has no defined aspect behaviour
    wide = seq.copy()
    wide[..., 0] = 0.5 + (seq[..., 0] - 0.5) * 1.4
    np.testing.assert_allclose(_feats(wide), _feats(seq), atol=1e-4)


def test_mirror_permutation_round_trip():
    seq = _seq()
    twice = seq[:, hl.MIRROR_PERM][:, hl.MIRROR_PERM]
    np.testing.assert_array_equal(np.isnan(twice), np.isnan(seq))
    np.testing.assert_allclose(np.nan_to_num(twice), np.nan_to_num(seq))


def test_left_handed_signer_matches_right_handed():
    right = _seq(seed=3)
    left = _seq(seed=3, left_handed=True)                # same performance, mirrored
    fr, _, ir = hf.sequence_to_features(right)
    fl, _, il = hf.sequence_to_features(left)
    assert not ir["flipped"] and il["flipped"]
    np.testing.assert_allclose(fl, fr, atol=1e-4)


def test_dominant_hand_goes_to_slot_zero():
    seq = _seq(sign=next(s for s in SIGNS if not s["two_handed"]))
    swapped = seq[:, hl.MIRROR_PERM].copy()
    swapped[..., 0] = 1 - swapped[..., 0]
    f = _feats(swapped)
    assert f[:, hf.DOM_PRESENT].sum() > 0 and f[:, hf.NONDOM_PRESENT].sum() == 0


def test_trim_drops_handless_edges():
    seq = _seq()
    pad = np.full((5, hl.N_SELECTED, 3), np.nan, dtype=seq.dtype)
    pad[:, hl.POSE.start:] = seq[0, hl.POSE.start:]      # body visible, no hands
    padded = np.concatenate([pad, seq, pad])
    np.testing.assert_allclose(_feats(padded), _feats(seq), atol=1e-6)


@pytest.mark.parametrize("missing", ["pose", "face", "pose+face", "everything"])
def test_nan_safety(missing):
    seq = _seq()
    if "pose" in missing or missing == "everything":
        seq[:, hl.POSE] = np.nan
    if "face" in missing or missing == "everything":
        seq[:, hl.FACE] = np.nan
    if missing == "everything":
        seq[:] = np.nan
    f, mask, info = hf.sequence_to_features(seq)
    assert np.isfinite(f).all()
    if missing == "everything":
        assert info["empty"] and mask.sum() == 1
    if "face" in missing:
        assert (f[:, hf.FACE_VALID] == 0).all() and (f[:, hf.FACE_DIST] == 0).all()


def test_pose_dropout_frames_are_carried():
    seq = _seq()
    seq[2:4, hl.POSE] = np.nan
    f = _feats(seq)
    assert np.isfinite(f).all()


def test_long_sequences_subsampled_short_padded():
    long_seq = np.concatenate([_seq(seed=i) for i in range(6)])
    f, mask, _ = hf.sequence_to_features(long_seq)
    assert mask.all()
    f, mask, info = hf.sequence_to_features(_seq())
    assert mask.sum() == info["n_frames"] < hf.MAX_LEN and (f[~mask] == 0).all()


def test_augmentation_goes_through_same_features():
    rng = np.random.default_rng(0)
    f, mask, _ = hf.sequence_to_features(_seq(), rng=rng)
    assert f.shape == (hf.MAX_LEN, hf.FEATURE_DIM) and np.isfinite(f).all()


def test_resample_to_fps():
    seq = _seq()
    t30 = np.arange(len(seq)) / 30.0
    np.testing.assert_array_equal(np.nan_to_num(hf.resample_to_fps(seq, t30)),
                                  np.nan_to_num(seq))
    t15 = np.arange(len(seq)) / 15.0
    assert abs(len(hf.resample_to_fps(seq, t15)) - 2 * len(seq)) <= 2


# ------------------------------------------------------------------ parity
def _fake_landmarks(arr):
    if np.isnan(arr).any():
        return None
    return SimpleNamespace(landmark=[SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                                     for p in arr])


def _fake_holistic_results(raw, i, refine_face=False):
    face = raw["face"][i]
    if refine_face and not np.isnan(face).any():
        face = np.concatenate([face, np.zeros((10, 3), dtype=face.dtype)])   # 478 points
    return SimpleNamespace(left_hand_landmarks=_fake_landmarks(raw["left_hand"][i]),
                           right_hand_landmarks=_fake_landmarks(raw["right_hand"][i]),
                           pose_landmarks=_fake_landmarks(raw["pose"][i]),
                           face_landmarks=_fake_landmarks(face))


@pytest.mark.parametrize("left_handed", [False, True])
def test_live_adapter_matches_parquet_loader(left_handed):
    """A GISLR parquet, rendered frame by frame as mediapipe Holistic results
    objects (what the demo receives), must featurise identically to the same
    parquet read by the offline loader."""
    rng = np.random.default_rng(7)
    df = selected_to_long_df(_seq(seed=5, left_handed=left_handed), rng)
    offline = parquet_to_selected(df)

    raw = parquet_to_raw(df)
    live = np.stack([hf.frame_from_holistic(_fake_holistic_results(raw, i, refine_face=i % 2 == 0))
                     for i in range(len(offline))])
    np.testing.assert_array_equal(np.isnan(live), np.isnan(offline))
    fo, mo, _ = hf.sequence_to_features(offline)
    fl, ml, _ = hf.sequence_to_features(live)
    np.testing.assert_array_equal(mo, ml)
    np.testing.assert_array_equal(fo, fl)


def test_parquet_loader_ignores_row_order():
    rng = np.random.default_rng(1)
    df = selected_to_long_df(_seq(), rng)
    shuffled = df.sample(frac=1.0, random_state=0)
    np.testing.assert_array_equal(np.nan_to_num(parquet_to_selected(shuffled)),
                                  np.nan_to_num(parquet_to_selected(df)))
