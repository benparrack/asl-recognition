"""
The live word path, replayed headlessly. This is the parity proof for words
mode: GISLR validation sequences are rendered frame by frame as mediapipe
Holistic results objects (what demo.py receives), pushed through
frame_from_holistic and WordRecogniser.update() exactly as the demo does, and
the predictions must match evaluate_words' offline pipeline on the same
sequences.

Uses conftest.synthetic_words (a synthetic GISLR cache plus a small trained
checkpoint), so it checks the code path, not recognition quality.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data import holistic_features as hf
from data import holistic_landmarks as hl
from data.preprocess_gislr import load_sequence
from data.word_dataset import WordDataset, load_word_index
from evaluate_words import collect, load_word_checkpoint
from holistic_fakes import fake_holistic_results
from words_live import SegParams, WordClassifier, WordRecogniser

FPS = 30.0
WH = (640, 480)


def selected_to_raw(seq):
    """Scatter a cached (T, 84, 3) sequence back into Holistic's raw layout.
    Unselected points of a detected part get dummy values (a real detection
    has all of them); an undetected part stays all-NaN."""
    t = len(seq)
    raw = {k: np.full((t, n, 3), np.nan, np.float32) for k, n in hl.RAW_TYPES.items()}
    parts = {"left_hand": (hl.LEFT_HAND, list(range(hl.N_HAND))),
             "right_hand": (hl.RIGHT_HAND, list(range(hl.N_HAND))),
             "pose": (hl.POSE, hl.POSE_IDX), "face": (hl.FACE, hl.FACE_IDX)}
    for name, (sl, idx) in parts.items():
        present = ~np.isnan(seq[:, sl]).any(axis=(1, 2))
        raw[name][present] = 0.5
        raw[name][np.ix_(present, idx)] = seq[present, sl]
    return raw


def replay(rec, raw, n, t0=0.0):
    """Feed n frames through the demo's per-frame path; return the events."""
    events = []
    for i in range(n):
        frame = hf.frame_from_holistic(fake_holistic_results(raw, i, refine_face=i % 3 == 0))
        st = rec.update(frame, t0 + i / FPS, WH)
        if st.event:
            events.append(st.event)
    return events


@pytest.fixture(scope="module")
def live(synthetic_words):
    cache, _, ckpt_path = synthetic_words
    model, ckpt = load_word_checkpoint(ckpt_path, "cpu")
    index, _ = load_word_index(cache)
    val = index[index["split"] == "val"].reset_index(drop=True)
    ds = WordDataset(val, cache, ckpt["feature_set"], ckpt["max_len"])
    y, offline = collect(model, DataLoader(ds, batch_size=64), torch.device("cpu"))
    return WordClassifier(model, ckpt, "cpu"), val, cache, offline


def test_manual_replay_matches_offline_evaluation(live):
    clf, val, cache, offline = live
    rec = WordRecogniser(clf, SegParams(min_confidence=0.0), auto=False)
    for i in range(len(val)):
        seq = load_sequence(cache, val.iloc[i])
        rec.toggle(0.0)
        assert replay(rec, selected_to_raw(seq), len(seq)) == []
        event = rec.toggle(len(seq) / FPS)
        np.testing.assert_allclose(clf.probs(np.stack(rec_frames(seq)),
                                             np.arange(len(seq)) / FPS),
                                   offline[i], atol=1e-5)
        assert event["outcome"] == "accepted"
        assert event["top"][0][0] == clf.classes[int(offline[i].argmax())]
        assert event["top"][0][1] == pytest.approx(float(offline[i].max()), abs=1e-5)
        assert len(event["top"]) == 3


def rec_frames(seq):
    raw = selected_to_raw(seq)
    return [hf.frame_from_holistic(fake_holistic_results(raw, i)) for i in range(len(seq))]


def test_resampling_live_rate_to_reference(live):
    """A clip captured at 15 fps is resampled to the 30 fps the model saw."""
    clf, val, cache, _ = live
    seq = load_sequence(cache, val.iloc[0])
    t = np.arange(len(seq)) / FPS
    slow = seq[::2]
    _, mask_slow, _ = clf.features(slow, t[::2])
    _, mask_full, _ = clf.features(seq, t)
    assert abs(int(mask_slow.sum()) - int(mask_full.sum())) <= 2   # trim may shift by a frame


def _with_rest(seq, n_rest=20):
    """Pad a sequence with 'hands in lap' frames: body visible, no hands."""
    rest = np.repeat(seq[:1].copy(), n_rest, axis=0)
    rest[:, hl.LEFT_HAND] = np.nan
    rest[:, hl.RIGHT_HAND] = np.nan
    first = np.flatnonzero(~np.isnan(seq[:, :2 * hl.N_HAND]).all(axis=(1, 2)))
    body = seq[first[0]:first[-1] + 1]
    return np.concatenate([rest, body, rest]), body


def test_auto_segmentation_finds_the_sign(live):
    clf, val, cache, offline = live
    agree = 0
    for i in range(len(val)):
        full, body = _with_rest(load_sequence(cache, val.iloc[i]))
        rec = WordRecogniser(clf, SegParams(min_confidence=0.0, lull_speed=0.0))
        events = replay(rec, selected_to_raw(full), len(full))
        assert len(events) == 1 and events[0]["reason"] == "hands_down"
        hand = ~(np.isnan(body[:, hl.LEFT_HAND]).any(axis=(1, 2))
                 & np.isnan(body[:, hl.RIGHT_HAND]).any(axis=(1, 2)))
        assert events[0]["n_hand_frames"] == hand.sum()      # every hand frame captured
        agree += events[0]["top"][0][0] == clf.classes[int(offline[i].argmax())]
    assert agree == len(val)


def test_short_blips_are_discarded_not_classified(live):
    clf, val, cache, _ = live
    full, body = _with_rest(load_sequence(cache, val.iloc[0]))
    blip = np.concatenate([full[:20], body[:3], full[-20:]])
    rec = WordRecogniser(clf)
    events = replay(rec, selected_to_raw(blip), len(blip))
    assert [e["outcome"] for e in events] == ["too_short"] and events[0]["top"] == []


def test_held_pose_does_not_retrigger(live):
    """A motion-lull ending with the hands still up must not start a new clip
    until the hands drop or move again."""
    clf, val, cache, _ = live
    _, body = _with_rest(load_sequence(cache, val.iloc[0]))
    frame = next(f for f in body if not np.isnan(f[hl.RIGHT_HAND]).any())
    still = np.repeat(frame[None], 150, axis=0)
    rec = WordRecogniser(clf)
    events = replay(rec, selected_to_raw(still), len(still))
    assert [e["reason"] for e in events] == ["motion_lull"]
