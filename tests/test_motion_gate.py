"""
Replay harness for the live J/Z gate (src/motion_gate.py).

Synthetic landmark streams are fed frame by frame through MotionGate.update() --
the same call demo.py makes per webcam frame -- with a MotionClassifierV2
wrapping a model trained on synthetic clips. This checks segmentation and the
classify/hold/release logic end to end, offline. It does not say anything about
real-hand accuracy.
"""

import numpy as np
import pytest

import config
from data.synthetic_motion import make_clip
from motion_gate import (GateParams, MotionClassifierV1, MotionClassifierV2, MotionGate,
                         hand_speed_v2)


def stream(*segments, fps=30.0):
    """Concatenate clips into one continuous stream of (raw_or_None, t, (w, h)).
    Each segment continues from where the previous one's hand ended."""
    frames, t0, pos = [], 0.0, (0.45, 0.3)
    for kind, shape, dur in segments:
        raw, t, w, h = make_clip(kind, shape, duration=dur, fps=fps, start_xy=pos,
                                 jitter=0.0005, rng=np.random.default_rng(len(frames)))
        for i in range(len(t)):
            frames.append((raw[i], t0 + t[i], (w, h)))
        t0 += t[-1] + 1.0 / fps
        pos = (raw[-1, 0, 0], raw[-1, 0, 1])      # the wrist is the clip's anchor point
    return frames


def run(gate, frames):
    events, statuses = [], []
    for raw, t, wh in frames:
        st = gate.update(raw, t, wh)
        statuses.append(st)
        if st.event:
            events.append((t, st.event))
    return events, statuses


@pytest.fixture
def v2_gate(synthetic_motion_model):
    import torch
    clf = MotionClassifierV2(synthetic_motion_model, torch.device("cpu"))
    return MotionGate(clf, GateParams.for_classifier(clf, min_confidence=0.5), verbose=False)


def test_hand_speed_is_scale_invariant():
    raw, t, w, h = make_clip("Z", "D", hand_size=0.05)
    raw2, _, _, _ = make_clip("Z", "D", hand_size=0.10)
    s1 = hand_speed_v2(raw[10], raw[11], t[11] - t[10], (w, h))
    s2 = hand_speed_v2(raw2[10], raw2[11], t[11] - t[10], (w, h))
    assert s1 == pytest.approx(s2, rel=1e-3)


def test_static_hold_never_triggers(v2_gate):
    events, statuses = run(v2_gate, stream(("still", "I", 3.0)))
    assert events == []
    assert not any(s.recording for s in statuses)


@pytest.mark.parametrize("letter,shape", [("J", "I"), ("Z", "D")])
def test_gate_triggers_ends_and_classifies(v2_gate, letter, shape):
    frames = stream(("still", shape, 1.0), (letter, shape, 1.2), ("still", shape, 2.0))
    events, statuses = run(v2_gate, frames)

    assert len(events) == 1, events
    t_event, ev = events[0]
    assert ev["outcome"] == "accepted" and ev["label"] == letter
    # Ended after the stroke (starts at 1.0 s, lasts 1.2 s), well before the stream ends.
    assert 1.8 < t_event < 2.8
    # Static classification is paused while recording and for hold_seconds after.
    i_evt = [i for i, (_, t, _) in enumerate(frames) if t == t_event][0]
    assert statuses[i_evt].in_hold
    first_after_hold = next(i for i, (_, t, _) in enumerate(frames)
                            if t >= t_event + v2_gate.params.hold_seconds)
    assert not statuses[first_after_hold].skip_static


def test_between_letter_movement_is_none_and_releases_immediately(v2_gate):
    frames = stream(("still", "A", 0.8), ("shift", "B", 0.6), ("still", "B", 1.5))
    events, statuses = run(v2_gate, frames)
    assert events, "the movement should trigger the gate"
    t_event, ev = events[0]
    assert ev["outcome"] == "none"
    i_evt = [i for i, (_, t, _) in enumerate(frames) if t == t_event][0]
    # No hold: the very next frame goes straight back to static classification.
    assert not statuses[i_evt].in_hold
    assert not statuses[i_evt + 1].skip_static


def test_brief_pause_does_not_fragment_clip():
    """Min-frames guard from the v1 live bug: a pause right after the trigger
    must not end the clip before min_frames have been buffered."""
    calls = []

    def fake_classify(frames, times, wh):
        calls.append(len(frames))
        return "Z", 0.99

    clf_params = GateParams(threshold=1.0, measure=hand_speed_v2, min_frames=10,
                            min_low_streak=3, max_frames=90)
    gate = MotionGate(fake_classify, clf_params, verbose=False)
    frames = stream(("still", "D", 0.5), ("Z", "D", 0.15), ("still", "D", 0.1),
                    ("Z", "D", 0.8), ("still", "D", 1.0))
    events, _ = run(gate, frames)
    assert len(events) == 1 and calls[0] >= 10


def test_undetected_frames_keep_timing_and_dont_crash(v2_gate):
    frames = stream(("still", "I", 1.0), ("J", "I", 1.2), ("still", "I", 1.5))
    frames = [(None if 40 <= i <= 43 else r, t, wh) for i, (r, t, wh) in enumerate(frames)]
    events, _ = run(v2_gate, frames)
    assert any(e["outcome"] == "accepted" and e["label"] == "J" for _, e in events)


def test_v1_mode_keeps_old_behaviour():
    """v1 checkpoints: centroid displacement in image units, no NONE class."""
    import torch
    from models.motion import MotionGRU

    torch.manual_seed(0)
    clf = MotionClassifierV1(MotionGRU().eval(), max_len=25, device=torch.device("cpu"))
    params = GateParams.for_classifier(clf, min_confidence=0.0)
    assert params.threshold == 0.025 and params.preroll == 0
    gate = MotionGate(clf, params, verbose=False)
    events, _ = run(gate, stream(("still", "I", 0.5), ("J", "I", 1.0), ("still", "I", 1.5)))
    assert events and events[0][1]["label"] in config.MOTION_LETTERS
