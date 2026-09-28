"""
The J/Z motion gate, pulled out of demo.py so it can be replay-tested offline.

demo.py feeds it one frame at a time (raw landmarks or None, a timestamp, the
frame size); the replay tests feed it synthetic clips through the exact same
update() call. There is one implementation of the gate, not a demo copy and a
test copy.

STATE MACHINE
=============
    idle --(hand speed > threshold)--> recording
    recording --(speed low for min_low_streak frames AND >= min_frames buffered,
                 or max_frames buffered)--> classify
    classify --> J/Z with conf >= min_confidence : hold (static paused hold_seconds)
             --> NONE                            : release immediately, no hold
             --> low confidence                  : release immediately
             --> too few frames                  : discard, release

MOTION MEASURE
==============
v1 thresholded the raw centroid displacement in image-fraction units, so the
same gesture triggered or didn't depending on how close the hand was to the
camera. v2 divides by the hand's own size (wrist->middle-MCP) and by the frame
interval: "hand-lengths per second", which is distance- and FPS-invariant.
v1 checkpoints keep the v1 measure so their behaviour is unchanged.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

import config
from data import motion_features as mf
from data.landmarks import normalise_landmarks


# ------------------------------------------------------------------ motion measures
def centroid_displacement_v1(prev_raw, raw, dt, frame_wh) -> float:
    """v1: frame-to-frame centroid displacement in image-fraction units."""
    if prev_raw is None or raw is None:
        return 0.0
    return float(np.linalg.norm(raw[:, :2].mean(axis=0) - prev_raw[:, :2].mean(axis=0)))


def hand_speed_v2(prev_raw, raw, dt, frame_wh) -> float:
    """v2: centroid speed in hand-lengths per second (scale- and FPS-invariant)."""
    if prev_raw is None or raw is None or dt <= 0:
        return 0.0
    aspect = frame_wh[0] / float(frame_wh[1])
    a = raw[:, :2] * (aspect, 1.0)
    b = prev_raw[:, :2] * (aspect, 1.0)
    size = np.linalg.norm(a[mf.MIDDLE_MCP] - a[mf.WRIST])
    if size < 1e-6:
        return 0.0
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)) / size / dt)


# ------------------------------------------------------------------ classifiers
class MotionClassifierV1:
    """v1 checkpoint: normalised landmark vectors, tail zero-padded to max_len."""

    feature_version = 1

    def __init__(self, model, max_len: int, device):
        self.model, self.max_len, self.device = model, max_len, device
        self.classes = config.MOTION_LETTERS

    def __call__(self, frames, timestamps, frame_wh):
        vecs = [normalise_landmarks(f) for f in frames if f is not None]
        seq = np.array(vecs, dtype=np.float32)
        if len(seq) >= self.max_len:
            seq = seq[:self.max_len]
        else:
            seq = np.concatenate([seq, np.zeros((self.max_len - len(seq), seq.shape[1]),
                                                dtype=np.float32)])
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.from_numpy(seq)[None].to(self.device)), 1)[0]
        conf, idx = probs.max(0)
        return self.classes[int(idx)], float(conf)


class MotionClassifierV2:
    """v2 checkpoint: raw clip -> motion_features.clip_to_features -> model."""

    feature_version = 2

    def __init__(self, model, device, resample_len: int = mf.RESAMPLE_LEN,
                 classes=tuple(config.MOTION_V2_CLASSES)):
        self.model, self.device, self.resample_len = model, device, resample_len
        self.classes = list(classes)

    def __call__(self, frames, timestamps, frame_wh):
        raw = np.stack([f if f is not None else np.full((21, 3), np.nan) for f in frames])
        feats = mf.clip_to_features(raw, np.asarray(timestamps), frame_wh[0], frame_wh[1],
                                    self.resample_len)
        if feats is None:
            return None, 0.0
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.from_numpy(feats)[None].to(self.device)), 1)[0]
        conf, idx = probs.max(0)
        return self.classes[int(idx)], float(conf)


def load_motion_classifier(checkpoint_path, device):
    """Dispatch on the checkpoint's feature_version (absent -> v1)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    version = ckpt.get("feature_version", 1)

    if version == 1:
        if ckpt.get("classes") != config.MOTION_LETTERS:
            raise RuntimeError(f"Motion checkpoint classes {ckpt.get('classes')} != "
                               f"config.MOTION_LETTERS {config.MOTION_LETTERS}")
        from models.motion import MotionGRU
        model = MotionGRU(hidden_size=ckpt["args"]["hidden_size"], dropout=ckpt["args"]["dropout"])
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        print(f"[demo] motion v1 gru, epoch {ckpt['epoch']}, val acc {ckpt['val_acc']:.4f}, "
              f"max_len={ckpt['max_len']}")
        return MotionClassifierV1(model, ckpt["max_len"], device)

    if version == 2:
        if ckpt.get("classes") != config.MOTION_V2_CLASSES:
            raise RuntimeError(f"Motion checkpoint classes {ckpt.get('classes')} != "
                               f"config.MOTION_V2_CLASSES {config.MOTION_V2_CLASSES}")
        if ckpt["feature_dim"] != mf.FEATURE_DIM:
            raise RuntimeError(f"Checkpoint feature_dim {ckpt['feature_dim']} != "
                               f"motion_features.FEATURE_DIM {mf.FEATURE_DIM}: features changed "
                               "since this model was trained")
        from models.motion import MotionGRUv2
        model = MotionGRUv2(input_size=ckpt["feature_dim"], hidden_size=ckpt["hidden_size"],
                            dropout=ckpt["dropout"])
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        print(f"[demo] motion v2 gru, epoch {ckpt['epoch']}, val acc {ckpt['val_acc']:.4f}, "
              f"resample_len={ckpt['resample_len']}")
        return MotionClassifierV2(model, device, ckpt["resample_len"], ckpt["classes"])

    raise RuntimeError(f"unknown motion feature_version {version}")


# ------------------------------------------------------------------ the gate
@dataclass
class GateParams:
    threshold: float
    measure: callable
    min_frames: int = 8
    min_low_streak: int = 3
    max_frames: int = 25
    hold_seconds: float = 1.5
    min_confidence: float = 0.6
    preroll: int = 0
    # v1 only buffered DETECTED frames, so its max-length cutoff counted those;
    # v2 keeps undetected frames (as NaN) to keep timing honest.
    max_counts_detected_only: bool = False

    @classmethod
    def for_classifier(cls, classifier, threshold=None, min_confidence=0.6, fps_hint=30.0):
        if classifier.feature_version == 1:
            return cls(threshold=0.025 if threshold is None else threshold,
                       measure=centroid_displacement_v1, max_frames=classifier.max_len,
                       min_confidence=min_confidence, max_counts_detected_only=True)
        return cls.v2_default(threshold, min_confidence, fps_hint)

    @classmethod
    def v2_default(cls, threshold=None, min_confidence=0.6, fps_hint=30.0):
        # ~2.5 s max matches the 2.0 s capture window plus slack; 4-frame low
        # streak (~130 ms) rides out the brief pause at Z's corners.
        return cls(threshold=1.0 if threshold is None else threshold,
                   measure=hand_speed_v2, min_frames=10, min_low_streak=4,
                   max_frames=int(2.5 * fps_hint), min_confidence=min_confidence, preroll=3)


@dataclass
class GateStatus:
    recording: bool
    in_hold: bool
    event: dict | None = None          # set on the frame a clip is resolved
    buffered: int = 0

    @property
    def skip_static(self) -> bool:
        return self.recording or self.in_hold


@dataclass
class MotionGate:
    classify: callable
    params: GateParams
    verbose: bool = True

    recording: bool = field(default=False, init=False)
    hold_until: float = field(default=-np.inf, init=False)
    held_letter: tuple | None = field(default=None, init=False)

    def __post_init__(self):
        self._prev_raw = None
        self._prev_t = None
        self._frames: list = []
        self._times: list = []
        self._low = 0
        self._preroll = deque(maxlen=max(self.params.preroll, 1))

    def _log(self, msg):
        if self.verbose:
            print(f"[motion] {msg}")

    def _reset(self):
        self.recording = False
        self._frames, self._times, self._low = [], [], 0

    def update(self, raw, t: float, frame_wh) -> GateStatus:
        p = self.params
        if t < self.hold_until:
            return GateStatus(recording=False, in_hold=True)

        dt = (t - self._prev_t) if self._prev_t is not None else 0.0
        speed = p.measure(self._prev_raw, raw, dt, frame_wh)
        if raw is not None:
            self._prev_raw, self._prev_t = raw, t

        event = None
        if not self.recording:
            if raw is not None and speed > p.threshold:
                self.recording = True
                pre = list(self._preroll) if p.preroll else []
                self._frames = [f for f, _ in pre] + [raw]
                self._times = [ts for _, ts in pre] + [t]
                self._low = 0
                self._log(f"triggered, speed {speed:.4f}")
            if p.preroll:
                self._preroll.append((raw, t))
        else:
            self._frames.append(raw)
            self._times.append(t)
            self._low = 0 if (raw is not None and speed > p.threshold) else self._low + 1

            n_detected = sum(f is not None for f in self._frames)
            n_for_max = n_detected if p.max_counts_detected_only else len(self._frames)
            # A brief pause mid-gesture must not end a clip that hasn't reached
            # the minimum length yet; only max_frames can end a short one.
            done = ((self._low >= p.min_low_streak and n_detected >= p.min_frames)
                    or n_for_max >= p.max_frames)
            if done:
                event = self._resolve(t, n_detected, frame_wh)
                self._reset()
                self._preroll.clear()

        return GateStatus(recording=self.recording, in_hold=t < self.hold_until,
                          event=event, buffered=len(self._frames))

    def _resolve(self, t, n_detected, frame_wh) -> dict:
        p = self.params
        if n_detected < p.min_frames:
            self._log(f"clip too short ({n_detected} frames < {p.min_frames}), discarded")
            return {"outcome": "too_short", "frames": n_detected}

        pred, conf = self.classify(self._frames, self._times, frame_wh)
        if pred is None:
            self._log("clip unusable after featurisation, discarded")
            return {"outcome": "too_short", "frames": n_detected}
        if pred == config.MOTION_NONE:
            self._log(f"NONE (not J/Z), confidence {conf:.2f}, {n_detected} frames -- released")
            return {"outcome": "none", "label": pred, "confidence": conf, "frames": n_detected}
        if conf < p.min_confidence:
            self._log(f"{pred} REJECTED, confidence {conf:.2f} < min-confidence "
                      f"{p.min_confidence}, {n_detected} frames")
            return {"outcome": "rejected", "label": pred, "confidence": conf, "frames": n_detected}

        self.hold_until = t + p.hold_seconds
        self.held_letter = (pred, conf)
        self._log(f"{pred} accepted, confidence {conf:.2f}, {n_detected} frames")
        return {"outcome": "accepted", "label": pred, "confidence": conf, "frames": n_detected}


# ------------------------------------------------------------------ offline segmentation
def gate_segment(raw, timestamps, frame_wh, params: GateParams | None = None):
    """
    Run the live gate over a recorded clip and return the part of it the gate
    would have handed to the classifier: (raw_segment, timestamps_segment), or
    None if the gate never fired on at least min_frames detected frames.

    Why: capture records a fixed 2 s window (still hand, letter, still hand),
    but live the model only ever sees what the gate cuts out, i.e. roughly
    trigger-minus-preroll to the first low-speed streak. Training on the whole
    window would teach it that a J fills about half the time axis, and live it
    fills nearly all of it. Cutting training clips with this same gate code keeps
    the two identical. If the gate fires more than once, the longest segment
    wins. A segment still open when the clip ends is kept: live, the gate would
    have run on a little longer.
    """
    params = params or GateParams.v2_default()
    captured = []

    def grab(frames, times, wh):
        captured.append((frames, times))
        return None, 0.0            # "unusable": no hold, the gate just resets

    gate = MotionGate(grab, params, verbose=False)
    for f, t in zip(raw, timestamps):
        gate.update(None if np.isnan(f).any() else f, float(t), frame_wh)
    if gate.recording and sum(f is not None for f in gate._frames) >= params.min_frames:
        captured.append((gate._frames, gate._times))
    if not captured:
        return None
    frames, times = max(captured, key=lambda c: len(c[0]))
    seg = np.stack([f if f is not None else np.full((21, 3), np.nan) for f in frames])
    return seg.astype(np.float32), np.asarray(times, dtype=np.float64)
