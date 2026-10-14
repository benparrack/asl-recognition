"""
Live isolated-sign (word) recognition: segmentation plus classification, one
frame at a time, pulled out of demo.py so it can be replay-tested offline.

demo.py feeds it `holistic_features.frame_from_holistic(results)`, a timestamp
and the frame size. The replay test (tests/test_words_live.py) feeds GISLR
validation sequences rendered as fake Holistic results through the same
update() call and checks the predictions against evaluate_words' offline
numbers. There is one live path, not a demo copy and a test copy.

CLASSIFICATION (WordClassifier)
===============================
    buffered frames (T, 84, 3) + timestamps
      -> resample_to_fps(REFERENCE_FPS)        live rate -> the ~30 fps GISLR rate
      -> sequence_to_features(max_len)         the SAME function WordDataset uses
      -> leading n_channels for the checkpoint's feature set
      -> model -> softmax -> top-k
Standardisation lives inside the model as buffers (models/words.py), so there's
nothing else to apply.

SEGMENTATION (WordRecogniser)
=============================
GISLR clips are single signs, roughly trimmed, so the live side must cut a
sign out of a continuous stream. Two ways, in order of robustness:

  manual  toggle() starts a clip and toggle() again ends and classifies it
          (cv2 has no key-up event, so it's a toggle, not hold-to-record).
  auto    idle --(a hand raised into the signing space for start_frames)--> recording
          recording --(no raised hand for end_absent frames)
                    or (>= min_frames AND hand speed below lull_speed for lull_frames)
                    or (max_frames)--> classify
          After a clip ends in a motion lull with the hands still up, auto
          start is disarmed until the hands drop or move again. Otherwise a
          held final handshape would retrigger at once.

"Raised" means a hand is detected with its wrist above shoulder height plus
rest_drop shoulder-widths (hands in the lap or at the sides are resting). If
the pose is missing, any detected hand counts as raised. Speeds are in
shoulder-widths per second, the scale the features use, so thresholds don't
depend on distance from the camera or on FPS. This is the same reason the J/Z
v2 gate uses hand-lengths per second.

The min-frames safeguards come from the J/Z gate: a clip with fewer than
min_hand_frames hand-detected frames is discarded ("too_short"), not
classified. A classifier fed three frames is guaranteed to answer something.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from data import holistic_features as hf
from data import holistic_landmarks as hl

_WRISTS = (hl.LEFT_HAND.start + hf.WRIST, hl.RIGHT_HAND.start + hf.WRIST)
_MIDDLE_MCP = 9
_SHOULDERS = (hl.pose_row(hl.POSE_SHOULDER_L), hl.pose_row(hl.POSE_SHOULDER_R))


class WordClassifier:
    """A loaded word checkpoint (evaluate_words.load_word_checkpoint) as a
    clip -> probabilities function."""

    def __init__(self, model, ckpt: dict, device):
        self.model, self.device = model.eval(), device
        self.classes: list[str] = ckpt["classes"]
        self.max_len: int = ckpt["max_len"]
        self.n_channels: int = ckpt["n_channels"]

    def features(self, frames: np.ndarray, timestamps=None):
        if timestamps is not None and len(frames) > 1:
            frames = hf.resample_to_fps(frames, np.asarray(timestamps))
        feats, mask, info = hf.sequence_to_features(frames, self.max_len)
        return feats[:, :self.n_channels], mask, info

    @torch.no_grad()
    def probs(self, frames: np.ndarray, timestamps=None) -> np.ndarray:
        feats, mask, _ = self.features(frames, timestamps)
        logits = self.model(torch.from_numpy(feats)[None].to(self.device),
                            torch.from_numpy(mask)[None].to(self.device))
        return torch.softmax(logits.float(), 1)[0].cpu().numpy()

    def top_k(self, frames, timestamps=None, k: int = 3) -> list[tuple[str, float]]:
        p = self.probs(frames, timestamps)
        return [(self.classes[i], float(p[i])) for i in np.argsort(-p)[:k]]


@dataclass
class SegParams:
    start_frames: int = 2        # consecutive raised frames to start a clip
    pre_roll: int = 4            # frames kept from before the trigger
    end_absent: int = 6          # frames without a raised hand that end a clip (~0.2 s)
    min_frames: int = 15         # a lull can't end a clip shorter than this (~0.5 s)
    lull_frames: int = 12        # frames below lull_speed that end a clip (~0.4 s)
    lull_speed: float = 0.4      # shoulder-widths per second
    max_frames: int = 120        # hard cap (~4 s); longer clips are subsampled anyway
    min_hand_frames: int = 5     # fewer hand frames than this -> discard, don't classify
    rest_drop: float = 1.2       # wrist this far below the shoulders (shoulder-widths) = resting
    min_confidence: float = 0.35  # top-1 below this -> reported, not accepted
    top_k: int = 3


@dataclass
class WordStatus:
    state: str                        # idle | recording | manual
    buffered: int
    raised: bool
    speed: float
    event: dict | None = None         # set on the frame a clip is closed


@dataclass
class _Clip:
    frames: list = field(default_factory=list)
    times: list = field(default_factory=list)


def _hand_scale(frame, wh) -> float | None:
    """Shoulder width in pixels, or 2.5 x hand length if the pose is missing."""
    s = frame[list(_SHOULDERS), :2] * wh
    if not np.isnan(s).any():
        d = float(np.linalg.norm(s[0] - s[1]))
        if d > 1e-6:
            return d
    for hand in (hl.RIGHT_HAND, hl.LEFT_HAND):
        h = frame[hand, :2] * wh
        if not np.isnan(h).any():
            d = float(np.linalg.norm(h[_MIDDLE_MCP] - h[hf.WRIST]))
            if d > 1e-6:
                return 2.5 * d
    return None


def raised_and_speed(frame, prev, dt, frame_wh, rest_drop) -> tuple[bool, float]:
    wh = np.asarray(frame_wh, dtype=np.float64)
    scale = _hand_scale(frame, wh)
    wrists = frame[list(_WRISTS), :2] * wh
    present = ~np.isnan(wrists).any(axis=1)
    if not present.any() or scale is None:
        return False, 0.0
    sh = frame[list(_SHOULDERS), 1] * wh[1]
    if np.isnan(sh).any():
        raised = True
    else:
        raised = bool((wrists[present, 1] < sh.mean() + rest_drop * scale).any())
    speed = 0.0
    if prev is not None and dt > 0:
        pw = prev[list(_WRISTS), :2] * wh
        both = present & ~np.isnan(pw).any(axis=1)
        if both.any():
            speed = float(np.linalg.norm(wrists[both] - pw[both], axis=1).max() / scale / dt)
    return raised, speed


class WordRecogniser:
    def __init__(self, classifier: WordClassifier, params: SegParams | None = None,
                 auto: bool = True):
        self.clf = classifier
        self.p = params or SegParams()
        self.auto = auto
        self.reset()

    def reset(self):
        self.state = "idle"
        self.clip = _Clip()
        self.history: deque = deque(maxlen=self.p.pre_roll)
        self.prev, self.prev_t = None, None
        self.raised_streak = self.absent_streak = self.lull_streak = self.move_streak = 0
        self.armed = True

    # ------------------------------------------------------------ manual
    def toggle(self, t: float) -> dict | None:
        """Start a manual clip, or end the current clip (manual or auto) and
        classify it. Returns the event when a clip is closed."""
        if self.state == "idle":
            self.state, self.clip = "manual", _Clip()
            return None
        return self._close("manual_stop")

    def flush(self, reason: str = "end_of_input") -> dict | None:
        """Close and classify a clip still open when the input ends."""
        return None if self.state == "idle" else self._close(reason)

    # ------------------------------------------------------------ per frame
    def update(self, frame: np.ndarray, t: float, frame_wh=(1, 1)) -> WordStatus:
        dt = t - self.prev_t if self.prev_t is not None else 0.0
        raised, speed = raised_and_speed(frame, self.prev, dt, frame_wh, self.p.rest_drop)
        self.prev, self.prev_t = frame, t
        self.raised_streak = self.raised_streak + 1 if raised else 0
        self.absent_streak = 0 if raised else self.absent_streak + 1
        self.move_streak = self.move_streak + 1 if speed > 2 * self.p.lull_speed else 0
        if not raised or self.move_streak >= 3:
            self.armed = True
        event = None

        if self.state == "manual":
            self._append(frame, t)
            if len(self.clip.frames) >= 10 * self.p.max_frames:     # runaway toggle
                event = self._close("max_frames")
        elif self.state == "recording":
            self._append(frame, t)
            self.lull_streak = self.lull_streak + 1 if speed < self.p.lull_speed else 0
            n = len(self.clip.frames)
            if self.absent_streak >= self.p.end_absent:
                event = self._close("hands_down")
            elif n >= self.p.min_frames and self.lull_streak >= self.p.lull_frames:
                event = self._close("motion_lull")
                self.armed = False
            elif n >= self.p.max_frames:
                event = self._close("max_frames")
                self.armed = False
        elif (self.auto and self.armed and raised
              and self.raised_streak >= self.p.start_frames):
            self.state, self.lull_streak = "recording", 0
            self.clip = _Clip(frames=[f for f, _ in self.history],
                              times=[s for _, s in self.history])
            self._append(frame, t)

        if self.state == "idle":
            self.history.append((frame, t))
        return WordStatus(self.state, len(self.clip.frames), raised, speed, event)

    def _append(self, frame, t):
        self.clip.frames.append(frame)
        self.clip.times.append(t)

    def _close(self, reason: str) -> dict:
        frames, times = self.clip.frames, self.clip.times
        self.state, self.clip = "idle", _Clip()
        self.history.clear()
        self.lull_streak = 0
        n_hand = sum(1 for f in frames if not (np.isnan(f[hl.LEFT_HAND]).any()
                                              and np.isnan(f[hl.RIGHT_HAND]).any()))
        event = dict(reason=reason, n_frames=len(frames), n_hand_frames=n_hand,
                     duration=(times[-1] - times[0]) if times else 0.0, top=[])
        if n_hand < self.p.min_hand_frames:
            event["outcome"] = "too_short"
            return event
        event["frames"], event["times"] = np.stack(frames), np.asarray(times)   # kept for teaching
        event["top"] = self.clf.top_k(event["frames"], event["times"], self.p.top_k)
        event["outcome"] = ("accepted" if event["top"][0][1] >= self.p.min_confidence
                            else "low_confidence")
        return event
