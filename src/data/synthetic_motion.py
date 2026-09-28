"""
Synthetic hand-landmark clips for OFFLINE verification of the motion pipeline.

These check that the code does what it claims (trajectory survives
featurisation, invariances hold, the gate segments and classifies a stream),
not that anything works on a real hand. Real-world accuracy only comes from
Ben's recordings -- see HUMAN_TODO.md.

A clip is a fixed handshape template (21 landmarks) translated along a path in
image coordinates, in exactly the raw format capture_motion_v2.py saves:
(T, 21, 3) image-normalised landmarks, NaN for undetected frames, timestamps,
frame width/height.
"""

from __future__ import annotations

import numpy as np

# Finger chains (MCP, PIP, DIP, TIP) in MediaPipe order.
_FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
            "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
_FINGER_X = {"index": -0.30, "middle": -0.10, "ring": 0.10, "pinky": 0.30}

HANDSHAPES = {
    "I": {"pinky"},             # pinky up: the J handshape
    "D": {"index"},             # index up: the Z handshape
    "B": {"index", "middle", "ring", "pinky"},
    "A": set(),
}


def hand_template(shape: str = "I") -> np.ndarray:
    """(21, 3) hand in 'hand units' (wrist->middle-MCP = 1), y pointing DOWN like
    image coordinates, so extended fingers have negative y."""
    extended = HANDSHAPES[shape]
    lm = np.zeros((21, 3))
    lm[0] = (0.0, 0.0, 0.0)
    lm[1:5] = [(-0.35, -0.25, -0.02), (-0.55, -0.45, -0.03),
               (-0.65, -0.65, -0.04), (-0.70, -0.85, -0.05)]
    for name, (mcp, pip, dip, tip) in _FINGERS.items():
        x = _FINGER_X[name]
        lm[mcp] = (x, -1.0, 0.0)
        if name in extended:
            lm[pip], lm[dip], lm[tip] = (x, -1.35, -0.01), (x, -1.6, -0.02), (x, -1.8, -0.03)
        else:
            lm[pip], lm[dip], lm[tip] = (x, -1.15, -0.10), (x, -0.95, -0.15), (x, -0.85, -0.10)
    return lm


def _path(kind: str, u: np.ndarray) -> np.ndarray:
    """Path offsets in hand units for progress u in [0, 1]. (N, 2)."""
    if kind == "J":
        # Straight down, then hook back and up.
        out = np.zeros((len(u), 2))
        down = u < 0.55
        out[down, 1] = 2.2 * u[down] / 0.55
        a = (u[~down] - 0.55) / 0.45 * np.pi              # 0..pi around a hook
        out[~down, 0] = -0.8 * (1 - np.cos(a))
        out[~down, 1] = 2.2 + 0.8 * np.sin(a)
        return out
    if kind == "Z":
        pts = np.array([[0, 0], [2.0, 0], [0, 2.0], [2.0, 2.0]], dtype=float)
        seg = np.minimum((u * 3).astype(int), 2)
        f = u * 3 - seg
        return pts[seg] + (pts[seg + 1] - pts[seg]) * f[:, None]
    if kind == "wave":
        return np.stack([1.0 * np.sin(2 * np.pi * 2 * u), np.zeros_like(u)], axis=1)
    if kind == "shift":                                    # reposition in a line
        return np.stack([1.5 * u, -0.7 * u], axis=1)
    if kind == "still":
        return np.zeros((len(u), 2))
    raise ValueError(kind)


def make_clip(path: str = "J", shape: str = "I", *, duration: float = 1.2, fps: float = 30.0,
              start_xy=(0.5, 0.35), hand_size: float = 0.08, frame_w: int = 640,
              frame_h: int = 480, rest_before: float = 0.0, rest_after: float = 0.0,
              twist_deg: float | None = None, jitter: float = 0.0,
              drop_frames=(), rng: np.random.Generator | None = None):
    """
    Returns (raw (T,21,3), timestamps (T,), frame_w, frame_h).

    hand_size is the wrist->middle-MCP length as a fraction of frame HEIGHT.
    twist_deg rotates the hand progressively during the stroke (J has a wrist
    twist; default 40 deg for J, 0 otherwise).
    """
    rng = rng or np.random.default_rng(0)
    if twist_deg is None:
        twist_deg = 40.0 if path == "J" else 0.0

    n_before, n_move, n_after = (int(round(x * fps)) for x in (rest_before, duration, rest_after))
    u = np.concatenate([np.zeros(n_before), np.linspace(0, 1, n_move), np.ones(n_after)])
    T = len(u)
    t = np.arange(T) / fps

    template = hand_template(shape)
    offsets = _path(path, u) * hand_size                               # square units
    aspect = frame_w / float(frame_h)
    raw = np.empty((T, 21, 3))
    for i in range(T):
        a = np.deg2rad(twist_deg * u[i])
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        pts = template[:, :2] @ R.T * hand_size + offsets[i]
        pts = pts + rng.normal(0, jitter, pts.shape)
        raw[i, :, 0] = start_xy[0] + pts[:, 0] / aspect               # back to fraction of width
        raw[i, :, 1] = start_xy[1] + pts[:, 1]
        raw[i, :, 2] = template[:, 2] * hand_size
    for i in drop_frames:
        raw[i] = np.nan
    return raw, t, frame_w, frame_h


def random_clip(label: str, rng: np.random.Generator, shape: str | None = None):
    """A randomised clip for a class label (J / Z / NONE), used to build small
    synthetic train/test sets. `shape` fixes the handshape for every class
    (the trajectory-only test) -- by default J uses I, Z uses D, like real signing."""
    if label == "J":
        path, default_shape = "J", "I"
    elif label == "Z":
        path, default_shape = "Z", "D"
    else:
        path = rng.choice(["still", "wave", "shift"])
        default_shape = rng.choice(["I", "D", "B", "A"])
    return make_clip(
        path, shape or default_shape,
        duration=rng.uniform(0.8, 1.6),
        start_xy=(rng.uniform(0.3, 0.6), rng.uniform(0.2, 0.45)),
        hand_size=rng.uniform(0.05, 0.11),
        rest_before=rng.uniform(0, 0.3), rest_after=rng.uniform(0, 0.3),
        jitter=0.002, rng=rng,
    )
