"""
Motion-letter (J, Z, NONE) features, v2. THE one definition used by training,
diagnostics, and the live demo.

WHY v2 EXISTS
=============
v1 fed the GRU per-frame `normalise_landmarks()` vectors: wrist at the origin,
unit scale. That throws away where the hand IS, so the only thing left for the
model to learn was handshape -- and handshape alone separates J (I-shape) from
Z (1-shape) at 100% (NOTES.md, 2026-09-28). The model never needed motion.

v2 keeps handshape (so the model can still use it where it's legitimate) but
adds explicit TRAJECTORY channels that survive the normalisation:

    handshape   63   normalise_landmarks() per frame -- same function the
                     letters MLP uses, so "handshape" means the same thing
                     everywhere in the project
    trajectory   6   wrist, index-tip, pinky-tip (x, y), each relative to its
                     own position in the clip's first frame, divided by the
                     clip's median hand size (wrist->middle-MCP). Position- and
                     distance-invariant, keeps the path, and is exactly zero
                     for a held hand -- so no handshape leaks in here.
    velocity     6   first differences of the trajectory channels
    rotation     2   angle of the wrist->middle-MCP vector relative to the first
                     frame, plus its first difference (J includes a wrist twist)

Only x/y go into trajectory/rotation. MediaPipe's hand z is a relative depth
estimate anchored to the wrist, much noisier than x/y, and a 2D path is all J
and Z are defined by.

TIME
====
Every clip is resampled to RESAMPLE_LEN frames by linear interpolation over the
recorded TIMESTAMPS (not frame indices, so a dropped camera frame doesn't
compress time). Undetected frames are NaN in the raw clip and get interpolated
across; leading/trailing undetected frames are trimmed. Tail zero-padding (v1)
is gone: it made short and long clips look structurally different for reasons
unrelated to the letter.

Clip duration is deliberately NOT a feature. Training clips are cut by the gate
code offline and live clips by the same gate online, but the live cut depends on
--motion-threshold and the camera's frame rate, so a duration channel would be a
train/serve mismatch waiting to happen.

ASPECT RATIO
============
MediaPipe x is a fraction of frame width, y a fraction of frame height, so on a
640x480 frame one unit of x is 1.33 units of y. Trajectory and rotation are
computed after scaling x by width/height, so a circle is a circle.

NEVER MIRROR
============
The direction of the path is part of the letter's identity (Z is traced
left-to-right from the signer's view), so augment_raw_clip() has no flip.
Capture and demo both mirror the camera frame the same way (cv2.flip before
detection), which is what keeps direction consistent between them.
"""

from __future__ import annotations

import numpy as np

import config
from data.landmarks import normalise_landmarks

FEATURE_VERSION = 2
RESAMPLE_LEN = 32
MIN_VALID_FRAMES = 4        # fewer detected frames than this -> not a usable clip

# MediaPipe Hands landmark indices used below.
WRIST = 0
INDEX_TIP = 8
MIDDLE_MCP = 9
PINKY_TIP = 20
TRAJECTORY_POINTS = (WRIST, INDEX_TIP, PINKY_TIP)

HANDSHAPE_DIM = config.LANDMARK_FEATURE_SIZE            # 63
TRAJECTORY_DIM = 2 * len(TRAJECTORY_POINTS)              # 6
FEATURE_DIM = HANDSHAPE_DIM + 2 * TRAJECTORY_DIM + 2     # 77

# Channel groups, so diagnostics can zero one group and see what's left.
HANDSHAPE_SLICE = slice(0, HANDSHAPE_DIM)
TRAJECTORY_SLICE = slice(HANDSHAPE_DIM, HANDSHAPE_DIM + TRAJECTORY_DIM)
VELOCITY_SLICE = slice(HANDSHAPE_DIM + TRAJECTORY_DIM, HANDSHAPE_DIM + 2 * TRAJECTORY_DIM)
ROTATION_SLICE = slice(HANDSHAPE_DIM + 2 * TRAJECTORY_DIM, FEATURE_DIM)
MOTION_SLICE = slice(HANDSHAPE_DIM, FEATURE_DIM)         # everything but handshape


def resample_clip(raw: np.ndarray, timestamps: np.ndarray, length: int = RESAMPLE_LEN):
    """
    (T, 21, 3) raw landmarks with NaN rows for undetected frames -> (length, 21, 3),
    evenly spaced in time between the first and last detected frame.

    Returns None when fewer than MIN_VALID_FRAMES frames were detected, or the
    detected frames span zero time.
    """
    raw = np.asarray(raw, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64)
    valid = ~np.isnan(raw).any(axis=(1, 2))

    if valid.sum() < MIN_VALID_FRAMES:
        return None

    t_valid = t[valid]
    if t_valid[-1] - t_valid[0] <= 1e-6:
        return None

    flat = raw[valid].reshape(valid.sum(), -1)                  # (Tv, 63)
    grid = np.linspace(t_valid[0], t_valid[-1], length)
    out = np.empty((length, flat.shape[1]))
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(grid, t_valid, flat[:, j])
    return out.reshape(length, 21, 3)


def clip_to_features(raw: np.ndarray, timestamps: np.ndarray,
                     frame_w: int, frame_h: int,
                     length: int = RESAMPLE_LEN) -> np.ndarray | None:
    """
    Raw clip -> (length, FEATURE_DIM) float32 feature sequence, or None if the
    clip has too few detected frames to say anything.

    raw         (T, 21, 3) MediaPipe image-normalised landmarks, NaN where the
                hand wasn't detected
    timestamps  (T,) seconds, any origin
    frame_w/h   pixel size of the frame the landmarks came from
    """
    seq = resample_clip(raw, timestamps, length)
    if seq is None:
        return None

    handshape = np.stack([normalise_landmarks(f) for f in seq])        # (L, 63)

    aspect = frame_w / float(frame_h)
    xy = seq[:, :, :2].copy()
    xy[:, :, 0] *= aspect                                             # square units

    hand_vec = xy[:, MIDDLE_MCP] - xy[:, WRIST]                       # (L, 2)
    scale = float(np.median(np.linalg.norm(hand_vec, axis=1)))
    if scale < 1e-6:
        return None

    # Each point relative to ITS OWN first position, not the first wrist: a
    # fingertip's offset from the wrist is handshape (I vs D put the tip in
    # different places), and handshape must not leak into the channels the
    # trajectory-only diagnostic relies on. A held hand gives exactly zero here.
    origin = xy[0, list(TRAJECTORY_POINTS)]
    traj = (xy[:, TRAJECTORY_POINTS] - origin) / scale                # (L, 3, 2)
    traj = traj.reshape(length, -1)                                   # (L, 6)
    vel = np.diff(traj, axis=0, prepend=traj[:1])                     # (L, 6)

    angle = np.unwrap(np.arctan2(hand_vec[:, 1], hand_vec[:, 0]))
    angle = angle - angle[0]
    d_angle = np.diff(angle, prepend=angle[:1])
    rot = np.stack([angle, d_angle], axis=1)                          # (L, 2)

    feats = np.concatenate([handshape, traj, vel, rot], axis=1).astype(np.float32)
    assert feats.shape == (length, FEATURE_DIM)
    return feats


def augment_raw_clip(raw: np.ndarray, timestamps: np.ndarray, frame_w: int, frame_h: int,
                     rng: np.random.Generator):
    """
    Train-time augmentation on the RAW clip, before featurisation, so every
    augmentation passes through the exact same feature code as real data.

    - rotation (+-10 deg) and scale (0.85-1.15) jitter about the clip's centre,
      the SAME transform on every frame (a per-frame jitter would inject fake
      motion)
    - time-stretch: a smooth monotone warp of the timestamps
    - random temporal crop of up to 10% at each end: jitter on where the gate
      starts and stops (clips arrive already gate-cut, so a bigger crop would
      start eating the letter itself)
    - random frame drop (frames become NaN, i.e. "undetected")
    - small Gaussian landmark noise
    No mirroring, ever -- see the module docstring.
    """
    raw = np.array(raw, dtype=np.float64, copy=True)
    t = np.array(timestamps, dtype=np.float64, copy=True)
    T = raw.shape[0]
    aspect = frame_w / float(frame_h)

    # Rotation + scale in square units about the clip's mean position.
    theta = np.deg2rad(rng.uniform(-10, 10))
    s = rng.uniform(0.85, 1.15)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    xy = raw[:, :, :2].copy()
    xy[:, :, 0] *= aspect
    centre = np.nanmean(xy.reshape(-1, 2), axis=0)
    xy = (xy - centre) @ rot.T * s + centre
    xy[:, :, 0] /= aspect
    raw[:, :, :2] = xy
    raw[:, :, 2] *= s

    # Time-stretch: t' = t + a * D * sin(pi * u), u in [0,1]. |a| < 1/pi keeps
    # it monotone.
    D = t[-1] - t[0]
    if D > 0:
        u = (t - t[0]) / D
        t = t + rng.uniform(-0.12, 0.12) * D * np.sin(np.pi * u)
        t = t * rng.uniform(0.85, 1.15)

    # Temporal crop.
    lo = int(rng.uniform(0, 0.10) * T)
    hi = T - int(rng.uniform(0, 0.10) * T)
    raw, t = raw[lo:hi], t[lo:hi]

    # Frame drop.
    drop = rng.random(raw.shape[0]) < rng.uniform(0, 0.15)
    raw[drop] = np.nan

    raw += rng.normal(0, 0.002, raw.shape)
    return raw, t
