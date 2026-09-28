"""
Word-level features from MediaPipe Holistic landmarks. The ONE implementation
used by GISLR training, evaluation, and the live words demo.

Input everywhere is a sequence of selected frames, (T, N_SELECTED, 3) in the
data/holistic_landmarks.py layout, NaN where something wasn't detected. Offline,
preprocess_gislr.parquet_to_selected builds it. Live, frame_from_holistic() below
builds one frame from `mp.solutions.holistic` results. Everything after that
point is shared.

PIPELINE (sequence_to_features)
===============================
1. canonicalise()  handedness. The dominant hand is the one seen in clearly more
                   frames, or, when both are seen (two-handed signs), the one
                   whose wrist travels further. If it's the left
                   hand, mirror x (x -> 1 - x) and swap every left/right pair
                   (hands, shoulders, eyes, ...). After this the dominant hand is
                   always Holistic's right_hand slot, so a left-handed signer
                   and a right-handed one look the same to the model, and so do
                   selfie-mirrored and unmirrored recordings.
2. trim()          drop leading/trailing frames with no hand at all.
3. (train only) augment_sequence()
4. frame_features() per-frame vector, layout below.
5. fit_length()    uniform subsample to MAX_LEN if longer; zero-pad plus mask if
                   shorter.

LIVE-DEMO CONSEQUENCE (see DECISIONS.md)
========================================
The letters demo cv2.flips the frame BEFORE detection, because its training
images were mirrored. The words path must NOT. Holistic runs on the raw camera
frame (as the GISLR recordings did), the flip is for display only, and
canonicalise() does the rest. Flipping first would swap Holistic's left/right
labels and then canonicalise() would "fix" the wrong thing.

BODY-RELATIVE COORDINATES (aspect-free)
=======================================
GISLR doesn't record frame size, and phone videos differ from a 4:3 webcam, so
no aspect correction is possible directly. Instead x and y are normalised
SEPARATELY by body measures in their own axis:
    X = (x - shoulder_mid_x) / shoulder_width_x
    Y = (y - shoulder_mid_y) / neck_height_y      (shoulder_mid_y - nose_y)
    Z = z / shoulder_width_x
An aspect change scales x relative to y. That scales x and shoulder_width_x
together, so X is unchanged, and Y never involved x. Hand shape is computed in
these coordinates too. Width and neck height are per-sequence medians (stable);
the midpoint is per frame, carried forward/back over pose dropouts.

PER-FRAME LAYOUT (FEATURE_DIM = 153)
====================================
Built as nested prefixes, so each ablation feature set is a leading slice:
    [  0: 63]  dominant hand shape (wrist-centred, max-distance 1)   | dom_shape
    [ 63]      dominant hand present                                 |
    [ 64:127]  non-dominant hand shape                               | both_shape
    [127]      non-dominant hand present                             |
    [128:136]  location: wrist + index tip (X, Y), each hand         | location
    [136:138]  dominant wrist - non-dominant wrist                   |
    [138]      pose (shoulders) detected this frame                  |
    [139:144]  dominant index tip -> chin, nose, forehead, eyes,     | full
               mouth distances                                       |
    [144]      face detected this frame                              |
    [145:153]  velocity: first difference of the 8 location channels |
Missing hands give zeros plus a 0 presence flag, never NaN.
"""

from __future__ import annotations

import numpy as np

from data import holistic_landmarks as hl
from data.landmarks import normalise_landmarks

FEATURE_VERSION = 1
MAX_LEN = 64
REFERENCE_FPS = 30.0        # live clips are resampled to this before featurising

WRIST, THUMB_TIP, INDEX_TIP = 0, 4, 8

SHAPE_DIM = 63
FEATURE_SETS = {             # name -> number of leading channels
    "dom_shape": 64,
    "both_shape": 128,
    "location": 139,
    "full": 153,
}
FEATURE_DIM = FEATURE_SETS["full"]
DOM_PRESENT, NONDOM_PRESENT = 63, 127
LOCATION = slice(128, 136)
HAND_TO_HAND = slice(136, 138)
POSE_VALID = 138
FACE_DIST = slice(139, 144)
FACE_VALID = 144
VELOCITY = slice(145, 153)

DOM, NONDOM = hl.RIGHT_HAND, hl.LEFT_HAND      # after canonicalise()

_FALLBACK_WIDTH, _FALLBACK_NECK = 0.25, 0.18


# ------------------------------------------------------------------ live adapter
def _landmark_array(lms, n: int) -> np.ndarray:
    if lms is None:
        return np.full((n, 3), np.nan, dtype=np.float32)
    pts = lms.landmark[:n]
    return np.array([(p.x, p.y, p.z) for p in pts], dtype=np.float32)


def frame_from_holistic(results) -> np.ndarray:
    """
    `mp.solutions.holistic.Holistic().process(rgb)` results -> one selected
    frame (N_SELECTED, 3), NaN where a part wasn't detected. Face meshes with
    refined irises (478 points) are cut to the 468 GISLR used.
    """
    raw = {
        "left_hand": _landmark_array(results.left_hand_landmarks, hl.N_HAND)[None],
        "right_hand": _landmark_array(results.right_hand_landmarks, hl.N_HAND)[None],
        "pose": _landmark_array(results.pose_landmarks, hl.N_POSE_RAW)[None],
        "face": _landmark_array(results.face_landmarks, hl.N_FACE_RAW)[None],
    }
    return hl.select_from_raw(raw)[0]


def resample_to_fps(frames: np.ndarray, timestamps: np.ndarray,
                    fps: float = REFERENCE_FPS) -> np.ndarray:
    """Live clips arrive at whatever rate the camera and Holistic manage;
    velocity channels assume ~REFERENCE_FPS. Nearest-frame resampling (not
    interpolation, which would smear NaN "not detected" frames). Identity
    when the input is already at fps."""
    t = np.asarray(timestamps, dtype=np.float64)
    t = t - t[0]
    n = int(np.floor(t[-1] * fps + 1e-6)) + 1
    grid = np.arange(n) / fps
    idx = np.clip(np.searchsorted(t, grid - 0.5 / fps, side="left"), 0, len(t) - 1)
    # pick whichever neighbour is nearest
    prev = np.clip(idx - 1, 0, len(t) - 1)
    idx = np.where(np.abs(t[prev] - grid) < np.abs(t[idx] - grid), prev, idx)
    return frames[idx]


# ------------------------------------------------------------------ sequence steps
def _present(seq: np.ndarray, part: slice) -> np.ndarray:
    return ~np.isnan(seq[:, part]).any(axis=(1, 2))


def canonicalise(seq: np.ndarray) -> tuple[np.ndarray, bool]:
    """Mirror so the dominant hand is in the right_hand slot. Returns (seq, flipped).

    Dominant = the hand seen in clearly more frames (the other in < half as
    many: a one-handed sign, the idle hand out of frame or undetected). When both
    are seen about equally, it's a two-handed sign, and frame counts would favour
    the steadier base hand, which is usually the NON-dominant one. Then the hand
    whose wrist travels further wins. Remaining ties stay unflipped."""
    n_left, n_right = _present(seq, hl.LEFT_HAND).sum(), _present(seq, hl.RIGHT_HAND).sum()
    if min(n_left, n_right) < 0.5 * max(n_left, n_right):
        flip = n_left > n_right
    else:
        def travel(part):
            w = seq[:, part][:, WRIST, :2]
            w = w[~np.isnan(w).any(axis=1)]
            return float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum()) if len(w) > 1 else 0.0
        flip = travel(hl.LEFT_HAND) > travel(hl.RIGHT_HAND)
    if not flip:
        return seq, False
    out = seq[:, hl.MIRROR_PERM].copy()
    out[..., 0] = 1.0 - out[..., 0]
    return out, True


def trim(seq: np.ndarray) -> np.ndarray:
    """Drop leading/trailing frames with neither hand. Empty stays empty."""
    any_hand = _present(seq, hl.LEFT_HAND) | _present(seq, hl.RIGHT_HAND)
    if not any_hand.any():
        return seq[:0]
    idx = np.flatnonzero(any_hand)
    return seq[idx[0]:idx[-1] + 1]


def _fill(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward- then back-fill NaN rows of (T, k). Returns (filled, valid-mask)."""
    valid = ~np.isnan(a).any(axis=1)
    if not valid.any():
        return a, valid
    idx = np.where(valid, np.arange(len(a)), -1)
    idx = np.maximum.accumulate(idx)
    idx[idx < 0] = np.flatnonzero(valid)[0]
    return a[idx], valid


def body_frame(seq: np.ndarray):
    """Per-frame shoulder midpoint (T,2), and per-sequence width / neck height,
    with pose -> face -> constant fallbacks. Also returns the pose-valid mask."""
    sl = seq[:, hl.pose_row(hl.POSE_SHOULDER_L), :2]
    sr = seq[:, hl.pose_row(hl.POSE_SHOULDER_R), :2]
    nose = seq[:, hl.pose_row(hl.POSE_NOSE), :2]
    both, pose_valid = _fill(np.concatenate([sl, sr, nose], axis=1))
    T = len(seq)
    if pose_valid.any():
        mid = (both[:, 0:2] + both[:, 2:4]) / 2
        width = float(np.median(np.abs(both[pose_valid, 0] - both[pose_valid, 2])))
        neck = float(np.median(mid[pose_valid, 1] - both[pose_valid, 5]))
    else:
        eyes = seq[:, [hl.face_row(hl.FACE_EYE1_OUTER), hl.face_row(hl.FACE_EYE2_OUTER)], :2]
        chin_fh = seq[:, [hl.face_row(hl.FACE_CHIN), hl.face_row(hl.FACE_FOREHEAD)], :2]
        f, fv = _fill(np.concatenate([eyes.reshape(T, -1), chin_fh.reshape(T, -1)], axis=1))
        if fv.any():
            width = float(np.median(np.abs(f[fv, 0] - f[fv, 2]))) * 2.6
            face_h = float(np.median(f[fv, 5] - f[fv, 7]))
            neck = face_h * 1.1
            mid = np.stack([(f[:, 0] + f[:, 2]) / 2, f[:, 5] + neck], axis=1)
        else:
            width, neck = _FALLBACK_WIDTH, _FALLBACK_NECK
            mid = np.tile([0.5, 0.6], (T, 1))
    width = width if width > 1e-4 else _FALLBACK_WIDTH
    neck = neck if neck > 1e-4 else _FALLBACK_NECK
    return mid, width, neck, pose_valid


def frame_features(seq: np.ndarray) -> np.ndarray:
    """Canonicalised, trimmed (T, N_SELECTED, 3) -> (T, FEATURE_DIM) float32."""
    T = len(seq)
    out = np.zeros((T, FEATURE_DIM), dtype=np.float32)
    if T == 0:
        return out
    mid, width, neck, pose_valid = body_frame(seq)
    scale = np.array([width, neck, width])

    def to_body(pts):                                     # (T, k, 3) -> body coords
        b = pts.copy()
        b[..., 0] -= mid[:, None, 0]
        b[..., 1] -= mid[:, None, 1]
        return b / scale

    hands = {}
    for name, part, shape_at, flag in (("dom", DOM, 0, DOM_PRESENT),
                                       ("nondom", NONDOM, 64, NONDOM_PRESENT)):
        h = to_body(seq[:, part])                         # (T, 21, 3)
        present = ~np.isnan(h).any(axis=(1, 2))
        for i in np.flatnonzero(present):
            out[i, shape_at:shape_at + SHAPE_DIM] = normalise_landmarks(h[i])
        out[:, flag] = present
        hands[name] = (h, present)

    loc = np.zeros((T, 8), dtype=np.float32)
    for k, name in enumerate(("dom", "nondom")):
        h, present = hands[name]
        pts = h[:, [WRIST, INDEX_TIP], :2].reshape(T, 4)
        loc[present, 4 * k:4 * k + 4] = pts[present]
    out[:, LOCATION] = loc

    (hd, pd_), (hn, pn) = hands["dom"], hands["nondom"]
    both = pd_ & pn
    out[both, HAND_TO_HAND] = hd[both, WRIST, :2] - hn[both, WRIST, :2]
    out[:, POSE_VALID] = pose_valid

    face_pts = np.stack([
        seq[:, hl.face_row(hl.FACE_CHIN), :2],
        seq[:, hl.face_row(hl.FACE_NOSE_TIP), :2],
        seq[:, hl.face_row(hl.FACE_FOREHEAD), :2],
        seq[:, [hl.face_row(i) for i in hl.FACE_EYES], :2].mean(axis=1),
        seq[:, [hl.face_row(i) for i in hl.FACE_LIPS], :2].mean(axis=1),
    ], axis=1)                                            # (T, 5, 2); NaN if face missing
    f, face_valid = _fill(face_pts.reshape(T, -1))
    if face_valid.any():
        fb = to_body(np.concatenate([f.reshape(T, 5, 2), np.zeros((T, 5, 1))], axis=2))[..., :2]
        tip = hd[:, INDEX_TIP, :2]
        d = np.linalg.norm(tip[:, None, :] - fb, axis=2)  # (T, 5)
        out[pd_, FACE_DIST] = d[pd_]
    out[:, FACE_VALID] = face_valid

    vel = np.diff(loc, axis=0, prepend=loc[:1])
    for k, name in enumerate(("dom", "nondom")):
        present = hands[name][1]
        ok = present & np.concatenate([[True], present[:-1]])
        vel[~ok, 4 * k:4 * k + 4] = 0
    out[:, VELOCITY] = vel
    return out


def fit_length(feats: np.ndarray, max_len: int = MAX_LEN):
    """(T, D) -> ((max_len, D), mask (max_len,) bool). Longer sequences are
    uniformly subsampled (keeps the whole sign); shorter ones zero-padded."""
    T, D = feats.shape
    out = np.zeros((max_len, D), dtype=np.float32)
    mask = np.zeros(max_len, dtype=bool)
    if T == 0:
        mask[0] = True                                   # one all-zero "empty" step
        return out, mask
    if T > max_len:
        feats = feats[np.linspace(0, T - 1, max_len).round().astype(int)]
        T = max_len
    out[:T], mask[:T] = feats, True
    return out, mask


# ------------------------------------------------------------------ augmentation
def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Train-time augmentation on a canonicalised, trimmed sequence, before
    featurisation (so augmented samples go through the real feature code):
      - affine jitter (rotation +-8 deg, scale 0.9-1.1, shift +-0.03), the same
        transform for every frame, so it adds no fake motion
      - time-stretch 0.8-1.25 by nearest-frame resampling
      - random frame drop (0-10 %)
      - with p=0.15, drop the non-dominant hand entirely
    No mirroring: canonicalisation already fixed handedness.
    """
    seq = seq.copy()
    T = len(seq)
    if T == 0:
        return seq
    theta = np.deg2rad(rng.uniform(-8, 8))
    s = rng.uniform(0.9, 1.1)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    xy = seq[..., :2]
    c = np.nanmean(xy.reshape(-1, 2), axis=0)
    seq[..., :2] = (xy - c) @ R.T * s + c + rng.uniform(-0.03, 0.03, 2)
    seq[..., 2] *= s

    n = max(2, int(round(T * rng.uniform(0.8, 1.25))))
    seq = seq[np.linspace(0, T - 1, n).round().astype(int)]
    keep = rng.random(len(seq)) >= rng.uniform(0, 0.10)
    keep[0] = keep[-1] = True
    seq = seq[keep]
    if rng.random() < 0.15:
        seq[:, NONDOM] = np.nan
    return seq


# ------------------------------------------------------------------ the entry point
def sequence_to_features(seq: np.ndarray, max_len: int = MAX_LEN,
                         rng: np.random.Generator | None = None):
    """
    Selected-layout sequence -> (features (max_len, FEATURE_DIM), mask (max_len,),
    info dict). Pass rng to augment (training only). Offline and live both call this.
    """
    seq = np.asarray(seq, dtype=np.float32)
    canon, flipped = canonicalise(seq)
    canon = trim(canon)
    if rng is not None:
        canon = augment_sequence(canon, rng)
    feats, mask = fit_length(frame_features(canon), max_len)
    return feats, mask, {"flipped": flipped, "n_frames": len(canon), "empty": len(canon) == 0}
