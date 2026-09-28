"""
Which MediaPipe Holistic landmarks the word pipeline keeps, and in what order.

This module is the single definition shared by the GISLR preprocessor (offline),
holistic_features.py (both), and the live demo adapter. If the lists change,
LANDMARK_SPEC_HASH changes, and a word checkpoint trained on the old selection
refuses to load, instead of silently reading the wrong points.

WHY THESE POINTS
================
A full Holistic frame is 543 points, 468 of them face mesh. Most of the face is
irrelevant to sign identity, and carrying it would make the cache ~6x bigger and
the model input mostly noise. Kept:

    both hands      21 + 21   everything handshape and hand motion needs
    upper body       9        nose, mouth corners, shoulders, elbows, wrists:
                              the reference frame for "where is the hand"
                              (shoulder midpoint + width). Legs/hips dropped:
                              they're usually out of frame for a webcam.
    face subset     33        lips outline, nose tip, eye corners, eyebrows,
                              chin, forehead: the places signs are made at
                              (MOTHER at the chin, FATHER at the forehead, ...)

LAYOUT OF ONE SELECTED FRAME (N_SELECTED = 84 rows of x, y, z)
=============================================================
    rows  0..20   left_hand   (GISLR/Holistic "left_hand" type)
    rows 21..41   right_hand
    rows 42..50   pose  (POSE_IDX order)
    rows 51..83   face  (FACE_IDX order)

Slot order here is Holistic's labelling. holistic_features.canonicalise() is
what turns it into "dominant, non-dominant".

Raw GISLR row offsets (type -> first row within a 543-row frame) follow the
competition's documented order: face 0..467, left_hand 468..488,
pose 489..521, right_hand 522..542. preprocess_gislr.py does not rely on that
order; it places rows by (type, landmark_index).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

# ------------------------------------------------------------------ raw layout
N_HAND = 21
N_POSE_RAW = 33
N_FACE_RAW = 468
RAW_TYPES = {"face": N_FACE_RAW, "left_hand": N_HAND, "pose": N_POSE_RAW, "right_hand": N_HAND}

# ------------------------------------------------------------------ selection
# Pose (MediaPipe Pose indices).
POSE_NOSE = 0
POSE_MOUTH_L, POSE_MOUTH_R = 9, 10
POSE_SHOULDER_L, POSE_SHOULDER_R = 11, 12
POSE_ELBOW_L, POSE_ELBOW_R = 13, 14
POSE_WRIST_L, POSE_WRIST_R = 15, 16
POSE_IDX = [POSE_NOSE, POSE_MOUTH_L, POSE_MOUTH_R, POSE_SHOULDER_L, POSE_SHOULDER_R,
            POSE_ELBOW_L, POSE_ELBOW_R, POSE_WRIST_L, POSE_WRIST_R]
# Left/right pairs, swapped when a frame is mirrored.
POSE_PAIRS = [(POSE_MOUTH_L, POSE_MOUTH_R), (POSE_SHOULDER_L, POSE_SHOULDER_R),
              (POSE_ELBOW_L, POSE_ELBOW_R), (POSE_WRIST_L, POSE_WRIST_R)]

# Face mesh (468-point FaceMesh indices).
FACE_NOSE_TIP = 1
FACE_CHIN = 152
FACE_FOREHEAD = 10
# FaceMesh indices are anatomical, not image-side: 33/133 and brow 70/105/107 are
# one eye, 263/362 and 300/334/336 the other. Named _1/_2 so nobody reads "L" as
# image-left; only the pairing matters (for mirroring).
FACE_EYE1_OUTER, FACE_EYE1_INNER = 33, 133
FACE_EYE2_INNER, FACE_EYE2_OUTER = 362, 263
FACE_BROW1 = [70, 105, 107]
FACE_BROW2 = [300, 334, 336]
# Outer lip contour: upper arc left->right, then lower arc left->right.
FACE_LIPS_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
FACE_LIPS_LOWER = [146, 91, 181, 84, 17, 314, 405, 321, 375]
FACE_IDX = ([FACE_NOSE_TIP, FACE_CHIN, FACE_FOREHEAD,
             FACE_EYE1_OUTER, FACE_EYE1_INNER, FACE_EYE2_INNER, FACE_EYE2_OUTER]
            + FACE_BROW1 + FACE_BROW2 + FACE_LIPS_UPPER + FACE_LIPS_LOWER)
FACE_PAIRS = ([(FACE_EYE1_OUTER, FACE_EYE2_OUTER), (FACE_EYE1_INNER, FACE_EYE2_INNER)]
              + list(zip(FACE_BROW1, FACE_BROW2))
              + [(61, 291), (185, 409), (40, 270), (39, 269), (37, 267),
                 (146, 375), (91, 321), (181, 405), (84, 314)])
FACE_EYES = [FACE_EYE1_OUTER, FACE_EYE1_INNER, FACE_EYE2_INNER, FACE_EYE2_OUTER]
FACE_LIPS = FACE_LIPS_UPPER + FACE_LIPS_LOWER

# ------------------------------------------------------------------ selected layout
LEFT_HAND = slice(0, N_HAND)
RIGHT_HAND = slice(N_HAND, 2 * N_HAND)
POSE = slice(2 * N_HAND, 2 * N_HAND + len(POSE_IDX))
FACE = slice(POSE.stop, POSE.stop + len(FACE_IDX))
N_SELECTED = FACE.stop                                    # 84


def pose_row(pose_index: int) -> int:
    """Row in a selected frame holding MediaPipe Pose landmark `pose_index`."""
    return POSE.start + POSE_IDX.index(pose_index)


def face_row(face_index: int) -> int:
    """Row in a selected frame holding FaceMesh landmark `face_index`."""
    return FACE.start + FACE_IDX.index(face_index)


def _mirror_permutation() -> np.ndarray:
    """Row permutation that goes with x -> 1 - x: swap the hand slots and every
    left/right pose and face pair, so "left shoulder" is still the left one."""
    perm = np.arange(N_SELECTED)
    perm[LEFT_HAND], perm[RIGHT_HAND] = np.arange(RIGHT_HAND.start, RIGHT_HAND.stop), \
        np.arange(LEFT_HAND.start, LEFT_HAND.stop)
    for a, b in POSE_PAIRS:
        perm[pose_row(a)], perm[pose_row(b)] = pose_row(b), pose_row(a)
    for a, b in FACE_PAIRS:
        perm[face_row(a)], perm[face_row(b)] = face_row(b), face_row(a)
    return perm


MIRROR_PERM = _mirror_permutation()
assert (MIRROR_PERM[MIRROR_PERM] == np.arange(N_SELECTED)).all()   # an involution


def _spec_hash() -> str:
    spec = {"pose": POSE_IDX, "face": FACE_IDX, "hand": N_HAND,
            "order": ["left_hand", "right_hand", "pose", "face"]}
    return hashlib.sha1(json.dumps(spec).encode()).hexdigest()[:12]


LANDMARK_SPEC_HASH = _spec_hash()


def select_from_raw(frames_raw: dict[str, np.ndarray]) -> np.ndarray:
    """
    {'left_hand': (T,21,3), 'right_hand': (T,21,3), 'pose': (T,33,3),
     'face': (T,468,3)} -> (T, N_SELECTED, 3), NaN kept where missing.
    The one place the selection is applied, for both parquet and live frames.
    """
    return np.concatenate([
        frames_raw["left_hand"],
        frames_raw["right_hand"],
        frames_raw["pose"][:, POSE_IDX],
        frames_raw["face"][:, FACE_IDX],
    ], axis=1)
