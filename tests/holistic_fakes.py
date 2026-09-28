"""Stand-ins for mediapipe Holistic results objects, built from raw landmark
arrays, so live-path code can be driven without a camera."""

from types import SimpleNamespace

import numpy as np


def fake_landmarks(arr):
    if np.isnan(arr).any():
        return None
    return SimpleNamespace(landmark=[SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                                     for p in arr])


def fake_holistic_results(raw, i, refine_face=False):
    """raw: parquet_to_raw-style dict of (T, n, 3) arrays; frame i as a results object."""
    face = raw["face"][i]
    if refine_face and not np.isnan(face).any():
        face = np.concatenate([face, np.zeros((10, 3), dtype=face.dtype)])   # 478 points
    return SimpleNamespace(left_hand_landmarks=fake_landmarks(raw["left_hand"][i]),
                           right_hand_landmarks=fake_landmarks(raw["right_hand"][i]),
                           pose_landmarks=fake_landmarks(raw["pose"][i]),
                           face_landmarks=fake_landmarks(face))
