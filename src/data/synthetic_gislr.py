"""
A small synthetic dataset in the GISLR (Kaggle `asl-signs`) on-disk format.

WHY THIS EXISTS
===============
The real download was blocked (403, competition rules not accepted; see
HUMAN_TODO.md). Instead of waiting, the word pipeline (preprocess -> features ->
training -> evaluation -> live replay) is built and tested end to end against
data with the competition's documented schema:

    train.csv                         path, participant_id, sequence_id, sign
    sign_to_prediction_index_map.json {sign: index}
    train_landmark_files/<participant_id>/<sequence_id>.parquet
        long format, one row per landmark per frame:
        frame, row_id ("<frame>-<type>-<landmark_index>"), type, landmark_index, x, y, z
        type in {face (468), left_hand (21), pose (33), right_hand (21)}
        a missing hand/face/pose is present as rows with NaN x, y, z

Numbers from it say the CODE works (training learns, splits are disjoint,
live == offline). They say nothing about real ASL accuracy.

Each synthetic sign is a combination of dominant-hand shape, location on the
body, movement, and whether the other hand joins in, so the one-hand vs
two-hand and static vs dynamic breakdowns have something to find. Some
participants are left-handed (mirrored), which exercises handedness
canonicalisation.
"""

from __future__ import annotations

import itertools
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from data import holistic_landmarks as hl
from data.synthetic_motion import hand_template

SHAPES = ["I", "D", "B", "A"]
LOCATIONS = {"chin": (0.0, 0.08), "forehead": (0.0, -0.10), "chest": (0.0, 0.30),
             "side": (0.18, 0.25)}          # offsets from the nose, fraction of frame
MOVES = ["still", "tap", "circle", "sweep"]
TYPE_ORDER = ["face", "left_hand", "pose", "right_hand"]


def sign_table(n_signs: int, seed: int = 0) -> list[dict]:
    """n distinct (shape, location, move, two_handed) combinations, named sign_000..."""
    combos = list(itertools.product(SHAPES, LOCATIONS, MOVES, (False, True)))
    order = np.random.default_rng(seed).permutation(len(combos))[:n_signs]
    return [dict(name=f"sign_{i:03d}", shape=combos[j][0], location=combos[j][1],
                 move=combos[j][2], two_handed=combos[j][3]) for i, j in enumerate(order)]


def _body(rng):
    """Selected-layout pose and face rows (unmirrored, subject facing camera:
    the subject's right side is image-left)."""
    nose = np.array([0.5, 0.30]) + rng.normal(0, 0.02, 2)
    s = rng.uniform(0.85, 1.15)                              # distance to camera
    pose = {hl.POSE_NOSE: (0, 0), hl.POSE_MOUTH_L: (0.02, 0.05), hl.POSE_MOUTH_R: (-0.02, 0.05),
            hl.POSE_SHOULDER_L: (0.13, 0.20), hl.POSE_SHOULDER_R: (-0.13, 0.20),
            hl.POSE_ELBOW_L: (0.17, 0.40), hl.POSE_ELBOW_R: (-0.17, 0.40),
            hl.POSE_WRIST_L: (0.15, 0.55), hl.POSE_WRIST_R: (-0.15, 0.55)}
    face = {hl.FACE_NOSE_TIP: (0, 0), hl.FACE_CHIN: (0, 0.09), hl.FACE_FOREHEAD: (0, -0.09),
            hl.FACE_EYE1_OUTER: (-0.05, -0.03), hl.FACE_EYE1_INNER: (-0.02, -0.03),
            hl.FACE_EYE2_INNER: (0.02, -0.03), hl.FACE_EYE2_OUTER: (0.05, -0.03)}
    for k, (a, b) in enumerate(zip(hl.FACE_BROW1, hl.FACE_BROW2)):
        face[a], face[b] = (-0.05 + 0.015 * k, -0.055), (0.05 - 0.015 * k, -0.055)
    for k, i in enumerate(hl.FACE_LIPS_UPPER):
        face[i] = (-0.03 + 0.06 * k / 10, 0.04 - 0.006 * np.sin(np.pi * k / 10))
    for k, i in enumerate(hl.FACE_LIPS_LOWER):
        face[i] = (-0.025 + 0.05 * k / 8, 0.05 + 0.006 * np.sin(np.pi * k / 8))
    return nose, s, pose, face


def _hand(shape, wrist_xy, size, angle, rng):
    t = hand_template(shape)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    out = np.empty((21, 3))
    out[:, :2] = wrist_xy + (t[:, :2] @ R.T) * size + rng.normal(0, 0.002, (21, 2))
    out[:, 2] = t[:, 2] * size
    return out


def make_sequence(sign: dict, rng: np.random.Generator, left_handed: bool = False) -> np.ndarray:
    """(T, N_SELECTED, 3) selected-layout frames for one performance of `sign`,
    NaN where a hand is absent. Mirrored if left_handed."""
    T_sign = int(rng.integers(12, 40))
    lead, tail = int(rng.integers(0, 5)), int(rng.integers(0, 5))
    T = lead + T_sign + tail
    nose, s, pose, face = _body(rng)
    size = 0.045 * s * rng.uniform(0.9, 1.1)

    frames = np.full((T, hl.N_SELECTED, 3), np.nan)
    for p, (dx, dy) in pose.items():
        frames[:, hl.pose_row(p), :2] = nose + np.array([dx, dy]) * s
        frames[:, hl.pose_row(p), 2] = 0.0
    for f, (dx, dy) in face.items():
        frames[:, hl.face_row(f), :2] = nose + np.array([dx, dy]) * s
        frames[:, hl.face_row(f), 2] = 0.0
    frames[:, hl.POSE.start:, :2] += rng.normal(0, 0.002, (T, hl.N_SELECTED - hl.POSE.start, 2))

    # Dominant = subject's right hand = image-left of the body, "right_hand" slot.
    loc = np.array(LOCATIONS[sign["location"]]) * s
    loc[0] = -loc[0]
    u = np.linspace(0, 1, T_sign)
    if sign["move"] == "still":
        path = np.zeros((T_sign, 2))
    elif sign["move"] == "tap":
        path = np.stack([np.zeros(T_sign), 0.03 * np.abs(np.sin(2 * np.pi * u))], 1)
    elif sign["move"] == "circle":
        path = 0.05 * np.stack([np.cos(2 * np.pi * u) - 1, np.sin(2 * np.pi * u)], 1)
    else:
        path = np.stack([0.12 * u, -0.04 * u], 1)
    path = path * s * rng.uniform(0.8, 1.2)
    start = nose + loc + rng.normal(0, 0.01, 2)
    for k in range(T_sign):
        frames[lead + k, hl.RIGHT_HAND] = _hand(sign["shape"], start + path[k], size,
                                                rng.normal(0, 0.05), rng)
        if sign["two_handed"]:
            base = nose + np.array([0.06, 0.32]) * s
            frames[lead + k, hl.LEFT_HAND] = _hand("B", base - path[k] * [1, 0.5], size,
                                                   np.pi / 2, rng)

    # Detection dropouts inside the sign.
    for k in np.flatnonzero(rng.random(T_sign) < 0.05):
        frames[lead + k, hl.RIGHT_HAND] = np.nan

    if left_handed:
        frames = frames[:, hl.MIRROR_PERM].copy()
        frames[..., 0] = 1.0 - frames[..., 0]
    return frames


def selected_to_long_df(frames: np.ndarray, rng: np.random.Generator,
                        first_frame: int = 0) -> pd.DataFrame:
    """Expand selected-layout frames to the full 543-row long parquet format.
    Unselected face/pose points get plausible filler near the head; hands keep
    their NaNs, as in the real data."""
    T = len(frames)
    parts = []
    for ti in range(T):
        f = first_frame + ti
        by_type = {
            "left_hand": frames[ti, hl.LEFT_HAND],
            "right_hand": frames[ti, hl.RIGHT_HAND],
            "pose": np.tile(frames[ti, hl.pose_row(hl.POSE_NOSE)], (hl.N_POSE_RAW, 1))
            + rng.normal(0, 0.05, (hl.N_POSE_RAW, 3)),
            "face": np.tile(frames[ti, hl.face_row(hl.FACE_NOSE_TIP)], (hl.N_FACE_RAW, 1))
            + rng.normal(0, 0.03, (hl.N_FACE_RAW, 3)),
        }
        by_type["pose"][hl.POSE_IDX] = frames[ti, hl.POSE]
        by_type["face"][hl.FACE_IDX] = frames[ti, hl.FACE]
        for typ in TYPE_ORDER:
            xyz = by_type[typ]
            n = len(xyz)
            parts.append(pd.DataFrame({
                "frame": np.full(n, f, dtype=np.int16),
                "row_id": [f"{f}-{typ}-{i}" for i in range(n)],
                "type": typ,
                "landmark_index": np.arange(n, dtype=np.int16),
                "x": xyz[:, 0].astype(np.float32), "y": xyz[:, 1].astype(np.float32),
                "z": xyz[:, 2].astype(np.float32),
            }))
    return pd.concat(parts, ignore_index=True)


def write_dataset(root: Path, n_signs: int = 12, n_participants: int = 8, reps: int = 3,
                  left_handed_frac: float = 0.25, as_zip: bool = False, seed: int = 0) -> Path:
    """
    Write a GISLR-format dataset under root. Returns the directory (or, with
    as_zip, root/asl-signs.zip laid out like the Kaggle archive).
    Participant ids are 5-digit ints like the real ones.
    """
    rng = np.random.default_rng(seed)
    signs = sign_table(n_signs, seed)
    pids = [int(p) for p in rng.choice(np.arange(10000, 99999), n_participants, replace=False)]
    lefties = set(pids[: int(round(left_handed_frac * n_participants))])

    files, rows = {}, []
    seq_id = int(rng.integers(1_000_000, 2_000_000))
    for pid in pids:
        for sign in signs:
            for _ in range(reps):
                seq_id += int(rng.integers(1, 1000))
                frames = make_sequence(sign, rng, left_handed=pid in lefties)
                rel = f"train_landmark_files/{pid}/{seq_id}.parquet"
                files[rel] = selected_to_long_df(frames, rng, first_frame=int(rng.integers(0, 50)))
                rows.append(dict(path=rel, participant_id=pid, sequence_id=seq_id,
                                 sign=sign["name"]))

    train_csv = pd.DataFrame(rows)
    sign_map = {s["name"]: i for i, s in enumerate(signs)}
    meta = {"signs": signs, "left_handed_participants": sorted(lefties)}

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if as_zip:
        import io
        zpath = root / "asl-signs.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            z.writestr("train.csv", train_csv.to_csv(index=False))
            z.writestr("sign_to_prediction_index_map.json", json.dumps(sign_map))
            for rel, df in files.items():
                buf = io.BytesIO()
                df.to_parquet(buf, index=False)
                z.writestr(rel, buf.getvalue())
        (root / "synthetic_meta.json").write_text(json.dumps(meta, indent=2))
        return zpath

    train_csv.to_csv(root / "train.csv", index=False)
    (root / "sign_to_prediction_index_map.json").write_text(json.dumps(sign_map))
    for rel, df in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(root / rel, index=False)
    (root / "synthetic_meta.json").write_text(json.dumps(meta, indent=2))
    return root
