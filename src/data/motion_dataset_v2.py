"""
Loading and splitting for motion v2 clips (J / Z / NONE) written by
capture_motion_v2.py.

Kept separate from motion_dataset.py (v1) rather than branching inside it: the
on-disk format is different (raw landmarks + timestamps vs normalised vectors),
the classes are different, and v1 has to keep working unchanged for the old
checkpoint.

Clips are first cut down by the live gate's own code (load_model_clip), then
featurised by data.motion_features.clip_to_features -- the same function the
demo calls -- and augmentation happens on the raw clip BEFORE that call, so
augmented samples go through the exact feature code real ones do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config
from data.dataset import split_by_signer
from data.motion_features import RESAMPLE_LEN, augment_raw_clip, clip_to_features

REQUIRED_COLUMNS = ["path", "label", "subtype", "signer_id", "session_id",
                    "n_frames", "n_detected", "frame_w", "frame_h", "format_version"]


def load_motion_v2_metadata(csv_path=config.MOTION_V2_METADATA_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No motion v2 metadata at {csv_path}. "
            "Record clips with src/data/capture_motion_v2.py first (see HUMAN_TODO.md)."
        )
    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {sorted(missing)}")

    df = df[(df["format_version"] == 2) & df["label"].isin(config.MOTION_V2_CLASSES)]
    df["subtype"] = df["subtype"].fillna("")
    return df.reset_index(drop=True)


def load_raw_clip(path):
    """npz -> (raw (T,21,3), timestamps (T,), frame_w, frame_h)."""
    d = np.load(path)
    return d["raw"], d["timestamps"], int(d["frame_w"]), int(d["frame_h"])


def load_model_clip(path):
    """load_raw_clip, then cut down to what the live gate would classify
    (motion_gate.gate_segment). Clips the gate never fires on (a held hand)
    stay whole: live they'd never reach the model, but they're still honest
    NONE examples. Everything that trains or evaluates a v2 model goes through
    here, so train, val, test and the diagnostics all see gate-cut clips."""
    from motion_gate import gate_segment

    raw, t, w, h = load_raw_clip(path)
    seg = gate_segment(raw, t, (w, h))
    if seg is None:
        return raw, t, w, h
    return seg[0], seg[1], w, h


class MotionClipDatasetV2(Dataset):
    """
    One clip -> (features (RESAMPLE_LEN, FEATURE_DIM), class index).

    Raw clips are loaded once and kept in memory (a few hundred clips of ~60
    frames is a few MB). With augment=True every __getitem__ draws a fresh
    augmentation; if an augmentation leaves too few detected frames the
    un-augmented clip is used instead, so a sample is never silently dropped.
    """

    def __init__(self, df: pd.DataFrame, root=config.MOTION_V2_DIR,
                 augment: bool = False, seed: int = config.SEED,
                 length: int = RESAMPLE_LEN):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.length = length
        self.rng = np.random.default_rng(seed)
        self.clips = [load_model_clip(root / p) for p in self.df["path"]]
        self.labels = [config.MOTION_V2_CLASS_TO_IDX[l] for l in self.df["label"]]

        self.base = []
        for i, (raw, t, w, h) in enumerate(self.clips):
            f = clip_to_features(raw, t, w, h, length)
            if f is None:
                raise ValueError(f"clip {self.df['path'][i]} has too few detected frames")
            self.base.append(f)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        feats = self.base[idx]
        if self.augment:
            raw, t, w, h = self.clips[idx]
            a_raw, a_t = augment_raw_clip(raw, t, w, h, self.rng)
            aug = clip_to_features(a_raw, a_t, w, h, self.length)
            if aug is not None:
                feats = aug
        return torch.from_numpy(feats), self.labels[idx]

    def feature_stats(self):
        """Per-channel mean/std over the un-augmented clips (call on TRAIN only)."""
        stacked = np.concatenate(self.base, axis=0)
        return stacked.mean(axis=0), stacked.std(axis=0)


def split_motion_v2(df: pd.DataFrame, seed: int = config.SEED):
    """Signer-disjoint split, reusing split_by_signer unchanged. For one person,
    record each session under its own signer id (ben_m1, ben_m2, ...)."""
    return split_by_signer(df, seed=seed)
