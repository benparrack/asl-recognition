"""
Loading and splitting for the motion-letter (J, Z) landmark-sequence clips
capture_motion.py produces. Parallels src/data/dataset.py but for a different
data shape: variable-length (T, 63) sequences instead of single images/vectors.

Reuses split_by_signer() from dataset.py unchanged -- it only touches the
signer_id column, so the same signer-disjoint discipline applies here with no
duplicated logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config
from data.dataset import split_by_signer

REQUIRED_COLUMNS = ["path", "label", "signer_id", "session_id", "n_frames"]


def load_motion_metadata() -> pd.DataFrame:
    if not config.MOTION_METADATA_CSV.exists():
        raise FileNotFoundError(
            f"No motion metadata at {config.MOTION_METADATA_CSV}. "
            "Run src/data/capture_motion.py first."
        )

    df = pd.read_csv(config.MOTION_METADATA_CSV)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"motion_metadata.csv is missing required columns: {sorted(missing)}")

    df = df[df["label"].isin(config.MOTION_LETTERS)].reset_index(drop=True)
    return df


class MotionSequenceDataset(Dataset):
    """
    (T, 63) landmark sequence -> class index, padded/truncated to a fixed
    length so a plain DataLoader can batch them.

    Padding with zero-vectors at the tail is a simplification (no attention
    mask, no pack_padded_sequence) -- reasonable here because every clip is
    close to `max_len` already (recorded frame counts were 16-25 for a 1s
    capture window), so the padded fraction is small. Worth revisiting with
    proper sequence masking if clip lengths ever get more variable.
    """

    def __init__(self, df: pd.DataFrame, max_len: int):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        data = np.load(config.MOTION_DIR / row["path"])
        seq = data["landmarks"].astype(np.float32)   # (T, 63)

        T = seq.shape[0]
        if T >= self.max_len:
            seq = seq[:self.max_len]
        else:
            pad = np.zeros((self.max_len - T, seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        x = torch.from_numpy(seq)
        y = config.MOTION_CLASS_TO_IDX[row["label"]]
        return x, y


def build_motion_dataloaders(seed: int = config.SEED, batch_size: int = 8):
    """
    metadata -> signer-disjoint split -> DataLoaders.

    Returns (train_loader, val_loader, test_loader, max_len). max_len must be
    saved in the checkpoint -- inference (demo.py) needs to pad/truncate live
    sequences the exact same way, or the model sees a different input shape
    than it trained on.
    """
    df = load_motion_metadata()
    max_len = int(df["n_frames"].max())

    train_df, val_df, test_df = split_by_signer(df, seed=seed)

    print(f"[motion-data] train {len(train_df):>4}  val {len(val_df):>4}  "
          f"test {len(test_df):>4}  (max_len={max_len})")
    print(f"[motion-data] signers -> train {train_df['signer_id'].nunique()}  "
          f"val {val_df['signer_id'].nunique()}  test {test_df['signer_id'].nunique()}")

    train_ds = MotionSequenceDataset(train_df, max_len)
    val_ds = MotionSequenceDataset(val_df, max_len)
    test_ds = MotionSequenceDataset(test_df, max_len)

    common = dict(batch_size=batch_size, num_workers=0, pin_memory=True)

    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
        max_len,
    )
