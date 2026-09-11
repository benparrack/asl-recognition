"""
Dataset loading and -- most importantly -- splitting.

READ THIS BEFORE TOUCHING THE SPLIT LOGIC
=========================================
The single most common way student vision projects produce meaningless results
is a random train/test split over frames.

Video is recorded at 30 fps. Frame 100 and frame 101 of the same clip are nearly
identical images. A random split puts one in train and the other in test, so the
model is evaluated on data it has effectively memorised. You will see 99%+
accuracy and it will mean nothing -- the model will collapse the first time it
sees a new person's hands.

This is textbook data leakage: the same failure mode as fitting a scaler on the
full dataset before splitting.

The correct unit of splitting is the SIGNER. A signer appears in exactly one of
train / val / test. The resulting accuracy will be substantially lower and it is
the honest number. Report both, explain the gap, and you have a genuine result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config
from data.landmarks import extract_dataset_landmarks


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------
REQUIRED_COLUMNS = ["path", "label", "signer_id", "session_id"]


def load_metadata() -> pd.DataFrame:
    """
    Load the metadata CSV describing every sample.

    Expected columns:
        path        relative path to the image, from PROCESSED_DIR
        label       single letter, e.g. "A"
        signer_id   stable identifier for the person signing
        session_id  recording session (lighting/background/outfit constant within)

    If your source dataset does not record who signed what, that is itself a
    finding worth a sentence in your report -- it means the published accuracy
    figures for that dataset are probably optimistic.
    """
    if not config.METADATA_CSV.exists():
        raise FileNotFoundError(
            f"No metadata at {config.METADATA_CSV}. "
            "Run src/data/build_metadata.py first."
        )

    df = pd.read_csv(config.METADATA_CSV)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {sorted(missing)}")

    df = df[df["label"].isin(config.CLASSES)].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def split_by_signer(df: pd.DataFrame, seed: int = config.SEED):
    """
    Partition the dataframe so that no signer appears in more than one split.

    Returns (train_df, val_df, test_df).
    """
    signers = np.array(sorted(df["signer_id"].unique()))

    if len(signers) < 3:
        raise ValueError(
            f"Only {len(signers)} distinct signer(s) found. A signer-disjoint split "
            "is impossible. Either obtain a multi-signer dataset (WLASL, ASL-LEX) or "
            "fall back to a session-disjoint split and state the limitation clearly."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(signers)

    n = len(signers)
    n_test = max(1, int(round(n * config.TEST_SIGNER_FRACTION)))
    n_val = max(1, int(round(n * config.VAL_SIGNER_FRACTION)))

    test_signers = set(signers[:n_test])
    val_signers = set(signers[n_test:n_test + n_val])
    train_signers = set(signers[n_test + n_val:])

    if not train_signers:
        raise ValueError("No signers left for training; adjust the split fractions.")

    splits = (
        df[df["signer_id"].isin(train_signers)].reset_index(drop=True),
        df[df["signer_id"].isin(val_signers)].reset_index(drop=True),
        df[df["signer_id"].isin(test_signers)].reset_index(drop=True),
    )

    assert not (train_signers & val_signers)
    assert not (train_signers & test_signers)
    assert not (val_signers & test_signers)

    return splits


def split_randomly_DO_NOT_USE(df: pd.DataFrame, seed: int = config.SEED):
    """
    Deliberately-wrong random split, kept for one purpose only: running it and
    reporting the inflated accuracy alongside the honest signer-disjoint number.

    That comparison is a result. Quantifying how much leakage inflates your score
    demonstrates you understand evaluation, and it is the kind of thing that
    distinguishes a strong report. Do not use this for your headline figure.
    """
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_test = int(n * config.TEST_SIGNER_FRACTION)
    n_val = int(n * config.VAL_SIGNER_FRACTION)
    return (
        shuffled[n_test + n_val:].reset_index(drop=True),
        shuffled[n_test:n_test + n_val].reset_index(drop=True),
        shuffled[:n_test].reset_index(drop=True),
    )


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------
def build_transforms(train: bool):
    """
    Augmentation applied to training data only.

    Think carefully about which augmentations are valid for THIS problem:

      - Horizontal flip: TEMPTING BUT WRONG BY DEFAULT. Flipping turns a right
        hand into a left hand. Signers do have a dominant hand, so a flip changes
        the semantics of the image. Either omit it, or flip deliberately to build
        left-handed coverage and document the choice.
      - Rotation: mild (+/- 15 deg) is realistic camera tilt. Large rotations
        change meaning -- K vs P differ only in orientation.
      - Colour jitter: valuable. It stops the model keying on skin tone or
        lighting, which is both an accuracy issue and a fairness issue.
      - Random crop / scale: valuable, simulates varying distance from camera.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            #transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
        ])

    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
    ])


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
class ASLImageDataset(Dataset):
    """Images -> class index. Used by the scratch CNN and the transfer model."""

    def __init__(self, df: pd.DataFrame, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.transform = build_transforms(train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(config.PROCESSED_DIR / row["path"]).convert("RGB")
        return self.transform(image), config.CLASS_TO_IDX[row["label"]]


class ASLLandmarkDataset(Dataset):
    """
    63-dimensional MediaPipe landmark vectors -> class index.

    Normalisation matters more here than you would expect. Raw landmark
    coordinates are in image space, so they encode where the hand happened to be
    and how large it appeared. Both are irrelevant to the handshape.
    See normalise_landmarks() in landmarks.py.
    """

    def __init__(self, df: pd.DataFrame, landmarks: np.ndarray):
        assert len(df) == len(landmarks), "metadata and landmark array are misaligned"
        self.df = df.reset_index(drop=True)
        self.landmarks = landmarks.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.landmarks[idx])
        y = config.CLASS_TO_IDX[self.df.iloc[idx]["label"]]
        return x, y


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def build_dataloaders(kind: str = "image", seed: int = config.SEED, split: str = "signer"):
    """
    Convenience wrapper: metadata -> split -> DataLoaders.

    kind:  "image" for the CNN paths, "landmark" for the MLP path.
    split: "signer" for the honest signer-disjoint split (default),
           "random" for pipeline smoke testing ONLY -- it leaks.
    """
    df = load_metadata()
    landmarks_by_row_id = None

    if kind == "landmark":
        if not config.LANDMARKS_NPZ.exists():
            print(f"[landmarks] {config.LANDMARKS_NPZ} not found -- extracting now "
                  "(one-time cost, cached to disk afterward)")
            extract_dataset_landmarks(df, config.PROCESSED_DIR, config.LANDMARKS_NPZ)

        cached = np.load(config.LANDMARKS_NPZ)
        landmarks, detected = cached["landmarks"], cached["detected"]
        assert len(landmarks) == len(df), (
            "landmarks.npz row count doesn't match metadata.csv -- delete "
            f"{config.LANDMARKS_NPZ} and re-run if the metadata changed"
        )

        fail_rate = 1.0 - detected.mean()
        print(f"[landmarks] detection failure rate: {fail_rate:.1%} "
              f"({(~detected).sum()}/{len(detected)})")

        # A stable row id survives the split_by_signer() filtering/reindexing
        # below, so we can look each row's landmark vector back up afterward.
        df = df.assign(_row_id=np.arange(len(df)))
        landmarks_by_row_id = {rid: vec for rid, vec in zip(df["_row_id"], landmarks)}
        df = df[detected].reset_index(drop=True)

    if split == "signer":
        train_df, val_df, test_df = split_by_signer(df, seed=seed)
    elif split == "random":
        print("[data] WARNING: random split -- frames from the same session appear "
              "in both train and test. Valid for pipeline testing only, never for "
              "a reported result.")
        train_df, val_df, test_df = split_randomly_DO_NOT_USE(df, seed=seed)
    else:
        raise ValueError(f"unknown split: {split}")

    print(f"[data] train {len(train_df):>6}  "
          f"val {len(val_df):>6}  test {len(test_df):>6}")
    print(f"[data] signers -> train {train_df['signer_id'].nunique()}  "
          f"val {val_df['signer_id'].nunique()}  test {test_df['signer_id'].nunique()}")

    if kind == "image":
        train_ds = ASLImageDataset(train_df, train=True)
        val_ds = ASLImageDataset(val_df, train=False)
        test_ds = ASLImageDataset(test_df, train=False)
    elif kind == "landmark":
        def landmarks_for(split_df: pd.DataFrame) -> np.ndarray:
            return np.stack([landmarks_by_row_id[rid] for rid in split_df["_row_id"]])

        train_ds = ASLLandmarkDataset(train_df, landmarks_for(train_df))
        val_ds = ASLLandmarkDataset(val_df, landmarks_for(val_df))
        test_ds = ASLLandmarkDataset(test_df, landmarks_for(test_df))
    else:
        raise ValueError(f"unknown kind: {kind}")

    common = dict(batch_size=config.BATCH_SIZE,
                  num_workers=config.NUM_WORKERS,
                  pin_memory=True)

    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


