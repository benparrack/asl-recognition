"""
GISLR word sequences from the preprocess_gislr.py cache, as model-ready tensors.

Every sample goes cached landmarks -> holistic_features.sequence_to_features ->
the leading channels of the chosen feature set, so the ablation's four feature
sets are the same computation with different slices, not four pipelines.

Also computes per-sign PROPERTIES (how often the non-dominant hand is present,
how much the dominant hand moves) from TRAIN sequences. The evaluation uses
them to break accuracy down by one- vs two-handed and static vs dynamic signs.
They're descriptive and never used for training or model selection.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config
from data import holistic_features as hf
from data.preprocess_gislr import load_sequence

TWO_HAND_THRESHOLD = 0.5    # non-dominant hand present in > 50 % of a sign's frames


def load_word_index(cache_dir: Path = config.GISLR_CACHE_DIR) -> tuple[pd.DataFrame, list[str]]:
    """index.csv joined with split.csv, plus the class list (index = label)."""
    cache_dir = Path(cache_dir)
    for name in ("index.csv", "split.csv"):
        if not (cache_dir / name).exists():
            raise FileNotFoundError(f"{cache_dir / name} missing. Run "
                                    "src/data/preprocess_gislr.py first (HUMAN_TODO.md).")
    index = pd.read_csv(cache_dir / "index.csv")
    split = pd.read_csv(cache_dir / "split.csv")
    index = index.merge(split, on="participant_id", how="left", validate="many_to_one")
    if index["split"].isna().any():
        raise ValueError("participants in index.csv missing from split.csv")
    classes = (index[["label", "sign"]].drop_duplicates().sort_values("label")["sign"].tolist())
    if len(classes) != index["label"].max() + 1:
        raise ValueError("sign labels are not a contiguous 0..N-1 range")
    return index, classes


EXTRA_PID_OFFSET = 100_000   # keeps extra-dataset participant ids apart from GISLR's


def load_combined_index(cache_dir: Path, extra_cache_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    GISLR plus a second cache (preprocess_asl_citizen.py --vocab gislr) in
    GISLR's label space. Extra rows keep their own signer split, take the
    GISLR label of their sign name, and get participant_id + EXTRA_PID_OFFSET.
    `shard` becomes an absolute path, so WordDataset/load_sequence read either
    cache whatever cache_dir they're given. `dataset` says which one a row is.
    """
    index, classes = load_word_index(cache_dir)
    extra, _ = load_word_index(extra_cache_dir)
    unknown = set(extra["sign"]) - set(classes)
    if unknown:
        raise ValueError(f"{extra_cache_dir}: signs not in GISLR: {sorted(unknown)[:10]}")
    index = index.assign(dataset="gislr", shard=[str(Path(cache_dir).resolve() / s)
                                                for s in index["shard"]])
    extra = extra.assign(dataset=Path(extra_cache_dir).name,
                         label=extra["sign"].map({s: i for i, s in enumerate(classes)}),
                         participant_id=extra["participant_id"] + EXTRA_PID_OFFSET,
                         shard=[str(Path(extra_cache_dir).resolve() / s) for s in extra["shard"]])
    if set(extra["participant_id"]) & set(index["participant_id"]):
        raise ValueError("participant ids collide between the two caches")
    return pd.concat([index, extra[index.columns]], ignore_index=True), classes


class WordDataset(Dataset):
    """
    (features (max_len, n_channels) float32, mask (max_len,) bool, label).

    augment=True draws a fresh augmentation per access. The RNG is created
    lazily inside each DataLoader worker from torch.initial_seed(), which
    differs per worker and per epoch, so workers don't repeat each other.
    """

    def __init__(self, index: pd.DataFrame, cache_dir: Path = config.GISLR_CACHE_DIR,
                 feature_set: str = "full", max_len: int = hf.MAX_LEN, augment: bool = False):
        self.index = index.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.n_channels = hf.FEATURE_SETS[feature_set]
        self.max_len = max_len
        self.augment = augment
        self._shards: dict = {}
        self._rng = None
        self._rng_pid = None

    def __len__(self) -> int:
        return len(self.index)

    def _get_rng(self):
        if self._rng is None or self._rng_pid != os.getpid():
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
            self._rng_pid = os.getpid()
        return self._rng

    def raw(self, idx: int) -> np.ndarray:
        return load_sequence(self.cache_dir, self.index.iloc[idx], self._shards)

    def __getitem__(self, idx: int):
        rng = self._get_rng() if self.augment else None
        feats, mask, _ = hf.sequence_to_features(self.raw(idx), self.max_len, rng)
        return (torch.from_numpy(feats[:, :self.n_channels]), torch.from_numpy(mask),
                int(self.index.at[idx, "label"]))

    def feature_stats(self, max_samples: int = 3000, seed: int = config.SEED):
        """Per-channel mean/std over real (unmasked) frames of un-augmented
        samples. Call on TRAIN only."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self), size=min(max_samples, len(self)), replace=False)
        rows = []
        for i in idx:
            f, m, _ = hf.sequence_to_features(self.raw(int(i)), self.max_len)
            rows.append(f[m, :self.n_channels])
        stacked = np.concatenate(rows)
        return stacked.mean(0), stacked.std(0)


def sign_properties(index: pd.DataFrame, classes: list[str],
                    cache_dir: Path = config.GISLR_CACHE_DIR, per_sign: int = 30,
                    seed: int = config.SEED) -> pd.DataFrame:
    """
    Per sign, from up to `per_sign` TRAIN sequences:
      nondom_frac    median fraction of frames with the non-dominant hand present
      motion         median of the mean dominant-wrist speed (body units per frame)
      two_handed     nondom_frac > TWO_HAND_THRESHOLD
      motion_bucket  static / moderate / dynamic (terciles of motion across signs)
    """
    train = index[index["split"] == "train"]
    shards: dict = {}
    rows = []
    for label, g in train.groupby("label"):
        g = g.sample(min(per_sign, len(g)), random_state=seed)
        nd, mo = [], []
        for _, r in g.iterrows():
            f, m, _ = hf.sequence_to_features(load_sequence(cache_dir, r, shards))
            f = f[m]
            nd.append(float(f[:, hf.NONDOM_PRESENT].mean()))
            dom = f[:, hf.DOM_PRESENT] > 0
            v = f[dom][:, hf.VELOCITY][:, :2]
            mo.append(float(np.linalg.norm(v, axis=1).mean()) if len(v) else 0.0)
        rows.append(dict(label=int(label), sign=classes[int(label)],
                         nondom_frac=float(np.median(nd)), motion=float(np.median(mo))))
    props = pd.DataFrame(rows)
    props["two_handed"] = props["nondom_frac"] > TWO_HAND_THRESHOLD
    props["motion_bucket"] = pd.qcut(props["motion"].rank(method="first"), 3,
                                     labels=["static", "moderate", "dynamic"]).astype(str)
    return props.sort_values("label").reset_index(drop=True)


def load_or_compute_sign_properties(index, classes, cache_dir, out_path: Path) -> pd.DataFrame:
    if out_path.exists():
        props = pd.read_csv(out_path)
        if len(props) == index.loc[index["split"] == "train", "label"].nunique():
            return props
    props = sign_properties(index, classes, cache_dir)
    props.to_csv(out_path, index=False)
    return props
