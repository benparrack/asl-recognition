"""
Preprocess the GISLR (Kaggle `asl-signs`) landmark parquets into a compact cache.

Run (after the download in HUMAN_TODO.md):
    python src/data/preprocess_gislr.py                  # skip participants already cached
    python src/data/preprocess_gislr.py --overwrite      # redo everything
    python src/data/preprocess_gislr.py --verify-only    # schema check + stats, no cache

Reads  config.GISLR_RAW_DIR: either the Kaggle archive asl-signs.zip or its
       extracted contents (train.csv, train_landmark_files/...)
Writes config.GISLR_CACHE_DIR:
    <participant_id>.npy      float16 (total_frames, N_SELECTED, 3), every
                              sequence of that participant back to back
    <participant_id>.csv      that shard's index rows
    index.csv                 all rows: sequence_id, participant_id, sign, label,
                              shard, offset, n_frames, left_frac, right_frac
    split.csv                 participant_id -> train / val / test
       results/gislr_schema.json   verified schema + dataset stats (for NOTES.md)

WHY READ FROM THE ZIP
=====================
The archive is tens of GB and extracting it roughly doubles that, which breaks
the ">20 GB free" headroom on this machine (DECISIONS.md). zipfile gives random
access to members, so each worker opens the archive and reads only its own
participant's parquets. An extracted directory works too, if one exists.

WHY SHARD BY PARTICIPANT
========================
21 participants suit ~20 worker processes, with one output file per worker and
no locking. The shard is also the unit of the split, so a participant's data
never straddles train and test even at the file level.

WHAT IS KEPT
============
Only the landmarks listed in data/holistic_landmarks.py (84 of 543), in that
module's order, with NaN preserved (NaN means "not detected" and the feature
code handles it). No normalisation or canonicalisation happens here. Those live
in holistic_features.py, so the demo can apply exactly the same code to live
frames, and features can change without re-reading 90k parquets.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import config  # noqa: E402
from data import holistic_landmarks as hl  # noqa: E402
from data.dataset import split_by_signer  # noqa: E402

ZIP_NAME = "asl-signs.zip"
PARQUET_COLUMNS = ["frame", "row_id", "type", "landmark_index", "x", "y", "z"]
TRAIN_CSV_COLUMNS = ["path", "participant_id", "sequence_id", "sign"]
_TYPE_OFFSET = {"left_hand": 0, "right_hand": hl.N_HAND,
                "pose": 2 * hl.N_HAND, "face": 2 * hl.N_HAND + hl.N_POSE_RAW}
_N_RAW = sum(hl.RAW_TYPES.values())                      # 543


# ------------------------------------------------------------------ source
class GislrSource:
    """The Kaggle archive or its extracted directory, behind one interface."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.zip_path = self.raw_dir / ZIP_NAME
        self._zip = None
        if (self.raw_dir / "train.csv").exists():
            self.kind = "dir"
        elif self.zip_path.exists():
            self.kind = "zip"
        else:
            raise FileNotFoundError(
                f"No GISLR data in {self.raw_dir}: expected train.csv or {ZIP_NAME}. "
                "See HUMAN_TODO.md for the download command.")

    def _open(self, rel: str):
        if self.kind == "dir":
            return open(self.raw_dir / rel, "rb")
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path)
        return io.BytesIO(self._zip.read(rel))

    def train_csv(self) -> pd.DataFrame:
        with self._open("train.csv") as f:
            return pd.read_csv(f)

    def sign_map(self) -> dict:
        with self._open("sign_to_prediction_index_map.json") as f:
            return json.load(f)

    def parquet(self, rel: str) -> pd.DataFrame:
        with self._open(rel) as f:
            return pd.read_parquet(f)


# ------------------------------------------------------------------ parsing
def check_parquet_schema(df: pd.DataFrame, rel: str = "") -> None:
    missing = set(PARQUET_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{rel}: parquet missing columns {sorted(missing)} (has {list(df.columns)})")
    unknown = set(df["type"].unique()) - set(hl.RAW_TYPES)
    if unknown:
        raise ValueError(f"{rel}: unexpected landmark types {sorted(unknown)}")
    for typ, n in hl.RAW_TYPES.items():
        idx = df.loc[df["type"] == typ, "landmark_index"]
        if len(idx) and (idx.min() < 0 or idx.max() >= n):
            raise ValueError(f"{rel}: {typ} landmark_index outside 0..{n - 1}")


def parquet_to_raw(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Long-format sequence -> {'left_hand': (T,21,3), 'right_hand': ..., 'pose':
    (T,33,3), 'face': (T,468,3)}, frames in ascending frame order, NaN where
    rows are missing or NaN. Rows are placed by (type, landmark_index), never
    by their position in the file.
    """
    frames = np.sort(df["frame"].unique())
    fpos = np.searchsorted(frames, df["frame"].to_numpy())
    offset = df["type"].map(_TYPE_OFFSET).to_numpy()
    col = offset + df["landmark_index"].to_numpy()
    full = np.full((len(frames), _N_RAW, 3), np.nan, dtype=np.float32)
    full[fpos, col] = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    out, start = {}, 0
    for typ in ("left_hand", "right_hand", "pose", "face"):
        n = hl.RAW_TYPES[typ]
        out[typ] = full[:, start:start + n]
        start += n
    return out


def parquet_to_selected(df: pd.DataFrame) -> np.ndarray:
    """Long-format sequence -> (T, N_SELECTED, 3) float32. The offline loader."""
    return hl.select_from_raw(parquet_to_raw(df))


# ------------------------------------------------------------------ workers
def process_participant(raw_dir: str, pid: int, rows: pd.DataFrame, cache_dir: str,
                        sign_map: dict) -> pd.DataFrame:
    src = GislrSource(Path(raw_dir))
    arrays, index, offset = [], [], 0
    for r in rows.itertuples(index=False):
        df = src.parquet(r.path)
        check_parquet_schema(df, r.path)
        seq = parquet_to_selected(df)
        left = ~np.isnan(seq[:, hl.LEFT_HAND]).any(axis=(1, 2))
        right = ~np.isnan(seq[:, hl.RIGHT_HAND]).any(axis=(1, 2))
        arrays.append(seq.astype(np.float16))
        index.append(dict(sequence_id=r.sequence_id, participant_id=pid, sign=r.sign,
                          label=sign_map[r.sign], shard=f"{pid}.npy", offset=offset,
                          n_frames=len(seq), left_frac=float(left.mean()),
                          right_frac=float(right.mean())))
        offset += len(seq)
    cache = Path(cache_dir)
    data = np.concatenate(arrays) if arrays else np.zeros((0, hl.N_SELECTED, 3), np.float16)
    tmp = cache / f"{pid}.tmp.npy"
    np.save(tmp, data)
    tmp.rename(cache / f"{pid}.npy")                     # a shard exists only when complete
    out = pd.DataFrame(index)
    out.to_csv(cache / f"{pid}.csv", index=False)
    return out


def load_shard(cache_dir: Path, shard: str) -> np.ndarray:
    return np.load(Path(cache_dir) / shard, mmap_mode="r")


def load_sequence(cache_dir: Path, row, shards: dict | None = None) -> np.ndarray:
    """One cached sequence as float32 (T, N_SELECTED, 3). Pass a dict to reuse
    memory-mapped shards across calls."""
    shards = {} if shards is None else shards
    if row["shard"] not in shards:
        shards[row["shard"]] = load_shard(cache_dir, row["shard"])
    o, n = int(row["offset"]), int(row["n_frames"])
    return np.asarray(shards[row["shard"]][o:o + n], dtype=np.float32)


# ------------------------------------------------------------------ split + stats
def participant_split(index: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """participant_id -> split, via the project's split_by_signer."""
    df = index[["participant_id"]].drop_duplicates().rename(columns={"participant_id": "signer_id"})
    tr, va, te = split_by_signer(df, seed=seed)
    parts = [pd.DataFrame({"participant_id": d["signer_id"], "split": name})
             for name, d in (("train", tr), ("val", va), ("test", te))]
    return pd.concat(parts, ignore_index=True).sort_values(["split", "participant_id"])


def dataset_stats(train: pd.DataFrame, index: pd.DataFrame | None, sample: pd.DataFrame) -> dict:
    stats = {
        "train_csv_columns": list(train.columns),
        "n_sequences": len(train),
        "n_participants": int(train["participant_id"].nunique()),
        "n_signs": int(train["sign"].nunique()),
        "per_participant": train["participant_id"].value_counts().describe().round(1).to_dict(),
        "per_sign": train["sign"].value_counts().describe().round(1).to_dict(),
        "sample_parquet": {
            "columns": list(sample.columns),
            "dtypes": {c: str(t) for c, t in sample.dtypes.items()},
            "rows_per_frame_by_type": (sample.groupby("type").size()
                                       / sample["frame"].nunique()).to_dict(),
            "nan_fraction_by_type": sample.groupby("type")["x"].apply(
                lambda s: float(s.isna().mean())).to_dict(),
        },
    }
    if index is not None and len(index):
        stats["frames"] = index["n_frames"].describe().round(1).to_dict()
        stats["left_hand_frac_mean"] = float(index["left_frac"].mean())
        stats["right_hand_frac_mean"] = float(index["right_frac"].mean())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="GISLR parquet -> float16 landmark cache")
    parser.add_argument("--raw-dir", type=Path, default=config.GISLR_RAW_DIR)
    parser.add_argument("--cache-dir", type=Path, default=config.GISLR_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--stats-out", type=Path, default=config.RESULTS_DIR / "gislr_schema.json")
    args = parser.parse_args()

    src = GislrSource(args.raw_dir)
    train = src.train_csv()
    missing = set(TRAIN_CSV_COLUMNS) - set(train.columns)
    if missing:
        raise ValueError(f"train.csv missing columns {sorted(missing)} (has {list(train.columns)})")
    sign_map = src.sign_map()
    if set(train["sign"]) - set(sign_map):
        raise ValueError("train.csv has signs missing from sign_to_prediction_index_map.json")
    sample = src.parquet(train["path"].iloc[0])
    check_parquet_schema(sample, train["path"].iloc[0])
    print(f"[gislr] source: {src.kind} ({args.raw_dir})")
    print(f"[gislr] {len(train)} sequences, {train['participant_id'].nunique()} participants, "
          f"{train['sign'].nunique()} signs; schema OK")

    if args.verify_only:
        stats = dataset_stats(train, None, sample)
        args.stats_out.write_text(json.dumps(stats, indent=2, default=str))
        print(json.dumps(stats, indent=2, default=str))
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [pid for pid in sorted(train["participant_id"].unique())
            if args.overwrite or not (args.cache_dir / f"{pid}.csv").exists()]
    print(f"[gislr] {len(todo)} participant shards to build "
          f"({train['participant_id'].nunique() - len(todo)} cached)")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(todo) or 1))) as pool:
        futs = {pool.submit(process_participant, str(args.raw_dir), int(pid),
                            train[train["participant_id"] == pid], str(args.cache_dir),
                            sign_map): pid for pid in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            out = fut.result()
            print(f"[gislr] {i}/{len(todo)} participant {futs[fut]}: {len(out)} sequences, "
                  f"{out['n_frames'].sum()} frames ({time.time() - t0:.0f}s)")

    index = pd.concat([pd.read_csv(args.cache_dir / f"{pid}.csv")
                       for pid in sorted(train["participant_id"].unique())], ignore_index=True)
    index.to_csv(args.cache_dir / "index.csv", index=False)
    split = participant_split(index)
    split.to_csv(args.cache_dir / "split.csv", index=False)

    stats = dataset_stats(train, index, sample)
    stats["split_participants"] = split.groupby("split")["participant_id"].apply(list).to_dict()
    args.stats_out.write_text(json.dumps(stats, indent=2, default=str))
    print(f"[gislr] index -> {args.cache_dir / 'index.csv'} ({len(index)} rows)")
    print(f"[gislr] split -> {args.cache_dir / 'split.csv'}: "
          f"{split['split'].value_counts().to_dict()} participants")
    print(f"[gislr] stats -> {args.stats_out}")


if __name__ == "__main__":
    main()
