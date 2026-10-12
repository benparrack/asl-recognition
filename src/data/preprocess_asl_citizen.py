"""
Extract Holistic landmarks from ASL Citizen (Microsoft, 83k webcam videos of
2,731 signs by 52 signers) into the same cache format as preprocess_gislr.py,
so WordDataset / train_words.py read it unchanged.

Run (after downloading ASL_Citizen.zip to config.ASL_CITIZEN_ZIP):
    python src/data/preprocess_asl_citizen.py              # skip shards already cached
    python src/data/preprocess_asl_citizen.py --limit 2    # 2 shards, timing check

Why: GISLR almost never has the second hand (NOTES.md). A 27-video sample of
ASL Citizen had both hands on every frame with hands up for two-handed signs.

Videos go through the live path (eval_word_videos.video_landmarks: Holistic on
the un-flipped frame, frame_from_holistic), then resample_to_fps to the 30 fps
GISLR rate (some clips are 15 fps), the same way demo clips are. Read straight
from the zip, so it never needs extracting.

Writes config.ASL_CITIZEN_CACHE_DIR:
    <pid>_<k>.npy / .csv   float16 (frames, N_SELECTED, 3) + index rows; a
                           participant is cut into shards of --shard-size videos
                           so an interrupted run loses minutes, not hours
    index.csv              sequence_id, participant_id, sign, label, gloss,
                           shard, offset, n_frames, left_frac, right_frac
    split.csv              the dataset's own signer-disjoint split (35/6/11)
Extraction runs at ~10 frames/s per process, so all 83k videos take ~13 h.
--vocab gislr (the default) keeps only glosses that are GISLR signs (~216 signs,
~8k videos, ~1.5 h) and names them as GISLR does, so the two datasets share
sign names; --vocab all keeps every gloss lower-cased. `label` is the sign's
index among the kept signs in sorted order.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
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
from data import holistic_features as hf  # noqa: E402
from data import holistic_landmarks as hl  # noqa: E402

PREFIX = "ASL_Citizen/"
# Numbered variants (DRINK1, DRINK2) are merged, like GISLR's one label per word.
# GISLR name -> ASL Citizen gloss, where they differ only in form, not sign
GISLR_ALIASES = {"dad": "FATHER", "mom": "MOTHER", "shoe": "SHOES", "eye": "EYES",
                 "grandma": "GRANDMOTHER", "grandpa": "GRANDFATHER", "kitty": "KITTEN",
                 "minemy": "MINE"}


def norm(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalpha())


def gislr_names(glosses, gislr_signs) -> dict[str, str]:
    """ASL Citizen gloss -> GISLR sign name, for glosses that are GISLR signs."""
    by_norm = {norm(GISLR_ALIASES.get(g, g)): g for g in gislr_signs}
    return {gl: by_norm[norm(gl)] for gl in glosses if norm(gl) in by_norm}


def read_splits(zf: zipfile.ZipFile, gislr_signs=None) -> pd.DataFrame:
    parts = []
    for split in ("train", "val", "test"):
        with zf.open(f"{PREFIX}splits/{split}.csv") as f:
            parts.append(pd.read_csv(f).assign(split=split))
    df = pd.concat(parts, ignore_index=True).rename(columns={
        "Participant ID": "participant", "Video file": "video", "Gloss": "gloss"})
    df["participant_id"] = df["participant"].str.lstrip("P").astype(int)
    if (df.groupby("participant_id")["split"].nunique() > 1).any():
        raise ValueError("a participant appears in more than one official split")
    if gislr_signs is None:
        df["sign"] = df["gloss"].str.lower()
    else:
        df["sign"] = df["gloss"].map(gislr_names(df["gloss"].unique(), gislr_signs))
        df = df.dropna(subset=["sign"]).reset_index(drop=True)
    classes = sorted(df["sign"].unique())
    df["label"] = df["sign"].map({s: i for i, s in enumerate(classes)})
    return df


def process_shard(zip_path: str, shard: str, rows: pd.DataFrame, cache_dir: str,
                  complexity: int) -> pd.DataFrame:
    import mediapipe as mp
    from eval_word_videos import video_landmarks

    arrays, index, offset = [], [], 0
    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as tmp, \
            mp.solutions.holistic.Holistic(static_image_mode=False,
                                           model_complexity=complexity) as h:
        for r in rows.itertuples(index=False):
            path = Path(tmp) / "clip.mp4"
            path.write_bytes(zf.read(PREFIX + "videos/" + r.video))
            h.reset()                                  # tracking must not carry across clips
            frames, times, _ = video_landmarks(path, h)
            if len(frames) < 2:
                print(f"[aslc] skip unreadable {r.video}", flush=True)
                continue
            seq = hf.resample_to_fps(np.stack(frames), times)
            left = ~np.isnan(seq[:, hl.LEFT_HAND]).any(axis=(1, 2))
            right = ~np.isnan(seq[:, hl.RIGHT_HAND]).any(axis=(1, 2))
            arrays.append(seq.astype(np.float16))
            index.append(dict(sequence_id=r.video, participant_id=r.participant_id, sign=r.sign,
                              label=r.label, gloss=r.gloss, shard=f"{shard}.npy", offset=offset,
                              n_frames=len(seq), left_frac=float(left.mean()),
                              right_frac=float(right.mean())))
            offset += len(seq)
    cache = Path(cache_dir)
    data = np.concatenate(arrays) if arrays else np.zeros((0, hl.N_SELECTED, 3), np.float16)
    np.save(cache / f"{shard}.tmp.npy", data)
    (cache / f"{shard}.tmp.npy").rename(cache / f"{shard}.npy")
    out = pd.DataFrame(index)
    out.to_csv(cache / f"{shard}.csv", index=False)    # written last: marks the shard done
    return out


def plan_shards(df: pd.DataFrame, size: int) -> dict[str, pd.DataFrame]:
    shards = {}
    for pid, g in df.groupby("participant_id"):
        for k, start in enumerate(range(0, len(g), size)):
            shards[f"{pid}_{k:02d}"] = g.iloc[start:start + size]
    return shards


def main() -> None:
    parser = argparse.ArgumentParser(description="ASL Citizen videos -> landmark cache")
    parser.add_argument("--zip", type=Path, default=config.ASL_CITIZEN_ZIP)
    parser.add_argument("--cache-dir", type=Path, default=config.ASL_CITIZEN_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--shard-size", type=int, default=200)
    parser.add_argument("--holistic-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--vocab", choices=["gislr", "all"], default="gislr")
    parser.add_argument("--gislr-index", type=Path, default=config.GISLR_INDEX_CSV)
    parser.add_argument("--limit", type=int, default=None, help="only build this many shards")
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip) as zf:
        df = read_splits(zf, None if args.vocab == "all" else
                         sorted(pd.read_csv(args.gislr_index)["sign"].unique()))
    shards = plan_shards(df, args.shard_size)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in shards if not (args.cache_dir / f"{s}.csv").exists()][:args.limit]
    print(f"[aslc] {len(df)} videos, {df['sign'].nunique()} signs, "
          f"{df['participant_id'].nunique()} signers; {len(todo)}/{len(shards)} shards to build",
          flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(todo) or 1))) as pool:
        futs = {pool.submit(process_shard, str(args.zip), s, shards[s], str(args.cache_dir),
                            args.holistic_complexity): s for s in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            out = fut.result()
            el = time.time() - t0
            print(f"[aslc] {i}/{len(todo)} shard {futs[fut]}: {len(out)} videos "
                  f"({el / 60:.0f} min, ~{el / i * (len(todo) - i) / 60:.0f} min left)", flush=True)

    done = [s for s in shards if (args.cache_dir / f"{s}.csv").exists()]
    if len(done) < len(shards):
        print(f"[aslc] {len(shards) - len(done)} shards still to build; index not written")
        return
    index = pd.concat([pd.read_csv(args.cache_dir / f"{s}.csv") for s in done], ignore_index=True)
    index.to_csv(args.cache_dir / "index.csv", index=False)
    split = df[["participant_id", "split"]].drop_duplicates().sort_values(["split", "participant_id"])
    split.to_csv(args.cache_dir / "split.csv", index=False)
    print(f"[aslc] index -> {args.cache_dir / 'index.csv'} ({len(index)} rows); split "
          f"{split['split'].value_counts().to_dict()} signers")


if __name__ == "__main__":
    main()
