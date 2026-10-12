"""preprocess_asl_citizen on a tiny fake archive (blank videos, so no hands):
checks the cache layout WordDataset reads, not landmark quality."""

import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from data.capture_words import write_clip
from data.preprocess_gislr import load_sequence
from data.word_dataset import load_word_index

ROOT = Path(__file__).resolve().parents[1]


def test_preprocess_builds_word_cache(tmp_path):
    rows = {"train": [("P1", "a-APPLE.mp4", "APPLE"), ("P1", "b-BOOK.mp4", "BOOK"),
                      ("P1", "c-BOOK.mp4", "BOOK")],
            "val": [("P7", "d-APPLE.mp4", "APPLE")], "test": [("P12", "e-BOOK.mp4", "BOOK")]}
    archive = tmp_path / "ASL_Citizen.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for split, rs in rows.items():
            zf.writestr(f"ASL_Citizen/splits/{split}.csv", pd.DataFrame(
                rs, columns=["Participant ID", "Video file", "Gloss"]).assign(
                **{"ASL-LEX Code": "X"}).to_csv(index=False))
            for _, video, _ in rs:
                clip = tmp_path / video
                write_clip(clip, [np.full((48, 64, 3), 90, np.uint8)] * 15, fps=15.0)
                zf.write(clip, f"ASL_Citizen/videos/{video}")
    cache = tmp_path / "cache"
    subprocess.run([sys.executable, str(ROOT / "src/data/preprocess_asl_citizen.py"),
                    "--zip", str(archive), "--cache-dir", str(cache), "--workers", "2",
                    "--shard-size", "2", "--holistic-complexity", "0", "--vocab", "all"],
                   check=True, capture_output=True)
    index, classes = load_word_index(cache)
    assert classes == ["apple", "book"] and len(index) == 5
    assert set(index["shard"]) == {"1_00.npy", "1_01.npy", "7_00.npy", "12_00.npy"}
    assert dict(zip(index["participant_id"], index["split"])) == {1: "train", 7: "val", 12: "test"}
    seq = load_sequence(cache, index.iloc[0])
    assert seq.shape[1:] == (84, 3) and 25 <= len(seq) <= 31     # 1 s at 15 fps -> ~30 fps
    assert np.isnan(seq).all()                                     # blank frames: nothing detected


def test_gislr_names_matches_forms_and_aliases():
    from data.preprocess_asl_citizen import gislr_names
    got = gislr_names(["THANKYOU", "FATHER", "TV", "CALL1", "SHOES"],
                      ["thankyou", "dad", "tv", "callonphone", "shoe"])
    assert got == {"THANKYOU": "thankyou", "FATHER": "dad", "TV": "tv", "SHOES": "shoe"}
