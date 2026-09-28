"""preprocess_gislr on a synthetic archive in the Kaggle layout: zip and
directory sources, the cache round trip, idempotency, and a participant-disjoint
split."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data import holistic_landmarks as hl
from data.preprocess_gislr import (
    GislrSource, check_parquet_schema, load_sequence, parquet_to_selected,
)
from data.synthetic_gislr import write_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gislr_zip(tmp_path_factory):
    d = tmp_path_factory.mktemp("gislr_raw")
    write_dataset(d, n_signs=4, n_participants=5, reps=2, as_zip=True)
    return d


def _run(raw, cache, stats, *extra):
    return subprocess.run([sys.executable, str(ROOT / "src/data/preprocess_gislr.py"),
                           "--raw-dir", str(raw), "--cache-dir", str(cache),
                           "--stats-out", str(stats), "--workers", "4", *extra],
                          capture_output=True, text=True, check=True).stdout


def test_cache_round_trip_split_and_idempotency(gislr_zip, tmp_path):
    cache = tmp_path / "cache"
    out = _run(gislr_zip, cache, tmp_path / "s.json")
    assert "5 participant shards to build" in out

    index = pd.read_csv(cache / "index.csv")
    assert len(index) == 5 * 4 * 2
    src = GislrSource(gislr_zip)
    train = src.train_csv()
    for _, row in index.sample(5, random_state=0).iterrows():
        rel = train.loc[train["sequence_id"] == row["sequence_id"], "path"].item()
        expected = parquet_to_selected(src.parquet(rel)).astype(np.float16).astype(np.float32)
        got = load_sequence(cache, row)
        assert got.shape == (row["n_frames"], hl.N_SELECTED, 3)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
        np.testing.assert_array_equal(np.nan_to_num(got), np.nan_to_num(expected))

    split = pd.read_csv(cache / "split.csv")
    assert split["participant_id"].is_unique
    assert set(split["participant_id"]) == set(index["participant_id"])
    assert set(split["split"]) == {"train", "val", "test"}

    assert "0 participant shards to build" in _run(gislr_zip, cache, tmp_path / "s.json")
    assert "5 participant shards to build" in _run(gislr_zip, cache, tmp_path / "s.json",
                                                   "--overwrite")


def test_directory_source(tmp_path):
    d = write_dataset(tmp_path / "raw", n_signs=2, n_participants=3, reps=1)
    src = GislrSource(d)
    assert src.kind == "dir"
    train = src.train_csv()
    assert list(train.columns) == ["path", "participant_id", "sequence_id", "sign"]
    df = src.parquet(train["path"].iloc[0])
    check_parquet_schema(df)
    assert set(src.sign_map()) == set(train["sign"])


def test_schema_check_rejects_bad_parquet():
    bad = pd.DataFrame({"frame": [0], "row_id": ["0-hand-0"], "type": ["hand"],
                        "landmark_index": [0], "x": [0.1], "y": [0.1], "z": [0.0]})
    with pytest.raises(ValueError, match="unexpected landmark types"):
        check_parquet_schema(bad)
    with pytest.raises(ValueError, match="missing columns"):
        check_parquet_schema(bad.drop(columns=["z"]))


def test_missing_source_explains_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="HUMAN_TODO"):
        GislrSource(tmp_path)
