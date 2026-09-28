"""
The v2 on-disk format end to end, without a camera: synthetic clips go through
capture_motion_v2.save_clip / append_metadata, back out through the v2 dataset,
into a checkpoint built by train_motion_v2.build_checkpoint, and are loaded by
the same load_motion_classifier the demo uses.
"""

import argparse

import numpy as np
import pandas as pd
import torch

import config
from data import motion_features as mf
from data.capture_motion_v2 import (
    METADATA_HEADER, append_metadata, next_clip_id, none_prompt, save_clip,
)
from data.motion_dataset_v2 import (
    REQUIRED_COLUMNS, MotionClipDatasetV2, load_motion_v2_metadata, split_motion_v2,
)
from data.synthetic_motion import random_clip
from models.motion import MotionGRUv2
from motion_gate import MotionClassifierV2, load_motion_classifier


def _frames_from_raw(raw):
    return [None if np.isnan(f).any() else f for f in raw]


def _write_dataset(root, n_signers=5, per_label=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_signers):
        signer = f"synth_{s}"
        for label in config.MOTION_V2_CLASSES:
            out_dir = root / signer / "sess" / label
            for k in range(per_label):
                raw, t, w, h = random_clip(label, rng)
                subtype = ""
                if label == config.MOTION_NONE:
                    subtype, _ = none_prompt(k, rng)
                rows.append(save_clip(out_dir, root, label, next_clip_id(out_dir, label),
                                      _frames_from_raw(raw), list(t + 1000.0), w, h,
                                      subtype=subtype, signer=signer, session="sess"))
    csv_path = root / "meta.csv"
    append_metadata(rows, csv_path)
    return csv_path


def test_metadata_has_required_columns():
    assert set(REQUIRED_COLUMNS) <= set(METADATA_HEADER)


def test_none_prompts_cycle_all_subtypes():
    rng = np.random.default_rng(0)
    seen = {none_prompt(i, rng)[0] for i in range(len(config.MOTION_NONE_SUBTYPES))}
    assert seen == set(config.MOTION_NONE_SUBTYPES)


def test_save_clip_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    raw, t, w, h = random_clip("J", rng)
    frames = _frames_from_raw(raw)
    frames[3] = None
    row = save_clip(tmp_path / "J", tmp_path, "J", 0, frames, list(t + 50.0), w, h,
                    signer="a", session="b")
    d = np.load(tmp_path / row["path"])
    assert d["raw"].shape == (len(frames), 21, 3)
    assert np.isnan(d["raw"][3]).all() and not d["detected"][3]
    assert d["timestamps"][0] == 0.0                     # origin moved to clip start
    assert int(d["format_version"]) == 2
    # features from the saved file equal features from the in-memory clip
    raw_mem = np.stack([f if f is not None else np.full((21, 3), np.nan) for f in frames])
    np.testing.assert_allclose(
        mf.clip_to_features(d["raw"], d["timestamps"], int(d["frame_w"]), int(d["frame_h"])),
        mf.clip_to_features(raw_mem, t, w, h), atol=1e-5)
    assert next_clip_id(tmp_path / "J", "J") == 1


def test_dataset_split_and_checkpoint_roundtrip(tmp_path):
    csv_path = _write_dataset(tmp_path)
    df = load_motion_v2_metadata(csv_path)
    assert len(df) == 5 * 3 * 4
    tr, va, te = split_motion_v2(df)
    sig = [set(d["signer_id"]) for d in (tr, va, te)]
    assert not (sig[0] & sig[1]) and not (sig[0] & sig[2]) and not (sig[1] & sig[2])

    ds = MotionClipDatasetV2(tr, root=tmp_path, augment=True)
    x, y = ds[0]
    assert x.shape == (mf.RESAMPLE_LEN, mf.FEATURE_DIM) and 0 <= y < 3

    import train_motion_v2
    model = MotionGRUv2(input_size=mf.FEATURE_DIM, hidden_size=8)
    model.set_normalisation(*ds.feature_stats())
    args = argparse.Namespace(hidden_size=8, dropout=0.3)
    ckpt = train_motion_v2.build_checkpoint(model, 1, 0.5, 1.0, args, {"train": []})
    path = tmp_path / "m.pt"
    torch.save(ckpt, path)

    clf = load_motion_classifier(path, torch.device("cpu"))
    assert isinstance(clf, MotionClassifierV2)
    # normalisation buffers survive the round trip
    torch.testing.assert_close(clf.model.feat_mean, model.feat_mean)
    raw, t, w, h = random_clip("Z", np.random.default_rng(3))
    label, conf = clf(_frames_from_raw(raw), t, (w, h))
    assert label in config.MOTION_V2_CLASSES and 0 < conf <= 1


def test_loader_rejects_feature_dim_mismatch(tmp_path):
    model = MotionGRUv2(input_size=mf.FEATURE_DIM, hidden_size=8)
    import train_motion_v2
    ckpt = train_motion_v2.build_checkpoint(model, 1, 0.5, 1.0,
                                            argparse.Namespace(hidden_size=8, dropout=0.3), {})
    ckpt["feature_dim"] = mf.FEATURE_DIM - 1
    torch.save(ckpt, tmp_path / "bad.pt")
    import pytest
    with pytest.raises(RuntimeError, match="feature_dim"):
        load_motion_classifier(tmp_path / "bad.pt", torch.device("cpu"))
