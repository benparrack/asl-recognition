"""diagnose_motion.py on synthetic data: the probes must separate "a still
handshape can do it" from "the path is needed", and the checkpoint ablations
must run on a real v2 checkpoint file."""

import argparse

import numpy as np
import torch

import config
from conftest import write_synthetic_motion_dataset
from data.motion_dataset_v2 import load_motion_v2_metadata


def test_diagnose_probes_and_checkpoint(tmp_path, synthetic_motion_model):
    import diagnose_motion as dm
    import train_motion_v2

    csv_path = write_synthetic_motion_dataset(tmp_path, n_signers=5, per_label=8)
    df = load_motion_v2_metadata(csv_path)

    ckpt = train_motion_v2.build_checkpoint(
        synthetic_motion_model, 1, 1.0, 0.0,
        argparse.Namespace(hidden_size=16, dropout=0.1), {})
    torch.save(ckpt, tmp_path / "m.pt")

    r = dm.diagnose(df, tmp_path, tmp_path / "m.pt", split="val")
    assert set(r["probes"]) == set(dm.PROBES)
    # Synthetic J and Z have distinct paths, so the path alone should do it.
    assert r["probes"]["trajectory_only"]["mean"] > 0.8

    c = r["checkpoint"]
    assert set(c["ablations"]) == {"normal", "time_shuffled",
                                   "handshape_neutralised", "motion_neutralised"}
    # the synthetic model was trained on this generator; it must use motion
    assert c["ablations"]["normal"] > c["ablations"]["motion_neutralised"]
    assert set(c["false_positive_rates"]) == {"hold_I->J", "hold_D->Z"}
    rows = set(c["confusion_by_subtype"])
    assert {"J", "Z"} <= rows and any(k.startswith("NONE:") for k in rows)


def test_shuffle_time_keeps_frames():
    import diagnose_motion as dm
    X = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    S = dm.shuffle_time(X)
    assert S.shape == X.shape
    for a, b in zip(X, S):
        assert sorted(map(tuple, a)) == sorted(map(tuple, b))
