"""Word models, the training CLI and checkpoint loading, on a synthetic GISLR
cache (conftest.synthetic_words). These check the code, not recognition."""

import json

import numpy as np
import pytest
import torch

from data import holistic_features as hf
from data.word_dataset import WordDataset, load_word_index, sign_properties
from evaluate_words import default_guess_sinks, evaluate_split, load_word_checkpoint
from models.words import build_word_model


@pytest.mark.parametrize("name", ["gru", "transformer"])
def test_padding_does_not_change_output(name):
    torch.manual_seed(0)
    model = build_word_model(name, 20, 5, 32).eval()
    x = torch.randn(2, 32, 20)
    mask = torch.zeros(2, 32, dtype=torch.bool)
    mask[0, :10], mask[1, :32] = True, True
    ref = model(x, mask)
    x2 = x.clone()
    x2[0, 10:] = 1e3 * torch.randn(22, 20)      # garbage in the padded steps
    torch.testing.assert_close(model(x2, mask), ref)
    assert ref.shape == (2, 5)


def test_dataset_slices_feature_sets(synthetic_words):
    cache, _, _ = synthetic_words
    index, classes = load_word_index(cache)
    assert len(classes) == 8
    full = WordDataset(index, cache, "full")[0]
    for fs, n in hf.FEATURE_SETS.items():
        x, m, y = WordDataset(index, cache, fs)[0]
        assert x.shape == (hf.MAX_LEN, n) and m.dtype == torch.bool
        torch.testing.assert_close(x, full[0][:, :n])    # one computation, sliced


def test_training_cli_outputs(synthetic_words):
    cache, out, ckpt_path = synthetic_words
    ckpt = torch.load(ckpt_path, weights_only=False)
    index, classes = load_word_index(cache)
    assert ckpt["classes"] == classes
    assert set(ckpt["split_participants"]) == {"train", "val", "test"}
    parts = [set(v) for v in ckpt["split_participants"].values()]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    history = json.loads((out / "words_transformer_full_history.json").read_text())
    assert len(history) == 12
    report = json.loads((out / "words_transformer_full_val_eval.json").read_text())
    assert report["split"] == "val" and report["acc"] > 0.5      # 8 easy synthetic signs
    assert set(report["breakdowns"]) == {"two_handed", "motion_bucket"}


def test_checkpoint_round_trip_and_refusal(synthetic_words, tmp_path):
    cache, _, ckpt_path = synthetic_words
    model, ckpt = load_word_checkpoint(ckpt_path, "cpu")
    torch.testing.assert_close(model.feat_mean, torch.tensor(ckpt["normalisation"]["mean"]))
    index, classes = load_word_index(cache)
    props = sign_properties(index, classes, cache)
    report = evaluate_split(model, ckpt, index, "val", cache, torch.device("cpu"), props,
                            workers=0)
    assert report["acc"] == pytest.approx(ckpt["val_acc"], abs=1e-6)

    for key, bad in [("landmark_spec_hash", "000000000000"), ("feature_version", -1)]:
        broken = dict(ckpt, **{key: bad})
        torch.save(broken, tmp_path / "bad.pt")
        with pytest.raises(RuntimeError):
            load_word_checkpoint(tmp_path / "bad.pt", "cpu")


def test_default_guess_sinks():
    cm = np.diag([10, 10, 10])
    cm[0, 2] = cm[1, 2] = 8
    sinks = default_guess_sinks(cm, ["a", "b", "c"])
    assert [s["sign"] for s in sinks] == ["c"] and sinks[0]["absorbed_from"] == 16


def test_ablation_aggregates_existing_runs(synthetic_words):
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    _, out, _ = synthetic_words
    src = out / "words_transformer_full_val_eval.json"
    shutil.copy(src, out / "words_transformer_full_ablation_val_eval.json")
    res = subprocess.run([sys.executable, str(Path(__file__).parents[1] / "src/ablation_words.py"),
                          "--aggregate-only", "--out-dir", str(out)],
                         capture_output=True, text=True, check=True).stdout
    assert "missing words_transformer_dom_shape_ablation_val_eval.json" in res
    rows = json.loads((out / "words_ablation.json").read_text())
    assert [r["features"] for r in rows] == ["full"]
    assert "| full | 153 |" in (out / "words_ablation.md").read_text()


def test_finetune_cli_holds_out_a_session(synthetic_words, tmp_path):
    """finetune_words.py on fake personal clips (landmark caches standing in for
    decoded videos): the holdout session is never trained on, and the output
    checkpoint loads like any other word checkpoint."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    import numpy as np
    from data.preprocess_gislr import load_sequence
    from data.word_dataset import load_word_index
    from evaluate_words import load_word_checkpoint

    cache, _, ckpt_path = synthetic_words
    index, classes = load_word_index(cache)
    clips = tmp_path / "word_videos"
    for sess, pid in zip(["s1", "s2"], index["participant_id"].unique()):
        for i, (_, row) in enumerate(index[index["participant_id"] == pid].head(8).iterrows()):
            path = clips / row["sign"] / f"{sess}_{i:02d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()                                   # cache is newer, so never decoded
            np.save(path.with_suffix(".npy"), load_sequence(cache, row))
    res = subprocess.run([sys.executable, str(Path(__file__).parents[1] / "src/finetune_words.py"),
                          "--checkpoint", str(ckpt_path), "--dir", str(clips),
                          "--cache-dir", str(cache), "--out-dir", str(tmp_path / "out"),
                          "--epochs", "2", "--repeat", "2", "--replay", "16",
                          "--workers", "0", "--no-amp"], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    run = ckpt_path.stem.removesuffix("_best") + "_ft"
    report = json.loads((tmp_path / "out" / f"{run}.json").read_text())
    assert report["holdout"] == "s2" and report["train_sessions"] == ["s1"]
    assert report["after"]["holdout"]["n"] == 8
    _, ckpt = load_word_checkpoint(tmp_path / "out" / f"{run}.pt", "cpu")
    assert ckpt["classes"] == classes and ckpt["finetune"]["holdout"] == "s2"
