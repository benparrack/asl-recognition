"""Put the repo root (config.py) and src/ (the `data`/`models` packages) on
sys.path, src/ FIRST so `import data` resolves to src/data, not the dataset
directory data/ at the repo root (see PROJECT_HANDOFF.md §3)."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402


def train_synthetic_motion_model(n_per_class=60, epochs=25, zero_handshape=False,
                                 shape=None, seed=0):
    """Train a tiny MotionGRUv2 on synthetic J/Z/NONE clips (CPU, a few seconds).
    Returns (model, feature transform used). Shared by the feature and gate tests."""
    import config
    from data import motion_features as mf
    from data.synthetic_motion import random_clip
    from models.motion import MotionGRUv2

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X, y = [], []
    for label in config.MOTION_V2_CLASSES:
        for _ in range(n_per_class):
            raw, t, w, h = random_clip(label, rng, shape=shape)
            if rng.random() < 0.5:
                raw, t = mf.augment_raw_clip(raw, t, w, h, rng)
            f = mf.clip_to_features(raw, t, w, h)
            if f is None:
                continue
            X.append(f)
            y.append(config.MOTION_V2_CLASS_TO_IDX[label])
    X = torch.from_numpy(np.stack(X))
    if zero_handshape:
        X[:, :, mf.HANDSHAPE_SLICE] = 0
    y = torch.tensor(y)

    model = MotionGRUv2(input_size=mf.FEATURE_DIM, hidden_size=16, dropout=0.1)
    flat = X.reshape(-1, X.shape[-1])
    model.set_normalisation(flat.mean(0), flat.std(0))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            opt.step()
    model.eval()
    return model


def frames_from_raw(raw):
    """(T, 21, 3) with NaN rows -> the list-of-(array | None) the capture loop builds."""
    return [None if np.isnan(f).any() else f for f in raw]


def write_synthetic_motion_dataset(root, n_signers=5, per_label=4, seed=0):
    """Write synthetic J/Z/NONE clips in the capture_motion_v2 on-disk format
    (one signer id per fake session) and return the metadata CSV path."""
    import config
    from data.capture_motion_v2 import append_metadata, next_clip_id, none_prompt, save_clip
    from data.synthetic_motion import random_clip

    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_signers):
        signer = f"synth_{s}"
        for label in config.MOTION_V2_CLASSES:
            out_dir = root / signer / "sess" / label
            for k in range(per_label):
                raw, t, w, h = random_clip(label, rng)
                subtype = none_prompt(k, rng)[0] if label == config.MOTION_NONE else ""
                rows.append(save_clip(out_dir, root, label, next_clip_id(out_dir, label),
                                      frames_from_raw(raw), list(t + 1000.0), w, h,
                                      subtype=subtype, signer=signer, session="sess"))
    csv_path = root / "meta.csv"
    append_metadata(rows, csv_path)
    return csv_path


@pytest.fixture(scope="session")
def synthetic_motion_model():
    return train_synthetic_motion_model()


@pytest.fixture(scope="session")
def synthetic_words(tmp_path_factory):
    """A preprocessed synthetic GISLR cache and a word checkpoint trained on it
    through the real train_words.py CLI. Returns (cache_dir, out_dir, ckpt_path).
    Numbers from this model only check the code, not sign recognition."""
    import subprocess
    from data.synthetic_gislr import write_dataset

    base = tmp_path_factory.mktemp("words")
    write_dataset(base / "raw", n_signs=8, n_participants=7, reps=4, seed=1)
    subprocess.run([sys.executable, str(_ROOT / "src/data/preprocess_gislr.py"),
                    "--raw-dir", str(base / "raw"), "--cache-dir", str(base / "cache"),
                    "--stats-out", str(base / "stats.json"), "--workers", "2"],
                   check=True, capture_output=True)
    subprocess.run([sys.executable, str(_ROOT / "src/train_words.py"),
                    "--cache-dir", str(base / "cache"), "--out-dir", str(base / "out"),
                    "--epochs", "12", "--workers", "0", "--dim", "64", "--no-amp"],
                   check=True, capture_output=True)
    return base / "cache", base / "out", base / "out" / "words_transformer_full_best.pt"
