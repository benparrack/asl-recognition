"""
Is the motion model actually using motion? Diagnostics for J / Z / NONE (v2).

Run:
    python src/diagnose_motion.py                               # probes only
    python src/diagnose_motion.py --checkpoint checkpoints/motion_v2_best.pt
    python src/diagnose_motion.py --checkpoint ... --split test  # ONCE, at the very end

WHY
===
v1 hit 100% val accuracy while learning only handshape: a logistic regression
on the FIRST FRAME alone also scored 100% held out (NOTES.md, 2026-09-28). A
headline accuracy can't tell those two situations apart. This script can:

1. Shortcut probes. Leave-one-signer-out logistic regression on train+val
   signers only (the test split is never read here):
       first_frame      handshape of frame 0              -> can a still photo do it?
       mean_frame       handshape averaged over the clip  -> ditto, without timing
       trajectory_only  motion channels, handshape removed -> is the path enough?
       time_shuffled    all channels, frames shuffled     -> does order matter?
   Healthy v2 data: first/mean-frame near chance on J-vs-NONE(hold_I) and
   Z-vs-NONE(hold_D); trajectory_only high; time_shuffled below trajectory_only.

2. Checkpoint ablations on one split (default val):
       normal / time-shuffled / handshape-neutralised / motion-neutralised
   "Neutralised" sets those channels to the training mean, i.e. zero after
   the model's own standardisation -- the model sees "average", not a spike.
   A model that uses motion drops sharply when motion is neutralised and
   barely moves when handshape is.

3. Confusion matrix with NONE broken out by sub-type, and the two false-positive
   rates that matter live: hold_I -> J and hold_D -> Z (a held handshape
   mistaken for its motion letter; exactly what v1 would do).

--split test is refused if results/motion_v2_test_used.json exists: the test
split is evaluated exactly once, for the final model, and never tuned on after.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from data import motion_features as mf  # noqa: E402
from data.motion_dataset_v2 import (  # noqa: E402
    load_model_clip, load_motion_v2_metadata, split_motion_v2,
)

TEST_MARKER = config.RESULTS_DIR / "motion_v2_test_used.json"
PROBES = ("first_frame", "mean_frame", "trajectory_only", "time_shuffled")


# ------------------------------------------------------------------ features
def featurise(df: pd.DataFrame, root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Un-augmented (N, L, FEATURE_DIM) features and (N,) labels. Clips that
    can't be featurised are an error, not a silent drop."""
    X, y = [], []
    for p, label in zip(df["path"], df["label"]):
        raw, t, w, h = load_model_clip(root / p)
        f = mf.clip_to_features(raw, t, w, h)
        if f is None:
            raise ValueError(f"{p}: too few detected frames to featurise")
        X.append(f)
        y.append(config.MOTION_V2_CLASS_TO_IDX[label])
    return np.stack(X), np.array(y)


def shuffle_time(X: np.ndarray, seed: int = config.SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([x[rng.permutation(len(x))] for x in X])


def probe_matrix(X: np.ndarray, probe: str) -> np.ndarray:
    if probe == "first_frame":
        return X[:, 0, mf.HANDSHAPE_SLICE]
    if probe == "mean_frame":
        return X[:, :, mf.HANDSHAPE_SLICE].mean(axis=1)
    if probe == "trajectory_only":
        return X[:, :, mf.MOTION_SLICE].reshape(len(X), -1)
    if probe == "time_shuffled":
        return shuffle_time(X).reshape(len(X), -1)
    raise ValueError(probe)


# ------------------------------------------------------------------ probes
def run_probes(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Leave-one-group-out logistic-regression accuracy for each probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = {}
    for probe in PROBES:
        M = probe_matrix(X, probe)
        folds = {}
        for g in np.unique(groups):
            held = groups == g
            if len(np.unique(y[~held])) < 2:
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            clf.fit(M[~held], y[~held])
            folds[str(g)] = float((clf.predict(M[held]) == y[held]).mean())
        out[probe] = {"mean": float(np.mean(list(folds.values()))) if folds else None,
                      "folds": folds}
    return out


# ------------------------------------------------------------------ checkpoint
@torch.no_grad()
def predict(model, X: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    probs = torch.softmax(model(torch.from_numpy(X).float().to(device)), dim=1).cpu().numpy()
    return probs.argmax(1), probs.max(1)


def neutralise(X: np.ndarray, sl: slice, mean: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[:, :, sl] = mean[sl]
    return X


def ablations(model, X, y, device) -> dict:
    mean = model.feat_mean.cpu().numpy()
    variants = {
        "normal": X,
        "time_shuffled": shuffle_time(X),
        "handshape_neutralised": neutralise(X, mf.HANDSHAPE_SLICE, mean),
        "motion_neutralised": neutralise(X, mf.MOTION_SLICE, mean),
    }
    return {k: float((predict(model, v, device)[0] == y).mean()) for k, v in variants.items()}


def confusion_by_subtype(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """Rows: J, Z, and NONE:<subtype>. Columns: predicted class. Counts."""
    rows = np.where(df["label"] == config.MOTION_NONE,
                    config.MOTION_NONE + ":" + df["subtype"].astype(str), df["label"])
    pred = [config.MOTION_V2_IDX_TO_CLASS[int(i)] for i in y_pred]
    cm = pd.crosstab(pd.Series(rows, name="true"), pd.Series(pred, name="pred"))
    return cm.reindex(columns=config.MOTION_V2_CLASSES, fill_value=0)


def false_positive_rates(df: pd.DataFrame, y_pred: np.ndarray) -> dict:
    pred = np.array([config.MOTION_V2_IDX_TO_CLASS[int(i)] for i in y_pred])
    out = {}
    for subtype, letter in (("hold_I", "J"), ("hold_D", "Z")):
        mask = (df["label"] == config.MOTION_NONE).to_numpy() & (df["subtype"] == subtype).to_numpy()
        out[f"{subtype}->{letter}"] = (float((pred[mask] == letter).mean()) if mask.any() else None,
                                       int(mask.sum()))
    return out


def load_v2_model(path, device):
    from models.motion import MotionGRUv2
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("feature_version") != mf.FEATURE_VERSION:
        raise RuntimeError(f"{path} is not a v2 motion checkpoint")
    model = MotionGRUv2(input_size=ckpt["feature_dim"], hidden_size=ckpt["hidden_size"],
                        dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


# ------------------------------------------------------------------ main
def diagnose(df: pd.DataFrame, root: Path, checkpoint=None, split: str = "val",
             device=torch.device("cpu")) -> dict:
    train_df, val_df, test_df = split_motion_v2(df)
    report = {"n_clips": {"train": len(train_df), "val": len(val_df), "test": len(test_df)}}

    # Probes: train+val only, leave-one-signer-out.
    dev_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    X, y = featurise(dev_df, root)
    report["probes"] = run_probes(X, y, dev_df["signer_id"].to_numpy())
    report["chance"] = float(np.bincount(y).max() / len(y))

    if checkpoint is not None:
        model, ckpt = load_v2_model(checkpoint, device)
        eval_df = (val_df if split == "val" else test_df).reset_index(drop=True)
        Xe, ye = featurise(eval_df, root)
        y_pred, _ = predict(model, Xe, device)
        report["checkpoint"] = {
            "path": str(checkpoint), "epoch": ckpt["epoch"], "split": split,
            "signers": sorted(eval_df["signer_id"].unique().tolist()),
            "ablations": ablations(model, Xe, ye, device),
            "false_positive_rates": false_positive_rates(eval_df, y_pred),
            "confusion_by_subtype": confusion_by_subtype(eval_df, y_pred).to_dict(orient="index"),
        }
    return report


def print_report(r: dict) -> None:
    print(f"clips: {r['n_clips']}   majority-class chance {r['chance']:.3f}")
    print("\nleave-one-signer-out logistic probes (train+val signers):")
    for probe, v in r["probes"].items():
        folds = "  ".join(f"{k}={a:.2f}" for k, a in v["folds"].items())
        mean = "n/a" if v["mean"] is None else f"{v['mean']:.3f}"
        print(f"  {probe:<16} {mean}   [{folds}]")
    c = r.get("checkpoint")
    if c:
        print(f"\ncheckpoint {c['path']} (epoch {c['epoch']}) on {c['split']} {c['signers']}:")
        for k, a in c["ablations"].items():
            print(f"  {k:<22} {a:.3f}")
        print("  false positives:")
        for k, (rate, n) in c["false_positive_rates"].items():
            print(f"    {k:<12} {'n/a' if rate is None else f'{rate:.3f}'}  (n={n})")
        print("  confusion (rows true, cols predicted):")
        print(pd.DataFrame(c["confusion_by_subtype"]).T.to_string())


def main():
    parser = argparse.ArgumentParser(description="Motion v2 shortcut diagnostics")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--metadata", type=Path, default=config.MOTION_V2_METADATA_CSV)
    parser.add_argument("--root", type=Path, default=config.MOTION_V2_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.split == "test":
        if args.checkpoint is None:
            parser.error("--split test needs --checkpoint")
        if TEST_MARKER.exists():
            sys.exit(f"Refusing: the motion v2 test split was already used "
                     f"({TEST_MARKER.read_text().strip()}). Report that result.")

    df = load_motion_v2_metadata(args.metadata)
    report = diagnose(df, args.root, args.checkpoint, args.split, config.DEVICE)
    print_report(report)

    out = args.out or config.RESULTS_DIR / f"motion_v2_diagnosis_{args.split}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[done] -> {out}")
    if args.split == "test":
        TEST_MARKER.write_text(json.dumps({"checkpoint": args.checkpoint,
                                           "result": str(out)}) + "\n")


if __name__ == "__main__":
    main()
