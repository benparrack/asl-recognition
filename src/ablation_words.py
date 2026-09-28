"""
The feature-set ablation: how much does the letters pipeline's one-hand,
no-location view of the world cost on words?

Run:
    python src/ablation_words.py                      # trains 4 runs, then tabulates
    python src/ablation_words.py --aggregate-only     # just rebuild the table

Same model, split, epochs and seed for each feature set (holistic_features.FEATURE_SETS):
    dom_shape    dominant-hand shape only (what the letters MLP sees)
    both_shape   + non-dominant hand shape
    location     + body-relative hand location, hand-to-hand vector
    full         + face distances, velocities

Each run is a normal train_words.py invocation (tag "ablation"), so every
number here traces back to a checkpoint and a *_val_eval.json. Validation only;
the test split is not touched.

Output: results/words_ablation.json and results/words_ablation.md
(val top-1/top-5, plus one- vs two-handed and static/moderate/dynamic accuracy).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from data import holistic_features as hf  # noqa: E402

TAG = "ablation"


def table(rows: list[dict]) -> str:
    buckets = ["static", "moderate", "dynamic"]
    head = ("| features | channels | val top-1 | val top-5 | one-handed | two-handed | "
            + " | ".join(buckets) + " |")
    lines = [head, "|" + "---|" * (6 + len(buckets))]
    for r in rows:
        b = r["breakdowns"]
        th = b["two_handed"]
        cell = lambda d, k: f"{d[k]['acc']:.3f} (n={d[k]['n']})" if k in d else "n/a"  # noqa: E731
        lines.append(
            f"| {r['features']} | {r['n_channels']} | {r['acc']:.3f} | {r['top5']:.3f} | "
            f"{cell(th, 'False')} | {cell(th, 'True')} | "
            + " | ".join(cell(b["motion_bucket"], k) for k in buckets) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Word feature-set ablation")
    parser.add_argument("--model", choices=["gru", "transformer"], default="transformer")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=config.GISLR_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="further args passed to train_words.py after --")
    args = parser.parse_args()
    res_dir = args.out_dir or config.RESULTS_DIR

    if not args.aggregate_only:
        for fs in hf.FEATURE_SETS:
            cmd = [sys.executable, str(Path(__file__).parent / "train_words.py"),
                   "--model", args.model, "--features", fs, "--epochs", str(args.epochs),
                   "--tag", TAG, "--cache-dir", str(args.cache_dir)]
            if args.out_dir:
                cmd += ["--out-dir", str(args.out_dir)]
            cmd += [a for a in args.extra if a != "--"]
            print(f"[ablation] {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)

    rows = []
    for fs in hf.FEATURE_SETS:
        p = res_dir / f"words_{args.model}_{fs}_{TAG}_val_eval.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
        else:
            print(f"[ablation] missing {p.name}")
    (res_dir / "words_ablation.json").write_text(json.dumps(
        [{k: r[k] for k in ("run", "features", "n_channels", "acc", "top5", "breakdowns",
                            "epoch")} for r in rows], indent=2))
    md = table(rows)
    (res_dir / "words_ablation.md").write_text(
        f"Word feature ablation ({args.model}, val split, participant-disjoint)\n\n{md}\n")
    print(md)


if __name__ == "__main__":
    main()
