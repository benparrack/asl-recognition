"""
Build a contact sheet of processed crops so you can eyeball many at once.

Run:
    python src/data/contact_sheet.py                    # 8 random per class
    python src/data/contact_sheet.py --per-class 16
    python src/data/contact_sheet.py --by session       # group rows by session
    python src/data/contact_sheet.py --raw              # sheet from raw frames

Writes results/contact_sheet.png and opens it.

WHAT TO LOOK FOR
================
  - Is the hand centred, with margin on all sides?
  - Are fingertips clipped? -> increase --padding in preprocess.py
  - Is there a lot of black padding? -> hand was near the frame edge a lot
  - Does one session look very different (colour cast, blur, framing)?
    That is a feature your model can learn instead of the handshape.
  - Are any crops of something that is not your hand? MediaPipe false
    positives happen; they are training-set poison and worth deleting.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # render to file without needing a display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Contact sheet of dataset images")
    parser.add_argument("--per-class", type=int, default=8, help="images per row")
    parser.add_argument("--by", default="label", choices=["label", "session", "signer_id"],
                        help="what each row of the sheet represents")
    parser.add_argument("--raw", action="store_true", help="use raw frames instead of crops")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    csv_path = config.RAW_METADATA_CSV if args.raw else config.METADATA_CSV
    image_root = config.RAW_DIR if args.raw else config.PROCESSED_DIR

    df = pd.read_csv(csv_path)
    group_col = "session_id" if args.by == "session" else args.by
    groups = sorted(df[group_col].unique())

    n_rows, n_cols = len(groups), args.per_class
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.6, n_rows * 1.8))

    # plt.subplots collapses dimensions when a count is 1, which breaks the
    # axes[r][c] indexing below. Force it back to 2-D.
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[a] for a in axes]

    for r, group in enumerate(groups):
        subset = df[df[group_col] == group]
        sample = subset.sample(min(n_cols, len(subset)), random_state=args.seed)

        for c in range(n_cols):
            ax = axes[r][c]
            ax.axis("off")

            if c >= len(sample):
                continue

            row = sample.iloc[c]
            ax.imshow(Image.open(image_root / row["path"]))

            if c == 0:
                ax.set_title(f"{group}", loc="left", fontsize=10, fontweight="bold")

    kind = "raw" if args.raw else "processed"
    fig.suptitle(f"{kind} images, grouped by {group_col}", fontsize=12)
    fig.tight_layout()

    out = config.RESULTS_DIR / f"contact_sheet_{kind}_{group_col}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[sheet] wrote {out}")

    try:
        subprocess.run(["xdg-open", str(out)], check=False)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
