# Current Project State (handoff notes)

Updated 2026-09-28, at the end of the autonomous run on branch
`feature/motion-v2-and-words` (not pushed). The longer references are
PROJECT_HANDOFF.md (§0 is the status), NOTES.md (dated findings),
DECISIONS.md (judgement calls), HUMAN_TODO.md (what's left for Ben, in order) and
AUTONOMOUS_RUN_REPORT.md.

## Where things stand

- **Static letters (24):** done. LandmarkMLP 0.960 val (RF 0.907, CNN 0.897,
  TransferNet 0.398), near-flawless live. `checkpoints/mlp_pugeault_best.pt`.
  The letters test split has never been used.
- **J/Z v1:** retired. A logistic regression on the first frame alone separates J from Z at 1.000,
  so it was recognising the I/D handshape, not the motion (NOTES.md, "J/Z motion
  v1 diagnosis"). `checkpoints/motion_gru_best.pt` still loads in the demo.
- **J/Z v2:** code complete. That covers trajectory/velocity/rotation features,
  a NONE class with hard negatives (hold_I, hold_D, ...), the capture script,
  MotionGRUv2, the scale-free gate in `src/motion_gate.py`, and
  `diagnose_motion.py` (probes plus ablations). Tested on synthetic clips only.
  **No recordings yet, so no v2 checkpoint.**
- **Words (GISLR, 250 signs):** code complete, from the preprocessor through
  features, GRU/transformer models, the 4-way feature ablation, evaluation with
  one-/two-handed and static/dynamic breakdowns, and a live words mode in
  `demo.py` (`w` switches mode, `t` for manual start/stop). Also `--video` and
  headless replay parity. Tested on a synthetic GISLR-format set only.
  **Kaggle returns 403 for `asl-signs` (rules not accepted), so there's no word
  checkpoint and no real val/test number.**
- Holistic runs at ~24 FPS (complexity 1) on Ben's 640×480 frames
  (`results/words_fps.json`).

## Setup / sanity check

```bash
cd ~/school/ml-project/asl-recognition
source ../.venv/bin/activate
python -m pytest                     # whole suite: no webcam, no dataset
python src/demo.py --checkpoint checkpoints/mlp_pugeault_best.pt   # letters demo
```

For ad-hoc imports use `PYTHONPATH=src:.`. `src/data/` shadows any
top-level `data` package.

## Next steps (Ben)

Follow HUMAN_TODO.md in order:
1. Record J/Z/NONE: 5 sessions, `ben_m1`..`ben_m5`.
2. Train v2 and read the diagnosis table.
3. Accept the Kaggle rules, then download, preprocess and train the word
   models, run the ablation, test once, and live-test ~20 signs with
   `eval_word_videos.py` / `--video`.
4. Decide on pushing.
5. Rotate the leaked Kaggle token.

Both new test splits are one-shot. Marker files in `results/` refuse a second
test evaluation.
