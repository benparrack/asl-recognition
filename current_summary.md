# Current Project State (handoff notes)

## Where things stand
All 24 static letters (A-Y minus J/Z) work well — validated offline via a
three(+one)-model comparison (LandmarkMLP best at 0.960 val acc, RandomForest
0.907, CNN 0.879-0.897 across ablations, TransferNet 0.398) and live via
webcam in a genuinely different environment (a moving train), with
near-flawless recognition across the whole alphabet except K, which improved
to 90%+ once Ben adjusted his handshape.

J and Z (the two motion letters) are being added via a gated two-stage
design: the static per-frame classifier runs as normal, plus a separate
motion-detector gate (frame-to-frame hand-centroid displacement) that, when
triggered, buffers a landmark sequence and classifies it with a small
`MotionGRU` (`src/models/motion.py`). This whole pipeline
(capture -> dataset -> train -> live demo) has been built end-to-end and the
trained motion model reached 0.9667 best val accuracy.

## Live testing status (in progress)
- Ben captured motion clips for J/Z across 3 signer sessions (`ben_j1`,
  `ben_j2`, `ben_j3`) using `src/data/capture_motion.py`.
- First live test of the gate in `src/demo.py` looked "stuck" on
  "recording J/Z..." — diagnosed as the gate silently discarding
  clips as too-short and immediately re-triggering, with no visible feedback.
  Added diagnostic print statements (`[motion] triggered`, `accepted`,
  `REJECTED`, `clip too short`).
- Second live test (with prints) showed the gate WAS working (`Z accepted`
  multiple times, 0.64-0.76 confidence) but many attempts were discarded as
  "clip too short" — diagnosed as one continuous gesture getting fragmented
  by a brief natural pause mid-gesture satisfying `MOTION_MIN_LOW_STREAK`
  before enough frames had accumulated.
- Fix applied: `gesture_done` in `src/demo.py` now requires
  `len(motion_buffer) >= MOTION_MIN_FRAMES` before the low-motion-streak path
  can end a clip early; only the max-length cutoff (`motion_max_len`) can end
  a clip before that minimum. Syntax-checked, not yet confirmed by a live
  retest.
- **Open question, not yet resolved**: no "J accepted" event has appeared in
  any live log yet — only "Z accepted". Need to deliberately test J to
  confirm it actually works live, not just Z.
- Also noticed: running with `--motion-checkpoint` active makes static-letter
  recognition feel worse, because any hand movement above the displacement
  threshold (e.g. moving between letters) triggers the motion gate and
  pauses static classification for up to `MOTION_HOLD_SECONDS` (1.5s) while
  it buffers/classifies. This is expected gate behavior, not a regression —
  to get pure static-only behavior back, just omit `--motion-checkpoint`:
  ```bash
  python src/demo.py --checkpoint checkpoints/mlp_pugeault_best.pt
  ```
  Full command with motion enabled:
  ```bash
  python src/demo.py --checkpoint checkpoints/mlp_pugeault_best.pt --motion-checkpoint checkpoints/motion_gru_best.pt
  ```

## How to get set up in a new terminal
```bash
cd /home/ben/school/ml-project/asl-recognition
source ../.venv/bin/activate
```
(venv lives one directory up, at `/home/ben/school/ml-project/.venv`)

## Checkpoints available
`checkpoints/` contains: `cnn_best.pt`, `cnn_noaug_best.pt`,
`cnn_pugeault_best.pt`, `cnn_pugeault_bs128_best.pt`,
`cnn_pugeault_dropout03_best.pt`, `cnn_smoketest_best.pt`,
`cnn_throwaway_best.pt`, `mlp_pugeault_best.pt` (best static model),
`motion_gru_best.pt` (J/Z model), `transfer_pugeault_best.pt`.

## Pending / not yet done
- Retest `demo.py` with the `gesture_done` fix; confirm fragmentation issue
  is resolved.
- Deliberately test J specifically to get a confirmed "J accepted" event.
- Further threshold tuning if needed: `--motion-threshold` (default 0.025),
  `MOTION_MIN_LOW_STREAK` (3), `MOTION_MIN_FRAMES` (8),
  `MOTION_HOLD_SECONDS` (1.5) — all in `src/demo.py`.
- Kaggle token rotation on kaggle.com — flagged earlier after a live token
  was found committed to `NOTES.md` history; redacted from the file going
  forward but rotation on kaggle.com itself has never been confirmed done.
- Once J/Z gate is stable: commit new motion-letter code/results, update
  NOTES.md/README.md with J/Z results.
- Test sets (static and motion) remain deliberately untouched, per project
  discipline ("touched exactly once, at the very end").

## Notes on working style
- Ben does not want unsolicited explanations/teaching — answer only what's
  asked, keep it direct.
- This is not a git repo at the top level containing this file's directory
  in the sense checked here — confirm before assuming `git` history reverts
  are available as an undo mechanism.
