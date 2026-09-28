# Autonomous run report (2026-09-28)

This run executed PROMPT.md on branch `feature/motion-v2-and-words`, with nothing
pushed. It asked no questions: judgement calls are in DECISIONS.md and blockers in
HUMAN_TODO.md.

**Headline:** the J/Z v2 and word-recognition pipelines are built end to end and
pass 65 headless tests. **Neither has been trained on real data.** For J/Z v2,
recordings need a person and a webcam. For words, Kaggle returns 403 for
`asl-signs` because the competition rules haven't been accepted on this account.
So there is **no word checkpoint, no real val/test number and no real ablation
result**. Every word number below is a synthetic code check.

## Done and verified by a test

| Item | Verified by |
|---|---|
| J/Z v1 shortcut diagnosed: first-frame logistic regression = 1.000, 3/3 folds | reproduced from `data/motion/*.npz`, NOTES.md |
| v2 motion features (per-point trajectory, wrist path, velocity, rotation) | `test_motion_features.py`: a held hand gives zero trajectory; a traced J/Z doesn't; scale/translation invariance |
| v2 capture → dataset → MotionGRUv2 → training (signer-disjoint, NONE class) | `test_motion_v2_pipeline.py` on synthetic clips |
| J/Z gate moved to `motion_gate.py` (v1 and v2 checkpoints, scale-free threshold, NONE releases) | `test_motion_gate.py` replay tests |
| `diagnose_motion.py` (probes, time-shuffle/neutralisation ablations, hard-negative FPs; one-shot test guard) | `test_diagnose_motion.py` |
| GISLR preprocessor (zip or directory, per-participant shards, participant-disjoint split, `--verify-only`) | `test_preprocess_gislr.py` on the synthetic GISLR-format fixture |
| Word features: one function for train and live, handedness canonicalisation, body-relative x/y | `test_holistic_features.py` (left-handed equals mirrored right-handed; aspect-ratio stretch invariance; parquet ↔ live adapter parity) |
| Word models (BiGRU baseline, conv+transformer), padding inert | `test_words.py` |
| `train_words.py` / `evaluate_words.py` (checkpoint provenance checks, one-/two-handed and static/dynamic breakdowns, one-shot test guard) | `test_words.py` |
| `ablation_words.py` (4 feature sets, same budget) | `test_words.py` (`--aggregate-only`) |
| Live words: segmentation + classifier in `words_live.py` | `test_words_live.py`: manual replay matches offline probabilities to 1e-5; auto segmentation captures every hand frame; blips are discarded; a held pose doesn't retrigger; 15 → 30 fps resampling |
| demo.py words mode, `--video`, `--no-display` | `test_demo_load.py` (headless runs in both modes; the letters demo still loads) |
| `eval_word_videos.py` (personal-video scorer) | `test_words_live.py::test_personal_video_scorer_matches_offline` |
| Holistic FPS measured | `bench_words_fps.py` → `results/words_fps.json` |

`python -m pytest`: **65 passed** (about 50 s, no webcam, no dataset).

## Verifiable only live or with data

- **J/Z v2 accuracy and whether it really uses motion.** Needs the 5 recording
  sessions (HUMAN_TODO §1–2). The diagnosis table there says what "good" looks like.
- **J/Z gate threshold** (1.0 hand-lengths/s) on a real webcam.
- **Any real word accuracy:** val, test, the two-hand ablation and the
  one-/two-handed breakdown (HUMAN_TODO §3a–3e).
- **Segmentation thresholds** (lull 0.4 shoulder-widths/s, 6 hands-down frames),
  the 0.35 acceptance confidence, and mirroring on a real camera. Replay on Ben's
  PEACE stills couldn't test lulls, because the frames aren't consecutive video.
- **GISLR frame rate.** 30 fps is assumed; check it against the `frame` column
  spacing once the data exists.
- **Live-loop FPS with the model in it.** Holistic alone measured 24 FPS, and the
  model adds about 3 ms per clip, so it should be fine.

## Results

| What | Number | Status |
|---|---|---|
| Letters: LandmarkMLP val | 0.960 | unchanged from before this run; the letters test split is untouched |
| Holistic FPS, 640×480, complexity 0 / 1 / 2 | 30.4 / 24.4 / 11.3 | measured on Ben's frames |
| Word model latency per clip (GRU / transformer, cuda) | 3.0 / 2.4 ms | measured |
| Synthetic words, transformer/full val top-1 | 0.996 | **code check, not a result** |
| Synthetic ablation top-1: dom_shape / both_shape / location / full | 0.237 / 0.492 / 0.971 / 0.988 | **code check**: shows the ablation can separate feature sets; says nothing about real signs |
| Real word val / test | — | blocked (403) |
| J/Z v2 val / test | — | no recordings |

## Failures, blockers and definition-of-done gaps

- **GISLR 403.** Retried several times, both bulk and single-file. Listing works,
  so auth is fine and the rules aren't accepted. Public mirrors were rejected:
  they're someone else's preprocessing and have no participant ids
  (DECISIONS.md).
- **Definition of done, not met:** "a word checkpoint with val/test numbers, the
  ablation and replay-verified inference". Only the replay-verified inference
  exists. The rest is a command sequence in HUMAN_TODO §3.
- **Definition of done, met:** pytest passes, the letters demo loads headlessly,
  J/Z v2 is complete and tested synthetically, and the docs are updated.
- Neither new test split has been touched, and none can be touched by accident:
  marker files guard them.

## Branch and commits

Branch `feature/motion-v2-and-words`, from `main`. **Nothing pushed** (the history
contains a leaked Kaggle token; see HUMAN_TODO §4–5). History was not rewritten, and
no checkpoints, data or results were deleted.

```
git log --oneline main..feature/motion-v2-and-words
```

At the time of writing (docs commits follow these):

```
04bf2ed Add a scorer for personal word videos through the live path
84d78aa Add the words-mode FPS benchmark and record it (Holistic ~24 FPS at complexity 1)
cd4f52a Add words mode, --video and headless running to the demo
b7992c4 Add live word recognition with segmentation and a headless replay parity test
f496d89 Add the word feature-set ablation runner
f870aea Add word models, dataset, training and evaluation with breakdowns
a4295e2 Add holistic word features with handedness canonicalisation and parity tests
9b3b991 Add GISLR preprocessing with shared landmark selection and a synthetic fixture
5e7b837 Add motion v2 diagnostics and the recording plan
401accd Move the J/Z gate into motion_gate.py; support v1 and v2 checkpoints in the demo
53cbb6a Add v2 motion capture, dataset, MotionGRUv2 and training script
c528a06 Add v2 motion features (trajectory, velocity, rotation) with synthetic tests
327348b Document J/Z v1 shortcut diagnosis and GISLR download blocker
c765385 Ignore logs/ and data/external/
acb87a0 WIP: J/Z motion gate v1 (pre-v2 snapshot)
```
