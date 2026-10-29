# Autonomous Work Session: J/Z v2 + Two-Handed Word Recognition

You are picking up an ASL recognition project and working **fully autonomously**.
The owner (Ben) will not be available to answer questions during this session.

## 0. Operating rules (read first, follow throughout)

1. **Never ask questions.** When something is ambiguous, pick the most defensible
   option, then log the choice and one line of reasoning in `DECISIONS.md` (create
   it; append-only, dated entries). Keep going.
2. **Blockers don't stop the session.** If a step truly can't be done (missing
   credential, no human to sign on camera, a download fails after 3 attempts),
   log it in `HUMAN_TODO.md` with the exact commands Ben must run. Then move to the
   next step that doesn't depend on it.
3. **Nothing needs a human signing on camera.** Never open the webcam. Anything
   that needs Ben's hands (recording clips, live testing) becomes code you finish
   and verify offline, plus a precise entry in `HUMAN_TODO.md`.
4. **Git:** work on a new branch `feature/motion-v2-and-words`. Make small, logical
   commits with descriptive messages. **Do not push.** The history holds a leaked
   Kaggle token (see below), so pushing is Ben's decision. Don't rewrite history
   and don't delete existing checkpoints, data or result files.
5. **Environment:** the venv is one level up. Run `source ../.venv/bin/activate`
   from the repo root before any Python. Install missing packages into the venv
   with `pip install` only; never `--break-system-packages`, never system pip.
6. **Evaluation discipline (non-negotiable, it's this project's core rule):**
   every split is **signer/participant-disjoint**, never random. Random splits
   were measured to inflate accuracy by ~15 points here. Report validation numbers
   during development. Touch a test split **exactly once**, for the final chosen
   model of a phase, and never tune afterwards. The existing letter and J/Z test
   splits stay untouched.
7. **Train/serve parity:** any feature transform used in training must be
   imported from **one shared function** by both the offline pipeline and the live
   demo. Never re-implement it in the demo.
8. **Style:** match the surrounding code, which is documented modules with
   docstrings explaining *why*. Ben doesn't want tutorial-style narration in chat.
   Put findings in the docs listed in §7.
9. **Compute budget:** RTX 4060 Laptop (8 GB VRAM), 22 CPU cores, 30 GB RAM,
   ~96 GB free disk. The laptop GPU slows under thermal throttling in long runs;
   that's expected, not a bug. Cap hyperparameter search at about 10 training
   runs per phase. Aim for a well-understood good model, not a leaderboard score.
10. **Long-running commands:** run downloads and long training in the background
    and log to files under `logs/` (gitignored). Don't let a hung process stall
    the session.

## 1. Orientation (do this before any code)

Read these, in order:
- `PROJECT_HANDOFF.md`: the full history, repo layout, import-path gotchas,
  config, and results table. It's the long-form source of truth.
- `current_summary.md`: the state of the J/Z motion work, which is newer than
  the handoff.
- `NOTES.md`, `README.md`, `config.py`.
- `src/data/landmarks.py`, `src/models/baselines.py`, `src/models/motion.py`,
  `src/data/motion_dataset.py`, `src/data/capture_motion.py`,
  `src/train_motion.py`, `src/train.py`, `src/demo.py`, `src/evaluate.py`.

Known gotchas that have already cost debugging time:
- `config.py` lives at the **repo root**, outside `src/`. Every script uses a
  `sys.path` bootstrap; copy the pattern from an existing script at the same depth.
- **Name collision:** the repo-root `data/` directory vs the `src/data/` package.
  `import data...` resolves to the wrong one unless `src/` is first on
  `sys.path`. Leave this unfixed unless you have a clean reason; if you fix it,
  do it in its own commit.
- `demo.py` refuses to load a checkpoint whose saved class list doesn't match
  config. Keep that safety pattern, and store feature/version metadata *in*
  every new checkpoint.
- pandas 2.x `groupby().apply()` silently drops the grouping columns. Iterate
  groups explicitly.
- `train_acc` from the training loop runs with dropout active and understates
  accuracy; compare eval-mode numbers only.
- A single-signer validation set is noisy; big epoch-to-epoch swings are normal.

Then, before any other work:
1. `git checkout -b feature/motion-v2-and-words`.
2. Commit the existing uncommitted WIP **as-is** in one commit ("WIP: J/Z motion
   gate v1 (pre-v2 snapshot)"): the modified `.gitignore`, `config.py`,
   `src/demo.py`, plus the untracked `current_summary.md`,
   `results/motion_gru_history.json`, `src/data/capture_motion.py`,
   `src/data/motion_dataset.py`, `src/models/motion.py`, `src/train_motion.py`,
   and this `PROMPT.md`. First check that none of these contain secrets.
3. Add `logs/` and `data/external/` to `.gitignore`.
4. `pip install pyarrow` (needed for the parquet dataset in §3).

## 2. Current state and the key diagnosis

**Working well:** 24 static letters (A–Y without J/Z). `LandmarkMLP` on 63
wrist-normalised MediaPipe Hands features, checkpoint
`checkpoints/mlp_pugeault_best.pt`: 0.960 signer-disjoint val acc, near-flawless
live. **Don't break this path.** The letters demo command must keep working:
```bash
python src/demo.py --checkpoint checkpoints/mlp_pugeault_best.pt
```

**The J/Z motion model (v1) doesn't use motion.** The design is a gated
two-stage system: a centroid-displacement gate buffers a clip, then
`MotionGRU` classifies it. It reported 0.967 val acc, but:
- `capture_motion.py` stores only `normalise_landmarks()` output per frame
  (wrist at the origin, unit scale). **The hand's trajectory through space is
  destroyed at capture time.** Saved `.npz` files contain only a `landmarks` key.
- A logistic regression on the **first frame alone** separates J vs Z at
  **100%** with session-held-out CV. J (pinky up, the I handshape) and Z (index
  up, the 1/D handshape) differ by handshape, so the GRU learned handshape, not
  motion.
- Consequences: it can't separate J from a held I or Z from a held D/1. The gate
  is a forced binary choice with no "neither" class, so any movement between
  letters gets labelled J or Z. The validation "signer" is just another session
  of Ben (signers `ben`, `ben_j2`, `ben_j3`, 15 clips each of J and Z).
- The v1 data **can't be repaired**: raw positions weren't saved. New data must
  be recorded by Ben, so v2 is "build everything, verify offline, hand Ben a
  recording checklist".

**Two-handed words aren't possible with the current features.**
`build_detector()` uses `max_num_hands=1` and `extract_landmarks()` takes hand
`[0]`, so every model sees exactly one hand. Wrist-centring also discards *where*
the hand is, which often decides a word's meaning (MOTHER vs FATHER is the same
handshape at the chin vs the forehead). Words need a new feature pipeline: both
hands in canonical slots, plus body/face-relative location.

**Decision already made (don't re-litigate):** there's no separate "static
words" classifier. Almost no ASL words are static. One sequence model over
per-frame holistic features covers static words as low-motion sequences.

## 3. Phase A: J/Z motion v2 (code complete, verified offline; recording goes to Ben)

Goal: a motion-letter system that uses trajectory, has a rejection class, and
can't win on handshape alone.

A1. **Capture v2.** Extend `src/data/capture_motion.py`, or create
`capture_motion_v2.py` if cleaner:
- Save **raw** per-frame landmarks `(T, 21, 3)` in image-normalised coords,
  per-frame timestamps, a per-frame detected mask (keep undetected frames as
  NaN rather than dropping them, so timing stays honest), and frame width and
  height.
- Add a `NONE` label with on-screen prompts cycling through sub-types: hold I
  still, hold D still, hold a random static letter, move between two letters,
  and wave or reposition the hand. Record the sub-type in metadata. `NONE` is
  what makes the gate reject non-J/Z movement.
- Metadata schema gets a `format_version=2` column. Write to a **new** CSV/dir
  (e.g. `data/motion_v2/`) so v1 and v2 never mix. Add config entries.
- Default ~2.0 s clips with a visible countdown. v1 used 1.0 s with 16–25
  frames, which is tight for a full J.

A2. **Shared motion features.** Add them in `src/data/motion_features.py`, one
function used by training *and* the demo:
- Per frame: the existing 63-dim wrist-normalised handshape (so handshape stays
  available), **plus trajectory channels**. These are the index-tip, pinky-tip
  and wrist positions relative to the clip's first detected wrist position,
  divided by a robust hand-size scale (e.g. median wrist→middle-MCP distance
  over the clip). That makes them position- and distance-invariant but keeps
  the path. Add first differences (velocities) and a wrist-rotation proxy
  (angle of the wrist→middle-MCP vector over time; J includes a wrist twist).
- Resample every clip to a fixed length (e.g. 32) by linear interpolation over
  time, with NaN frames interpolated or masked. Don't use tail zero-padding,
  since it hides length information unevenly.
- Augmentations (train only): small rotation/scale jitter applied consistently
  to all frames, time-stretch, random frame drop, and small mirror-free noise.
  **Never mirror J/Z**: the direction of the path is part of the letter's
  identity.

A3. **Model and training.** Generalise `MotionGRU` or add `MotionGRUv2` with
`num_classes=3` (J, Z, NONE). Keep it tiny. Use a masked/attention-pooled
readout or a bidirectional GRU. Update `train_motion.py` (or add
`train_motion_v2.py`) to use the v2 dataset. Save a checkpoint containing:
classes, `feature_version`, resample length, feature dim and a normalisation
spec. Use a session/signer-disjoint split via `split_by_signer`.

A4. **Anti-shortcut diagnostic.** Add `src/diagnose_motion.py`, which Ben runs
after recording. It reports:
- first-frame-only and mean-frame (order-free) logistic accuracy
- trajectory-only accuracy (handshape channels zeroed)
- the full model's accuracy
- a confusion matrix covering NONE sub-types

Include a "time-shuffled" control, where frames are permuted within each clip.
Accuracy dropping under shuffling is the evidence the model uses motion. Also
compute NONE-hold-I→J and NONE-hold-D→Z false-positive rates specifically.

A5. **Offline verification without Ben:** write a synthetic-data test
(`tests/test_motion_features.py`, pytest; install into the venv if missing).
Generate fake landmark clips with a static handshape and translate them along a
J-hook, a Z-zigzag or no path. Assert that:
- the features differ across paths for an identical handshape
- the features are invariant to global translation and scale
- a tiny model trained on synthetic data beats chance using trajectory channels
  alone

These tests check the pipeline, not real-world accuracy; say so in the docs.

A6. **Demo integration (v2).** In `src/demo.py`, dispatch on the checkpoint's
`feature_version`: v1 checkpoints keep the old behaviour, v2 uses the new
features. Replace the gate's centroid-displacement threshold with a
**scale-invariant** measure (displacement divided by hand size). A `NONE`
prediction releases the gate immediately and returns to static classification
with no hold, which removes the "static letters feel worse" problem.
Keep the existing `[motion]` debug prints and add the NONE outcome.
Verify with a **replay harness**: feed synthetic clips frame-by-frame through
the exact gate plus the classify code path (refactor the gate into a testable
class/function if needed; don't duplicate it). Assert that the gate triggers,
ends and classifies correctly.

A7. **HUMAN_TODO entry:** exact recording commands for Ben. At least 3 sessions
on different days or lighting, each with ~20 clips each of J, Z and NONE (NONE
spread across the sub-types). Follow with the exact train, diagnose and demo
commands and what "good" looks like, e.g. shuffled-control accuracy clearly
below full accuracy, and NONE false-positive rate < 10%.

Commit Phase A in logical pieces.

## 4. Phase B: Word-level dataset (Google Isolated Sign Language Recognition)

Kaggle competition `asl-signs` (already accessible; the CLI is authenticated via
`~/.kaggle/`). It has ~94k clips of 250 signs from 21 participants, stored as
**MediaPipe Holistic landmarks**. There's no video, which fits this project's
landmark approach exactly.

B1. Check the total size (`kaggle competitions files -c asl-signs`, paging
through, or just download and watch). Confirm disk headroom (keep >20 GB free
after extraction). Download to `data/external/gislr/` in the background with
logging, then unzip. If a 403 or rules error appears, log it in `HUMAN_TODO.md`
("accept competition rules at kaggle.com/competitions/asl-signs") and skip to
Phase A work or the §5 design work that doesn't need data.
**Do not** download `asl-fingerspelling`; it's very large and out of scope.

B2. **Verify the schema yourself; don't trust this summary.** Expected:
`train.csv` with `path, participant_id, sequence_id, sign`;
`sign_to_prediction_index_map.json`; per-sequence parquet with
`frame, row_id, type, landmark_index, x, y, z`, where `type` ∈ {face (468),
left_hand (21), pose (33), right_hand (21)} for 543 landmarks per frame, and a
missing hand shows as NaN. Record the verified schema, frame-length stats,
per-participant counts and per-sign counts in `NOTES.md`.

B3. **Preprocess once, in parallel** (multiprocessing across ~20 cores) into a
compact cache, e.g. one float16 `.npy`/memmap of selected landmarks per sequence
plus an index CSV. Keep both hands (42), upper-body pose (shoulders, elbows,
wrists, nose and similar; drop the legs) and a small face subset (lips outline,
nose tip, eye corners, eyebrows: roughly 30–40 points). Document the exact index
lists **in code** as named constants, because the demo must use the same ones.
Make it idempotent with an `--overwrite` flag, matching `preprocess.py`
conventions.

B4. **Participant-disjoint split** reusing `split_by_signer` (map `participant_id`
→ `signer_id`). The fractions in config give roughly 15/3/3 participants. Save
the split assignment to a file so it's reproducible and auditable.

## 5. Phase C: Holistic feature pipeline v2 (shared offline and live)

Create `src/data/holistic_features.py`. This is the most important
train/serve-parity module in the project.

C1. **Canonicalise handedness.** Determine the dominant hand per sequence (the
hand with more detected frames, or more motion). If it's the left hand, mirror x
(`x → 1-x`) for all landmarks and swap the left/right slots, so the dominant
hand always sits in slot 0. This makes the model invariant to Ben's handedness
and to any selfie-mirroring in the dataset. Log the fraction of sequences
flipped.
**Live-demo implication:** the current demo `cv2.flip`s frames before
detection. For the words path, run Holistic on the **un-flipped** frame, flip
only for display, and apply the identical canonicalisation. Write this down in
the docstring and `DECISIONS.md`.

C2. **Per-frame features:**
- per-hand shape: wrist-centred, hand-size-normalised 21×3 (the same idea as
  the existing `normalise_landmarks`), zeros when missing
- per-hand presence flag
- per-hand **location**: the wrist and a fingertip or two, relative to the
  shoulder midpoint (or nose) and scaled by shoulder width
- hand-to-hand vector (dominant wrist → non-dominant wrist, shoulder-scaled)
- dominant hand → face-point distances (chin, nose, forehead-ish, eye), so
  locations like MOTHER/FATHER are explicit
- first differences (velocity) of the location channels

Handle NaN pose/face frames (carry forward, or zero with a flag).

C3. **Sequence handling:** resample or crop to a fixed max length (e.g. 64
frames) with an attention mask. Drop frames where no hand is present at the
clip's start and end.

C4. **Unit tests** (`tests/test_holistic_features.py`): check translation/scale
invariance, the mirror-canonicalisation round-trip, NaN safety, and that the
output dim matches a constant. Add a **parity test** that runs a raw live-format
frame (a dict of arrays like the demo will build from `mp.solutions.holistic`
results) through the demo adapter, and a parquet-derived frame through the
offline loader, and asserts identical features for identical landmarks.

## 6. Phase D: Word model

D1. **Baselines first:** a tiny BiGRU over the C-features, then a stronger model:
a 1D depthwise-conv + small transformer encoder with masked mean pooling. Both
are well-proven designs for this dataset family. Use label smoothing, AdamW and
cosine LR, with augmentations: time-stretch, random frame drop, small
affine jitter applied consistently across the sequence, and random dropout of
the non-dominant hand. Train on the GPU with AMP; keep the batch size inside
8 GB.

D2. **The experiment Ben explicitly wants answered** ("how well does the MLP
approach work for two-handed words?"). Run the ablation with the same model,
split and budget for each feature set:
1. dominant hand shape only (the current letter pipeline's view of the world)
2. both hands' shape
3. both hands + body-relative location
4. (3) + face distances + velocities (full)

Report signer-disjoint val acc and top-5 acc for each, and break results down by
signs where the non-dominant hand is active in more than X% of frames vs
one-handed signs. That quantifies exactly how much the one-hand, no-location
design costs on words. Also report a **static-vs-dynamic** breakdown: bucket
signs by median dominant-hand motion energy and give accuracy per bucket.

D3. Put the training entry point in `src/train_words.py`, following
`train.py`/`train_motion.py` conventions: `--tag`, history JSON in `results/`,
best checkpoint in `checkpoints/` with the class list, feature spec version,
landmark index constants' hash, max length and normalisation stats. Evaluate
with per-class metrics, the most-confused pairs and a confusion-matrix figure
(reuse `evaluate.py` helpers where they fit; extend them where they don't). The
letters study found "default guess sink" classes, so check for that pattern here
too and note it.

D4. **Final test:** for the single chosen model, evaluate the held-out test
participants **once** and record the result. For context, strong Kaggle
solutions reached roughly 0.8–0.9 on unseen participants with heavy ensembling;
a single small model in the 0.7s–0.8s is a solid outcome. Don't chase the
leaderboard.

## 7. Phase E: Demo integration (words)

E1. Add a **words mode** to `src/demo.py`, or a sibling `src/demo_words.py` if
`demo.py` gets unwieldy (log whichever you choose in `DECISIONS.md`). Use
`mp.solutions.holistic` on the un-flipped frame, the shared `holistic_features`
adapter, and a rolling buffer.
Segmentation, in order of robustness:
- (a) a key toggle to start and stop a sign (cv2 has no key-up event, so use a
  toggle)
- (b) automatic mode: a sign starts when a hand rises into the signing space
  and ends when the hands rest or leave the frame, or after a motion-energy
  lull, with the same min-frames safeguards learned from the J/Z gate

Show the top-3 predictions with confidences. Append accepted words to the
existing sentence builder (words separated by spaces; letters still spell).
A key switches between letters mode (the existing Hands + MLP + optional
motion gate, unchanged) and words mode.

E2. Add `--video PATH` input to the demo so it can run on a file instead of the
webcam. With it, add a **headless replay test** that feeds a few GISLR
validation sequences, rendered as landmark streams through the same adapter
the live path uses, into the live inference function. Assert predictions match
the offline evaluation for those sequences. This is the parity proof, since
you can't test live.

E3. Performance: measure Holistic + model FPS on this machine using a synthetic
or blank frame loop and record it. If it's under ~15 FPS, add `model_complexity`
and frame-skip options.

## 8. Documentation and handoff (required at the end, even if phases are incomplete)

- `NOTES.md`: append dated sections covering the J/Z v1 diagnosis (the
  first-frame 100% result is a report-worthy finding, like the earlier
  split-leakage one), the verified GISLR schema, the feature design, the
  ablation table, static-vs-dynamic and one-vs-two-hand breakdowns, and the
  test result.
- `README.md`: update the scope (letters + J/Z + words), with commands for each
  mode.
- `PROJECT_HANDOFF.md` and `current_summary.md`: bring them up to date. The next
  session should be able to start from these alone.
- `DECISIONS.md`: every judgement call you made.
- `HUMAN_TODO.md`: everything that needs Ben, in priority order, each with exact
  commands and the expected output or what success looks like. Must include:
  1. recording J/Z/NONE v2 data (§A7)
  2. training and diagnosing the v2 motion model
  3. live-testing words mode on himself (suggest ~20 known signs recorded as a
     personal test set, ideally with `--video` recordings so it's repeatable;
     this mirrors how the letters were validated "on the train")
  4. deciding whether to push the branch
  5. **security:** confirm the Kaggle API token that was committed to `NOTES.md`
     history has been rotated on kaggle.com (it has never been confirmed)
- `AUTONOMOUS_RUN_REPORT.md`: a concise summary covering what was done, what was
  verified and how, the results table, what failed or was skipped and why, and
  the branch name and commit list. Be honest: separate "verified by test" from
  "written but only verifiable live".

## 9. Priority order if time or compute runs short

1. §1 setup + the §2 diagnosis written into NOTES.md
2. Phase A code + tests + HUMAN_TODO recording plan
3. Phase B download + preprocess
4. Phase C features + parity tests
5. Phase D, the full model first, then the ablation
6. Phase E
7. Docs (§8). **Always finish §8, whatever else is incomplete.**

## 10. Definition of done

- The branch has clean, logical commits, and `pytest` passes.
- The letters demo command from §2 still works unchanged (at minimum, it
  imports and loads the checkpoint headlessly).
- A trained word model checkpoint exists, with val and one-time test numbers,
  the ablation answering the two-hand question, and a replay-verified live
  inference path.
- The J/Z v2 pipeline is complete and tested on synthetic data, ready for Ben's
  recordings.
- All §8 documents are written and accurate.
