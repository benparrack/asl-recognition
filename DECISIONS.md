# Decisions log

Append-only. One entry per judgement call made without Ben available, with the
reasoning in a line or two. Newest at the bottom.

## 2026-09-28 (autonomous session, branch `feature/motion-v2-and-words`)

- **GISLR download blocked (403), no mirror used.** `kaggle competitions files -c asl-signs`
  works (auth fine) but every download call (bulk and single-file, 3 attempts) returns
  403 Forbidden, the signature of unaccepted competition rules. Public Kaggle mirrors
  were checked (`markwijkhuizen/gislr-dataset-public`, `cristaliss/islr-hands-only-dataset`,
  `aapokossi/saved-tfdataset-...`): all are someone else's *preprocessed* tensors
  (dominant-hand only, fixed landmark subset, no participant ids), so they can't answer
  the two-hand ablation or feed a pipeline whose landmark indices must match the live
  demo. Logged in HUMAN_TODO; Phases B–E are built and verified against a synthetic
  fixture with the exact competition schema instead.
- **Read parquet straight out of the zip, don't extract.** The competition archive is
  roughly 40+ GB and extraction would need a similar amount again; with ~96 GB free that
  breaks the ">20 GB free after extraction" rule. `preprocess_gislr.py` reads members
  from the zip (or an extracted directory, whichever exists), so the extracted copy is
  never needed.
- **Motion v2 trajectory channels: each point relative to its own first position**, not
  to the first wrist position as PROMPT §A2 literally says. Relative to the wrist, a
  fingertip's offset encodes handshape (I vs D place the tip differently), which leaked
  handshape into the "trajectory-only" diagnostic. Caught by the synthetic test
  (a held hand gave non-zero trajectory). The wrist channel is unchanged.
- **No clip-duration feature in motion v2.** Training clips are fixed 2 s windows while
  live clips are gate-segmented, so duration would be a train/serve mismatch.
- **Motion normalisation lives inside the model** as buffers (`feat_mean`/`feat_std`),
  and is also written out in the checkpoint's `normalisation` spec for auditing. The
  demo can't skip or mis-apply it.
- **The motion gate moved to `src/motion_gate.py`**, which both `demo.py` and the replay
  tests call through the same `MotionGate.update()`. v1 checkpoints keep v1 behaviour
  (centroid displacement, 0.025 threshold); v2 uses hand-lengths/second (default 1.0)
  so the threshold no longer depends on how far the hand is from the camera. A NONE
  result releases the gate with no hold.
- **Recording plan: 5 sessions, one signer id each** (`ben_m1`..`ben_m5`). With the
  existing 15 %/15 % fractions and at least one signer per split, 3 sessions gives
  1/1/1 and the model would train on one session. 5 gives 3/1/1.
- **The motion v2 test split is guarded by a marker file** (`results/motion_v2_test_used.json`).
  `diagnose_motion.py --split test` refuses a second run. The one-time rule is
  enforced, not left to memory.
- **Diagnostic "neutralisation" uses the training mean**, not literal zeros. The model
  standardises its input internally, so the training mean becomes 0 after
  normalisation, i.e. an "average" value rather than an out-of-distribution spike.
- **Word pipeline built and tested on a synthetic GISLR-format dataset**
  (`src/data/synthetic_gislr.py`) while the real download is blocked. It writes the
  documented schema (train.csv, sign map, long-format parquets, NaN hands, both a zip
  and a directory layout), and includes left-handed participants and two-handed signs.
  Every number it produces is a code check, not a result, and is labelled that way everywhere.
- **Landmark selection: 84 of 543 points** (both hands, 9 upper-body pose points, 33
  face points). The exact lists are named constants in `src/data/holistic_landmarks.py`,
  and their hash goes into word checkpoints. The cache stores raw selected landmarks
  (NaN kept), not features, so features can change without re-reading ~94k parquets.
- **One cache shard per participant** (`<pid>.npy` float16 plus a per-shard CSV). 21
  participants suit ~20 workers, and a shard is written under a temp name then renamed,
  so an interrupted run never leaves a half-shard that looks complete.
- **Dominant hand = clearly more detected frames, else more wrist travel.** PROMPT
  §C1 offers either. Frame count alone picked the wrong hand on two-handed signs,
  because the static base hand is tracked more steadily than the moving dominant
  one (caught by `test_left_handed_signer_matches_right_handed`). Counts decide
  when one hand appears in under half as many frames as the other; otherwise travel decides.
- **Body-relative coordinates normalise x and y separately** (x by shoulder width,
  y by nose-to-shoulder height). GISLR has no frame sizes, and phone video differs
  from a 4:3 webcam, so a single isotropic scale would make location features
  depend on the camera's aspect ratio. Tested by stretching x by 1.4 with no
  change in any feature.
- **Words path: Holistic runs on the un-flipped frame**, and the flip is for display
  only (PROMPT §C1). The letters path keeps flipping before detection, which is
  unchanged.
- **Live word clips are resampled to 30 fps (nearest frame) before featurising**, because
  velocity channels are per-frame differences. GISLR's frame rate isn't in the data; 30
  is an assumption to check against the `frame` column spacing once the data is here.
- **Word models: a small conv+transformer as the main model, a BiGRU as the baseline.**
  Depthwise temporal convs followed by a 2-layer transformer is a scaled-down version of
  what did well on this competition. The BiGRU is the same family as the J/Z model, so the
  comparison has a simple reference point. Both pool with a masked mean, and padding is
  provably inert (`test_padding_does_not_change_output`). Defaults: width 192, 40 epochs,
  AdamW lr 1e-3 / wd 0.05, one-epoch warm-up then cosine, label smoothing 0.1, AMP. These
  are standard choices, untuned, because no real data was available to tune on.
- **Input standardisation for word models lives inside the model** (buffers, from ≤3000
  train sequences), for the same reason as motion v2.
- **The ablation is four separate runs with an `ablation` tag**: same model, seed, split and
  epochs; only the leading feature channels differ. `full` is retrained under the tag
  rather than reusing the main run, so all four have an identical budget.
  `ablation_words.py --aggregate-only` rebuilds the table from the eval JSONs.
- **One- vs two-handed and static vs dynamic are per-sign properties measured from TRAIN
  sequences** (non-dominant hand in >50 % of frames; dominant-wrist speed terciles), not
  labels anyone curated. They're descriptive only, never used for training or selection.
  Terciles give three equal-sized sign groups, whatever the absolute speeds are.
- **The word test split is guarded by `results/words_test_used.json`**, as for motion v2.
- **Words mode lives in `demo.py`, not a separate `demo_words.py`.** PROMPT §E1 wants a key
  that switches letters ↔ words in one session, which needs one loop. To keep `demo.py`
  readable, the live word logic is in `src/words_live.py` (replay-testable, like
  `motion_gate.py`), and the old letters loop body moved verbatim into `letters_step()`.
  The only change to letters behaviour: the J/Z gate receives the frame timestamp `ts`
  instead of calling `time.time()` itself. For the webcam `ts` is `time.time()`, so
  behaviour is identical; for `--video` it's the file's own clock.
- **Auto segmentation defaults** (`SegParams`): start after 2 raised-hand frames, with 4
  frames of pre-roll; end on 6 frames of hands down, or a lull below 0.4
  shoulder-widths/s for 12 frames once ≥15 frames are in, or a 120-frame cap; fewer
  than 5 hand frames → discarded. These are reasoned guesses at 30 fps, not measured,
  so they're exposed as `--words-lull-speed` and the manual `t` toggle is always
  available. After a lull ending, auto-start disarms until the hands drop or move
  again; otherwise a held last handshape retriggers every half-second.
- **Accepted words go straight into the sentence** (no SPACE confirm), since a sign
  already had an explicit start and end. Acceptance needs top-1 ≥ 0.35, a guess for
  250 classes that should be checked live. Below that the top-3 is shown but nothing
  is appended.
- **No frame-skip option.** Holistic measured 24 FPS at `model_complexity=1` on Ben's
  own 640×480 captures (30 FPS at 0, 11 at 2), above PROMPT §E3's ~15 FPS bar. Frame
  skipping would also thin the clip that velocity features are computed from.
  `--holistic-complexity` is exposed anyway because it's free. Blank frames report
  ~55 FPS, which is misleading: with nobody in frame the hand and face stages don't run.
- **The replay parity test streams from the preprocessed cache**, scattered back into
  Holistic's raw layout (unselected points get dummy values, as a real detection has all
  of them). That makes the offline side exactly `evaluate_words.collect` on the same
  arrays, so probabilities must match to 1e-5. Adapter parity from raw parquets is
  already covered in `test_holistic_features.py`.
- **`--video` uses the file's timestamps (CAP_PROP_POS_MSEC)**, not the wall clock, so a
  recorded sign has its true timing even though Holistic runs slower or faster than
  real time. At end of file an open clip is flushed and classified (`end_of_input`).
- **Added `src/eval_word_videos.py`** for the ~20-sign personal test. It scores each file
  both as one whole clip (model only) and with auto segmentation (model + segmenter),
  so a poor live number can be attributed to one or the other.

## 2026-09-28 (same session, follow-up while waiting for recordings)

- **Motion v2 training clips are cut by the live gate's own code**
  (`motion_gate.gate_segment`, applied in `motion_dataset_v2.load_model_clip`
  for train, val, test and diagnostics). Capture records a fixed 2 s window
  (still → letter → still), but live the model only sees the gate's cut, so
  otherwise a J would fill ~half the time axis in training and nearly all of it
  live. The previous stand-in, a random 0–15 % end crop, didn't close that gap.
  On synthetic clips the cut features are 2–3× closer to the bare letter's than
  the full window's. A test checks that `gate_segment` returns exactly the frames
  the live gate passes to the classifier. Clips the gate never fires on (held
  hands) stay whole. The augmentation end crop dropped to 10 %, as it now only
  jitters the gate boundaries. The checkpoint records `train_segmentation`.
  Caveat: training uses the default threshold 1.0. A very different
  `--motion-threshold` live would cut clips differently from training.
- **The capture screen reports after each J/Z clip whether the gate would have
  fired.** A trace too slow to trigger it would never be caught live, and would
  train on the full window, so it should be redone during the session, not found
  afterwards.
- **`r` on a NONE clip now redoes the same prompt.** It used to advance the
  sub-type rotation, so each redo skipped one sub-type.
