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
