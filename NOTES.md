#Notes for findings througout the developing experience

In ben_s1, the first video capture of 80 frames of each test position (FIST, PALM, PEACE),
it was found the around 6 frames failed the crophand() function.
The distribution in frame number failure was most around the middle and the end, most likely when I was moving
the most so I will try to move slower for the future tests, ben_s2 and bens3

## Preprocessing
- 9 sessions, 3 classes (FIST/PALM/PEACE), captured with capture_data.py
- Cropped with crop_hand_square, padding=0.25, resized to 256px
- MediaPipe detection failure rate: 2.5%
  -> this is the pipeline's error floor; end-to-end accuracy cannot exceed 97.5%
EOF

## Run 1: cnn, 20 epochs, full augmentation
- best val acc 0.700, train plateaued ~0.78
- val stuck at 0.333 (single-class prediction) for epochs 1-8
- hypothesis: RandomResizedCrop + rotation destroying already-cropped hands
- still improving at epoch 20 -> undertrained

## Run 3: cnn, 50 epochs, random split (smoke test)
- val acc 0.95 -- PIPELINE VERIFIED, not a generalization result
- random split leaks near-duplicate frames between train/test
- signer-disjoint on same data: 0.80
- leakage inflation: ~15 points
- NOTE: train_acc understated by Dropout(0.5) active in train mode

## Milestone: end-to-end demo working
- live webcam inference, 3 classes, ~90% observed
- live accuracy comparable to held-out test -> no significant train/serve skew
- confirms crop/transform consistency between preprocess.py and demo.py

## Per-class analysis (3-class smoke test, random split)
- FIST: recall 1.000, mean conf 0.967, zero confusions either direction
- PALM: recall 0.921, mean conf 0.784
- PEACE: recall 0.939, mean conf 0.717
- ALL 5 errors are PALM<->PEACE; FIST is fully separable
- confidence tracks difficulty -> model is well calibrated, not overconfident
- FIST predicted 34/105 (~1/3): no class bias, high confidence is earned

## Kaggle Token
REDACTED 2026-09-11 -- this was a live token committed in plaintext (commit
8745cf2, pushed to origin/main). Treat it as compromised regardless of repo
visibility; rotate via kaggle.com account settings if not already done. Not
restoring it here -- use `~/.kaggle/access_token` locally instead.

## Dataset switch: Pugeault & Bowden fingerspelling
- 5 signers (A-E), 24 letters, 200 imgs per signer per letter = 24,000 total
- perfectly balanced across both signers and labels
- dataset authors also exclude J and Z (motion letters) -- independent
  confirmation of the same design decision made in config.py
- RGB frames only (color_*); depth maps excluded via --pattern
- FIRST dataset in this project where split_by_signer() is meaningful

## Detection failure investigation
- crop_hand() on Pugeault dataset: 30.1% failure, heavily uneven
  (signer A kept 13/200 W; signer D lost ALL P images)
- CAUSE: images are already tight hand crops (60-160px), not scene captures.
  MediaPipe's detector expects contextual framing and degrades without it.
- Surviving subset was biased toward easy poses -> would have inflated accuracy
- FIX: --no-detect mode, letterbox to square, 0% loss, geometry preserved
- Reportable finding: detector-based preprocessing is wrong for pre-cropped data

## Run 4: cnn (32,64,128,256), 40 epochs, SIGNER-DISJOINT split
- train 14400 / val 4800 / test 4800, signers 3/1/1
- best val acc 0.879 @ epoch 27 (chance = 0.042)
- final train 0.987 -> 11-point generalization gap to unseen signer
- converged ~epoch 27, flat after; cosine LR schedule
- val volatile epochs 3-20 (0.598 to 0.858) -- single validation signer,
  so the estimate has n=1 person and is inherently noisy

## evaluate.py implemented (2026-09-11)
- collect_predictions/confusion_matrix/per_class_metrics/analyse_confusable_groups/
  plot_confusion_matrix all written and run against cnn_pugeault_best on val split
- accuracy recomputed by evaluate.py matched train.py's recorded val_acc exactly
  (0.8790) -- good cross-check that the two pipelines agree
- only 26.0% of CNN errors fall inside config.KNOWN_CONFUSABLE_GROUPS (151/581)
  -- lower than hoped; see confusion-pair analysis below for what the other
  74% actually is
- worst CNN classes by recall: C (0.400), K (0.570), Q (0.690), A (0.715), M (0.760)

## Run 5: transfer (frozen ResNet18 backbone, ImageNet pretrained), 40 epochs, SIGNER-DISJOINT
- same split as Run 4 (14400/4800/4800, signers 3/1/1)
- best val acc **0.3981** @ epoch 20 -- far below the CNN, plateaued from ~epoch
  10 onward despite train acc climbing to 0.76+
- only 12,312 trainable params (just the replaced head; backbone frozen)
- hypothesis: frozen ImageNet features (edges/textures/shapes) don't capture the
  fine-grained finger-position distinctions ASL handshapes need, made worse by
  feeding 128px images into a backbone that expects 224px
- worst classes: N (0.055 recall!), I (0.125), W (0.200), E (0.240), H (0.250) --
  several classes barely above chance
- genuinely reportable negative result for the three-model comparison, not a bug

## Landmark pipeline implemented (2026-09-11)
- extract_dataset_landmarks() in landmarks.py, build_dataloaders(kind="landmark")
  in dataset.py, LandmarkMLP + TransferNet in baselines.py all implemented
- landmark extraction over full 24000-image Pugeault set: **75.6% detection
  success (24.4% failure, 5862/24000)** -- notably better than the 30.1% failure
  crop_hand() hit earlier, because this ran on the already-letterboxed processed
  images rather than the raw tight Kinect crops
- Python stdout is fully buffered when redirected to a file (not a TTY) --
  progress prints during long-running background jobs don't appear until the
  buffer flushes or the process exits. Worth adding flush=True to long-running
  scripts' progress prints in future, or running with `python -u`.

## Run 6: mlp (63 landmarks -> 128 -> 64 -> 24), 40 epochs, SIGNER-DISJOINT
- train 10031 / val 3961 / test 4146 (smaller than image runs -- rows with failed
  landmark detection are dropped), signers 3/1/1
- best val acc **0.9599** @ epoch 36 -- beats the CNN (0.879) by 8 points with
  ~260x fewer trainable params (18,392 vs ~4.8M)
- trains ~15x faster per epoch than the CNN (~1.2s vs ~18s) -- GPU barely used,
  bottleneck is elsewhere
- THE headline result for the report: explicit feature engineering (MediaPipe
  landmarks, which discard background/lighting/skin tone by construction) beat
  end-to-end pixel learning at this data scale
- worst class: K (0.810 recall) -- still far above the CNN's worst (C at 0.400)
- only 22.0% of errors in known confusable groups (35/159) -- see analysis below

## Model 4: RandomForestClassifier baseline on same landmark vectors
- n_estimators=300, fit in 2.5s (train=10031, val=3961)
- val acc **0.9071** -- between CNN and MLP. Classical ML on good features gets
  within 5 points of the neural MLP almost for free; the neural net still wins,
  but not by a landslide. Worth stating both numbers in the report.

## Final three(+one)-model comparison (val split, SIGNER-DISJOINT, same split every time)
| model                    | val acc | trainable params |
|--------------------------|---------|-------------------|
| LandmarkMLP              | 0.9599  | 18,392            |
| RandomForest             | 0.9071  | n/a (300 trees)   |
| ASLNet (CNN, bs128/lr2e-3)| 0.8967 | ~4.8M             |
| ASLNet (CNN, baseline)   | 0.8790  | ~4.8M             |
| ASLNet (CNN, dropout0.3) | 0.8560  | ~4.8M             |
| TransferNet              | 0.3981  | 12,312            |

## Confusion-pair analysis (src/analyze_confusions.py), all three neural models
Looked past the "% in known groups" summary number at which SPECIFIC (true,pred)
pairs actually drive the errors.

- **Biggest single driver for the CNN**: `C -> O`, 116 of 581 errors (20% of ALL
  CNN errors), almost entirely one-directional (`O -> C` only 6 times). Not in
  `KNOWN_CONFUSABLE_GROUPS`.
- **Biggest single driver for the MLP**: `K -> R`, 29 of 159 errors (18%), also
  one-directional (0 reverse). Not in `KNOWN_CONFUSABLE_GROUPS`.
- **TransferNet**: `X` alone absorbs 362 wrong guesses, its single biggest sink;
  several top pairs (`N->X`, `T->X`, `I->X`) also outside the known groups.
- **The pattern, not just the pairs**: most top-error pairs are strongly
  ASYMMETRIC (e.g. M->N 42 times, N->M 0 times), not the roughly-symmetric
  pattern you'd expect from "these two handshapes genuinely look alike." Looks
  more like specific letters (O, R, X, and for the CNN also P/T) act as a
  default fallback the model collapses toward under uncertainty -- and this
  shows up across all three architectures (pixel CNN, pixel TransferNet,
  landmark MLP), suggesting something about those classes generally, not an
  architecture-specific quirk.
- What DOES match the existing known groups and looks like real mutual
  confusion (roughly symmetric, especially for TransferNet): A<->T, M<->N,
  U<->V<->R, G<->H<->Q.
- Candidate groups the data suggests adding to config.KNOWN_CONFUSABLE_GROUPS,
  if the report wants to incorporate this: {C, O}, {K, L, R, X} (or some
  subset), {W, V}, {I, Y}, {Q, P}. Not yet added to config.py -- an editorial
  call for the report, not made unilaterally.
- **RandomForest reinforces this, and strengthens the claim.** Its biggest error
  is also K -> X (77/368, 21% of its errors, one-directional, 0 reverse), and
  its over-prediction sinks (X: 78, P: 50, O: 48) substantially overlap with the
  neural models' (O, R, X, P). Since RF is trees + majority vote -- no gradient
  descent, no "default guess when uncertain" training dynamic -- this pattern
  showing up there too is evidence it reflects something structural about those
  letters' landmark geometry (K, X, O, P, R apparently sit somewhere less
  separable in the 63-dim landmark space), not an artifact specific to how
  neural nets get trained.

## Ablations (2026-09-11), both vs. Run 4 baseline (dropout=0.5, batch=64, lr=1e-3, val 0.8790)
- **dropout=0.3** (tag pugeault_dropout03): val acc **0.8560** -- WORSE than
  baseline. Train acc pinned near 0.99 the whole back half of training (vs.
  baseline's 0.987 final) -- less dropout let it overfit harder without
  improving generalization. 0.5 was already a reasonable choice; going lower
  hurts, not helps.
- **batch=128, lr=2e-3 (linearly scaled)** (tag pugeault_bs128): val acc
  **0.8967** -- BETTER than baseline, +1.8 points. This is the run that
  resolves the confound the original batch-size comparison (Run 4's own dev
  notes) flagged: the earlier bs64-vs-bs128 test changed batch size without
  scaling LR to match, so it wasn't a fair comparison. With LR properly scaled,
  bigger batches actually help here.
- **New candidate baseline for the CNN leg going forward: bs128/lr2e-3
  (0.8967), not the original bs64/lr1e-3 (0.8790).** Worth using this
  checkpoint (`cnn_pugeault_bs128_best.pt`) as "the" CNN result if/when the
  test set finally gets touched, rather than the original Run 4 checkpoint.

## Live domain-gap test (2026-09-11): landmarks succeed where pixels failed
- Ran demo.py with mlp_pugeault_best.pt live against Ben's laptop webcam, in a
  genuinely different environment (moving train, arbitrary lighting/background
  -- about as far from the 2011 Kinect dataset's conditions as this gets)
- Signed all 24 static letters (no J/Z). Nearly flawless -- every letter
  recognized correctly, ONE weak spot: K, and Ben suspects his own signing was
  the issue there (got 90%+ once he adjusted), not the model
- **This is the answer to the domain-gap question §6.5 raised**: the CNN
  trained on the same dataset scored 0.879 signer-disjoint val acc but was
  "confidently wrong" on Ben's own webcam -- cross-signer generalization does
  NOT imply cross-domain generalization for a pixel model. The landmark MLP
  generalizes across BOTH axes. Best supported explanation: MediaPipe itself
  (not the MLP) is what's actually doing the domain adaptation -- it's a
  general-purpose detector trained on diverse real-world images, so it was
  arguably a worse fit for the Pugeault dataset's atypical tiny pre-cropped
  Kinect crops (source of the 24.4% detection failure rate there) than for an
  ordinary webcam frame. The landmark MLP only ever has to learn on top of
  MediaPipe's already-normalised output, so it inherits that domain robustness
  for free.
- **K trouble matches the offline numbers exactly** -- K was the single worst
  class by recall for the CNN (0.40), RandomForest (0.48), and MLP (0.81)
  alike, and the dominant source of one-directional confusion pairs (K->X,
  K->L, K->R) across nearly every model in analyze_confusions.py. Convergent
  evidence from a completely different method (live human testing vs. offline
  dataset evaluation) that K's handshape is intrinsically harder to separate in
  landmark space, not an artifact of any one dataset or model.
- Practical implication: fine-tuning on Ben's own captured data may matter less
  than originally assumed for the MLP leg specifically -- it already works well
  out of the box. Still valuable for: a properly quantified cross-domain test
  set (rather than one person's qualitative impression), and because the
  CNN/TransferNet legs still likely need it if the domain gap question is to be
  answered for those too, not just the landmark path.

## Still open
- Kaggle API token was committed to git history (commit 8745cf2) and pushed to
  origin/main -- needs rotation via Kaggle account settings if not already done.
- demo.py now supports the landmark model live (`--checkpoint
  checkpoints/mlp_pugeault_best.pt`) -- not yet tested against a real camera.
  This is the actual test of cross-DOMAIN generalization (webcam vs. the 2011
  Kinect dataset), which the strong val accuracy above does NOT establish on its
  own -- Run 4/5's CNN scored well signer-disjoint but failed badly on Ben's own
  webcam, a different generalization axis entirely.
- src/data/capture_alphabet.py written: loops through all 24 static letters in
  one continuous webcam session (vs. one capture_data.py call per letter). Not
  yet run -- Ben plans to capture his own 24-letter dataset across ~4-5 sessions
  (using distinct --signer ids per session, e.g. ben_s1..ben_s5, since
  split_by_signer() needs >=3 distinct signer_id values and this is one person).
- test set still untouched, deliberately -- waiting on ablation results and
  live-test/domain-gap conclusions before finalizing which checkpoint is "the"
  model for the one-time test evaluation.

## J/Z motion v1 diagnosis (2026-09-28): the "motion" model doesn't use motion
- v1 MotionGRU reported 0.967 best val acc (session-disjoint: signers `ben`,
  `ben_j2`, `ben_j3` are three sessions of one person, 15 J + 15 Z clips each,
  16-25 frames per 1.0 s clip).
- **A logistic regression on the FIRST FRAME ALONE separates J from Z at 1.000
  held-out accuracy in every leave-one-session-out fold** (3/3 folds). The
  order-free mean-of-frames vector does the same (1.000, 3/3). Reproduced
  2026-09-28 directly from `data/motion/*.npz`.
- Why: J is traced with the I handshape (pinky up) and Z with the 1/D handshape
  (index up). The clips store only `normalise_landmarks()` output per frame
  (wrist at origin, unit scale), so **the hand's path through space is thrown
  away at capture time**. What's left is handshape, and handshape alone solves
  the task. The GRU learned "I-shape vs 1-shape", not "hook vs zigzag".
- Consequences for the live system: a held I is indistinguishable from J, a
  held D/1 from Z. The gate is a forced J-or-Z choice with no "neither" class,
  so any movement between letters gets labelled J or Z (this is the "static
  letters feel worse with --motion-checkpoint" symptom).
- Can't be repaired: raw positions were never saved. v2 (below) re-records
  with raw landmarks, adds a NONE class, and ships an anti-shortcut diagnostic.
- Report-worthy in the same way as the split-leakage finding: a 0.967 number
  from an honest session-disjoint split can still be the wrong kind of right.
  The split was fine; the *features* leaked the answer through a shortcut.

## J/Z motion v2 built, awaiting recordings (2026-09-28)
- Built and tested on synthetic clips only: `capture_motion_v2.py` (saves raw
  image-space landmarks and timestamps, adds a NONE class with hold_I / hold_D /
  between-letters prompts), `motion_features.py` v2 (trajectory relative to each
  point's own start, velocity, rotation, plus handshape), `MotionGRUv2`,
  `train_motion_v2.py`, the shared `motion_gate.py`, and `diagnose_motion.py`
  (first-frame / mean-frame / trajectory-only probes, time-shuffle and
  neutralisation ablations, hold_I→J and hold_D→Z false-positive rates).
- What the synthetic tests show: a model with the handshape channels zeroed
  still learns J/Z/NONE from trajectory alone (>0.8 on held-out clips, same
  handshape for every class). On synthetic clips, the diagnostic's
  trajectory_only probe is >0.8 and neutralising motion lowers accuracy.
  Shuffling frames changes the trajectory channels.
- No real v2 numbers yet; that needs Ben's 5 recording sessions (HUMAN_TODO §1–2).

## Word recognition (GISLR) status (2026-09-28): pipeline complete, real data blocked
- **Kaggle `asl-signs` downloads return 403** (listing works, so auth is fine; the
  competition rules haven't been accepted on this account). Retried at the start and
  end of the run. The competition's schema is therefore **documented, not yet
  verified**: `preprocess_gislr.py --verify-only` checks it (train.csv columns, sign
  map coverage, parquet columns, 543 landmarks/frame, NaN for undetected parts) and
  writes `results/gislr_schema.json` when the data exists.
- **No real word model, val number, ablation or test result exists yet.** Every
  word number below comes from `src/data/synthetic_gislr.py`, a generated dataset
  in the competition's exact file format whose "signs" are combinations of shape ×
  body location × movement × one/two hands. Those numbers show the code works, not
  that it recognises ASL.

### Feature design (`src/data/holistic_features.py`, one implementation for train and live)
- 84 of Holistic's 543 landmarks: both hands, 9 upper-body pose points, and 33 face
  points (nose, chin, forehead, eyes, brows, lips). The lists are named constants,
  and their hash is stored in every word checkpoint and checked on load.
- Handedness canonicalisation: the dominant hand (clearly more frames, else more
  wrist travel) is mirrored into one slot. A left-handed signer, a right-handed
  signer, and a selfie-mirrored recording give the same features (tested).
- Body-relative location, with x normalised by shoulder width and y by neck height
  **separately**, so a 4:3 webcam and a 9:16 phone agree (GISLR has no frame sizes).
- 153 channels as nested prefixes, so the ablation is four slices of one computation:
  `dom_shape` 64 (what the letters MLP sees) ⊂ `both_shape` 128 ⊂ `location` 139 ⊂
  `full` 153 (face distances, velocities).

### Synthetic code checks (NOT results)
- Transformer / full, 20 synthetic signs × 10 participants (6/2/2 split), 15 epochs:
  val 0.996, top-5 1.000. GRU / full, 8 epochs: val 0.908.
- The ablation runner on the same data (12 epochs each):

  | features | val top-1 | one-handed | two-handed |
  |---|---|---|---|
  | dom_shape | 0.237 | 0.204 | 0.265 |
  | both_shape | 0.492 | 0.481 | 0.500 |
  | location | 0.971 | 0.944 | 0.992 |
  | full | 0.988 | 0.981 | 0.992 |

  This only confirms each feature slice carries what the generator put in it: the
  synthetic signs differ by location and movement by construction, so `location`
  wins. It says **nothing** about the real one-hand vs two-hand question, which
  needs the real ablation (HUMAN_TODO §3d).
- Headless replay parity: GISLR-format val sequences streamed frame by frame as
  Holistic results objects through `words_live.WordRecogniser` (the demo's path)
  give the same probabilities as `evaluate_words` offline (|Δ| < 1e-5), and auto
  segmentation captures every hand frame of a sign padded with rest frames.

### Throughput (`src/bench_words_fps.py`, `results/words_fps.json`)
- Holistic on Ben's own 640×480 webcam captures (person and hand in frame):
  complexity 0 → 30 FPS, **1 → 24 FPS (default)**, 2 → 11 FPS. The word model
  adds ~2–3 ms per sign on the GPU, once per clip. Blank frames give a
  misleading ~55 FPS because the hand and face stages never run.
- Tested only offline: segmentation thresholds, the acceptance confidence, and real
  camera behaviour. See HUMAN_TODO §3f.

## GISLR real data: the second hand is almost never present (2026-10-11)

Preprocessed all 21 participants (94,477 sequences, 3.58M frames, 1.7 GB cache;
the 37 GB zip was deleted afterwards). Per participant, one hand is detected in
60–80 % of frames and the other in ~0–3 %. Left or right depending on the
signer, i.e. the dominant hand. Across all sequences only **0.5 % of frames
have both hands**, including clearly two-handed signs (BOOK, SHOE, CAR, BATH:
median non-dominant presence 0). The raw-parquet sample in
`results/gislr_schema.json` agrees (left_hand 100 % NaN), and the preprocessor
places rows by (type, landmark_index), so this is the dataset, not our code.

Consequences:
- `results/words_sign_properties.csv` marks all 250 signs one-handed, so the
  `two_handed` breakdown in every `*_val_eval.json` is empty. It's meaningless
  on this data.
- The ablation's `both_shape` step can't show a two-hand gain: there's almost
  no second hand to learn from. `location` and `full` are still informative.
- Live, Holistic on a webcam will often see both hands, which the model has
  almost never seen in training. If two-handed signs score badly live, try
  blanking the non-dominant hand at inference to match the training data.

GRU baseline (`--model gru --features full`, 40 epochs): best val **0.600**
top-1 / 0.826 top-5 (epoch 35), on 13,756 val sequences from 3 unseen
participants.

## J/Z motion v2 trained (2026-10-11)

5 sessions × 20 J / 20 Z / 20 NONE (sessions 2–5 at 1.5 s clips, 0.3 s
countdown; session 1 at 2.0 s). Split by session: train m1, m2, m4 / val m3 /
test m5 (test not yet used). `train_motion_v2.py --epochs 80`: val **1.000**
(epoch 76), 0/4 hold_I→J and 0/4 hold_D→Z false positives.

Diagnosis on val (`results/motion_v2_diagnosis_val.json`): time-shuffled 1.000,
handshape neutralised 0.950, motion neutralised 0.983. Neither cue alone is
needed, so the model uses handshape and motion **redundantly**. That's not the
v1 failure (v1 had no motion at all; here the trajectory-only probe gets
0.93), but it can't be called motion-driven. Probes: first_frame 0.946,
mean_frame 0.988. Gate-cut clips start mid-movement, so even frame one
differs from a still hold. Val has only 4 clips per NONE sub-type, so the
false-positive rates are coarse. The live gate is the real test, since only
moving segments ever reach the classifier.

## Word feature ablation (2026-10-11, transformer, 30 epochs, val)

| features | channels | val top-1 | val top-5 | one-handed | two-handed | static | moderate | dynamic |
|---|---|---|---|---|---|---|---|---|
| dom_shape | 64 | 0.556 | 0.792 | 0.556 (n=13756) | n/a | 0.519 (n=4512) | 0.567 (n=4583) | 0.582 (n=4661) |
| both_shape | 128 | 0.564 | 0.794 | 0.564 (n=13756) | n/a | 0.535 (n=4512) | 0.565 (n=4583) | 0.590 (n=4661) |
| location | 139 | 0.614 | 0.823 | 0.614 (n=13756) | n/a | 0.590 (n=4512) | 0.613 (n=4583) | 0.638 (n=4661) |
| full | 153 | 0.639 | 0.843 | 0.639 (n=13756) | n/a | 0.611 (n=4512) | 0.644 (n=4583) | 0.661 (n=4661) |

Location is the biggest single gain (+5.0). The second hand adds +0.8,
which is expected given it's ~absent from GISLR (section above). Face and
velocity add +2.5. Full transformer at 40 epochs: 0.642 val
(`checkpoints/words_transformer_full_best.pt`), the current best word model.
Test split not used yet.

Demo latency fix: the static letter classifier used to pause whenever the J/Z
gate was recording, and every move between letters triggers it, so each letter
change waited for a NONE verdict. Static now pauses only during the post-accept
hold (`GateStatus.skip_static`), and the smoother is cleared on an accepted J/Z.
Ben confirmed live that it's much faster.

## ASL Citizen fixes two-handed signs (2026-10-12)

ASL Citizen (Microsoft, 83k webcam videos, 2,731 glosses, 52 signers,
non-commercial research licence) mapped onto GISLR's vocabulary: 216 of the 250
signs, 8,188 videos (`preprocess_asl_citizen.py`, numbered variants merged,
8 aliases like dad→FATHER). Official signer-disjoint split, restricted to these
signs: train 35 signers / 3,925 videos, val 5 / 1,000, test 11 / 3,263
(only ~18 train videos per sign). 38 % of videos have both hands in >20 % of
frames vs 0.9 % in GISLR. Checked: no signer or video file in two splits, no
val landmark sequence identical to a train one.

Extraction caveat: this laptop throttles at ~104 °C (Holistic ~3 fps/process),
so after the first 774 videos the rest were extracted with `--max-fps 15`
(every other frame of 30 fps clips, then resampled to 30 fps as before). On a
test clip that halved the time with the same timestamps/resampled length.

All runs: transformer, full features, 40 epochs, GISLR's 250-way head; each
dataset keeps its own split; checkpoint selected on the mean of the two val
accuracies. "Ben's clips" = 100 webcam clips (20 signs × 5, one session,
`data/word_videos`), never trained on, scored whole-clip through the live path.

| run | GISLR val | ASLC val | ASLC one-handed | ASLC two-handed | Ben top-1 | Ben top-3 |
|---|---|---|---|---|---|---|
| GISLR only (`words_transformer_full`) | 0.642 | 0.548 | 0.767 | **0.245** | 0.37 | 0.59 |
| both (`aslc_r1`) | 0.650 | 0.900 | 0.914 | 0.881 | 0.67 | 0.76 |
| both, ASLC ×3 (`aslc_r3`) | 0.645 | **0.932** | **0.945** | 0.914 | **0.76** | **0.83** |
| ASLC only (`aslc_only`) | 0.337 | 0.916 | 0.905 | 0.931 | 0.60 | 0.76 |

(ASLC one/two-handed: 134 / 82 signs, from `results/words_sign_properties_asl_citizen.csv`.)

Two-handed accuracy on unseen signers goes 0.25 → 0.91, GISLR doesn't suffer,
and Ben's own clips double (0.37 → 0.76). The ranking on Ben's clips matches
ASLC val, so ASLC val is a usable proxy for live; ASLC-only does worse on
Ben's clips than its val suggests, so keep GISLR in the mix. ASLC val (0.93)
is optimistic for live: one uniform webcam setup, 216 classes. Test splits of
both datasets are still unused.

Tuning sweep on top of `aslc_r3` (val + Ben's clips, whole-clip top-1):

| run | GISLR val | ASLC val | ASLC two-handed | Ben top-1 |
|---|---|---|---|---|
| `aslc_r3` (ASLC ×3, 40 ep) | 0.645 | 0.932 | 0.914 | **0.76** |
| `aslc_r5_e60` (ASLC ×5, 60 ep) | 0.646 | 0.931 | 0.912 | 0.71 |
| `aslc_r3_d03` (dropout 0.3) | 0.647 | 0.928 | 0.919 | 0.67 |
| `aslc_r3_w256` (dim 256) | 0.645 | 0.940 | 0.936 | 0.68 |

All within noise on val (1,000 videos, SE ≈ 0.8 pts); none beats `aslc_r3` on
Ben's clips, so **`checkpoints/words_transformer_full_aslc_r3_best.pt` is the
word model**. Tuning has plateaued; more gains need data, not settings.

Live check (Ben, webcam): one-handed signs work well, two-handed worse. On his
clips `aslc_r3` gets two-handed 0.66 vs one-handed 0.86. Cause is mostly hand
DETECTION, not the classifier: BOOK (0/5) has both hands in 0–5 % of frames
(ASLC: 28 %) and CLEAN (0/5) in 3–31 %. In those clips Ben's hands are close to
a low camera, fill the frame and press together, and Holistic finds only one.
Where both hands are detected (SHOE, WET) it's 5/5 at 97 %+. Holistic
complexity 2 doesn't help. Remaining misses are near-neighbour signs
(dance→jump, car→horse, hello→bye). Fix is framing: sit back so head to waist
is in frame, camera at face height, sign at chest height.

A suspend/resume cycle mid-training left one run hung (GPU idle, no progress);
long jobs now run as `systemd-run --user` units via resumable scripts in `logs/`.
