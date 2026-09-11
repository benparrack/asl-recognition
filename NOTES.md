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
