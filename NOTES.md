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
