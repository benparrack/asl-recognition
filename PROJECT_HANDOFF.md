# ASL Fingerspelling Recognition — Project Handoff

Context document for continuing work on this project. Written at the end of the
setup-and-first-results phase.

---

## 1. Who and why

Ben, junior CS major, taking Machine Learning, AI, App Development, and
Data-Driven Analytics. This is a self-directed project for fun and also for learning.

Ben is new to Linux and to PyTorch.

**Update, 2026-09-11:** earlier guidance in this doc said to explain every terminal
command and let Ben write model-relevant code himself with guidance. Ben has since
clarified this is personal curiosity, not a school assignment, and no longer wants
unsolicited explanations or the write-it-yourself pattern by default — write code
directly and move through steps; he'll ask if he wants something explained.

He does **not** know ASL. This matters for validation (see §7).

---

## 2. Environment

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS, dual-booted alongside Windows |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB VRAM, driver 595.84, CUDA 13.2 |
| CPU | 22 cores |
| Python | 3.12 |
| venv | `~/school/ml-project/.venv` (one level ABOVE the repo) |
| Repo | `~/school/ml-project/asl-recognition` |
| GitHub | `github.com/benparrack/asl-recognition` |

Activate with `source ../.venv/bin/activate` from inside the repo. There's an
alias `asl` in `~/.bashrc` that does both the cd and the activation.

`torch.cuda.is_available()` returns True. Training runs on GPU at ~10 s/epoch on
14,400 images.

**Environment notes:**
- Ubuntu 24.04 refuses system-wide pip (`externally-managed-environment`). Never
  use `--break-system-packages`. Everything goes in the venv.
- `~/.bashrc` had ROS 2 (`/opt/ros/jazzy/setup.bash`) lines that hung every new
  terminal. They are commented out. A backup exists at `~/.bashrc.backup`.
- Kaggle CLI authenticates via `~/.kaggle/access_token` (raw token, not JSON —
  this version of the CLI does not use `kaggle.json`).
- The course's ML class uses a separate Docker setup (`ageron/handson-ml3`) that
  is unrelated to this project.

---

## 3. Repository layout

```
asl-recognition/
├── config.py                  # single source of truth: paths, classes, hyperparams
├── NOTES.md                   # running experiment log — KEEP UPDATING THIS
├── requirements.txt
├── .gitignore                 # excludes data/, checkpoints/, *.zip
├── src/
│   ├── train.py               # training loop + CLI
│   ├── evaluate.py            # STUB — highest priority work
│   ├── demo.py                # live webcam inference
│   ├── data/
│   │   ├── capture_data.py    # webcam capture -> raw/ + raw_metadata.csv
│   │   ├── build_metadata.py  # folder-structured dataset -> raw_metadata.csv
│   │   ├── preprocess.py      # raw/ -> processed/ + metadata.csv
│   │   ├── dataset.py         # Dataset classes, transforms, SPLITTING
│   │   ├── landmarks.py       # MediaPipe: detection, cropping, letterboxing
│   │   └── contact_sheet.py   # visual grid review of images
│   └── models/
│       ├── cnn.py             # ASLNet — the scratch CNN (COMPLETE)
│       └── baselines.py       # STUB — TransferNet + LandmarkMLP
├── data/raw/                  # gitignored
├── data/processed/            # gitignored
├── checkpoints/               # gitignored
└── results/                   # history JSONs, figures
```

### Import bootstrapping — read this before editing any script

`config.py` lives at the repo root, outside the `src/` package. Every script
needs a `sys.path` bootstrap:

- Files in `src/` → `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`
- Files in `src/data/` and `src/models/` → `parents[2]`, **plus**
  `parents[2] / "src"`

**Name collision, important:** there are two things called `data` — the dataset
directory `data/` at the repo root, and the Python package `src/data/`. Python
resolves `import data` to the root directory (a namespace package) unless
`src/` is explicitly on `sys.path` first. That produces
`ModuleNotFoundError: No module named 'data.landmarks'`.

A clean permanent fix (not yet done, deliberately deferred): move `config.py`
into `src/`, or rename `data/` to `datasets/`. Either removes all the path
surgery. Worth doing during a refactor, not mid-debugging.

---

## 4. Data

### 4a. Primary dataset — Pugeault & Bowden fingerspelling (CURRENT)

Kaggle: `mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out`
Extracted to `data/raw/dataset5/<SIGNER>/<letter>/color_*.png`

- **5 signers** (A, B, C, D, E) — this is what makes an honest split possible
- **24 letters** (dataset authors also exclude J and Z, independently confirming
  the design choice in `config.py`)
- Sampled at 200 images per signer per letter = **24,000 images**, perfectly
  balanced on both axes
- Source images are **already tight hand crops**, 60–160 px, non-square, and
  contain both RGB (`color_*`) and depth frames — the `--pattern 'color_*'` flag
  excludes depth

Built with:
```bash
python src/data/build_metadata.py --source data/raw/dataset5 --layout nested \
  --pattern 'color_*' --per-class 200 --force
python src/data/preprocess.py --no-detect --overwrite
```

### 4b. Secondary dataset — own webcam captures

3 "signers" (`ben_s1`, `ben_s2`, `ben_s3` — actually three consecutive sessions
at the same desk, all `session_id=desk`), 3 classes (FIST / PALM / PEACE),
~720 images. Raw images still on disk under `data/raw/ben_s*`. Its metadata is
backed up at `data/raw/raw_metadata_webcam.csv.backup`.

Used for the pipeline smoke test. `checkpoints/cnn_smoketest_best.pt` is trained
on it and **works well live**, unlike the Pugeault model (see §6).

---

## 5. Current config settings

```python
STATIC_LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")   # 24, no J/Z
NUM_CLASSES = 24
IMAGE_SIZE = 128
BATCH_SIZE = 64            # settled by ablation
NUM_WORKERS = 8            # settled by ablation
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
VAL_SIGNER_FRACTION = 0.15
TEST_SIGNER_FRACTION = 0.15
```

Model: `ASLNet(channels=(32, 64, 128, 256), dropout=0.5)`.
Conv→BN→ReLU→MaxPool blocks, then AdaptiveAvgPool2d(1), Flatten, Dropout,
Linear. ~94k params at 3 blocks; roughly 4× that at 4 blocks.

Augmentation currently: `Resize`, `RandomRotation(15)`, `ColorJitter`,
`ToTensor`, `Normalize`. **`RandomResizedCrop` is deliberately disabled** — it
clips fingertips on already-tight hand crops.

---

## 6. Results so far

| Run | Data | Split | Val acc | Notes |
|---|---|---|---|---|
| 1 | webcam 3-class | signer | 0.700 | full augmentation, train plateaued 0.78 |
| 2 | webcam 3-class | signer | 0.800 | augmentation reduced |
| 3 | webcam 3-class | **random** | 0.952 | pipeline verified; leaky, not reportable |
| 4 | Pugeault 24-class | **signer** | **0.879** | ← the real result, epoch 27/40 |

**Run 4 detail:** train 14,400 / val 4,800 / test 4,800; signers 3/1/1.
Final train acc 0.987, so an **11-point generalization gap** to an unseen signer.
Converged ~epoch 27, flat after. Chance = 0.042.

Checkpoint: `checkpoints/cnn_pugeault_best.pt`

**The test set has never been touched.** Deliberately. It gets used exactly once,
at the end, in `evaluate.py`.

### Findings worth reporting

1. **Leakage inflation ≈ 15 points.** Same 3-class data: 0.95 random split vs
   0.80 signer-disjoint. Quantified, not hand-waved.

2. **Detector-based preprocessing is wrong for pre-cropped data.** Running
   `crop_hand()` (MediaPipe) on the Pugeault images failed on **30.1%** of
   frames, and unevenly — signer A kept only 13/200 W images, signer D lost
   *every* P image. The surviving subset was biased toward easy poses. Cause:
   MediaPipe's detector expects contextual framing and degrades on images that
   are already tight crops. Fix: `--no-detect` mode that letterboxes instead,
   0% loss.

3. **Letterboxing vs stretching.** Source images have varying aspect ratios
   (84×126, 155×86, 64×146). A plain resize to square distorts each differently,
   and the distortion correlates with original framing — a spurious signal.
   `letterbox_square()` scales the long side and pads the short one.

4. **Dataloader ablation.** workers 4→12: 11.6→9.7 s/epoch, GPU util 63%→98%,
   temp 81°C→64°C. workers 12→8: no measurable difference, so 8 is kept (frees
   cores for the desktop). batch 64→128: slower *and* worse — pipeline was
   CPU-bound so batch size wasn't the bottleneck, and it halves gradient updates
   per epoch (225→112) without a compensating LR increase.
   Note: later-epoch slowdown (10 s → 24 s over 40 epochs) is **thermal
   throttling** on a laptop GPU, not a code issue.

5. **Domain gap — the most interesting negative result.** The Pugeault-trained
   model performs badly on Ben's laptop webcam even after framing was matched
   (tight non-square crop at padding 0.15, then letterboxed to 256). Predictions
   are **confidently wrong**: 20–90% confidence, letter usually incorrect. Not an
   uncertainty problem — the model finds strong evidence for wrong classes on
   out-of-distribution input.

   Remaining unaddressed differences: sensor (Kinect vs webcam), resolution
   (60–160 px upscaled vs 300–500 px downscaled), lighting, unseen hands.

   **The 3-class webcam model works fine live.** Same architecture, same code,
   same demo — the difference is entirely whether training and deployment
   conditions match. Cross-*domain* generalization is a distinct and harder
   problem than cross-*signer* generalization within one dataset.

   One untested hypothesis: simulate the training pipeline's resolution loss in
   the demo by downscaling the crop to ~110 px and back up to 256 before
   inference. Cheap to test, not yet tried.

---

## 7. Known issues and gotchas

- **`config.CLASSES` must match the checkpoint.** `load_model()` in `demo.py`
  raises if they differ. Switching between the 3-class and 24-class models means
  editing `config.py` (the old line is commented out, not deleted).
- **`demo.py` framing depends on which model is loaded.** Pugeault needs
  `square=False` + `letterbox_square()`; the webcam model needs `square=True`
  and no letterbox. This is currently a manual edit and would be better as a
  flag, or better still recorded in the checkpoint.
- **pandas 2.x `groupby().apply()` drops grouping columns.** This silently
  produced a metadata CSV missing `signer_id` and `label`. Fixed by iterating
  groups explicitly. Watch for this pattern elsewhere.
- **`train_acc` is understated** because it's computed with `model.train()`
  active, so Dropout(0.5) is zeroing half the features. Don't compare it directly
  against val accuracy. Use `evaluate()` in eval mode for a true training-set
  number.
- **Validation is a single signer**, so the val estimate is noisy — swings of
  15+ points between consecutive epochs are expected and are not a bug.
- **`.gitignore` must exclude `*.zip`.** A 357 MB Kaggle archive at the repo root
  was rejected by GitHub's 100 MB limit. Already fixed.
- **`preprocess.py` is idempotent** — it skips existing files. Use `--overwrite`
  when changing the cropping method, or you get a dataset with mixed
  preprocessing styles, which is worse than either style alone.

---

## 8. What's still stubbed

**`src/evaluate.py`** — all four functions raise `NotImplementedError`:
- `collect_predictions(model, loader, device)` → `(y_true, y_pred, y_prob)`
- `confusion_matrix(y_true, y_pred, num_classes)` → 24×24 int array
- `per_class_metrics(cm)` → precision / recall / F1 per class
- `analyse_confusable_groups(cm)` → fraction of error inside
  `config.KNOWN_CONFUSABLE_GROUPS`
- `plot_confusion_matrix(cm, classes, out_path)` → matplotlib figure

**`src/models/baselines.py`** — `TransferNet` (pretrained ResNet + new head) and
`LandmarkMLP` (63-dim input) are both stubs.

**`src/data/landmarks.py`** — `extract_dataset_landmarks()` is a stub.

**`src/data/dataset.py`** — `build_dataloaders(kind="landmark")` raises
`NotImplementedError`.

---

## 9. Next steps, in priority order

### Step 1 — `evaluate.py` (do this first)

87.9% is a number; the confusion matrix is a *finding*. The key question is
whether errors cluster inside `config.KNOWN_CONFUSABLE_GROUPS`:

```python
KNOWN_CONFUSABLE_GROUPS = [
    ["M", "N", "S", "T"],   # fist variants, thumb position only
    ["A", "S", "T"],        # closed-fist family
    ["U", "V", "R"],        # two extended fingers, spread/cross
    ["D", "F"],             # index vs ring contact with thumb
    ["K", "P"],             # same handshape, different orientation
    ["G", "H", "Q"],        # pointing variants
]
```

If, say, 60% of errors fall inside these groups, the model fails the way a human
ASL learner fails — which is a much stronger claim than any accuracy figure.

Run on **validation** while iterating. Touch the test set exactly once, at the
very end.

### Step 2 — the two baselines

The three-model comparison is the spine of the report:
1. `ASLNet` — scratch CNN (done, 0.879)
2. `TransferNet` — pretrained ResNet18, new head. Ask the professor whether a
   pretrained backbone is permitted; drop this if not.
3. `LandmarkMLP` — 63 normalized MediaPipe keypoints → small MLP.

**Expected interesting result:** the landmark MLP may *beat* the CNN on unseen
signers, because landmarks discard background, lighting, and skin tone by
construction. If so, that's a concrete result about feature engineering vs
end-to-end learning at small data scale — the best sentence in the report.

Caveat: MediaPipe detection fails on ~30% of these pre-cropped images, so the
landmark path may only work on a subset. That failure rate is itself a finding
and should be reported alongside.

Also worth 20 minutes: `sklearn.ensemble.RandomForestClassifier` on the same
landmark vectors, as a non-neural floor.

### Step 3 — ablations

At ~10 s/epoch, full 40-epoch runs cost 8 minutes. Run several, one variable at
a time, same seed:
- 3 blocks vs 4 blocks
- rotation on vs off
- dropout 0.3 / 0.5 / 0.7
- image size 128 vs 224
- batch 64 vs 128 **with** a scaled learning rate (the earlier comparison
  confounded batch size with effective LR)

Record every run in `NOTES.md` with its tag, so the table can be reconstructed.

### Step 4 — the domain gap (optional, high value)

Options, roughly increasing in effort:
- Test the resolution-degradation hypothesis in the demo (§6.5)
- Heavier augmentation (blur, noise, stronger color jitter) to force invariance
- Fine-tune the Pugeault model on a small set of Ben's own captures
- Train on landmarks instead of pixels — landmarks should transfer across
  sensors far better than raw images, which would be a strong demonstration of
  the same point

### Step 5 — the report

Structure it around the findings, not the accuracy:
- Scope statement: this is **glossing**, not translation. ASL is a distinct
  language with grammar carried partly by non-manual markers (eyebrows, head
  tilt, mouth morphemes) that a hands-only model cannot observe. J and Z are
  excluded because they require motion.
- Leakage quantified (~15 points)
- Detector-vs-letterbox preprocessing finding (30.1% biased loss)
- Three-model comparison
- Confusion analysis vs known-confusable groups
- Cross-signer (0.879) vs cross-domain (fails) distinction
- Limitations, stated plainly: the author does not sign; validation is one
  signer; the dataset is 2011-era Kinect captures of five people.

---

## 10. Suggested immediate action

Ask Ben whether he wants to write `collect_predictions` and `confusion_matrix`
himself with guidance (recommended — the professor constraint, and it's ~15 lines
of genuinely instructive code), or have them written and explained afterwards.

He has consistently preferred to write the model-relevant code with feedback,
and to have infrastructure handed to him. `evaluate.py` sits closer to the model
side of that line.
