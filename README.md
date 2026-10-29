# ASL Recognition

Recognises American Sign Language from a webcam: fingerspelled letters
(including the moving J and Z) and 2,331 isolated word signs. **SignDesk** is a
browser dashboard that puts it all together.

| Part | Model | Result (validation, unseen signers) |
|---|---|---|
| **Static letters** (24, no J/Z) | `LandmarkMLP` on 63 MediaPipe hand keypoints | 0.960; near-flawless live |
| **J and Z** (+ a NONE class) | `MotionGRUv2` on hand trajectories, behind a motion gate | 1.000 (small val set: 1 session) |
| **Words** (2,331 signs) | conv+transformer on Holistic landmarks; GISLR + ASL Citizen, two models blended | Ben's 100 webcam clips: 0.76 top-1 / 0.86 top-3 (0.92 with one taught example per sign) |

No test split has been used yet. Each one is evaluated once, when its model is
final (see [docs/TODO.md](docs/TODO.md)).

## Running it

The venv lives one level up, at `../.venv` (see [Setup](#setup)).

```bash
source ../.venv/bin/activate

python src/dashboard.py     # SignDesk at http://127.0.0.1:8790 (--video clip.mp4 to loop a file, --port N)
python -m pytest           # 95 tests, ~1 min; no webcam, dataset or checkpoint needed
```

**SignDesk** covers letters, J/Z and words in one page. Switch modes with L / W.
To pick a letter, hold it until the ring fills. After signing a word, press 1–3
to take another match. You can edit the text, copy it, read it aloud or show it
as captions (V). The Signs tab links each sign to how it's done, and the
Practice tab tracks which signs you miss. Profiles hold each signer's settings
and text. The Teach tab adapts the models to a profile from one example per
sign. Press H for every shortcut. Profiles are saved under `data/dashboard/`
(gitignored).

Framing matters for two-handed signs. Sit back so you're in frame from head to
waist, put the camera at about face height and sign at chest height. With your
hands close to the lens, Holistic often finds only one of two hands that touch.

The original OpenCV demo is still there. It's useful headless and for tuning:

```bash
python src/demo.py --checkpoint checkpoints/mlp_pugeault_best.pt \
    --motion-checkpoint checkpoints/motion_v2_best.pt \
    --words-checkpoint checkpoints/words_transformer_full_aslc_r3_best.pt
# 'w' switches letters/words, 't' starts/stops a sign by hand
# --video clip.mp4 [--no-display] runs on a file, printing events
```

## Results

### Letters: features beat pixels

Every model below used the same signer-disjoint split of the Pugeault & Bowden
fingerspelling dataset (5 signers, 24 static letters):

| Model | What it is | Val acc |
|---|---|---|
| **LandmarkMLP** | Small MLP on 63 MediaPipe keypoints (18k params) | **0.960** |
| RandomForest | Classical ML on the same landmarks | 0.907 |
| ASLNet | CNN from scratch (~4.8M params, batch 128, lr 2e-3) | 0.897 |
| TransferNet | Frozen ImageNet ResNet18, new head | 0.398 |

The tiny MLP beats the CNN by 6 points. MediaPipe landmarks drop the
background, lighting and skin tone by construction, and at this data scale that
beats learning from pixels end to end. Live, on a different camera in a
different room, the MLP read all 24 letters near-flawlessly. The CNN was
confidently wrong on the same test. MediaPipe does the domain adaptation, and
the MLP gets it for free. The one weak letter live (K) is also the worst class
offline in every model.

Only about 20–26% of errors fall inside the visually similar handshape groups
(`config.KNOWN_CONFUSABLE_GROUPS`). The bigger pattern
(`src/analyze_confusions.py`) is that O, X, P and R act as one-way "default
guess" sinks in every model, even the random forest. `C -> O` alone is 20% of
the CNN's errors, and the reverse almost never happens.

### J/Z: make sure it's motion

The first J/Z model scored 0.967, but a logistic regression on the **first
frame alone** scored 1.000. J starts from the I handshape and Z from D, so the
model was recognising handshape, not motion. v2 keeps the trajectory at
capture, adds a NONE class with hard negatives (hold I still, hold D still,
move between letters) and checks itself with ablations
(`src/diagnose_motion.py`). The model now uses motion and handshape
redundantly: the trajectory alone gets 0.93.

### Words: the second hand comes from ASL Citizen

GISLR (Kaggle `asl-signs`, 250 signs) almost never contains the second hand.
Only 0.5% of frames have both, even for BOOK or SHOE. Trained on GISLR alone,
two-handed signs from unseen signers scored 0.25. Adding ASL Citizen raised
that to 0.91, and Ben's own clips went from 0.37 to 0.76. The feature ablation
showed where the hands are matters most (+5 points), more than the second
hand's shape.

The full ASL Citizen vocabulary takes the model to 2,331 signs, but having 9×
more choices costs common signs accuracy. SignDesk therefore blends the big
model with the 250-sign one (`WordBlend` in `src/words_live.py`). That keeps
0.76 on Ben's clips and adds 2,081 signs at 0.735 val. Automatic segmentation
costs about 1 point on top of that.

[docs/NOTES.md](docs/NOTES.md) has every run, table and dead end.

## The rule that matters most: split by signer

Never split randomly across frames. At 30 fps, consecutive frames are near
duplicates. A random split puts one in train and its twin in test, and you get
99% accuracy that means nothing. `split_by_signer()` in `src/data/dataset.py`
enforces this. Every dataset here is split by signer, participant or recording
session, and each test split is used once.

## Setup

```bash
python3 -m venv ../.venv           # this machine keeps the venv outside the repo
source ../.venv/bin/activate
pip install -r requirements.txt    # requirements-lock.txt pins the exact working set
python -c "import torch; print(torch.cuda.is_available())"   # expect True
```

`mediapipe` is pinned to 0.10.14, because 0.10.31+ removed the legacy
`solutions` API that Holistic uses. `config.py` at the repo root holds the
paths, classes and hyperparameters. Scripts put the repo root and `src/` on
`sys.path` themselves, with `src/` first, so that `import data` finds
`src/data/` and not the dataset folder `data/`.

## Rebuilding the models

Datasets and checkpoints are gitignored. These are the commands that made them.
Run anything that takes hours as a `systemd-run --user` unit, so it outlives
the terminal. The local `logs/` folder (gitignored) has the resumable chain
scripts that were used.

**Letters.** Get the Pugeault & Bowden dataset from Kaggle
(`mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out`) and extract it to
`data/raw/dataset5/`.

```bash
python src/data/build_metadata.py --source data/raw/dataset5 --layout nested \
    --pattern 'color_*' --per-class 200 --force
python src/data/preprocess.py --no-detect --overwrite
python src/train.py --model mlp --tag pugeault          # also: cnn, transfer
python src/models/random_forest_baseline.py
python src/evaluate.py --checkpoint checkpoints/mlp_pugeault_best.pt --split val
python src/analyze_confusions.py
```

**J/Z.** Record 5 sessions, each with its own signer id (the split holds out
whole sessions). Every session takes about 10 minutes: 20 clips each of J, Z
and NONE. The full recording guide is §1 of
[docs/archive/HUMAN_TODO.md](docs/archive/HUMAN_TODO.md).

```bash
python src/data/capture_motion_v2.py --signer ben_m1 --session s1_desk_daylight   # ... up to m5
python src/train_motion_v2.py --epochs 80
python src/diagnose_motion.py --checkpoint checkpoints/motion_v2_best.pt
```

**Words.** First accept the competition rules at
<https://www.kaggle.com/competitions/asl-signs/rules>. Until you do, every
download returns 403. ASL Citizen (Microsoft, non-commercial research licence)
goes in `data/external/asl_citizen/ASL_Citizen.zip`. Holistic extraction takes
about 1.7 GB of RAM per worker.

```bash
kaggle competitions download -c asl-signs -p data/external/gislr
python src/data/preprocess_gislr.py --verify-only        # check the schema first
python src/data/preprocess_gislr.py                      # reads straight from the zip
python src/data/preprocess_asl_citizen.py --vocab all --workers 14 \
    --cache-dir data/processed/asl_citizen_all           # --vocab gislr for the 216-sign cache

python src/train_words.py --extra-cache-dir data/processed/asl_citizen --extra-repeat 3 --tag aslc_r3
python src/train_words.py --extra-cache-dir data/processed/asl_citizen_all --extra-repeat 3 --tag aslc_all_r3
python src/ablation_words.py --epochs 30                 # which features matter
python src/eval_word_videos.py --checkpoint checkpoints/words_transformer_full_aslc_all_r3_best.pt \
    --blend checkpoints/words_transformer_full_aslc_r3_best.pt     # your clips in data/word_videos/<sign>/
```

## Layout

```
config.py                    paths, classes, hyperparameters (single source of truth)
src/
├── dashboard.py             SignDesk server (stdlib HTTP + MJPEG + SSE)
├── dashboard_engine.py      camera loop and recognisers behind it
├── dashboard_web/           the page (no CDN, strict CSP)
├── personal.py              profiles + adapting models to a signer's examples
├── demo.py                  OpenCV live/--video demo
├── motion_gate.py           J/Z gate (shared by demo, dashboard, replay tests)
├── words_live.py            word segmentation, classification, WordBlend
├── train.py, evaluate.py, analyze_confusions.py           letters
├── train_motion_v2.py, diagnose_motion.py                 J/Z
├── train_words.py, evaluate_words.py, ablation_words.py,
│   eval_word_videos.py, finetune_words.py, bench_words_fps.py   words
├── train_motion.py          J/Z v1 (superseded; its checkpoint still loads)
├── models/                  cnn, baselines (transfer, MLP), random forest, motion, words
└── data/
    ├── dataset.py           loading + signer-disjoint splitting
    ├── landmarks.py         MediaPipe hand keypoints, cropping
    ├── build_metadata.py, preprocess.py, contact_sheet.py
    ├── capture_*.py         webcam capture (letters, alphabet, J/Z v1/v2, words)
    ├── motion_features.py, motion_dataset*.py, synthetic_motion.py
    ├── holistic_landmarks.py   which 84 of Holistic's 543 points words use
    ├── holistic_features.py    word features: one implementation for train and live
    ├── preprocess_gislr.py, preprocess_asl_citizen.py, word_dataset.py
    └── synthetic_gislr.py   GISLR-format fake data for tests
tests/                       pytest suite (synthetic data, fake Holistic)
results/                     metrics, histories, eval JSONs
docs/
├── NOTES.md                 dated findings: every run, number and dead end
├── DECISIONS.md             judgement calls and why
├── TODO.md                  what's still open
└── archive/                 one-off briefs and handoffs (2026-09), kept for history
data/, checkpoints/, logs/   gitignored
```

## Datasets

- **Pugeault & Bowden fingerspelling** (letters): 5 signers, 24 static letters,
  sampled to 24,000 RGB hand crops.
- **GISLR** (Kaggle `asl-signs`): 94k isolated signs, 250 words, 21
  participants, already as Holistic landmarks. Split by participant.
- **ASL Citizen** (Microsoft): 83k webcam videos, 2,731 glosses, 52 signers,
  official signer-disjoint split, non-commercial research licence. Merged into
  GISLR's label space: 216 of its 250 signs, plus 2,081 new ones.
- **Ben's recordings**: 100 word clips (20 signs × 5) for live-path evaluation
  (never trained on), 300 J/Z/NONE clips over 5 sessions, and the original
  720-image FIST/PALM/PEACE set used to get the first pipeline running.
