# ASL Fingerspelling Recognition

A computer vision system that classifies static ASL fingerspelling handshapes
from webcam input.

## Project structure

```
asl-recognition/
├── config.py               # paths, class list, hyperparameters — single source of truth
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── dataset.py      # loading + SIGNER-DISJOINT splitting (read the docstring)
│   │   ├── landmarks.py    # MediaPipe keypoint extraction and normalisation
│   │   ├── capture_data.py     # webcam capture, one class at a time
│   │   └── capture_alphabet.py # webcam capture, loops through all 24 letters
│   ├── models/
│   │   ├── cnn.py                   # MODEL 1 — your own CNN, the centrepiece
│   │   ├── baselines.py             # MODEL 2 transfer learning, MODEL 3 landmark MLP
│   │   └── random_forest_baseline.py # MODEL 4 — non-neural floor on the same landmarks
│   ├── train.py             # training loop
│   ├── evaluate.py          # confusion matrix, per-class metrics, error analysis
│   ├── analyze_confusions.py # which SPECIFIC pairs drive each model's errors
│   └── demo.py               # live webcam, both pixel and landmark models
├── data/raw/               # untouched downloads (gitignored)
├── data/processed/         # crops, landmarks, metadata.csv (gitignored)
├── checkpoints/            # saved weights (gitignored)
├── results/                # metrics and figures — these go in the report
└── notebooks/              # exploration only; final code lives in src/
```

## The three(+one)-model comparison

An accuracy number with nothing to compare it against is not a result. All four
trained on the same signer-disjoint split of the Pugeault & Bowden fingerspelling
dataset (5 signers, 24 static letters, no J/Z):

| Model | What it is | Val acc |
|---|---|---|
| **LandmarkMLP** | Small MLP on 63 MediaPipe keypoints | **0.960** |
| **RandomForest** | Classical ML, same landmark features | 0.907 |
| **ASLNet** | CNN from scratch (batch=128, lr=2e-3) | 0.897 |
| **TransferNet** | Frozen pretrained ResNet18, new head | 0.398 |

The tiny MLP (18k params) beats the from-scratch CNN (~4.8M params) by 8
points — the headline result. Feature engineering (MediaPipe landmarks, which
discard background/lighting/skin tone by construction) beat end-to-end pixel
learning at this data scale. TransferNet's weak result is itself a finding, not
a bug: a frozen ImageNet backbone doesn't capture the fine-grained finger-position
distinctions ASL handshapes need, especially fed 128px images when it expects 224px.

**Confusion patterns are more interesting than the raw accuracy.** Only
~20-26% of errors fall inside the visually-similar handshape groups in
`config.KNOWN_CONFUSABLE_GROUPS`. The bigger pattern (see `src/analyze_confusions.py`)
is that specific letters — O, X, P, R — act as one-directional "default guess"
sinks across every model tried, including the non-neural RandomForest. `C -> O`
alone is 20% of the CNN's total error, almost never the reverse.

**Live cross-domain test**: the landmark MLP was tested live against a real
webcam in a completely different environment from the training data (different
camera, lighting, background) and recognized all 24 static letters near-flawlessly
— the one weak spot (K) matches K being the worst class across every model
offline too. This is the opposite of what happened with the pixel-based CNN,
which scored well signer-disjoint but was confidently wrong on the same live test.
Best explanation: MediaPipe itself (general-purpose, not dataset-specific) is
doing the real domain adaptation — the landmark MLP inherits that robustness
for free, pixel models don't.

## The evaluation rule that matters most

**Split by signer, never randomly across frames.**

Video runs at 30 fps, so consecutive frames are near-duplicates. A random split
puts one in train and its twin in test, and you get 99% accuracy that means
nothing. This is the same data leakage the housing chapter warns about.

`split_by_signer()` enforces this. `split_randomly_DO_NOT_USE()` exists so you
can run it once, report the inflated number beside the honest one, and quantify
exactly how much leakage was worth. That comparison is itself a result.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available())"   # expect True
```
## Datasets

- **ASL Alphabet (Kaggle)** — ~87k images. Convenient, but largely one signer
  against one background with near-duplicate frames. Models hit 99% and then fail
  on a real webcam. Fine for getting the pipeline running; don't trust its numbers.
- **Sign Language MNIST** — 28×28 grayscale, no J/Z. Sanity checks only.
- **WLASL** — real video, many signers. The serious option, and the one that makes
  a signer-disjoint split possible.
- **My own recordings** - To test at first I made 720 images of my hand spread across
- 3 different shapes (FIST, PALM, and PEACE) so that I could train a model for the
- first time and get my bearings for how it worked, and also have a working demo.
