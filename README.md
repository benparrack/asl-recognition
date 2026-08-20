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
│   │   └── landmarks.py    # MediaPipe keypoint extraction and normalisation
│   ├── models/
│   │   ├── cnn.py          # MODEL 1 — your own CNN, the centrepiece
│   │   └── baselines.py    # MODEL 2 transfer learning, MODEL 3 landmark MLP
│   ├── train.py            # training loop
│   ├── evaluate.py         # confusion matrix, per-class metrics, error analysis
│   └── demo.py             # live webcam
├── data/raw/               # untouched downloads (gitignored)
├── data/processed/         # crops, landmarks, metadata.csv (gitignored)
├── checkpoints/            # saved weights (gitignored)
├── results/                # metrics and figures — these go in the report
└── notebooks/              # exploration only; final code lives in src/
```

## The three-model comparison

An accuracy number with nothing to compare it against is not a result. Train all
three on the same split:

| Model | What it is | Why it's here |
|---|---|---|
| **ASLNet** | CNN from scratch | Your work; the thing being assessed |
| **TransferNet** | Pretrained ResNet, new head | Upper bound; shows how much error is data-limited |
| **LandmarkMLP** | Small MLP on 63 keypoints | Often *wins* on unseen signers — the interesting result |

If the tiny MLP beats the CNN, that is a genuine finding about feature
engineering versus end-to-end learning at small data scale, and it is the most
report-worthy sentence you will write. Check with your professor whether the
pretrained backbone is permitted.

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
