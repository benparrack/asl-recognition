# ASL Fingerspelling Recognition

A computer vision system that classifies static ASL fingerspelling handshapes
from webcam input.

## Scope, stated honestly

This system performs **isolated fingerspelling recognition**. It classifies 24
static handshapes from single frames.

It does **not** translate ASL. ASL is a distinct language with its own grammar
and spatial syntax, and it relies on non-manual markers — eyebrow position, head
tilt, mouth morphemes, eye gaze — that carry grammatical meaning and that a
hands-only model cannot observe. Mapping handshapes to letters is *glossing*, not
translation.

Fingerspelling itself is also a small part of ASL, used mainly for proper nouns,
technical terms, and loanwords — not for ordinary conversation.

**J and Z are excluded** because they are produced with motion rather than a
static handshape. A single-frame classifier is structurally incapable of
representing them. This is a property of the language, not a limitation of the
implementation.

State all of this in your report. Overstated scope is the most common criticism
of sign-language recognition projects, and a clear statement of limits reads as
understanding rather than as apology.

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

## Suggested order of work

Do these roughly in sequence; each depends on the last actually working.

1. **Sanity check the architecture.** `python src/models/cnn.py` — shape check,
   then confirm the model can overfit 20 images. If it can't, you have a bug, and
   no amount of extra layers will fix it. Iterate here where it takes seconds.
2. **Build `metadata.csv`** with `path, label, signer_id, session_id`. Everything
   downstream depends on this. If your dataset doesn't record who signed what,
   that's a finding — it means its published accuracy figures are optimistic.
3. **Train the scratch CNN.** Get an end-to-end run finishing before you tune
   anything.
4. **Build the webcam demo.** Do this early. It's the fastest way to find out
   whether your test accuracy survives contact with reality.
5. **Add the two baselines** and run the comparison.
6. **Error analysis.** Confusion matrix, per-class F1, confusable-group breakdown.
   This is where the report gets its substance.
7. **Stretch:** feed frame sequences to handle J and Z; then isolated word signs.

## Datasets

- **ASL Alphabet (Kaggle)** — ~87k images. Convenient, but largely one signer
  against one background with near-duplicate frames. Models hit 99% and then fail
  on a real webcam. Fine for getting the pipeline running; don't trust its numbers.
- **Sign Language MNIST** — 28×28 grayscale, no J/Z. Sanity checks only.
- **WLASL** — real video, many signers. The serious option, and the one that makes
  a signer-disjoint split possible.
- **Your own recordings** — worth doing for a held-out test set, with the caveat
  below.

## Validation caveat

If you don't sign, your own recordings may contain incorrect handshapes, and you
won't be able to tell whether an error is the model's or yours. Mitigations, in
order of value:

- Learn the 24 static handshapes yourself (a weekend's work; lifeprint.com is the
  standard free reference). This also makes you far better at reading a confusion
  matrix, because you'll recognise *why* M/N/S/T collide.
- Source training data from datasets with real signers.
- Get one recorded session with a fluent signer — an ASL club, Deaf student
  organisation, or interpreter training program. Thirty minutes is worth more than
  hours of your own footage.
- State the limitation in the report. Unstated limitations get marked down;
  stated ones don't.
