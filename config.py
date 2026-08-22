"""
Central configuration. Import from here rather than hardcoding values in scripts,
so that a single change propagates everywhere and your experiments stay reproducible.
"""

from pathlib import Path
import torch

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # untouched downloaded datasets
RAW_METADATA_CSV = RAW_DIR / "raw_metadata.csv"
PROCESSED_DIR = DATA_DIR / "processed"  # cropped images, extracted landmarks
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in (RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# The metadata CSV is the backbone of the whole project. See src/data/build_metadata.py.
METADATA_CSV = PROCESSED_DIR / "metadata.csv"
LANDMARKS_NPZ = PROCESSED_DIR / "landmarks.npz"

# ---------------------------------------------------------------- classes
# J and Z are produced with motion, not a static handshape. A single-frame
# classifier is structurally incapable of representing them -- this is a
# property of the language, not a shortcoming of your model. Say so in the
# report; it reads as understanding rather than as an excuse.
STATIC_LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")   # 24 letters, no J, no Z
#STATIC_LETTERS = ["FIST", "PALM", "PEACE"]   # TEMPORARY: pipeline smoke test
MOTION_LETTERS = ["J", "Z"]

INCLUDE_MOTION_LETTERS = False   # flip to True only once you feed sequences
CLASSES = STATIC_LETTERS + (MOTION_LETTERS if INCLUDE_MOTION_LETTERS else [])
NUM_CLASSES = len(CLASSES)

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# Handshapes that are genuinely similar. Expect these to dominate your confusion
# matrix; a model that confuses these is making *human* mistakes, which is a far
# more interesting result than a uniform error rate.
KNOWN_CONFUSABLE_GROUPS = [
    ["M", "N", "S", "T"],   # fist variants, differing in thumb placement
    ["A", "S", "T"],        # closed-fist family
    ["U", "V", "R"],        # two extended fingers, differing in spread/cross
    ["D", "F"],             # index vs. ring contact with thumb
    ["K", "P"],             # same handshape, different orientation
    ["G", "H", "Q"],        # pointing variants
]

# ---------------------------------------------------------------- images
IMAGE_SIZE = 128        # square. 128 trains fast; try 224 for transfer learning.
IMAGE_CHANNELS = 3

# ImageNet statistics -- required for pretrained backbones, harmless otherwise.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------- landmarks
NUM_HAND_LANDMARKS = 21          # MediaPipe Hands returns 21 keypoints
LANDMARK_DIMS = 3                # x, y, z per keypoint
LANDMARK_FEATURE_SIZE = NUM_HAND_LANDMARKS * LANDMARK_DIMS   # 63

# ---------------------------------------------------------------- training
BATCH_SIZE = 64          # RTX 4060 (8 GB) handles this comfortably at 128px.
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8          # DataLoader subprocesses; ~= your CPU core count

SEED = 42                # fix every source of randomness -- see src/utils.py

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- splitting
# NEVER split randomly across frames. See src/data/dataset.py for why.
VAL_SIGNER_FRACTION = 0.15
TEST_SIGNER_FRACTION = 0.15
