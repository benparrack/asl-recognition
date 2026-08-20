"""
Hand landmark extraction via MediaPipe.

WHY THIS IS NOT CHEATING
========================
MediaPipe is a feature extractor, not a classifier. It converts an image into 21
keypoints; it has no idea what ASL is and outputs nothing about letters. The
model that maps those keypoints to letters is entirely yours.

In the vocabulary of the Chapter 2 workflow, this is feature engineering -- the
same category as combining two columns into a ratio. Confirm with your professor,
but "used an off-the-shelf keypoint detector as preprocessing, then trained my own
classifier" is a normal and defensible pipeline.

Its real value in your project is as a CONTRAST. Landmarks discard background,
lighting, and skin tone by construction, so a tiny MLP over 63 numbers often
generalises to unseen signers better than a CNN trained on raw pixels with limited
data. If that happens, you have a concrete result about feature engineering versus
end-to-end learning -- and that is a much more interesting report than one accuracy
number.
"""

from __future__ import annotations

import numpy as np

import config

# pip install mediapipe opencv-python
try:
    import mediapipe as mp
except ImportError:  # keep the module importable without the dependency
    mp = None

try:
    import cv2
except ImportError:
    cv2 = None


def build_detector(static_mode: bool = True, min_confidence: float = 0.5):
    """
    Create a MediaPipe Hands detector.

    static_mode=True  -> treats each frame independently (correct for a folder
                         of unrelated images)
    static_mode=False -> tracks between frames, faster and steadier for webcam
    """
    if mp is None:
        raise ImportError("mediapipe is not installed: pip install mediapipe")

    return mp.solutions.hands.Hands(
        static_image_mode=static_mode,
        max_num_hands=1,
        min_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )


def extract_landmarks(detector, image_rgb: np.ndarray) -> np.ndarray | None:
    """
    Return a (21, 3) array of keypoints, or None if no hand was detected.

    Track your detection failure rate. It is a genuine result: if MediaPipe fails
    on 8% of frames, your end-to-end system has an 8% floor on errors regardless
    of how good the classifier is. Report it.
    """
    result = detector.process(image_rgb)
    if not result.multi_hand_landmarks:
        return None

    hand = result.multi_hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)


def normalise_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Make the 21x3 keypoints invariant to hand position and apparent size.

    Raw MediaPipe coordinates are normalised to the image frame, so they encode
    where in the picture the hand was and how close it was to the camera. Neither
    is relevant to which letter is being signed -- feed them in raw and the model
    will happily learn that "letters signed on the left side of frame are more
    often B", which is nonsense that evaporates on new data.

    Two steps:
      1. Translate so the wrist (landmark 0) sits at the origin.
      2. Scale so the largest distance from the wrist is 1.

    Returns a flat 63-vector.

    OPTIONAL EXTENSION, worth a paragraph in the report: also normalise for
    ROTATION, by rotating so the wrist-to-middle-knuckle vector points a fixed
    direction. This helps for most letters but destroys the K/P distinction,
    which is orientation-only. Trying it and measuring the trade-off is a nice
    piece of analysis.
    """
    lm = landmarks.astype(np.float32).copy()

    lm -= lm[0]                                   # 1. wrist to origin

    scale = np.linalg.norm(lm, axis=1).max()      # 2. unit scale
    if scale > 1e-6:
        lm /= scale

    return lm.flatten()


def crop_hand(frame_bgr: np.ndarray, detector, padding: float = 0.25) -> np.ndarray | None:
    """
    Locate the hand in a BGR image and return a square crop around it.

    Returns None when no hand is detected or the crop would be degenerate.

    THIS FUNCTION HAS EXACTLY ONE DEFINITION FOR A REASON. Both the offline
    preprocessing script and the live webcam demo call it. If the demo cropped
    differently from preprocessing, the model would see a different distribution
    of images at inference than it trained on -- train/serve skew -- and accuracy
    would fall for reasons that look like a model bug but are not. One function,
    two callers, guaranteed consistent.

    ---------------------------------------------------------------- coordinates
    Three systems are in play, and mixing them is the main source of bugs here:

      1. MediaPipe returns NORMALISED coordinates: lm.x and lm.y are fractions of
         image width and height, roughly 0..1, independent of resolution.
      2. Slicing an image needs INTEGER PIXELS.
      3. NumPy indexes ROWS FIRST, so frame.shape is (height, width, channels)
         and the slice is frame[y1:y2, x1:x2] -- y before x.

    Point 3 is backwards from the (x, y) convention you are used to. Swapping
    them raises no error; it just silently crops the wrong region.
    """
    if cv2 is None:
        raise ImportError("opencv-python is not installed: pip install opencv-python")

    # 1. Detect. OpenCV loads BGR; MediaPipe expects RGB. Getting this wrong does
    #    not crash -- it just quietly degrades detection, which is worse.
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = detector.process(image_rgb)

    if not result.multi_hand_landmarks:
        return None

    # 2. Normalised coordinates -> pixels.
    h, w = frame_bgr.shape[:2]
    hand = result.multi_hand_landmarks[0]
    xs = [lm.x * w for lm in hand.landmark]
    ys = [lm.y * h for lm in hand.landmark]

    # 3. Bounding box, expressed as centre + half-size so squaring is trivial.
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    # max() of width and height makes the box square; padding stops fingertips
    # being clipped, since MediaPipe's box hugs the landmarks tightly and the
    # fingertips ARE the landmarks at the boundary.
    half = max(x_max - x_min, y_max - y_min) / 2.0
    half *= (1.0 + padding)

    # 4. Integers, then clamp to the frame.
    #    Clamping is not optional: a negative index in NumPy means "count from the
    #    end", so an unclamped negative silently wraps to the opposite side of the
    #    image and you get a crop of the wrong thing with no error.
    x1, x2 = int(round(cx - half)), int(round(cx + half))
    y1, y2 = int(round(cy - half)), int(round(cy + half))

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Reject slivers: a hand mostly out of frame yields a few-pixel crop that
    # causes confusing errors much further downstream.
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None

    # 5. Slice. y before x.
    return frame_bgr[y1:y2, x1:x2]


def crop_hand_square(frame_bgr: np.ndarray, detector, padding: float = 0.25):
    """
    As crop_hand(), but pads with a border instead of clamping, so the result is
    always square even when the hand sits against a frame edge.

    Why you might want this: clamping can return a non-square crop near the
    edges. Resize() will then stretch it, distorting the handshape -- and since
    hands near the edge are exactly the awkward cases, you are distorting your
    hardest examples.

    Whether it measurably helps is an empirical question. Trying both and
    reporting the difference is a cheap, legitimate ablation for your write-up.
    """
    if cv2 is None:
        raise ImportError("opencv-python is not installed: pip install opencv-python")

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = detector.process(image_rgb)

    if not result.multi_hand_landmarks:
        return None

    h, w = frame_bgr.shape[:2]
    hand = result.multi_hand_landmarks[0]
    xs = [lm.x * w for lm in hand.landmark]
    ys = [lm.y * h for lm in hand.landmark]

    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2.0 * (1.0 + padding)

    x1, x2 = int(round(cx - half)), int(round(cx + half))
    y1, y2 = int(round(cy - half)), int(round(cy + half))

    # How far the desired box overhangs each edge.
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    crop = frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

    if crop.size == 0:
        return None

    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )

    return crop


def extract_dataset_landmarks(df, image_root, out_path=config.LANDMARKS_NPZ):
    """
    Run extraction over every row of the metadata dataframe and cache the result.

    TODO (yours to write):
      1. Build a detector with static_mode=True.
      2. For each row: read the image with cv2, convert BGR -> RGB (OpenCV loads
         BGR and MediaPipe expects RGB -- forgetting this is a classic silent bug
         that just degrades accuracy without any error).
      3. extract_landmarks(), then normalise_landmarks().
      4. Record failures rather than silently dropping them: keep a boolean mask
         so the landmark array stays row-aligned with the dataframe.
      5. np.savez_compressed(out_path, landmarks=..., detected=...).

    Cache this. Extraction over tens of thousands of images takes a while and you
    do not want to repeat it on every run.
    """
    raise NotImplementedError("See the TODO above.")
