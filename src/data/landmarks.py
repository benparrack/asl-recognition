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
