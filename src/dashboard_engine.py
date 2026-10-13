"""
SignDesk engine: the camera loop, both recognisers and the transcript, behind
one lock. dashboard.py serves it to the browser; nothing here knows about HTTP.

Recognition is demo.py's letters_step / words_step, not a copy, so the
dashboard runs exactly the pipeline the demo and the replay tests run. What's
added on top is everything a usable tool needs around the raw predictions:

  letters  a smoothed letter held steady for `letters_hold` seconds is added
           once; it can't repeat until the letter changes or the hand drops
           (double letters: drop the hand briefly, or press Enter). J/Z from
           the motion gate are added the moment the gate accepts them. After a
           fingerspelled run, lowering the hand for `letters_space_after`
           seconds adds the space.
  words    an accepted sign is added as a word; its top-3 stay on screen and
           picking another one swaps the word in place.
  text     a Transcript with undo/redo, edited by commands from the UI.

Practice and editing modes still recognise and report events, but add nothing
to the text.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import config
from data import holistic_landmarks as hl
from data.dataset import build_transforms
from data.landmarks import build_detector
from demo import PredictionSmoother, letters_step, load_model, words_step
from motion_gate import GateParams, MotionGate, hand_speed_v2, load_motion_classifier
from words_live import WordClassifier, WordRecogniser, SegParams

STATE_DIR = config.DATA_DIR / "dashboard"
# Smoothed hand speed (hand-lengths/s) above which a letter isn't being held:
# Ben's still holds stay under ~0.25, a hand moving into frame is well above.
MOVING_SPEED = 0.8

# GISLR sign names that aren't plain words: (label shown, text added).
WORD_FORMS = {
    "callonphone": ("call (phone)", "call"),
    "frenchfries": ("french fries", "french fries"),
    "glasswindow": ("glass / window", "window"),
    "haveto": ("have to", "have to"),
    "hesheit": ("he / she / it", "he"),
    "icecream": ("ice cream", "ice cream"),
    "minemy": ("mine / my", "my"),
    "thankyou": ("thank you", "thank you"),
    "weus": ("we / us", "we"),
}


def word_label(name: str) -> str:
    return WORD_FORMS.get(name, (name, name))[0]


def word_text(name: str) -> str:
    return WORD_FORMS.get(name, (name, name))[1]


# ---------------------------------------------------------------- settings
@dataclass
class Settings:
    letters_min_confidence: float = 0.6
    letters_window: int = 15
    letters_auto_add: bool = True
    letters_hold: float = 0.8
    letters_auto_space: bool = True
    letters_space_after: float = 1.5
    motion_letters: bool = True
    words_auto_segment: bool = True
    words_min_confidence: float = SegParams.min_confidence
    words_auto_add: bool = True
    words_lull_speed: float = SegParams.lull_speed
    mirror: bool = True
    show_landmarks: bool = True
    camera: int = 0


BOUNDS = {
    "letters_min_confidence": (0.2, 0.95), "letters_window": (3, 45),
    "letters_hold": (0.2, 3.0), "letters_space_after": (0.5, 5.0),
    "words_min_confidence": (0.0, 0.95), "words_lull_speed": (0.0, 2.0), "camera": (0, 63),
}


def clean_settings(raw: dict, base: Settings) -> Settings:
    """Settings from untrusted JSON: unknown keys dropped, types coerced,
    numbers clamped. Bad values keep the current setting."""
    out = asdict(base)
    for f in fields(Settings):
        if f.name not in raw:
            continue
        v = raw[f.name]
        try:
            if f.type == "bool":
                if not isinstance(v, bool):
                    continue
            else:
                v = (int if f.type == "int" else float)(v)
                if not np.isfinite(v):
                    continue
                lo, hi = BOUNDS[f.name]
                v = type(v)(min(max(v, lo), hi))
        except (TypeError, ValueError):
            continue
        out[f.name] = v
    return Settings(**out)


# ---------------------------------------------------------------- transcript
class Transcript:
    """The text being built, with undo/redo. `last` remembers the most recent
    recognised word while it's still at the end, so it can be swapped for one
    of the other candidates."""

    def __init__(self, text: str = ""):
        self.text = text
        self.last: tuple[int, str, int] | None = None     # (start, text, event id)
        self._undo: list = []
        self._redo: list = []

    def _push(self):
        self._undo.append((self.text, self.last))
        del self._undo[:-500]
        self._redo.clear()

    def _sep(self) -> str:
        return "" if not self.text or self.text[-1] in " \n" else " "

    def add_letter(self, ch: str):
        self._push()
        self.text += ch
        self.last = None

    def add_word(self, word: str, event_id: int = -1):
        self._push()
        self.text += self._sep()
        self.last = (len(self.text), word, event_id)
        self.text += word + " "

    def replace_last_word(self, word: str, event_id: int) -> bool:
        """Swap the last recognised word, if it came from event_id and nothing
        was typed after it."""
        if not self.last or self.last[2] != event_id:
            return False
        start, old, _ = self.last
        if self.text[start:] != old + " ":
            return False
        self._push()
        self.text = self.text[:start] + word + " "
        self.last = (start, word, event_id)
        return True

    def space(self):
        if self.text and not self.text.endswith(" "):
            self._push()
            self.text += " "
            self.last = None

    def newline(self):
        self._push()
        self.text = self.text.rstrip(" ") + "\n"
        self.last = None

    def punct(self, p: str):
        if not self.text.strip():
            return
        self._push()
        self.text = self.text.rstrip(" ") + p + " "
        self.last = None

    def backspace(self):
        if self.text:
            self._push()
            self.text = self.text[:-1]
            self.last = None

    def delete_word(self):
        """Remove the last word and the whitespace after it."""
        if not self.text:
            return
        self._push()
        t = self.text.rstrip()
        i = len(t)
        while i and not t[i - 1].isspace():
            i -= 1
        self.text = t[:i]
        self.last = None

    def clear(self):
        if self.text:
            self._push()
            self.text, self.last = "", None

    def set_text(self, text: str):
        if text != self.text:
            self._push()
            self.text, self.last = text, None

    def undo(self):
        if self._undo:
            self._redo.append((self.text, self.last))
            self.text, self.last = self._undo.pop()

    def redo(self):
        if self._redo:
            self._undo.append((self.text, self.last))
            self.text, self.last = self._redo.pop()


# ---------------------------------------------------------------- letters
class LetterCommitter:
    """Fires once when the same letter has been shown for `hold` seconds.
    progress (0..1) drives the hold ring in the UI."""

    def __init__(self, hold: float):
        self.hold = hold
        self.reset()

    def reset(self):
        self.current, self.since, self.fired = None, 0.0, False
        self.progress = 0.0
        self.suppress_next = False

    def suppress(self):
        """Don't fire the next letter that appears (the handshape a J or Z
        ends in, still held after the motion letter was added)."""
        self.suppress_next = True

    def update(self, letter: str | None, t: float) -> str | None:
        if letter != self.current:
            self.current, self.since = letter, t
            self.fired = letter is not None and self.suppress_next
            if letter is not None:
                self.suppress_next = False
        if letter is None:
            self.suppress_next = False
            self.progress = 0.0
            return None
        if self.fired:
            self.progress = 1.0
            return None
        self.progress = min(1.0, (t - self.since) / self.hold)
        if self.progress >= 1.0:
            self.fired = True
            return letter
        return None


# ---------------------------------------------------------------- drawing
_HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
               (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
_ARM_EDGES = [(hl.POSE_SHOULDER_L, hl.POSE_SHOULDER_R), (hl.POSE_SHOULDER_L, hl.POSE_ELBOW_L),
              (hl.POSE_ELBOW_L, hl.POSE_WRIST_L), (hl.POSE_SHOULDER_R, hl.POSE_ELBOW_R),
              (hl.POSE_ELBOW_R, hl.POSE_WRIST_R)]
ACCENT = (246, 130, 59)          # BGR of the UI accent (#3b82f6)
ACCENT_2 = (153, 211, 52)        # second hand (#34d399)


def _draw_points(img, pts, edges, colour, joint=True, width=2):
    h, w = img.shape[:2]
    ok = ~np.isnan(pts[:, :2]).any(axis=1)
    p = np.round(np.nan_to_num(pts[:, :2]) * (w, h)).astype(int)
    for a, b in edges:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(p[a]), tuple(p[b]), colour, width, cv2.LINE_AA)
    if joint:
        for q in p[ok]:
            cv2.circle(img, tuple(q), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(q), 3, colour, 1, cv2.LINE_AA)


def draw_words_landmarks(img, lm):
    """lm: the (84, 3) selected Holistic frame, un-mirrored; img is mirrored."""
    pts = lm.copy()
    pts[:, 0] = 1.0 - pts[:, 0]
    pose = {idx: pts[hl.pose_row(idx)] for a in _ARM_EDGES for idx in a}
    for a, b in _ARM_EDGES:
        if not (np.isnan(pose[a][:2]).any() or np.isnan(pose[b][:2]).any()):
            _draw_points(img, np.stack([pose[a], pose[b]]), [(0, 1)], (170, 170, 170),
                         joint=False, width=2)
    _draw_points(img, pts[hl.LEFT_HAND], _HAND_EDGES, ACCENT_2)
    _draw_points(img, pts[hl.RIGHT_HAND], _HAND_EDGES, ACCENT)


def camera_indices() -> list[int]:
    """Video devices on this machine (Linux /dev/videoN); [0] elsewhere."""
    found = sorted(int(p.name[5:]) for p in Path("/dev").glob("video*") if p.name[5:].isdigit())
    return found or [0]


# ---------------------------------------------------------------- engine
class Engine:
    def __init__(self, letters_checkpoint, motion_checkpoint=None, words_checkpoint=None,
                 holistic_complexity: int = 1, video: str | None = None,
                 state_dir: Path = STATE_DIR):
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.device = config.DEVICE
        self.video = video
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.model, self.kind = load_model(str(letters_checkpoint), self.device)
        self.detector = build_detector(static_mode=False)
        self.transform = build_transforms(train=False)
        self._args = SimpleNamespace(padding=0.25)        # demo.py's default crop padding
        self.motion_clf = (load_motion_classifier(motion_checkpoint, self.device)
                           if motion_checkpoint and self.kind == "landmark" else None)

        self.words = self.holistic = None
        self.word_classes: list[str] = []
        self.models = {"letters": f"{Path(letters_checkpoint).stem} ({self.kind})",
                       "motion": Path(motion_checkpoint).stem if self.motion_clf else None,
                       "words": None, "device": str(self.device)}
        if words_checkpoint:
            import mediapipe as mp
            from evaluate_words import load_word_checkpoint
            wmodel, wckpt = load_word_checkpoint(str(words_checkpoint), self.device)
            self.words = WordRecogniser(WordClassifier(wmodel, wckpt, self.device), SegParams())
            self.holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False, model_complexity=holistic_complexity)
            self.word_classes = list(wckpt["classes"])
            self.models["words"] = f"{Path(words_checkpoint).stem} ({len(self.word_classes)} signs)"

        self.settings = self._load_settings()
        self.smoother = PredictionSmoother(self.settings.letters_window,
                                           self.settings.letters_min_confidence)
        self.committer = LetterCommitter(self.settings.letters_hold)
        self.gate = None
        self._apply_settings()

        tf = self.state_dir / "transcript.txt"
        self.transcript = Transcript(tf.read_text() if tf.exists() else "")
        self._saved_text = self.transcript.text

        self.mode = "letters"
        self.paused = self.practice = self.editing = False
        self.letter, self.confidence = None, 0.0
        self.hand = False
        self.motion_recording = False
        self.word_status = None
        self.candidates: list[dict] = []
        self.candidates_event = -1
        self.chosen = 0
        self.pending_space = False
        self.hand_gone_since: float | None = None
        self._prev_hand, self._speed = None, 0.0
        self.events: deque = deque(maxlen=60)
        self.event_id = 0
        self.fps = 0.0
        self.camera_ok, self.camera_error = False, None
        self.version = 0
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self._cap = None
        self._open_index = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ settings
    def _load_settings(self) -> Settings:
        f = self.state_dir / "settings.json"
        try:
            return clean_settings(json.loads(f.read_text()), Settings())
        except (OSError, ValueError, AttributeError):
            return Settings()

    def _apply_settings(self):
        s = self.settings
        if self.smoother.buffer.maxlen != s.letters_window:
            self.smoother = PredictionSmoother(s.letters_window, s.letters_min_confidence)
        self.smoother.min_confidence = s.letters_min_confidence
        self.committer.hold = s.letters_hold
        self.gate = None
        if self.motion_clf is not None and s.motion_letters:
            params = GateParams.for_classifier(self.motion_clf, min_confidence=s.letters_min_confidence)
            params.arm_still_frames = 4            # a hand moving into frame isn't a J/Z
            params.discard_on_exit = True           # ...nor one leaving it
            self.gate = MotionGate(self.motion_clf, params)
        if self.words is not None:
            self.words.auto = s.words_auto_segment
            self.words.p.min_confidence = s.words_min_confidence
            self.words.p.lull_speed = s.words_lull_speed

    def _save(self):
        try:
            (self.state_dir / "settings.json").write_text(json.dumps(asdict(self.settings), indent=1))
            if self.transcript.text != self._saved_text:
                (self.state_dir / "transcript.txt").write_text(self.transcript.text)
                self._saved_text = self.transcript.text
        except OSError as e:                       # a read-only disk mustn't stop recognition
            print(f"[dashboard] could not save state: {e}")

    # ------------------------------------------------------------ events
    def _event(self, **ev) -> dict:
        self.event_id += 1
        ev.update(id=self.event_id, t=time.time())
        self.events.append(ev)
        return ev

    @property
    def _adding(self) -> bool:
        return not (self.practice or self.editing)

    def _commit_letter(self, letter: str, confidence: float, source: str):
        add = self._adding
        if add:
            self.transcript.add_letter(letter)
            self.pending_space = True
        self._event(kind="letter", letter=letter, confidence=float(confidence),
                    source=source, committed=add)

    def _word_event(self, ev: dict):
        top = [dict(name=n, label=word_label(n), p=float(p)) for n, p in ev["top"]]
        add = ev["outcome"] == "accepted" and self.settings.words_auto_add and self._adding
        e = self._event(kind="word", outcome=ev["outcome"], reason=ev["reason"],
                        duration=float(ev["duration"]), top=top, committed=add)
        if top:
            self.candidates, self.candidates_event = top, e["id"]
            self.chosen = 0 if add else -1
        if add:
            self.transcript.add_word(word_text(top[0]["name"]), e["id"])

    # ------------------------------------------------------------ frames
    def _hand_speed(self, raw, ts, wh) -> float:
        """EMA of the hand's speed; 0 when it has just appeared or is gone."""
        prev, self._prev_hand = self._prev_hand, (raw, ts) if raw is not None else None
        if raw is None or prev is None:
            self._speed = 0.0
        else:
            self._speed = 0.35 * hand_speed_v2(prev[0], raw, ts - prev[1], wh) + 0.65 * self._speed
        return self._speed

    def _letters(self, frame, ts):
        s = self.settings
        frame, bbox, letter, conf, _, gate, raw = letters_step(
            frame, self.kind, self.model, self.detector, self.transform, self.smoother,
            self.gate, self._args, self.device, ts, self.letter, self.confidence)
        self.letter, self.confidence = letter, conf
        self.motion_recording = bool(gate is not None and gate.recording)
        if gate is not None and gate.event and gate.event["outcome"] == "accepted":
            self._commit_letter(gate.event["label"], gate.event["confidence"], "motion")
            self.committer.suppress()
        moving = self._hand_speed(raw, ts, frame.shape[1::-1]) > MOVING_SPEED
        fired = self.committer.update(None if self.motion_recording or moving else letter, ts)
        if fired and s.letters_auto_add:
            self._commit_letter(fired, conf, "hold")

        self.hand = bbox is not None
        if self.hand:
            self.hand_gone_since = None
        elif self.hand_gone_since is None:
            self.hand_gone_since = ts
        if (self.pending_space and s.letters_auto_space and self.hand_gone_since is not None
                and ts - self.hand_gone_since >= s.letters_space_after):
            self.pending_space = False
            if self._adding:
                self.transcript.space()
        if s.show_landmarks and raw is not None:
            _draw_points(frame, raw, _HAND_EDGES, ACCENT)
        elif s.show_landmarks and bbox is not None:            # pixel models: box only
            cv2.rectangle(frame, bbox[:2], bbox[2:], ACCENT, 2, cv2.LINE_AA)
        return frame

    def _words(self, frame, ts):
        frame, bbox, status, lm = words_step(frame, self.holistic, self.words, ts)
        self.word_status, self.hand = status, bbox is not None
        if status.event:
            self._word_event(status.event)
        if self.settings.show_landmarks:
            draw_words_landmarks(frame, lm)
        return frame

    def process(self, frame, ts: float):
        """One camera frame -> the display frame (mirrored unless mirror is off)."""
        with self.lock:
            text_before = self.transcript.text
            out = self._words(frame, ts) if self.mode == "words" else self._letters(frame, ts)
            if not self.settings.mirror:
                out = cv2.flip(out, 1)
            if self.transcript.text != text_before:
                self._save()
            return out

    # ------------------------------------------------------------ camera loop
    def start(self):
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        with self.cond:
            self.cond.notify_all()
        if self._thread:
            self._thread.join(timeout=3)
        self._release()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _release(self):
        if self._cap is not None:
            self._cap.release()
        self._cap, self._open_index = None, None

    def _set_camera(self, ok: bool, error: str | None = None):
        if (ok, error) != (self.camera_ok, self.camera_error):
            with self.cond:
                self.camera_ok, self.camera_error = ok, error
                self._bump()

    def _open(self) -> bool:
        idx = self.settings.camera
        self._cap = cv2.VideoCapture(self.video if self.video else idx)
        self._open_index = idx
        if not self._cap.isOpened():
            self._release()
            self._set_camera(False, f"Couldn't open {'video ' + self.video if self.video else f'camera {idx}'}. "
                                    "Check it's connected and not in use by another app.")
            return False
        self._set_camera(True)
        return True

    def _run(self):
        last_t, period = time.time(), 0.0
        while not self._stop.is_set():
            if self.paused:
                self._release()
                self._stop.wait(0.1)
                continue
            if self._cap is None or (not self.video and self._open_index != self.settings.camera):
                self._release()
                if not self._open():
                    self._stop.wait(1.0)
                    continue
                fps = self._cap.get(cv2.CAP_PROP_FPS) if self.video else 0
                period = 1.0 / fps if fps and fps > 0 else 0.0
            t0 = time.time()
            ok, frame = self._cap.read()
            if not ok:
                if self.video:                                    # loop a file
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._release()
                self._set_camera(False, "The camera stopped sending frames. Reconnecting…")
                self._stop.wait(0.5)
                continue
            out = self.process(frame, t0)
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 82])
            now = time.time()
            with self.cond:
                self.fps = 0.9 * self.fps + 0.1 / max(now - last_t, 1e-6) if self.fps else 1 / max(now - last_t, 1e-6)
                last_t = now
                if ok:
                    self.jpeg, self.frame_id = buf.tobytes(), self.frame_id + 1
                self._bump()
            if period:                                            # play a file at its own speed
                self._stop.wait(max(0.0, period - (time.time() - t0)))

    def _bump(self):
        self.version += 1
        self.cond.notify_all()

    # ------------------------------------------------------------ state
    def snapshot(self) -> dict:
        with self.lock:
            ws = self.word_status
            return {
                "v": self.version, "mode": self.mode, "paused": self.paused,
                "practice": self.practice, "editing": self.editing,
                "camera": {"ok": self.camera_ok, "error": self.camera_error,
                           "source": self.video or f"camera {self.settings.camera}",
                           "fps": round(self.fps, 1)},
                "hand": self.hand,
                "letters": {"letter": self.letter, "confidence": round(float(self.confidence), 3),
                            "hold": round(self.committer.progress, 3),
                            "held": self.committer.fired and self.letter is not None,
                            "motion": self.motion_recording,
                            "motion_available": self.motion_clf is not None},
                "words": {"available": self.words is not None,
                          "state": ws.state if ws else "idle",
                          "buffered": ws.buffered if ws else 0,
                          "raised": bool(ws.raised) if ws else False,
                          "candidates": self.candidates, "event": self.candidates_event,
                          "chosen": self.chosen},
                "transcript": {"text": self.transcript.text,
                               "can_undo": bool(self.transcript._undo),
                               "can_redo": bool(self.transcript._redo)},
                "events": list(self.events)[-40:],
                "settings": asdict(self.settings),
                "defaults": asdict(Settings()),
                "cameras": camera_indices(),
                "models": self.models,
            }

    def vocabulary(self) -> list[dict]:
        # The GISLR properties file has nondom_frac 0 for every sign (no second
        # hand in its cache), so two-handedness comes from ASL Citizen; signs
        # outside its 216 count as one-handed.
        two = {}
        props = config.RESULTS_DIR / "words_sign_properties_asl_citizen.csv"
        if props.exists():
            import pandas as pd
            two = dict(pd.read_csv(props)[["sign", "two_handed"]].itertuples(index=False))
        return [dict(name=n, label=word_label(n), text=word_text(n), two_handed=bool(two.get(n, False)))
                for n in self.word_classes]

    # ------------------------------------------------------------ commands
    def command(self, action: str, **kw) -> dict:
        """Apply one UI command. Raises ValueError for unknown/invalid ones."""
        with self.cond:
            self._command(action, kw)
            self._save()
            self._bump()
        return self.snapshot()

    def _reset_recognisers(self):
        self.smoother.buffer.clear()
        self.committer.reset()
        self.letter, self.confidence, self.motion_recording = None, 0.0, False
        if self.gate is not None:
            self._apply_settings()                       # a fresh gate, nothing buffered
        if self.words is not None:
            self.words.reset()
        self.word_status = None

    def _command(self, action, kw):
        t = self.transcript
        if action == "mode":
            mode = kw.get("mode")
            if mode not in ("letters", "words") or (mode == "words" and self.words is None):
                raise ValueError(f"mode {mode!r} not available")
            if mode != self.mode:
                self.mode = mode
                self._reset_recognisers()
                self.pending_space = False
        elif action == "pause":
            self.paused = bool(kw.get("paused", not self.paused))
            self._reset_recognisers()
        elif action == "practice":
            self.practice = bool(kw.get("on"))
        elif action == "editing":
            self.editing = bool(kw.get("on"))
        elif action == "settings":
            if kw.get("reset"):
                cam = self.settings.camera
                self.settings = Settings(camera=cam)
            else:
                self.settings = clean_settings(kw.get("values") or {}, self.settings)
            self._apply_settings()
        elif action == "add_letter":                     # the letter on screen, now
            if self.mode == "letters" and self.letter and self.hand:
                self._commit_letter(self.letter, self.confidence, "manual")
                self.committer.fired = True
        elif action == "sign":                           # words: start / stop by hand
            if self.words is None or self.mode != "words":
                raise ValueError("words mode is not active")
            ev = self.words.toggle(time.time())
            if ev:
                self._word_event(ev)
        elif action == "cancel_sign":
            if self.words is not None:
                self.words.reset()
                self.word_status = None
        elif action == "choose":                         # swap / add a candidate word
            i = kw.get("index")
            if not isinstance(i, int) or not 0 <= i < len(self.candidates):
                raise ValueError("no such candidate")
            word = word_text(self.candidates[i]["name"])
            if not t.replace_last_word(word, self.candidates_event):
                t.add_word(word, self.candidates_event)
            self.chosen = i
        elif action == "insert_word":
            name = kw.get("name")
            if name not in self.word_classes:
                raise ValueError("unknown sign")
            t.add_word(word_text(name))
        elif action == "space":
            t.space()
            self.pending_space = False
        elif action == "newline":
            t.newline()
            self.pending_space = False
        elif action == "punct":
            p = kw.get("char")
            if p not in (".", ",", "?", "!"):
                raise ValueError("unsupported punctuation")
            t.punct(p)
            self.pending_space = False
        elif action == "backspace":
            t.backspace()
        elif action == "delete_word":
            t.delete_word()
            self.pending_space = False
        elif action == "clear":
            t.clear()
            self.pending_space = False
        elif action == "set_text":
            text = kw.get("text")
            if not isinstance(text, str) or len(text) > 100_000:
                raise ValueError("text must be a string under 100k characters")
            t.set_text(text)
            self.pending_space = False
        elif action == "undo":
            t.undo()
        elif action == "redo":
            t.redo()
        else:
            raise ValueError(f"unknown action {action!r}")
