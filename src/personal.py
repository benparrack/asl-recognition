"""
SignDesk profiles: local per-signer accounts, their stored examples, and the
adapter that tailors the letter and word models to each signer.

HOW A PROFILE TAILORS THE MODELS
================================
Every stored example is run through the base model and the input to its last
Linear layer (its embedding) is kept, L2-normalised. A sign's prototype is the
mean of its examples' embeddings. At recognition time each sign with a
prototype gets a bonus on its log-probability:

    score(c) = log p_model(c) + beta * max(0, cos(embedding, prototype_c) - offset)

A clip that looks like *this signer's* version of a sign is pulled toward it;
one that looks like nothing they taught is left to the model. No retraining,
so teaching takes effect on the next frame and costs nothing to undo.

Numbers behind beta/offset (offline, 2026-10-13):
  words   Ben's 100 clips (20 signs x 5, one session). One example per sign:
          taught signs top-1 0.76 -> 0.93, untaught signs unchanged at 0.76
          (beta 40, offset 0.3). Lower offsets/higher beta gain ~1 point more
          on taught signs but start pulling untaught ones toward them.
  letters dataset5 signer E, never trained on. 3 takes of every letter:
          0.932 -> 0.968 (beta 80, offset 0.5). Letter embeddings sit closer
          together than word ones, so with only half the letters taught that
          offset drags the others down (0.93 -> 0.66); until every letter is
          taught the offset is 0.9 (taught 0.955, untaught 0.927).
Both are same-session numbers, so expect less across days and lighting.

STORAGE (all under data/dashboard/profiles/<id>/, gitignored)
=======
    profile.json        {"name", "created"}
    settings.json       the dashboard settings for this signer
    transcript.txt
    letters.npz         X (N, 63) normalised hand landmarks, y labels, take ids
    words/<sign>/<t>.npz  frames (T, points, 3) float16, times, source
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

LETTER_TAKE_FRAMES = 20        # landmark frames per letter take (~1 s held still)
MAX_LETTER_TAKES = 5           # per letter; the oldest is dropped
MAX_WORD_CLIPS = 10            # per sign; the oldest is dropped
SOURCES = ("teach", "correction", "practice")


# ---------------------------------------------------------------- adapter
class Adapter(nn.Module):
    """Wraps a classifier; forward returns log-probabilities plus the
    prototype bonus (callers that softmax the output get the adapted
    distribution). With no prototypes, or disabled, it returns the raw logits."""

    def __init__(self, model: nn.Module, beta: float, offset: float,
                 offset_all: float | None = None, n_classes: int | None = None):
        super().__init__()
        self.model = model
        head = model.head[-1] if hasattr(model, "head") else \
            [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
        head.register_forward_hook(self._grab)
        self.beta, self.offset = beta, offset
        self.offset_all, self.n_classes = offset_all, n_classes
        self.enabled = True
        self.protos: torch.Tensor | None = None      # (K, D) unit vectors
        self.proto_idx: torch.Tensor | None = None   # (K,) class indices
        self._emb = None

    def _grab(self, module, inputs, output):
        self._emb = inputs[0]

    @torch.no_grad()
    def embed(self, *args) -> torch.Tensor:
        self.model(*args)
        return F.normalize(self._emb.float(), dim=1)

    def set_prototypes(self, protos: dict[int, np.ndarray]):
        if not protos:
            self.protos = self.proto_idx = None
            return
        dev = next(self.model.parameters()).device
        idx = sorted(protos)
        self.protos = F.normalize(torch.tensor(np.stack([protos[i] for i in idx]),
                                               dtype=torch.float32, device=dev), dim=1)
        self.proto_idx = torch.tensor(idx, device=dev)

    def forward(self, *args):
        logits = self.model(*args)
        if not self.enabled or self.protos is None:
            return logits
        sim = F.normalize(self._emb.float(), dim=1) @ self.protos.T
        full = self.offset_all is not None and len(self.proto_idx) >= self.n_classes
        off = self.offset_all if full else self.offset
        out = torch.log_softmax(logits.float(), 1)
        out[:, self.proto_idx] += self.beta * (sim - off).clamp_min(0)
        return out


# ---------------------------------------------------------------- profiles
_ID = re.compile(r"^[a-z0-9-]{1,48}$")


def clean_name(name) -> str:
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 40:
        raise ValueError("a profile name is 1-40 characters")
    return " ".join(name.split())


class ProfileStore:
    """The profiles folder. Ids are slugs made here, never taken from a
    request unchecked, so they're safe as directory names."""

    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.root = self.state_dir / "profiles"
        self.root.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _migrate(self):
        """Before profiles, settings and the transcript lived in state_dir:
        they become the first profile's."""
        if self.list():
            return
        pid = self.create("Me")
        for f in ("settings.json", "transcript.txt"):
            if (self.state_dir / f).exists():
                shutil.move(str(self.state_dir / f), str(self.root / pid / f))

    def list(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir()):
            try:
                meta = json.loads((d / "profile.json").read_text())
                out.append(dict(id=d.name, name=meta["name"], created=meta.get("created", 0)))
            except (OSError, ValueError, KeyError):
                continue
        return sorted(out, key=lambda p: p["created"])

    def dir(self, pid) -> Path:
        if not isinstance(pid, str) or not _ID.match(pid) or not (self.root / pid / "profile.json").exists():
            raise ValueError("no such profile")
        return self.root / pid

    def create(self, name) -> str:
        name = clean_name(name)
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "signer"
        pid, n = base, 2
        while (self.root / pid).exists():
            pid, n = f"{base}-{n}", n + 1
        (self.root / pid).mkdir(parents=True)
        (self.root / pid / "profile.json").write_text(json.dumps({"name": name, "created": time.time()}))
        return pid

    def rename(self, pid, name):
        f = self.dir(pid) / "profile.json"
        meta = json.loads(f.read_text())
        meta["name"] = clean_name(name)
        f.write_text(json.dumps(meta))

    def delete(self, pid):
        shutil.rmtree(self.dir(pid))

    @property
    def current(self) -> str | None:
        try:
            pid = (self.state_dir / "current_profile").read_text().strip()
            self.dir(pid)
            return pid
        except (OSError, ValueError):
            ps = self.list()
            return ps[0]["id"] if ps else None

    @current.setter
    def current(self, pid):
        self.dir(pid)
        (self.state_dir / "current_profile").write_text(pid)


# ---------------------------------------------------------------- examples
class Samples:
    """One profile's stored examples."""

    def __init__(self, profile_dir: Path):
        self.dir = Path(profile_dir)
        self.words_dir = self.dir / "words"
        f = self.dir / "letters.npz"
        if f.exists():
            d = np.load(f)
            self.lx, self.ly, self.ltake = d["X"], list(d["y"]), d["take"]
        else:
            self.lx, self.ly, self.ltake = np.zeros((0, 63), np.float32), [], np.zeros(0, np.int64)

    # letters
    def add_letter_take(self, label: str, vecs: np.ndarray):
        take = int(self.ltake.max()) + 1 if len(self.ltake) else 0
        self.lx = np.concatenate([self.lx, vecs.astype(np.float32)])
        self.ly += [label] * len(vecs)
        self.ltake = np.concatenate([self.ltake, np.full(len(vecs), take)])
        takes = sorted({t for t, y in zip(self.ltake, self.ly) if y == label})
        if len(takes) > MAX_LETTER_TAKES:
            self._keep_letters(~np.isin(self.ltake, takes[:-MAX_LETTER_TAKES]))
        self._save_letters()

    def _keep_letters(self, keep: np.ndarray):
        self.lx, self.ltake = self.lx[keep], self.ltake[keep]
        self.ly = [y for y, k in zip(self.ly, keep) if k]

    def _save_letters(self):
        np.savez(self.dir / "letters.npz", X=self.lx, y=np.array(self.ly, dtype="<U2"), take=self.ltake)

    def letter_groups(self) -> dict[str, np.ndarray]:
        y = np.array(self.ly)
        return {c: self.lx[y == c] for c in sorted(set(self.ly))}

    # words
    def add_word_clip(self, label: str, frames: np.ndarray, times: np.ndarray, source: str) -> Path:
        d = self.words_dir / label
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{time.time_ns()}.npz"
        np.savez_compressed(f, frames=frames.astype(np.float16), times=np.asarray(times, np.float64),
                            source=source)
        for old in sorted(d.glob("*.npz"))[:-MAX_WORD_CLIPS]:
            old.unlink()
        return f

    def word_files(self) -> dict[str, list[Path]]:
        if not self.words_dir.exists():
            return {}
        out = {d.name: sorted(d.glob("*.npz")) for d in sorted(self.words_dir.iterdir()) if d.is_dir()}
        return {k: v for k, v in out.items() if v}

    @staticmethod
    def load_clip(f: Path):
        d = np.load(f)
        return d["frames"].astype(np.float32), d["times"]

    # both
    def forget(self, kind: str, label: str | None = None):
        if kind == "letters":
            keep = np.zeros(len(self.ly), bool) if label is None else np.array([y != label for y in self.ly], bool)
            self._keep_letters(keep)
            self._save_letters()
        elif label is None:
            shutil.rmtree(self.words_dir, ignore_errors=True)
        else:
            shutil.rmtree(self.words_dir / label, ignore_errors=True)

    def counts(self) -> dict:
        letters = {}
        for y, t in {(y, int(t)) for y, t in zip(self.ly, self.ltake)}:
            letters[y] = letters.get(y, 0) + 1
        return {"letters": letters, "words": {k: len(v) for k, v in self.word_files().items()}}


def letter_prototypes(adapter: Adapter, samples: Samples) -> dict[int, np.ndarray]:
    dev = next(adapter.model.parameters()).device
    out = {}
    for label, vecs in samples.letter_groups().items():
        if label in config.CLASSES and len(vecs):
            e = adapter.embed(torch.from_numpy(vecs).to(dev)).mean(0)
            out[config.CLASSES.index(label)] = e.cpu().numpy()
    return out
