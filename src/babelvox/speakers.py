"""Speaker profile management for BabelVox.

Save, load, search, and mix speaker voice embeddings as named profiles.
Storage is file-based: JSON metadata + .npy embedding per profile.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import asdict, dataclass, field

import numpy as np

logger = logging.getLogger("babelvox")

BUILTIN_SPEAKERS_DIR = os.path.join(os.path.dirname(__file__), "data", "speakers")

_INVALID_NAME_RE = re.compile(r'[/\\]|\.\.|[\x00-\x1f]')


def _validate_name(name: str) -> str:
    """Validate and normalize a speaker profile name."""
    name = name.strip().lower()
    if not name:
        raise ValueError("speaker name cannot be empty")
    if _INVALID_NAME_RE.search(name):
        raise ValueError(f"invalid speaker name: {name!r}")
    return name


@dataclass
class SpeakerProfile:
    """A named speaker voice profile with metadata."""
    name: str
    embedding: np.ndarray  # (1, 1024) float32
    description: str = ""
    language: str = ""
    gender: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    source_audio: str = ""


class SpeakerLibrary:
    """File-based speaker profile library.

    Each profile is stored as two files in ``library_dir``:
      - ``{name}.json`` — metadata (everything except embedding)
      - ``{name}.npy`` — numpy array (1, 1024) float32
    """

    def __init__(self, library_dir: str, copy_builtins: bool = True):
        self.library_dir = library_dir
        os.makedirs(library_dir, exist_ok=True)
        if copy_builtins:
            self._copy_builtin_speakers()

    def _copy_builtin_speakers(self) -> None:
        """Copy bundled example speakers to user library if not present."""
        if not os.path.isdir(BUILTIN_SPEAKERS_DIR):
            return
        for filename in os.listdir(BUILTIN_SPEAKERS_DIR):
            if not filename.endswith(".json"):
                continue
            name = filename[:-5]
            if os.path.isfile(self._json_path(name)):
                continue  # never overwrite user profiles
            json_src = os.path.join(BUILTIN_SPEAKERS_DIR, f"{name}.json")
            npy_src = os.path.join(BUILTIN_SPEAKERS_DIR, f"{name}.npy")
            if os.path.isfile(json_src) and os.path.isfile(npy_src):
                shutil.copy2(json_src, self._json_path(name))
                shutil.copy2(npy_src, self._npy_path(name))
                logger.info("Copied bundled speaker profile '%s'", name)

    def _json_path(self, name: str) -> str:
        return os.path.join(self.library_dir, f"{name}.json")

    def _npy_path(self, name: str) -> str:
        return os.path.join(self.library_dir, f"{name}.npy")

    def save(self, profile: SpeakerProfile) -> None:
        """Save a speaker profile to disk."""
        name = _validate_name(profile.name)
        profile.name = name

        meta = asdict(profile)
        del meta["embedding"]
        with open(self._json_path(name), "w") as f:
            json.dump(meta, f, indent=2)

        np.save(self._npy_path(name), profile.embedding)
        logger.info("Saved speaker profile '%s'", name)

    def load(self, name: str) -> SpeakerProfile:
        """Load a speaker profile by name."""
        name = _validate_name(name)
        json_path = self._json_path(name)
        npy_path = self._npy_path(name)

        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"speaker profile not found: {name!r}")

        with open(json_path) as f:
            meta = json.load(f)

        embedding = np.load(npy_path)
        return SpeakerProfile(embedding=embedding, **meta)

    def list_profiles(self) -> list[dict]:
        """List all speaker profiles (metadata only, no embeddings)."""
        profiles = []
        for filename in sorted(os.listdir(self.library_dir)):
            if filename.endswith(".json"):
                path = os.path.join(self.library_dir, filename)
                with open(path) as f:
                    profiles.append(json.load(f))
        return profiles

    def delete(self, name: str) -> None:
        """Delete a speaker profile."""
        name = _validate_name(name)
        json_path = self._json_path(name)
        npy_path = self._npy_path(name)

        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"speaker profile not found: {name!r}")

        os.remove(json_path)
        if os.path.isfile(npy_path):
            os.remove(npy_path)
        logger.info("Deleted speaker profile '%s'", name)

    def search(self, language=None, gender=None, tag=None) -> list[dict]:
        """Filter profiles by criteria."""
        results = []
        for profile in self.list_profiles():
            if language and profile.get("language", "").lower() != language.lower():
                continue
            if gender and profile.get("gender", "").lower() != gender.lower():
                continue
            if tag and tag.lower() not in [t.lower() for t in profile.get("tags", [])]:
                continue
            results.append(profile)
        return results


def mix_speakers(embeddings: list[np.ndarray],
                 weights: list[float]) -> np.ndarray:
    """Weighted average of speaker embeddings.

    Weights are normalized to sum to 1.0. Returns (1, 1024) array.
    """
    if len(embeddings) != len(weights):
        raise ValueError("embeddings and weights must have the same length")
    if not embeddings:
        raise ValueError("at least one embedding required")
    total = sum(weights)
    if total == 0:
        raise ValueError("weights must not all be zero")
    normed = [w / total for w in weights]
    result = sum(e * w for e, w in zip(embeddings, normed, strict=True))
    return result


def interpolate_speakers(a: np.ndarray, b: np.ndarray,
                         alpha: float) -> np.ndarray:
    """Linear interpolation between two speaker embeddings.

    alpha=0.0 returns a, alpha=1.0 returns b.
    """
    return (1.0 - alpha) * a + alpha * b
