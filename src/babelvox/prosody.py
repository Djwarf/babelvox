"""Prosody and emotion control for BabelVox.

Provides text manipulation hints, sampling parameter adjustment, and
waveform post-processing for expressiveness control.

Note: Qwen3-TTS has no explicit prosody inputs. Text manipulation is
heuristic/best-effort. Waveform post-processing (rate, pitch, volume)
provides guaranteed control via librosa.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import librosa
import numpy as np

VALID_EMOTIONS = {"happy", "sad", "angry", "surprised", "neutral"}


@dataclass
class ProsodyConfig:
    """Prosody control parameters for TTS generation."""
    rate: float = 1.0             # 0.25–4.0: speech rate multiplier
    pitch_semitones: float = 0.0  # -12 to +12: pitch shift
    volume: float = 1.0           # 0.0–2.0: amplitude multiplier
    emotion: str | None = None    # happy, sad, angry, surprised, neutral
    emphasis_words: list[str] = field(default_factory=list)


def apply_text_prosody(text: str, config: ProsodyConfig) -> str:
    """Modify text to hint the desired prosody to the model.

    This is heuristic — the model wasn't trained with emotion labels.
    Results will vary. Guaranteed prosody comes from waveform processing.
    """
    if config.emotion is None and not config.emphasis_words:
        return text

    # Apply emphasis words (capitalize specific words)
    for word in config.emphasis_words:
        text = re.sub(
            rf'\b{re.escape(word)}\b',
            word.upper(),
            text,
            flags=re.IGNORECASE,
        )

    emotion = (config.emotion or "").lower()

    if emotion == "happy":
        # Replace trailing periods with exclamation marks
        text = re.sub(r'\.\s*$', '!', text)
        # Add exclamation if no sentence-ending punctuation
        if text and text[-1] not in '.!?':
            text += '!'

    elif emotion == "sad":
        # Replace trailing periods/exclamation with ellipsis
        text = re.sub(r'[.!]\s*$', '...', text)
        if text and text[-1] not in '.!?':
            text += '...'

    elif emotion == "angry":
        # Capitalize emphasis words or first word if none specified
        if not config.emphasis_words:
            words = text.split()
            if words:
                words[0] = words[0].upper()
                text = ' '.join(words)
        # Ensure exclamation ending
        text = re.sub(r'\.\s*$', '!', text)
        if text and text[-1] not in '.!?':
            text += '!'

    elif emotion == "surprised":
        text = re.sub(r'\.\s*$', '!', text)
        if text and text[-1] not in '.!?':
            text += '!'

    return text


def adjust_sampling_for_emotion(emotion: str | None,
                                temperature: float,
                                top_k: int) -> tuple[float, int]:
    """Adjust sampling parameters based on emotion.

    Returns adjusted (temperature, top_k) tuple.
    """
    if not emotion:
        return temperature, top_k

    emotion = emotion.lower()

    if emotion in ("happy", "surprised"):
        return temperature * 1.1, top_k
    if emotion == "sad":
        return temperature * 0.9, top_k
    if emotion == "angry":
        return temperature * 1.05, max(1, int(top_k * 0.8))

    return temperature, top_k


def apply_waveform_prosody(wav: np.ndarray, sr: int,
                           config: ProsodyConfig) -> np.ndarray:
    """Apply guaranteed prosody effects to a waveform.

    Uses librosa for time-stretching and pitch-shifting.
    Volume is simple amplitude scaling with clipping.
    """
    if config.rate != 1.0:
        wav = librosa.effects.time_stretch(wav, rate=config.rate)

    if config.pitch_semitones != 0.0:
        wav = librosa.effects.pitch_shift(
            wav, sr=sr, n_steps=config.pitch_semitones)

    if config.volume != 1.0:
        wav = np.clip(wav * config.volume, -1.0, 1.0).astype(np.float32)

    return wav
