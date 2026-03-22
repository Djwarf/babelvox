"""Prosody and emotion control for BabelVox.

Provides text manipulation hints, sampling parameter adjustment, and
waveform post-processing for expressiveness control.

Emotion control uses five independent levers:
  1. Activation steering — emotion bias vector in embedding space
  2. Dynamic per-step sampling — prosodic contours via modulated temperature/rep_penalty
  3. Subtalker tuning — voice texture via code predictor parameters
  4. Text manipulation — punctuation/capitalization cues for the model
  5. Speaker embedding perturbation — long-form anti-monotony (in longform.py)
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import resample

logger = logging.getLogger("babelvox")

VALID_EMOTIONS = {"happy", "sad", "angry", "surprised", "neutral"}

# Fixed seeds for reproducible per-emotion bias vectors
_EMOTION_BIAS_SEEDS = {"happy": 42, "sad": 123, "angry": 456, "surprised": 789}


@dataclass
class ProsodyConfig:
    """Prosody control parameters for TTS generation."""
    rate: float = 1.0             # 0.25–4.0: speech rate multiplier
    pitch_semitones: float = 0.0  # accepted but not applied (see apply_waveform_prosody)
    volume: float = 1.0           # 0.0–2.0: amplitude multiplier
    emotion: str | None = None    # happy, sad, angry, surprised, neutral
    emphasis_words: list[str] = field(default_factory=list)


# ── Lever 1: Activation steering ─────────────────────────────────────

def get_emotion_bias(emotion: str | None, dim: int = 1024,
                     magnitude: float = 0.01) -> np.ndarray | None:
    """Return an emotion-specific bias vector for embedding-level steering.

    Each emotion gets a fixed random direction in embedding space (seeded
    for consistency). Applied at every generation step to subtly steer
    the talker toward emotion-characteristic codec tokens.

    Returns (1, 1, dim) float32 array, or None if no emotion.
    """
    if not emotion or emotion.lower() not in _EMOTION_BIAS_SEEDS:
        return None
    seed = _EMOTION_BIAS_SEEDS[emotion.lower()]
    rng = np.random.RandomState(seed)
    return (rng.randn(1, 1, dim) * magnitude).astype(np.float32)


# ── Lever 2: Dynamic per-step sampling ────────────────────────────────

def emotion_schedule(emotion: str | None, step: int,
                     base_temp: float, base_rep_pen: float,
                     base_top_k: int) -> tuple[float, float, int]:
    """Return per-step (temperature, repetition_penalty, top_k).

    Creates temporal prosodic contours — the key to expressive speech.
    Each emotion has a characteristic temporal shape.
    """
    if not emotion:
        return base_temp, base_rep_pen, base_top_k

    emotion = emotion.lower()

    if emotion == "happy":
        # Rhythmic energy wave
        wave = 0.15 * math.sin(step * math.pi / 20)
        return (base_temp * 1.2 + wave * 0.1,
                base_rep_pen + wave * 0.15,
                base_top_k)

    if emotion == "sad":
        # Flat and low throughout
        return base_temp * 0.65, 1.0, max(1, int(base_top_k * 0.6))

    if emotion == "angry":
        # Sharp periodic intensity peaks
        peak = 0.2 if step % 15 < 5 else 0.0
        return (base_temp * 1.1 + peak,
                base_rep_pen * 1.2 + peak,
                max(1, int(base_top_k * 0.5)))

    if emotion == "surprised":
        # High initial burst that decays
        decay = math.exp(-step / 30)
        return (base_temp * (1.0 + 0.5 * decay),
                base_rep_pen * (1.0 + 0.4 * decay),
                base_top_k)

    return base_temp, base_rep_pen, base_top_k


# ── Lever 3: Subtalker tuning ─────────────────────────────────────────

def get_emotion_subtalker_params(emotion: str | None) -> tuple[float, int]:
    """Return (subtalker_temperature, subtalker_top_k) for code predictor.

    Codes 1-15 control voice texture — breathiness, clarity, harshness.
    """
    if not emotion:
        return 0.9, 50

    emotion = emotion.lower()
    params = {
        "happy":    (1.1,  50),   # bright, lively
        "sad":      (0.7,  30),   # muted, soft
        "angry":    (1.2,  25),   # harsh, intense
        "surprised": (1.15, 50),  # bright, airy
    }
    return params.get(emotion, (0.9, 50))


# ── Lever 4: Text manipulation ────────────────────────────────────────

def apply_text_prosody(text: str, config: ProsodyConfig) -> str:
    """Modify text to hint the desired prosody to the model.

    Applies across the full text, not just trailing punctuation.
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
        # Replace ALL sentence-ending periods with exclamation
        text = re.sub(r'\.(\s)', r'!\1', text)
        text = re.sub(r'\.\s*$', '!', text)
        if text and text[-1] not in '.!?':
            text += '!'

    elif emotion == "sad":
        # Replace ALL sentence-ending periods with ellipsis
        text = re.sub(r'\.(\s)', r'...\1', text)
        text = re.sub(r'[.!]\s*$', '...', text)
        if text and text[-1] not in '.!?':
            text += '...'

    elif emotion == "angry":
        # Capitalize first word for emphasis
        if not config.emphasis_words:
            words = text.split()
            if words:
                words[0] = words[0].upper()
                text = ' '.join(words)
        # Replace ALL periods with exclamation
        text = re.sub(r'\.(\s)', r'!\1', text)
        text = re.sub(r'\.\s*$', '!', text)
        if text and text[-1] not in '.!?':
            text += '!'

    elif emotion == "surprised":
        # Replace ALL periods with exclamation
        text = re.sub(r'\.(\s)', r'!\1', text)
        text = re.sub(r'\.\s*$', '!', text)
        if text and text[-1] not in '.!?':
            text += '!'

    return text


# ── Sampling adjustment (base values for the schedule) ────────────────

def adjust_sampling_for_emotion(
        emotion: str | None,
        temperature: float,
        top_k: int,
        repetition_penalty: float = 1.05,
) -> tuple[float, int, float, float, int]:
    """Adjust sampling parameters based on emotion.

    Returns (temperature, top_k, repetition_penalty,
             subtalker_temperature, subtalker_top_k).
    These are BASE values — emotion_schedule() modulates them per-step.
    """
    sub_temp, sub_top_k = get_emotion_subtalker_params(emotion)

    if not emotion:
        return temperature, top_k, repetition_penalty, sub_temp, sub_top_k

    emotion = emotion.lower()

    if emotion == "happy":
        return temperature * 1.2, top_k, 1.2, sub_temp, sub_top_k
    if emotion == "sad":
        return temperature * 0.7, max(1, int(top_k * 0.6)), 1.0, sub_temp, sub_top_k
    if emotion == "angry":
        return temperature * 1.1, max(1, int(top_k * 0.5)), 1.2, sub_temp, sub_top_k
    if emotion == "surprised":
        return temperature * 1.3, top_k, 1.3, sub_temp, sub_top_k

    return temperature, top_k, repetition_penalty, sub_temp, sub_top_k


# ── Waveform post-processing ─────────────────────────────────────────

def apply_waveform_prosody(wav: np.ndarray, sr: int,
                           config: ProsodyConfig) -> np.ndarray:
    """Apply guaranteed prosody effects to a waveform.

    Rate changes use scipy resampling (changes speed + pitch together,
    artifact-free). Pitch-only shifting is not supported — the phase
    vocoder artifacts were too severe for speech.
    Volume is simple amplitude scaling with clipping.
    """
    if config.rate != 1.0:
        target_len = int(len(wav) / config.rate)
        if target_len > 0:
            wav = resample(wav, target_len).astype(np.float32)

    if config.pitch_semitones != 0.0:
        logger.debug("pitch_semitones ignored — phase vocoder artifacts "
                     "degrade speech quality; use rate instead")

    if config.volume != 1.0:
        wav = np.clip(wav * config.volume, -1.0, 1.0).astype(np.float32)

    return wav
