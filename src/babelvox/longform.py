"""Long-form text-to-speech synthesis for BabelVox.

Segments long text into manageable chunks, synthesizes each with
consistent voice, and concatenates with crossfade for seamless output.
Supports paragraph, sentence, natural, and chapter-based segmentation.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("babelvox")

# Abbreviations that shouldn't trigger sentence splits
_ABBREV_RE = re.compile(
    r'\b(?:Mr|Mrs|Ms|Dr|Prof|Jr|Sr|St|Ave|Blvd|vs|etc|approx|dept|govt'
    r'|inc|corp|ltd|est|vol|no|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec'
    r'|U\.S|U\.K)\.',
    re.IGNORECASE,
)

_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])')
_CHAPTER_RE = re.compile(r'^#{1,6}\s+.+$|^---+\s*$', re.MULTILINE)


@dataclass
class Segment:
    """A chunk of text to be synthesized."""
    text: str
    index: int
    type: str = "paragraph"  # "paragraph", "sentence", "chapter_marker"


@dataclass
class SegmentResult:
    """Timing metadata for a synthesized segment."""
    segment: Segment
    start_time: float
    end_time: float
    duration: float


@dataclass
class SynthesisProgress:
    """Progress update during long-form synthesis."""
    total_segments: int
    completed_segments: int
    current_segment_index: int
    elapsed_seconds: float
    total_audio_seconds: float


@dataclass
class SynthesisResult:
    """Complete result of long-form synthesis."""
    waveform: np.ndarray
    sample_rate: int = 24000
    segments: list[SegmentResult] = field(default_factory=list)
    total_duration: float = 0.0


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting abbreviations."""
    # Replace abbreviation periods with a placeholder
    placeholders = {}
    for i, match in enumerate(_ABBREV_RE.finditer(text)):
        placeholder = f"\x00ABBR{i}\x00"
        placeholders[placeholder] = match.group()

    masked = text
    for placeholder, original in placeholders.items():
        masked = masked.replace(original, placeholder, 1)

    # Split on sentence boundaries
    parts = _SENTENCE_END_RE.split(masked)

    # Restore abbreviations
    sentences = []
    for part in parts:
        for placeholder, original in placeholders.items():
            part = part.replace(placeholder, original)
        part = part.strip()
        if part:
            sentences.append(part)
    return sentences


def segment_text(text: str, strategy: str = "natural") -> list[Segment]:
    """Split text into segments using the specified strategy.

    Strategies:
        paragraph: Split on double newlines.
        sentence: Split on sentence-ending punctuation.
        natural: Paragraphs first, split long paragraphs (>300 chars) into sentences.
        chapter: Split on markdown headers/dividers; headers become chapter_marker segments.
    """
    if strategy == "paragraph":
        parts = [p.strip() for p in text.split("\n\n")]
        return [Segment(text=p, index=i, type="paragraph")
                for i, p in enumerate(parts) if p]

    if strategy == "sentence":
        sentences = _split_sentences(text)
        return [Segment(text=s, index=i, type="sentence")
                for i, s in enumerate(sentences) if s]

    if strategy == "natural":
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        segments = []
        idx = 0
        for para in paragraphs:
            if len(para) > 300:
                for sent in _split_sentences(para):
                    if sent:
                        segments.append(Segment(text=sent, index=idx, type="sentence"))
                        idx += 1
            else:
                segments.append(Segment(text=para, index=idx, type="paragraph"))
                idx += 1
        return segments

    if strategy == "chapter":
        segments = []
        idx = 0
        last_end = 0
        for match in _CHAPTER_RE.finditer(text):
            # Text before the header
            before = text[last_end:match.start()].strip()
            if before:
                for para in before.split("\n\n"):
                    para = para.strip()
                    if para:
                        segments.append(Segment(text=para, index=idx, type="paragraph"))
                        idx += 1
            # The header itself
            header = match.group().strip()
            if header and not header.startswith("---"):
                segments.append(Segment(text=header, index=idx, type="chapter_marker"))
                idx += 1
            last_end = match.end()

        # Remaining text after last header
        remaining = text[last_end:].strip()
        if remaining:
            for para in remaining.split("\n\n"):
                para = para.strip()
                if para:
                    segments.append(Segment(text=para, index=idx, type="paragraph"))
                    idx += 1
        return segments

    raise ValueError(f"unknown segmentation strategy: {strategy!r}")


def _segment_gap_ms(prev_type: str, curr_type: str) -> int:
    """Return silence gap duration (ms) between two segment types.

    Mimics natural human pacing: short breath between sentences,
    longer pause between paragraphs, full stop at chapter breaks.
    """
    if prev_type == "chapter_marker" or curr_type == "chapter_marker":
        return 1000
    if curr_type == "paragraph":
        return 600
    return 300  # sentence-to-sentence


def _soft_onset(wav: np.ndarray, fade_samples: int = 1200) -> np.ndarray:
    """Apply a short fade-in to prevent abrupt onset."""
    n = min(fade_samples, len(wav))
    if n > 0:
        wav = wav.copy()
        wav[:n] *= np.linspace(0, 1, n, dtype=np.float32)
    return wav


def crossfade_concat(waveforms: list[np.ndarray],
                     crossfade_samples: int = 4800,
                     gaps_ms: list[int] | None = None,
                     sample_rate: int = 24000) -> np.ndarray:
    """Concatenate waveforms with silence gaps and equal-power crossfade.

    Args:
        waveforms: List of audio arrays to join.
        crossfade_samples: Overlap length for crossfade (default 200ms).
        gaps_ms: List of N-1 silence durations (ms) between adjacent
            segments. If None, no silence is inserted (pure crossfade).
        sample_rate: Audio sample rate for gap calculation.
    """
    if not waveforms:
        return np.zeros(1, dtype=np.float32)
    if len(waveforms) == 1:
        return _soft_onset(waveforms[0])

    result_parts = [_soft_onset(waveforms[0])]
    for i in range(1, len(waveforms)):
        prev = result_parts[-1]
        curr = waveforms[i]

        # Insert silence gap if specified
        if gaps_ms and i - 1 < len(gaps_ms) and gaps_ms[i - 1] > 0:
            gap_samples = int(sample_rate * gaps_ms[i - 1] / 1000)
            result_parts.append(np.zeros(gap_samples, dtype=np.float32))
            # After a silence gap, apply soft onset to next segment
            curr = _soft_onset(curr)
            result_parts.append(curr)
            continue

        # Equal-power crossfade (constant perceptual loudness)
        ol = min(crossfade_samples, len(prev) // 2, len(curr) // 2)
        if ol > 0:
            t = np.linspace(0, np.pi / 2, ol, dtype=np.float32)
            fade_out = np.cos(t)
            fade_in = np.sin(t)
            blended = prev[-ol:] * fade_out + curr[:ol] * fade_in
            result_parts[-1] = prev[:-ol]
            result_parts.append(blended)
            result_parts.append(curr[ol:])
        else:
            result_parts.append(curr)

    return np.concatenate(result_parts)


class LongFormSynthesizer:
    """Orchestrates long-form text-to-speech synthesis.

    Segments text, synthesizes each segment with consistent voice,
    and concatenates with crossfade for seamless output.
    """

    def __init__(self, tts):
        self.tts = tts

    def synthesize(self, text, strategy="natural", speaker=None,
                   speaker_embed=None, language="English", prosody=None,
                   crossfade_samples=4800, max_new_tokens=512,
                   progress_callback=None, resume_from=0,
                   seed=None):
        """Synthesize long text as a single waveform.

        Returns a SynthesisResult with the concatenated waveform,
        per-segment timestamps, and total duration.
        """
        segments = segment_text(text, strategy)

        # Resolve speaker once for voice consistency
        if speaker and self.tts.speaker_library:
            speaker_embed = self.tts.speaker_library.load(speaker).embedding
        elif speaker_embed is None and self.tts.default_speaker is not None:
            speaker_embed = self.tts.default_speaker

        waveforms = []
        seg_types = []  # track types for gap computation
        segment_results = []
        t0 = time.time()
        current_time = 0.0

        logger.info("Long-form synthesis: %d segments (%s strategy)",
                     len(segments), strategy)

        prev_type = None
        for seg in segments[resume_from:]:
            if seg.type == "chapter_marker":
                segment_results.append(SegmentResult(
                    segment=seg, start_time=current_time,
                    end_time=current_time, duration=0.0))
                logger.debug("  [%d] chapter marker: %s", seg.index, seg.text)
                prev_type = "chapter_marker"
                continue

            if seed is not None:
                np.random.seed(seed + seg.index)

            # Per-segment variation to break monotony (Lever 5)
            seg_embed = speaker_embed
            seg_temp = None
            if speaker_embed is not None:
                rng = np.random.RandomState(42 + seg.index)
                seg_embed = speaker_embed + (
                    rng.randn(*speaker_embed.shape).astype(np.float32) * 0.02)
                seg_temp = 0.9 * (1.0 + rng.uniform(-0.1, 0.1))

            logger.debug("  [%d] synthesizing: %.60s...", seg.index, seg.text)
            gen_kwargs = dict(
                text=seg.text, language=language,
                speaker_embed=seg_embed, prosody=prosody,
                max_new_tokens=max_new_tokens)
            if seg_temp is not None:
                gen_kwargs["temperature"] = seg_temp
            wav, sr = self.tts.generate(**gen_kwargs)

            duration = len(wav) / sr
            # Account for gap before this segment in timestamps
            if waveforms and prev_type is not None:
                gap_secs = _segment_gap_ms(prev_type, seg.type) / 1000.0
                current_time += gap_secs
            segment_results.append(SegmentResult(
                segment=seg, start_time=current_time,
                end_time=current_time + duration, duration=duration))
            current_time += duration
            waveforms.append(wav)
            seg_types.append(seg.type)
            prev_type = seg.type

            if progress_callback:
                progress_callback(SynthesisProgress(
                    total_segments=len(segments),
                    completed_segments=len(waveforms),
                    current_segment_index=seg.index,
                    elapsed_seconds=time.time() - t0,
                    total_audio_seconds=current_time))

        if not waveforms:
            return SynthesisResult(waveform=np.zeros(1, dtype=np.float32))

        # Compute per-gap silence durations based on segment types
        gaps_ms = []
        for i in range(1, len(seg_types)):
            gaps_ms.append(_segment_gap_ms(seg_types[i - 1], seg_types[i]))

        full_wav = crossfade_concat(waveforms, crossfade_samples,
                                    gaps_ms=gaps_ms)
        logger.info("Long-form complete: %.1fs audio from %d segments",
                     len(full_wav) / 24000, len(waveforms))

        return SynthesisResult(
            waveform=full_wav, sample_rate=24000,
            segments=segment_results,
            total_duration=len(full_wav) / 24000)

    def synthesize_stream(self, text, strategy="natural", speaker=None,
                          speaker_embed=None, language="English",
                          prosody=None, max_new_tokens=512,
                          resume_from=0, seed=None):
        """Yield (waveform, sample_rate, progress) per segment.

        Appends natural silence gaps between segments for human-like pacing.
        """
        segments = segment_text(text, strategy)

        if speaker and self.tts.speaker_library:
            speaker_embed = self.tts.speaker_library.load(speaker).embedding
        elif speaker_embed is None and self.tts.default_speaker is not None:
            speaker_embed = self.tts.default_speaker

        t0 = time.time()
        current_time = 0.0
        completed = 0
        prev_type = None

        for seg in segments[resume_from:]:
            if seg.type == "chapter_marker":
                prev_type = "chapter_marker"
                continue

            if seed is not None:
                np.random.seed(seed + seg.index)

            # Per-segment variation (Lever 5)
            seg_embed = speaker_embed
            seg_temp = None
            if speaker_embed is not None:
                rng = np.random.RandomState(42 + seg.index)
                seg_embed = speaker_embed + (
                    rng.randn(*speaker_embed.shape).astype(np.float32) * 0.02)
                seg_temp = 0.9 * (1.0 + rng.uniform(-0.1, 0.1))

            gen_kwargs = dict(
                text=seg.text, language=language,
                speaker_embed=seg_embed, prosody=prosody,
                max_new_tokens=max_new_tokens)
            if seg_temp is not None:
                gen_kwargs["temperature"] = seg_temp
            wav, sr = self.tts.generate(**gen_kwargs)

            # Append silence gap for natural pacing
            if prev_type is not None:
                gap_ms = _segment_gap_ms(prev_type, seg.type)
                gap_samples = int(sr * gap_ms / 1000)
                wav = np.concatenate([np.zeros(gap_samples, dtype=np.float32),
                                      _soft_onset(wav)])
            else:
                wav = _soft_onset(wav)

            current_time += len(wav) / sr
            completed += 1
            prev_type = seg.type

            progress = SynthesisProgress(
                total_segments=len(segments),
                completed_segments=completed,
                current_segment_index=seg.index,
                elapsed_seconds=time.time() - t0,
                total_audio_seconds=current_time)

            yield wav, sr, progress
