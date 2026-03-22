"""Tests for long-form text-to-speech synthesis."""
import threading
from unittest.mock import MagicMock

import numpy as np

from babelvox.longform import (
    LongFormSynthesizer,
    SynthesisProgress,
    crossfade_concat,
    segment_text,
)

# ── Segmentation tests ───────────────────────────────────────────────

class TestSegmentText:
    def test_paragraph_split(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        segments = segment_text(text, "paragraph")
        assert len(segments) == 3
        assert segments[0].text == "First paragraph."
        assert segments[2].text == "Third paragraph."
        assert all(s.type == "paragraph" for s in segments)

    def test_sentence_split(self):
        text = "First sentence. Second sentence! Third sentence?"
        segments = segment_text(text, "sentence")
        assert len(segments) == 3
        assert segments[0].text == "First sentence."
        assert segments[1].text == "Second sentence!"

    def test_sentence_handles_abbreviations(self):
        text = "Mr. Smith went to Washington. He met Dr. Jones there."
        segments = segment_text(text, "sentence")
        assert len(segments) == 2
        assert "Mr. Smith" in segments[0].text
        assert "Dr. Jones" in segments[1].text

    def test_natural_splits_long_paragraphs(self):
        short = "Short paragraph."
        long_text = "This is a longer sentence that goes on and on. " * 10
        text = f"{short}\n\n{long_text}"
        segments = segment_text(text, "natural")
        assert segments[0].text == short
        assert segments[0].type == "paragraph"
        # Long paragraph should be split into sentences
        assert len(segments) > 2

    def test_natural_keeps_short_paragraphs(self):
        text = "Short one.\n\nShort two.\n\nShort three."
        segments = segment_text(text, "natural")
        assert len(segments) == 3
        assert all(s.type == "paragraph" for s in segments)

    def test_chapter_split(self):
        text = "# Chapter 1\n\nFirst content.\n\n## Section A\n\nMore content."
        segments = segment_text(text, "chapter")
        types = [s.type for s in segments]
        assert "chapter_marker" in types
        texts = [s.text for s in segments]
        assert "# Chapter 1" in texts
        assert "First content." in texts

    def test_chapter_markers_have_correct_type(self):
        text = "# Title\n\nBody text."
        segments = segment_text(text, "chapter")
        markers = [s for s in segments if s.type == "chapter_marker"]
        assert len(markers) == 1
        assert markers[0].text == "# Title"

    def test_empty_segments_stripped(self):
        text = "\n\nHello.\n\n\n\nWorld.\n\n"
        segments = segment_text(text, "paragraph")
        assert len(segments) == 2

    def test_single_paragraph(self):
        text = "Just one paragraph."
        segments = segment_text(text, "paragraph")
        assert len(segments) == 1

    def test_unknown_strategy_raises(self):
        import pytest
        with pytest.raises(ValueError, match="unknown"):
            segment_text("text", "invalid")


# ── Crossfade tests ──────────────────────────────────────────────────

class TestCrossfadeConcat:
    def test_crossfade_length(self):
        wav1 = np.ones(4800, dtype=np.float32)
        wav2 = np.ones(4800, dtype=np.float32)
        result = crossfade_concat([wav1, wav2], crossfade_samples=1200)
        # Should be shorter than sum by overlap amount
        assert len(result) < 9600
        assert len(result) >= 8400  # 9600 - 1200

    def test_crossfade_single_wav(self):
        wav = np.ones(4800, dtype=np.float32) * 0.5
        result = crossfade_concat([wav])
        # Soft onset applies fade-in, so start ramps up but end matches
        assert len(result) == len(wav)
        assert result[-1] == wav[-1]
        assert result[0] < wav[0]  # fade-in starts lower

    def test_crossfade_empty(self):
        result = crossfade_concat([])
        assert len(result) == 1  # zeros fallback

    def test_crossfade_smooth_transition(self):
        # Create two different-amplitude waveforms
        wav1 = np.ones(4800, dtype=np.float32) * 0.8
        wav2 = np.ones(4800, dtype=np.float32) * 0.2
        result = crossfade_concat([wav1, wav2], crossfade_samples=1200)
        # The transition region should have values between 0.2 and 0.8
        mid = len(wav1) - 600  # middle of crossfade
        assert 0.2 < result[mid] < 0.8


# ── LongFormSynthesizer tests ────────────────────────────────────────

def _make_mock_tts(wav_length=2400):
    tts = MagicMock()
    tts.generate.return_value = (np.zeros(wav_length, dtype=np.float32), 24000)
    tts.speaker_library = None
    tts.default_speaker = None
    tts._inference_lock = threading.Lock()
    return tts


class TestLongFormSynthesizer:
    def test_synthesize_basic(self):
        tts = _make_mock_tts()
        synth = LongFormSynthesizer(tts)
        result = synth.synthesize("Para one.\n\nPara two.\n\nPara three.")
        assert result.waveform is not None
        assert result.sample_rate == 24000
        assert result.total_duration > 0
        assert len(result.segments) == 3
        assert tts.generate.call_count == 3

    def test_progress_callback_called(self):
        tts = _make_mock_tts()
        synth = LongFormSynthesizer(tts)
        progress_updates = []
        synth.synthesize("Para one.\n\nPara two.",
                         progress_callback=progress_updates.append)
        assert len(progress_updates) == 2
        assert progress_updates[0].completed_segments == 1
        assert progress_updates[1].completed_segments == 2
        assert progress_updates[1].total_segments == 2

    def test_resume_from_skips_segments(self):
        tts = _make_mock_tts()
        synth = LongFormSynthesizer(tts)
        synth.synthesize("A.\n\nB.\n\nC.\n\nD.", resume_from=2)
        # Only segments 2 and 3 should be synthesized
        assert tts.generate.call_count == 2

    def test_chapter_markers_in_results(self):
        tts = _make_mock_tts()
        synth = LongFormSynthesizer(tts)
        result = synth.synthesize("# Chapter 1\n\nContent here.", strategy="chapter")
        markers = [s for s in result.segments if s.segment.type == "chapter_marker"]
        assert len(markers) == 1
        assert markers[0].duration == 0.0

    def test_timestamps_sequential(self):
        tts = _make_mock_tts(wav_length=2400)
        synth = LongFormSynthesizer(tts)
        result = synth.synthesize("A.\n\nB.\n\nC.", crossfade_samples=0)
        for i in range(1, len(result.segments)):
            assert result.segments[i].start_time >= result.segments[i - 1].start_time

    def test_empty_text(self):
        tts = _make_mock_tts()
        synth = LongFormSynthesizer(tts)
        result = synth.synthesize("")
        assert len(result.segments) == 0
        assert result.total_duration == 0.0

    def test_stream_yields_per_segment(self):
        tts = _make_mock_tts()
        synth = LongFormSynthesizer(tts)
        chunks = list(synth.synthesize_stream("A.\n\nB.\n\nC."))
        assert len(chunks) == 3
        for _wav, sr, progress in chunks:
            assert sr == 24000
            assert isinstance(progress, SynthesisProgress)

    def test_voice_consistency(self):
        """All generate() calls should use a speaker_embed close to original."""
        tts = _make_mock_tts()
        embed = np.ones((1, 1024), dtype=np.float32)
        tts.default_speaker = embed
        synth = LongFormSynthesizer(tts)
        synth.synthesize("A.\n\nB.\n\nC.")
        for call in tts.generate.call_args_list:
            seg_embed = call.kwargs.get("speaker_embed")
            # Per-segment perturbation adds tiny noise (magnitude 0.02)
            assert np.allclose(seg_embed, embed, atol=0.15)
