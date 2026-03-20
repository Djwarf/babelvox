"""Tests for speaker profile management."""
import io
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from babelvox.speakers import (
    SpeakerLibrary,
    SpeakerProfile,
    interpolate_speakers,
    mix_speakers,
)


@pytest.fixture
def tmp_library(tmp_path):
    return SpeakerLibrary(str(tmp_path), copy_builtins=False)


def _make_profile(name="alice", **kwargs):
    defaults = {
        "embedding": np.random.randn(1, 1024).astype(np.float32),
        "description": "Test speaker",
        "language": "English",
        "gender": "female",
        "tags": ["narrator"],
        "created_at": "2026-03-19T00:00:00Z",
        "source_audio": "alice.wav",
    }
    defaults.update(kwargs)
    return SpeakerProfile(name=name, **defaults)


# ── CRUD Tests ────────────────────────────────────────────────────────

class TestSaveAndLoad:
    def test_roundtrip(self, tmp_library):
        profile = _make_profile()
        tmp_library.save(profile)
        loaded = tmp_library.load("alice")
        assert loaded.name == "alice"
        assert loaded.description == "Test speaker"
        assert loaded.language == "English"
        assert loaded.gender == "female"
        assert loaded.tags == ["narrator"]
        assert loaded.source_audio == "alice.wav"
        assert loaded.embedding.shape == (1, 1024)
        np.testing.assert_array_almost_equal(loaded.embedding, profile.embedding)

    def test_name_is_lowercased(self, tmp_library):
        profile = _make_profile(name="Alice")
        tmp_library.save(profile)
        loaded = tmp_library.load("alice")
        assert loaded.name == "alice"

    def test_overwrite_existing(self, tmp_library):
        p1 = _make_profile(description="first")
        p2 = _make_profile(description="second")
        tmp_library.save(p1)
        tmp_library.save(p2)
        loaded = tmp_library.load("alice")
        assert loaded.description == "second"


class TestListProfiles:
    def test_list_empty(self, tmp_library):
        assert tmp_library.list_profiles() == []

    def test_list_multiple(self, tmp_library):
        tmp_library.save(_make_profile("alice"))
        tmp_library.save(_make_profile("bob"))
        tmp_library.save(_make_profile("carol"))
        profiles = tmp_library.list_profiles()
        names = [p["name"] for p in profiles]
        assert sorted(names) == ["alice", "bob", "carol"]

    def test_list_does_not_include_embedding(self, tmp_library):
        tmp_library.save(_make_profile())
        profiles = tmp_library.list_profiles()
        assert "embedding" not in profiles[0]


class TestDelete:
    def test_delete_existing(self, tmp_library):
        tmp_library.save(_make_profile())
        tmp_library.delete("alice")
        assert tmp_library.list_profiles() == []

    def test_delete_nonexistent_raises(self, tmp_library):
        with pytest.raises(FileNotFoundError):
            tmp_library.delete("nobody")

    def test_load_nonexistent_raises(self, tmp_library):
        with pytest.raises(FileNotFoundError):
            tmp_library.load("nobody")


class TestSearch:
    def test_search_by_language(self, tmp_library):
        tmp_library.save(_make_profile("alice", language="English"))
        tmp_library.save(_make_profile("bob", language="French"))
        results = tmp_library.search(language="English")
        assert len(results) == 1
        assert results[0]["name"] == "alice"

    def test_search_by_gender(self, tmp_library):
        tmp_library.save(_make_profile("alice", gender="female"))
        tmp_library.save(_make_profile("bob", gender="male"))
        results = tmp_library.search(gender="male")
        assert len(results) == 1
        assert results[0]["name"] == "bob"

    def test_search_by_tag(self, tmp_library):
        tmp_library.save(_make_profile("alice", tags=["narrator", "soft"]))
        tmp_library.save(_make_profile("bob", tags=["announcer"]))
        results = tmp_library.search(tag="narrator")
        assert len(results) == 1
        assert results[0]["name"] == "alice"

    def test_search_no_match(self, tmp_library):
        tmp_library.save(_make_profile())
        assert tmp_library.search(language="Klingon") == []


class TestNameValidation:
    def test_rejects_slash(self, tmp_library):
        with pytest.raises(ValueError):
            tmp_library.save(_make_profile("alice/bob"))

    def test_rejects_backslash(self, tmp_library):
        with pytest.raises(ValueError):
            tmp_library.save(_make_profile("alice\\bob"))

    def test_rejects_dotdot(self, tmp_library):
        with pytest.raises(ValueError):
            tmp_library.save(_make_profile(".."))

    def test_rejects_empty(self, tmp_library):
        with pytest.raises(ValueError):
            tmp_library.save(_make_profile(""))

    def test_rejects_whitespace_only(self, tmp_library):
        with pytest.raises(ValueError):
            tmp_library.save(_make_profile("   "))


# ── Utility Tests ─────────────────────────────────────────────────────

class TestMixSpeakers:
    def test_weighted_average(self):
        a = np.ones((1, 1024), dtype=np.float32)
        b = np.ones((1, 1024), dtype=np.float32) * 3
        result = mix_speakers([a, b], [1.0, 1.0])
        np.testing.assert_array_almost_equal(result, np.full((1, 1024), 2.0))

    def test_normalizes_weights(self):
        a = np.zeros((1, 1024), dtype=np.float32)
        b = np.ones((1, 1024), dtype=np.float32) * 5
        result = mix_speakers([a, b], [2, 3])
        # (0*0.4 + 5*0.6) = 3.0
        np.testing.assert_array_almost_equal(result, np.full((1, 1024), 3.0))

    def test_single_speaker(self):
        a = np.ones((1, 1024), dtype=np.float32) * 7
        result = mix_speakers([a], [1.0])
        np.testing.assert_array_almost_equal(result, a)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            mix_speakers([np.zeros((1, 1024))], [1.0, 2.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mix_speakers([], [])


class TestInterpolateSpeakers:
    def test_alpha_zero(self):
        a = np.ones((1, 1024), dtype=np.float32)
        b = np.zeros((1, 1024), dtype=np.float32)
        result = interpolate_speakers(a, b, 0.0)
        np.testing.assert_array_almost_equal(result, a)

    def test_alpha_one(self):
        a = np.ones((1, 1024), dtype=np.float32)
        b = np.zeros((1, 1024), dtype=np.float32)
        result = interpolate_speakers(a, b, 1.0)
        np.testing.assert_array_almost_equal(result, b)

    def test_midpoint(self):
        a = np.zeros((1, 1024), dtype=np.float32)
        b = np.ones((1, 1024), dtype=np.float32) * 4
        result = interpolate_speakers(a, b, 0.5)
        np.testing.assert_array_almost_equal(result, np.full((1, 1024), 2.0))


# ── Server Endpoint Tests ─────────────────────────────────────────────

class MockHandler:
    """Simulate BaseHTTPRequestHandler for server tests."""
    def __init__(self, body=None):
        if body is None:
            body = b""
        elif isinstance(body, dict):
            body = json.dumps(body).encode()
        self.headers = {"Content-Length": str(len(body))}
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.path = ""
        self._response_code = None
        self._response_headers = {}

    def send_response(self, code):
        self._response_code = code

    def send_header(self, k, v):
        self._response_headers[k] = v

    def end_headers(self):
        pass

    @property
    def response_body(self):
        return self.wfile.getvalue()

    @property
    def response_json(self):
        return json.loads(self.response_body)


class TestServerSpeakerEndpoints:
    @pytest.fixture
    def mock_tts(self, tmp_path):
        tts = MagicMock()
        tts.speaker_library = SpeakerLibrary(str(tmp_path), copy_builtins=False)
        tts.generate.return_value = (np.zeros(2400, dtype=np.float32), 24000)
        # Wire save_speaker to actually save
        def _save(name, ref_audio, **meta):
            profile = SpeakerProfile(
                name=name,
                embedding=np.zeros((1, 1024), dtype=np.float32),
                source_audio=ref_audio,
                **meta)
            tts.speaker_library.save(profile)
            return profile
        tts.save_speaker.side_effect = _save
        return tts

    def test_get_speakers_empty(self, mock_tts):
        profiles = mock_tts.speaker_library.list_profiles()
        assert profiles == []

    def test_save_and_list_speaker(self, mock_tts, tmp_path):
        from babelvox.server import handle_save_speaker
        # Create a dummy audio file
        audio_path = str(tmp_path / "voice.wav")
        with open(audio_path, "w") as f:
            f.write("fake")

        handler = MockHandler({"name": "alice", "ref_audio": audio_path,
                                "language": "English"})
        handle_save_speaker(handler, mock_tts)
        assert handler._response_code == 201

        profiles = mock_tts.speaker_library.list_profiles()
        assert len(profiles) == 1
        assert profiles[0]["name"] == "alice"

    def test_save_speaker_missing_name(self, mock_tts):
        from babelvox.server import handle_save_speaker
        handler = MockHandler({"ref_audio": "voice.wav"})
        handle_save_speaker(handler, mock_tts)
        assert handler._response_code == 400
        assert "name" in handler.response_json["error"]

    def test_save_speaker_missing_ref_audio(self, mock_tts):
        from babelvox.server import handle_save_speaker
        handler = MockHandler({"name": "alice"})
        handle_save_speaker(handler, mock_tts)
        assert handler._response_code == 400
        assert "ref_audio" in handler.response_json["error"]

    def test_delete_speaker(self, mock_tts):
        from babelvox.server import handle_delete_speaker
        mock_tts.speaker_library.save(_make_profile("alice"))
        handler = MockHandler()
        handle_delete_speaker(handler, mock_tts, "alice")
        assert handler._response_code == 200

        handler2 = MockHandler()
        handle_delete_speaker(handler2, mock_tts, "alice")
        assert handler2._response_code == 404

    def test_speaker_in_tts_request(self, mock_tts):
        from babelvox.server import validate_tts_params
        kwargs, error = validate_tts_params({"text": "hello", "speaker": "alice"})
        assert error is None
        assert kwargs["speaker"] == "alice"

    def test_batch_synthesis(self, mock_tts):
        from babelvox.server import handle_batch_request
        handler = MockHandler({"items": [
            {"text": "Hello"},
            {"text": "World"},
        ]})
        handle_batch_request(handler, mock_tts)
        assert handler._response_code == 200
        results = handler.response_json["results"]
        assert len(results) == 2
        assert results[0]["index"] == 0
        assert "audio" in results[0]
        assert "duration_secs" in results[0]

    def test_batch_empty_items(self, mock_tts):
        from babelvox.server import handle_batch_request
        handler = MockHandler({"items": []})
        handle_batch_request(handler, mock_tts)
        assert handler._response_code == 400

    def test_batch_partial_failure(self, mock_tts):
        from babelvox.server import handle_batch_request
        # First call succeeds, second fails
        mock_tts.generate.side_effect = [
            (np.zeros(2400, dtype=np.float32), 24000),
            RuntimeError("boom"),
        ]
        handler = MockHandler({"items": [
            {"text": "Hello"},
            {"text": "World"},
        ]})
        handle_batch_request(handler, mock_tts)
        results = handler.response_json["results"]
        assert "audio" in results[0]
        assert "error" in results[1]
