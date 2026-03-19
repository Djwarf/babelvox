"""Tests for SSE streaming endpoint."""
import io
import json
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from babelvox.server import handle_sse_stream, validate_tts_params


class MockHandler:
    """Simulate BaseHTTPRequestHandler for SSE tests."""

    def __init__(self):
        self.headers = {}
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


@pytest.fixture
def mock_tts():
    tts = MagicMock()
    chunks = [(np.zeros(2400, dtype=np.float32), 24000)] * 3
    tts.generate_stream.return_value = iter(chunks)
    tts._inference_lock = threading.Lock()
    return tts


def test_sse_stream_returns_events(mock_tts):
    handler = MockHandler()
    handler.path = "/tts/stream?text=hello"
    handle_sse_stream(handler, mock_tts)
    body = handler.response_body.decode()
    assert handler._response_code == 200
    assert handler._response_headers["Content-Type"] == "text/event-stream"
    assert "event: start" in body
    assert "event: audio" in body
    assert "event: done" in body


def test_sse_start_event_contains_format(mock_tts):
    handler = MockHandler()
    handler.path = "/tts/stream?text=hello&format=wav_chunks"
    handle_sse_stream(handler, mock_tts)
    body = handler.response_body.decode()
    # Find the start event data line
    for line in body.split("\n"):
        if line.startswith("data:") and "sample_rate" in line:
            data = json.loads(line[5:].strip())
            assert data["sample_rate"] == 24000
            assert data["format"] == "wav_chunks"
            break


def test_sse_done_event_has_duration(mock_tts):
    handler = MockHandler()
    handler.path = "/tts/stream?text=hello"
    handle_sse_stream(handler, mock_tts)
    body = handler.response_body.decode()
    # Find the done event
    lines = body.split("\n")
    for i, line in enumerate(lines):
        if line == "event: done":
            data_line = lines[i + 1]
            data = json.loads(data_line[5:].strip())
            assert "total_duration_secs" in data
            assert data["total_duration_secs"] == 0.3  # 3 chunks * 2400/24000
            break


def test_sse_missing_text_returns_400(mock_tts):
    handler = MockHandler()
    handler.path = "/tts/stream?language=English"
    handle_sse_stream(handler, mock_tts)
    assert handler._response_code == 400
    assert "text" in handler.response_json["error"]


def test_sse_generation_error_sends_error_event(mock_tts):
    mock_tts.generate_stream.side_effect = RuntimeError("fail")
    handler = MockHandler()
    handler.path = "/tts/stream?text=hello"
    handle_sse_stream(handler, mock_tts)
    body = handler.response_body.decode()
    assert "event: error" in body
    assert "generation failed" in body


def test_sse_cors_headers(mock_tts):
    handler = MockHandler()
    handler.path = "/tts/stream?text=hello"
    handle_sse_stream(handler, mock_tts, cors_origin="http://example.com")
    assert handler._response_headers["Access-Control-Allow-Origin"] == "http://example.com"


def test_sse_audio_events_are_base64(mock_tts):
    import base64
    handler = MockHandler()
    handler.path = "/tts/stream?text=hello"
    handle_sse_stream(handler, mock_tts)
    body = handler.response_body.decode()
    audio_data = []
    lines = body.split("\n")
    for i, line in enumerate(lines):
        if line == "event: audio":
            data_line = lines[i + 1]
            b64 = data_line[5:].strip()
            # Should be valid base64
            decoded = base64.b64decode(b64)
            audio_data.append(decoded)
    assert len(audio_data) == 3  # 3 chunks


class TestValidateTtsParams:
    def test_valid_params(self):
        kwargs, error = validate_tts_params({"text": "hello"})
        assert error is None
        assert kwargs["text"] == "hello"

    def test_missing_text(self):
        kwargs, error = validate_tts_params({})
        assert kwargs is None
        assert "text" in error

    def test_empty_text(self):
        kwargs, error = validate_tts_params({"text": "  "})
        assert kwargs is None
        assert "text" in error

    def test_string_numeric_params(self):
        """Query string params come as strings — should be converted."""
        kwargs, error = validate_tts_params({
            "text": "hello", "temperature": "0.9", "max_new_tokens": "100",
        })
        assert error is None
        assert kwargs["temperature"] == 0.9
        assert kwargs["max_new_tokens"] == 100

    def test_invalid_string_numeric(self):
        kwargs, error = validate_tts_params({
            "text": "hello", "temperature": "not_a_number",
        })
        assert kwargs is None
        assert "temperature" in error

    def test_ssml_flag(self):
        kwargs, error = validate_tts_params({"text": "<speak>hi</speak>", "ssml": True})
        assert error is None
        assert kwargs["ssml"] is True
