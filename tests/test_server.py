"""Essential HTTP API contract tests for babelvox server."""
import io
import json
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from babelvox.server import handle_tts_request


class MockHandler:
    """Simulate BaseHTTPRequestHandler for unit tests."""

    def __init__(self, body=None):
        if body is None:
            body = b""
        elif isinstance(body, dict):
            body = json.dumps(body).encode()
        self.headers = {"Content-Length": str(len(body))}
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
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
    tts.generate.return_value = (np.zeros(2400, dtype=np.float32), 24000)
    return tts


def test_missing_text_returns_400(mock_tts):
    handler = MockHandler({"language": "English"})
    handle_tts_request(handler, mock_tts)
    assert handler._response_code == 400
    assert "text" in handler.response_json["error"]


def test_invalid_json_returns_400(mock_tts):
    handler = MockHandler(b"not json{{{")
    handle_tts_request(handler, mock_tts)
    assert handler._response_code == 400
    assert handler.response_json["error"] == "invalid JSON"


def test_valid_request_returns_wav(mock_tts):
    handler = MockHandler({"text": "hello"})
    handle_tts_request(handler, mock_tts)
    assert handler._response_code == 200
    assert handler._response_headers["Content-Type"] == "audio/wav"
    mock_tts.generate.assert_called_once()


def test_generation_error_returns_500_without_leak(mock_tts):
    mock_tts.generate.side_effect = RuntimeError("secret internal error details")
    handler = MockHandler({"text": "hello"})
    handle_tts_request(handler, mock_tts)
    assert handler._response_code == 500
    assert handler.response_json["error"] == "internal server error"
    assert "secret" not in handler.response_json["error"]


def test_oversized_body_returns_413(mock_tts):
    handler = MockHandler(b"x")
    handler.headers["Content-Length"] = str(2_000_000)
    handle_tts_request(handler, mock_tts)
    assert handler._response_code == 413


def test_ref_audio_path_traversal_blocked(mock_tts):
    with tempfile.TemporaryDirectory() as allowed_dir:
        handler = MockHandler({"text": "hello", "ref_audio": "/etc/passwd"})
        handle_tts_request(handler, mock_tts, audio_dir=allowed_dir)
        assert handler._response_code == 400
        assert "not allowed" in handler.response_json["error"]


def test_ref_audio_missing_file_returns_400(mock_tts):
    handler = MockHandler({"text": "hello", "ref_audio": "/nonexistent/file.wav"})
    handle_tts_request(handler, mock_tts)
    assert handler._response_code == 400
    assert "not found" in handler.response_json["error"]


def test_invalid_numeric_params_returns_400(mock_tts):
    handler = MockHandler({"text": "hello", "temperature": -1})
    handle_tts_request(handler, mock_tts)
    assert handler._response_code == 400

    handler = MockHandler({"text": "hello", "max_new_tokens": 0})
    handle_tts_request(handler, mock_tts)
    assert handler._response_code == 400
