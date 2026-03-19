"""BabelVox HTTP server for cross-language TTS integration."""
import base64
import io
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf

logger = logging.getLogger("babelvox")

MAX_REQUEST_BYTES = 1_000_000  # 1 MB


def _cors_headers(origin="*"):
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


def _json_response(handler, code, obj, cors_origin="*"):
    body = json.dumps(obj).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    for k, v in _cors_headers(cors_origin).items():
        handler.send_header(k, v)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _wav_response(handler, wav, sr, cors_origin="*"):
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    data = buf.getvalue()
    handler.send_response(200)
    handler.send_header("Content-Type", "audio/wav")
    for k, v in _cors_headers(cors_origin).items():
        handler.send_header(k, v)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def encode_chunk(wav, sr, fmt="pcm_s16le"):
    """Encode a float32 waveform chunk to bytes.

    Formats:
        pcm_s16le: raw 16-bit signed little-endian PCM
        wav_chunks: complete WAV file with header
    """
    if fmt == "pcm_s16le":
        return (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue()


def validate_tts_params(body, audio_dir=None):
    """Validate TTS request parameters.

    Returns:
        tuple: (kwargs_dict, None) on success, or (None, error_message) on failure.
    """
    text = body.get("text", "")
    text = text.strip() if isinstance(text, str) else ""
    if not text:
        return None, "missing required field: text"

    kwargs = {"text": text}
    for key in ("language", "ref_audio", "ref_text"):
        if key in body:
            kwargs[key] = body[key]

    if body.get("ssml"):
        kwargs["ssml"] = True

    # Validate numeric parameters
    for key in ("max_new_tokens", "top_k"):
        if key in body:
            val = body[key]
            if isinstance(val, str):
                try:
                    val = int(val)
                except ValueError:
                    return None, f"{key} must be a positive integer"
            if not isinstance(val, int) or val < 1:
                return None, f"{key} must be a positive integer"
            kwargs[key] = val

    for key in ("temperature", "top_p", "repetition_penalty"):
        if key in body:
            val = body[key]
            if isinstance(val, str):
                try:
                    val = float(val)
                except ValueError:
                    return None, f"{key} must be a positive number"
            if not isinstance(val, (int, float)) or val <= 0:
                return None, f"{key} must be a positive number"
            kwargs[key] = val

    # Path traversal protection for ref_audio
    if "ref_audio" in kwargs:
        ref_path = os.path.realpath(kwargs["ref_audio"])
        if audio_dir is not None:
            allowed = os.path.realpath(audio_dir)
            if not ref_path.startswith(allowed + os.sep):
                return None, "ref_audio path not allowed"
        if not os.path.isfile(ref_path):
            return None, "ref_audio file not found"

    return kwargs, None


# ── SSE helpers ───────────────────────────────────────────────────────

def _sse_write(handler, event, data):
    """Write a single SSE event to the handler's output stream."""
    if isinstance(data, dict):
        data = json.dumps(data)
    handler.wfile.write(f"event: {event}\ndata: {data}\n\n".encode())
    handler.wfile.flush()


def handle_sse_stream(handler, tts, cors_origin="*", audio_dir=None):
    """Handle a GET /tts/stream SSE request."""
    parsed = urlparse(handler.path)
    params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

    kwargs, error = validate_tts_params(params, audio_dir=audio_dir)
    if error:
        _json_response(handler, 400, {"error": error}, cors_origin)
        return

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    for k, v in _cors_headers(cors_origin).items():
        handler.send_header(k, v)
    handler.end_headers()

    fmt = params.get("format", "pcm_s16le")
    _sse_write(handler, "start", {"sample_rate": 24000, "format": fmt})

    total_samples = 0
    try:
        for wav_chunk, sr in tts.generate_stream(**kwargs):
            audio_bytes = encode_chunk(wav_chunk, sr, fmt)
            _sse_write(handler, "audio", base64.b64encode(audio_bytes).decode())
            total_samples += len(wav_chunk)
    except Exception:
        logger.exception("SSE stream failed")
        _sse_write(handler, "error", {"message": "generation failed"})
        return

    _sse_write(handler, "done",
               {"total_duration_secs": round(total_samples / 24000, 2)})


# ── HTTP handler ──────────────────────────────────────────────────────

def _make_handler(tts, cors_origin="*", audio_dir=None):
    class TTSHandler(BaseHTTPRequestHandler):
        def do_OPTIONS(self):
            self.send_response(204)
            for k, v in _cors_headers(cors_origin).items():
                self.send_header(k, v)
            self.end_headers()

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                _json_response(self, 200, {"status": "ok"}, cors_origin)
            elif parsed.path == "/tts/stream":
                handle_sse_stream(self, tts, cors_origin=cors_origin,
                                  audio_dir=audio_dir)
            else:
                _json_response(self, 404, {"error": "not found"}, cors_origin)

        def do_POST(self):
            if self.path != "/tts":
                _json_response(self, 404, {"error": "not found"}, cors_origin)
                return
            handle_tts_request(self, tts, cors_origin=cors_origin,
                               audio_dir=audio_dir)

        def log_message(self, format, *args):
            logger.info(args[0])

    return TTSHandler


def handle_tts_request(handler, tts, cors_origin="*", audio_dir=None):
    """Parse JSON request body, call tts.generate(), return WAV response."""
    length = int(handler.headers.get("Content-Length", 0))
    if length > MAX_REQUEST_BYTES:
        _json_response(handler, 413, {"error": "request body too large"},
                       cors_origin)
        return

    try:
        body = json.loads(handler.rfile.read(length)) if length else {}
    except (json.JSONDecodeError, ValueError):
        _json_response(handler, 400, {"error": "invalid JSON"}, cors_origin)
        return

    kwargs, error = validate_tts_params(body, audio_dir=audio_dir)
    if error:
        _json_response(handler, 400, {"error": error}, cors_origin)
        return

    try:
        wav, sr = tts.generate(**kwargs)
    except Exception:
        logger.exception("TTS generation failed")
        _json_response(handler, 500, {"error": "internal server error"},
                       cors_origin)
        return

    _wav_response(handler, wav, sr, cors_origin)


def serve(tts, host="0.0.0.0", port=8765, cors_origin="*", audio_dir=None):
    """Start HTTP server wrapping a BabelVox instance."""
    handler_class = _make_handler(tts, cors_origin=cors_origin,
                                  audio_dir=audio_dir)
    server = HTTPServer((host, port), handler_class)
    logger.info("BabelVox server listening on http://%s:%d", host, port)
    logger.info("  POST /tts        - synthesize speech")
    logger.info("  GET  /tts/stream - streaming SSE")
    logger.info("  GET  /health     - health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    server.server_close()
