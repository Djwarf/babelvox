"""BabelVox HTTP server for cross-language TTS integration."""
import io
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

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


def _make_handler(tts, cors_origin="*", audio_dir=None):
    class TTSHandler(BaseHTTPRequestHandler):
        def do_OPTIONS(self):
            self.send_response(204)
            for k, v in _cors_headers(cors_origin).items():
                self.send_header(k, v)
            self.end_headers()

        def do_GET(self):
            if self.path == "/health":
                _json_response(self, 200, {"status": "ok"}, cors_origin)
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

    text = body.get("text", "").strip()
    if not text:
        _json_response(handler, 400,
                       {"error": "missing required field: text"}, cors_origin)
        return

    kwargs = {"text": text}
    for key in ("language", "ref_audio", "ref_text"):
        if key in body:
            kwargs[key] = body[key]

    # Validate numeric parameters
    for key in ("max_new_tokens", "top_k"):
        if key in body:
            if not isinstance(body[key], int) or body[key] < 1:
                _json_response(handler, 400,
                               {"error": f"{key} must be a positive integer"},
                               cors_origin)
                return
            kwargs[key] = body[key]

    for key in ("temperature", "top_p", "repetition_penalty"):
        if key in body:
            if not isinstance(body[key], (int, float)) or body[key] <= 0:
                _json_response(handler, 400,
                               {"error": f"{key} must be a positive number"},
                               cors_origin)
                return
            kwargs[key] = body[key]

    # Path traversal protection for ref_audio
    if "ref_audio" in kwargs:
        ref_path = os.path.realpath(kwargs["ref_audio"])
        if audio_dir is not None:
            allowed = os.path.realpath(audio_dir)
            if not ref_path.startswith(allowed + os.sep):
                _json_response(handler, 400,
                               {"error": "ref_audio path not allowed"},
                               cors_origin)
                return
        if not os.path.isfile(ref_path):
            _json_response(handler, 400,
                           {"error": "ref_audio file not found"}, cors_origin)
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
    logger.info("  POST /tts   - synthesize speech")
    logger.info("  GET  /health - health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    server.server_close()
