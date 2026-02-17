"""BabelVox HTTP server for cross-language TTS integration."""
import io
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

import soundfile as sf


def _cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


def _json_response(handler, code, obj):
    body = json.dumps(obj).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    for k, v in _cors_headers().items():
        handler.send_header(k, v)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _wav_response(handler, wav, sr):
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    data = buf.getvalue()
    handler.send_response(200)
    handler.send_header("Content-Type", "audio/wav")
    for k, v in _cors_headers().items():
        handler.send_header(k, v)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _make_handler(tts):
    class TTSHandler(BaseHTTPRequestHandler):
        def do_OPTIONS(self):
            self.send_response(204)
            for k, v in _cors_headers().items():
                self.send_header(k, v)
            self.end_headers()

        def do_GET(self):
            if self.path == "/health":
                _json_response(self, 200, {"status": "ok"})
            else:
                _json_response(self, 404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/tts":
                _json_response(self, 404, {"error": "not found"})
                return
            handle_tts_request(self, tts)

        def log_message(self, format, *args):
            print(f"[babelvox] {args[0]}")

    return TTSHandler


def handle_tts_request(handler, tts):
    """Parse JSON request body, call tts.generate(), return WAV response."""
    try:
        length = int(handler.headers.get("Content-Length", 0))
        body = json.loads(handler.rfile.read(length)) if length else {}
    except (json.JSONDecodeError, ValueError):
        _json_response(handler, 400, {"error": "invalid JSON"})
        return

    text = body.get("text", "").strip()
    if not text:
        _json_response(handler, 400, {"error": "missing required field: text"})
        return

    kwargs = {"text": text}
    for key in ("language", "ref_audio", "max_new_tokens",
                "temperature", "top_k", "top_p", "repetition_penalty"):
        if key in body:
            kwargs[key] = body[key]

    try:
        wav, sr = tts.generate(**kwargs)
    except Exception as e:
        _json_response(handler, 500, {"error": str(e)})
        return

    _wav_response(handler, wav, sr)


def serve(tts, host="0.0.0.0", port=8765):
    """Start HTTP server wrapping a BabelVox instance."""
    handler_class = _make_handler(tts)
    server = HTTPServer((host, port), handler_class)
    print(f"BabelVox server listening on http://{host}:{port}")
    print(f"  POST /tts   - synthesize speech")
    print(f"  GET  /health - health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    server.server_close()
