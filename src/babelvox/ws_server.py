"""BabelVox WebSocket server for real-time audio streaming.

Requires the 'ws' extra: pip install babelvox[ws]

Protocol:
    Client sends JSON text messages with TTS parameters.
    Server responds with binary audio chunks + JSON control messages.
"""
import asyncio
import functools
import json
import logging
import threading

from babelvox.server import encode_chunk, validate_tts_params

logger = logging.getLogger("babelvox")


async def _stream_audio(websocket, tts, kwargs, cancel_event):
    """Run generate_stream() and send audio chunks over WebSocket."""
    fmt = kwargs.pop("format", "pcm_s16le")
    await websocket.send(json.dumps({
        "event": "start", "sample_rate": 24000, "format": fmt,
    }))

    loop = asyncio.get_event_loop()
    total_samples = 0

    def _next_chunk(gen):
        """Get next chunk from generator (runs in thread pool)."""
        try:
            return next(gen)
        except StopIteration:
            return None

    try:
        gen = tts.generate_stream(
            **kwargs, cancel_event=cancel_event, split_on_silence=True,
            min_chunk_frames=6, max_chunk_frames=48, crossfade_samples=1200)
        while True:
            result = await loop.run_in_executor(None, _next_chunk, gen)
            if result is None:
                break
            wav_chunk, sr = result
            total_samples += len(wav_chunk)
            audio_bytes = encode_chunk(wav_chunk, sr, fmt)
            await websocket.send(audio_bytes)
    except Exception:
        logger.exception("WebSocket stream failed")
        try:  # noqa: SIM105
            await websocket.send(json.dumps({
                "event": "error", "message": "generation failed",
            }))
        except Exception:
            pass  # Connection may already be closed
        return

    duration = round(total_samples / 24000, 2)
    await websocket.send(json.dumps({
        "event": "done", "total_duration_secs": duration,
    }))


async def ws_handler(websocket, tts, audio_dir=None):
    """Handle a single WebSocket connection."""
    cancel_event = threading.Event()
    logger.info("WebSocket client connected: %s", websocket.remote_address)

    try:
        async for message in websocket:
            if not isinstance(message, str):
                continue

            try:
                req = json.loads(message)
            except (json.JSONDecodeError, ValueError):
                await websocket.send(json.dumps({
                    "event": "error", "message": "invalid JSON",
                }))
                continue

            # Cancel support
            if req.get("event") == "cancel":
                cancel_event.set()
                logger.debug("Cancel requested by client")
                continue

            # New TTS request — reset cancel flag
            cancel_event.clear()

            kwargs, error = validate_tts_params(req, audio_dir=audio_dir)
            if error:
                await websocket.send(json.dumps({
                    "event": "error", "message": error,
                }))
                continue

            # Extract format (not a generate_stream param)
            if "format" in req:
                kwargs["format"] = req["format"]

            await _stream_audio(websocket, tts, kwargs, cancel_event)
    except Exception:
        logger.debug("WebSocket connection closed")


def ws_serve(tts, host="0.0.0.0", port=8766, audio_dir=None):
    """Start WebSocket server (blocking). Intended to run in a thread.

    Requires the 'ws' extra: pip install babelvox[ws]
    """
    try:
        import websockets  # noqa: F811
    except ImportError:
        raise ImportError(
            "WebSocket support requires the 'ws' extra: "
            "pip install babelvox[ws]"
        ) from None

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    handler = functools.partial(ws_handler, tts=tts, audio_dir=audio_dir)

    async def _start():
        return await websockets.serve(handler, host, port)

    server = loop.run_until_complete(_start())
    logger.info("WebSocket server listening on ws://%s:%d", host, port)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        loop.run_until_complete(server.wait_closed())
        loop.close()
