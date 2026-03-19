"""Tests for WebSocket streaming server."""
import asyncio
import json
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
import websockets

from babelvox.ws_server import ws_handler


@pytest.fixture
def mock_tts():
    tts = MagicMock()
    chunks = [(np.zeros(2400, dtype=np.float32), 24000)] * 3
    tts.generate_stream.return_value = iter(chunks)
    tts._inference_lock = threading.Lock()
    return tts


@pytest.fixture
def mock_tts_factory():
    """Factory that creates a fresh mock_tts each time generate_stream is called."""
    tts = MagicMock()
    tts._inference_lock = threading.Lock()

    def _make_iter(**kwargs):
        return iter([(np.zeros(2400, dtype=np.float32), 24000)] * 3)

    tts.generate_stream.side_effect = _make_iter
    return tts


async def _start_server(tts, port):
    """Start a test WebSocket server and return the server object."""
    import functools
    handler = functools.partial(ws_handler, tts=tts)
    server = await websockets.serve(handler, "127.0.0.1", port)
    return server


def _find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.asyncio
async def test_basic_stream(mock_tts_factory):
    port = _find_free_port()
    server = await _start_server(mock_tts_factory, port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"text": "hello"}))

            messages = []
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                if isinstance(msg, str):
                    data = json.loads(msg)
                    messages.append(data)
                    if data.get("event") in ("done", "error"):
                        break
                else:
                    messages.append(("binary", len(msg)))

            # Should have: start, 3 binary chunks, done
            events = [m.get("event") if isinstance(m, dict) else m[0] for m in messages]
            assert events[0] == "start"
            assert events[-1] == "done"
            binary_count = sum(1 for e in events if e == "binary")
            assert binary_count == 3
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_missing_text_returns_error(mock_tts):
    port = _find_free_port()
    server = await _start_server(mock_tts, port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"language": "English"}))
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(msg)
            assert data["event"] == "error"
            assert "text" in data["message"]
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_invalid_json_returns_error(mock_tts):
    port = _find_free_port()
    server = await _start_server(mock_tts, port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send("not valid json{{{")
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(msg)
            assert data["event"] == "error"
            assert "JSON" in data["message"]
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_pcm_format(mock_tts_factory):
    port = _find_free_port()
    server = await _start_server(mock_tts_factory, port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"text": "hello", "format": "pcm_s16le"}))

            # First message should be start event
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(msg)
            assert data["event"] == "start"
            assert data["format"] == "pcm_s16le"

            # Next should be binary PCM
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            assert isinstance(msg, bytes)
            # PCM s16le: 2400 samples * 2 bytes = 4800 bytes
            assert len(msg) == 4800
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_wav_format(mock_tts_factory):
    port = _find_free_port()
    server = await _start_server(mock_tts_factory, port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"text": "hello", "format": "wav_chunks"}))

            # Skip start event
            await asyncio.wait_for(ws.recv(), timeout=5)

            # Binary message should be a WAV file (starts with RIFF)
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            assert isinstance(msg, bytes)
            assert msg[:4] == b"RIFF"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_done_event_has_duration(mock_tts_factory):
    port = _find_free_port()
    server = await _start_server(mock_tts_factory, port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"text": "hello"}))

            done_data = None
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                if isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("event") == "done":
                        done_data = data
                        break

            assert done_data is not None
            assert done_data["total_duration_secs"] == 0.3  # 3 * 2400/24000
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_generation_error_sends_error_event(mock_tts):
    mock_tts.generate_stream.side_effect = RuntimeError("boom")
    port = _find_free_port()
    server = await _start_server(mock_tts, port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"text": "hello"}))

            # Should get start then error
            messages = []
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                if isinstance(msg, str):
                    data = json.loads(msg)
                    messages.append(data)
                    if data.get("event") in ("done", "error"):
                        break

            error_msgs = [m for m in messages if m.get("event") == "error"]
            assert len(error_msgs) == 1
            assert "generation failed" in error_msgs[0]["message"]
    finally:
        server.close()
        await server.wait_closed()
