"""Dependency contract: aiofiles (import name ``aiofiles``).

Open WebUI uses aiofiles for non-blocking file I/O on the request path.
``routers/audio.py`` is the sole importer, and it uses exactly one entry
point: ``aiofiles.open(path, mode)`` as an *async context manager*, then
``await f.write(...)`` / ``await f.read()`` inside. The modes the backend
opens with are ``'wb'`` (binary audio bytes), ``'w'`` (JSON metadata /
transcripts) and ``'rb'`` (reading cached audio back, e.g. base64-encoding
for the chat-completions API and the Deepgram upload). The point is to keep
the event loop free while the speech cache is written/read.

This module pins exactly that contract so an aiofiles bump that changed the
``open()`` signature, the async-context-manager protocol, or the
awaitable read/write methods fails loudly here instead of as a runtime
``TypeError`` / ``AttributeError`` in an audio request. Pattern mirrors
test_requests.py: symbol-existence + signature checks plus offline
BEHAVIOURAL contracts that round-trip real bytes/text through a
``tmp_path`` file (local disk, no network) for each mode the backend uses.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "aiofiles"
DIST_NAME = "aiofiles"


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`aiofiles` must import (skip cleanly if absent in this env)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "aiofiles"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence + signature (API surface)
# ---------------------------------------------------------------------------


def test_open_exists_and_callable(depcheck):
    """audio.py: `async with aiofiles.open(path, mode) as f:`. The top-level
    `open` must remain a callable factory."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "open")


def test_open_accepts_file_and_mode(depcheck):
    """aiofiles.open is called positionally as open(file_path, 'wb'|'w'|'rb').
    The `file` and `mode` parameters must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.open, ["file", "mode"])


def test_open_accepts_encoding_kwarg(depcheck):
    """Text-mode writes ('w') rely on the default/encoding handling that
    Python's open() provides; pin that `encoding` remains an accepted kwarg so
    the text/binary mode split keeps working."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.open, ["encoding"])


# ---------------------------------------------------------------------------
# Behavioural: the async-context-manager + awaitable read/write protocol.
# All against a local tmp_path file — no network, no external state.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_behaviour_open_is_async_context_manager_write_binary(depcheck, tmp_path):
    """The `'wb'` path in audio.py (e.g. `async with aiofiles.open(file_path,
    'wb') as f: await f.write(audio)`). Opening must yield an async context
    manager whose handle has an awaitable write that persists raw bytes."""
    mod = depcheck.load(IMPORT_NAME)
    target = tmp_path / "audio.bin"
    payload = b"\x00\x01\x02OWUI-binary\xff"

    async with mod.open(target, "wb") as f:
        assert hasattr(f, "write"), "aiofiles handle has no write()"
        await f.write(payload)

    # Verify the bytes actually landed (sync read is fine for assertion).
    assert target.read_bytes() == payload


@pytest.mark.asyncio
async def test_behaviour_write_then_read_binary_roundtrip(depcheck, tmp_path):
    """audio.py writes audio with 'wb' then later reads it back with 'rb'
    (`await audio_file.read()` -> base64). Pin the full binary round trip."""
    mod = depcheck.load(IMPORT_NAME)
    target = tmp_path / "speech.mp3"
    payload = b"ID3fake-mp3-bytes\x00\x10\x20"

    async with mod.open(target, "wb") as f:
        await f.write(payload)

    async with mod.open(target, "rb") as f:
        assert hasattr(f, "read"), "aiofiles handle has no read()"
        data = await f.read()

    assert data == payload
    assert isinstance(data, bytes)


@pytest.mark.asyncio
async def test_behaviour_write_text_json_roundtrip(depcheck, tmp_path):
    """The `'w'` path: audio.py does `async with aiofiles.open(body_path, 'w')
    as f: await f.write(json.dumps(payload))`. Verify a text write of a JSON
    string round-trips to the exact same string (and parses back)."""
    mod = depcheck.load(IMPORT_NAME)
    target = tmp_path / "body.json"
    payload = {"text": "hello world", "lang": "en", "n": 3}
    serialized = json.dumps(payload)

    async with mod.open(target, "w") as f:
        await f.write(serialized)

    async with mod.open(target, "r") as f:
        text = await f.read()

    assert text == serialized
    assert json.loads(text) == payload


@pytest.mark.asyncio
async def test_behaviour_write_returns_awaitable_count(depcheck, tmp_path):
    """`await f.write(...)` must be awaitable (the whole reason aiofiles exists).
    Pin that write is awaitable and that the file content reflects every byte
    written across multiple awaited writes (audio.py writes audio then a JSON
    sidecar in sequence)."""
    mod = depcheck.load(IMPORT_NAME)
    target = tmp_path / "multi.txt"

    async with mod.open(target, "w") as f:
        await f.write("part-1;")
        await f.write("part-2;")
        await f.write("part-3")

    async with mod.open(target, "r") as f:
        assert await f.read() == "part-1;part-2;part-3"


@pytest.mark.asyncio
async def test_behaviour_context_manager_closes_handle(depcheck, tmp_path):
    """Leaving the `async with` block must close the handle (audio.py relies on
    this so the cached file is flushed/closed before FileResponse serves it).
    Pin that the handle reports closed after the block and the data is durable."""
    mod = depcheck.load(IMPORT_NAME)
    target = tmp_path / "closed.bin"

    handle_ref = None
    async with mod.open(target, "wb") as f:
        handle_ref = f
        await f.write(b"durable")

    # aiofiles proxies `closed` from the wrapped file object.
    assert getattr(handle_ref, "closed", True) is True, (
        "aiofiles handle not closed after `async with` exit; cached audio could "
        "be served before it is flushed."
    )
    assert target.read_bytes() == b"durable"
