"""Regression: the 0.11.0 observability and startup fixes that no request can observe.

* Pure-ASGI middlewares (#26924, issue #26922): `SecurityHeadersMiddleware` was a
  `BaseHTTPMiddleware`, which re-buffers every response through an anyio memory stream and cut
  streamed audio short. Every HTTP middleware is pure ASGI now.
* Values in error logs (commit 6aebfd8, #26814): loguru's `diagnose` was left at its default, so
  every logged traceback printed the local variables beside it (API keys, message content).
  `start_logger` now passes `diagnose=LOGURU_DIAGNOSE`, off by default.
* Transcription chunk order (#27417, issue #27143): `transcribe` collected chunk results with
  `asyncio.as_completed`, so a long recording came back in completion order. Fixed by
  `asyncio.gather`. Splitting needs ffmpeg, so the order is pinned here.
* Licensed startup (commits 8f77533, 0c7ddbd): the lifespan fetched the license inline, blocking
  readiness on the license server, and an unreachable license host propagated out of `handler`
  instead of falling through to the next one.

The headers, audit log, webhook, provider log and feedback event fixes of the same release are
pinned over HTTP in integration/config/test_observability_middleware.py.

Discriminates: passes on bbfa876af; registering an `@app.middleware('http')` fails the middleware
audit, `diagnose` left on prints the probe's secret, `as_completed` joins the chunks out of order,
and an inline license fetch or a raising `handler` fails the license tests.
"""

from __future__ import annotations

import ast
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression

SECRET = "sk-probe-secret-7f3a"

# The secret arrives through argv, so it never appears in a source line of the traceback.
LOGGED_TRACEBACK_PROBE = """
import logging, sys

sys.path.insert(0, sys.argv[1])
from open_webui.utils.logger import start_logger


def call_provider(api_key):
    raise RuntimeError("the provider refused the request")


start_logger()
secret = sys.argv[2]
try:
    call_provider(secret)
except RuntimeError:
    logging.getLogger("open_webui.probe").exception("request failed")
"""


# pure-ASGI middleware stack


def test_no_registered_middleware_rebuffers_responses(owui_module):
    from starlette.middleware.base import BaseHTTPMiddleware

    registered = owui_module("open_webui.main").app.user_middleware
    assert registered, "open_webui.main.app registers no middleware; retarget this audit"

    rebuffering = [
        str(middleware)
        for middleware in registered
        if isinstance(middleware.cls, type) and issubclass(middleware.cls, BaseHTTPMiddleware)
    ]
    assert rebuffering == [], (
        "a BaseHTTPMiddleware re-buffers every response through a memory stream, which cut "
        f"streamed replies short (#26922): {rebuffering}"
    )


# tracebacks in the server log


def test_a_logged_traceback_does_not_print_local_values(open_webui_backend, tmp_path):
    # run from a file: loguru annotates only frames whose source it can read
    probe = tmp_path / "probe.py"
    probe.write_text(LOGGED_TRACEBACK_PROBE, encoding="utf-8")
    env = {name: value for name, value in os.environ.items() if name != "LOGURU_DIAGNOSE"}
    probed = subprocess.run(
        [sys.executable, str(probe), str(open_webui_backend), SECRET],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    logged = probed.stdout + probed.stderr

    assert "the provider refused the request" in logged, f"no traceback was logged:\n{logged}"
    assert SECRET not in logged, (
        "a logged traceback printed a local variable's value next to the frame, which is how "
        f"API keys reached the server log (#26814):\n{logged[-2000:]}"
    )


# transcription chunk order


@pytest.fixture
def audio_module(owui_module):
    return owui_module("open_webui.routers.audio")


def _transcribe(audio_module, monkeypatch, tmp_path: Path, chunk_count: int, handler) -> dict:
    """Transcribe a recording that splits into `chunk_count` chunks, each answered by `handler`."""
    recording = tmp_path / "recording.mp3"
    recording.write_bytes(b"not really audio")
    chunks = [tmp_path / f"chunk_{index}.mp3" for index in range(chunk_count)]
    for chunk in chunks:
        chunk.write_bytes(b"chunk")

    async def transcribe_chunk(request, file_path, metadata, user=None):
        return await handler(chunks.index(Path(file_path)))

    # ffprobe, ffmpeg and the speech-to-text provider are the I/O this stands in for
    monkeypatch.setattr(audio_module, "BYPASS_PYDUB_PREPROCESSING", False)
    monkeypatch.setattr(audio_module, "is_audio_conversion_required", lambda path: False)
    monkeypatch.setattr(
        audio_module, "split_audio", lambda path, max_bytes: [str(c) for c in chunks]
    )
    monkeypatch.setattr(audio_module, "transcription_handler", transcribe_chunk)

    transcribing = audio_module.transcribe(request=SimpleNamespace(), file_path=str(recording))
    return asyncio.run(asyncio.wait_for(transcribing, timeout=15))


def test_a_long_recording_is_transcribed_in_spoken_order(audio_module, monkeypatch, tmp_path):
    last_chunk_done = asyncio.Event()

    async def last_chunk_finishes_first(index: int) -> dict:
        if index == 2:
            last_chunk_done.set()
        else:
            await last_chunk_done.wait()
        return {"text": f"part{index}"}

    result = _transcribe(audio_module, monkeypatch, tmp_path, 3, last_chunk_finishes_first)

    assert result["text"] == "part0 part1 part2", (
        f"the chunks were joined in completion order, not spoken order (#27143): {result['text']}"
    )


def test_a_failing_chunk_fails_the_transcription(audio_module, monkeypatch, tmp_path):
    from fastapi import HTTPException

    async def provider_down(index: int) -> dict:
        raise RuntimeError("provider down")

    with pytest.raises(HTTPException) as refused:
        _transcribe(audio_module, monkeypatch, tmp_path, 2, provider_down)
    assert refused.value.status_code == 500


# licensed startup


def _lifespan(open_webui_backend) -> ast.AsyncFunctionDef:
    main = ast.parse((open_webui_backend / "open_webui" / "main.py").read_text(encoding="utf-8"))
    found = [
        node
        for node in main.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    ]
    assert found, "main.py has no lifespan function; retarget the license audit"
    return found[0]


def _call_name(call: ast.Call) -> str:
    function = call.func
    return function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")


def test_startup_does_not_wait_on_the_license_server(open_webui_backend):
    lifespan = _lifespan(open_webui_backend)
    parents = {child: node for node in ast.walk(lifespan) for child in ast.iter_child_nodes(node)}

    def enclosing(node: ast.AST) -> list[ast.AST]:
        chain = []
        while node in parents:
            node = parents[node]
            chain.append(node)
        return chain

    def enclosing_calls(node: ast.AST) -> set[str]:
        return {_call_name(outer) for outer in enclosing(node) if isinstance(outer, ast.Call)}

    fetches = [
        node
        for node in ast.walk(lifespan)
        if isinstance(node, ast.Name) and node.id == "get_license_data"
    ]
    assert fetches, "the lifespan no longer fetches the license; retarget this audit"
    inline = [fetch.lineno for fetch in fetches if "create_task" not in enclosing_calls(fetch)]
    assert inline == [], f"main.py:{inline} fetches the license inline, blocking readiness on it"

    tasks = {
        target.id
        for fetch in fetches
        for outer in enclosing(fetch)
        if isinstance(outer, ast.Assign)
        for target in outer.targets
        if isinstance(target, ast.Name)
    }
    unbounded = [
        node.lineno
        for node in ast.walk(lifespan)
        if isinstance(node, ast.Name)
        and node.id in tasks
        and any(isinstance(outer, ast.Await) for outer in enclosing(node))
        and "wait_for" not in enclosing_calls(node)
    ]
    assert unbounded == [], f"main.py:{unbounded} waits on the license fetch without a timeout"


def test_an_unreachable_license_host_falls_through_to_the_next(owui_module, monkeypatch):
    auth = owui_module("open_webui.utils.auth")
    attempted: list[str] = []

    def unreachable(url, **kwargs):
        attempted.append(url)
        raise OSError("name or service not known")

    monkeypatch.setattr(auth.requests, "post", unreachable)

    assert (
        auth.get_license_data(SimpleNamespace(state=SimpleNamespace()), "test-license-key") is False
    )
    assert len(set(attempted)) == 2, f"the second license host was never tried: {attempted}"
