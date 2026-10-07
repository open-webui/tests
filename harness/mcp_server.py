"""A Model Context Protocol tool server on a local port, built with the `mcp` SDK's own FastMCP.

`serving_mcp()` runs one over Streamable HTTP in a thread of the test process and yields its
URL, the one an admin enters for an MCP tool server connection. It offers a single tool, `echo`,
and stops again when the block ends. With `media=True` it also offers `snapshot`, answering
with `SNAPSHOT_PNG` as an image, and `chime`, answering with `CHIME_WAV` as audio; with
`failing=True`, `capsize`, which fails with `CAPSIZE_ERROR`; with `slow=True`, `ponder`, which
answers `PONDERED` after the number of seconds it is asked to wait. `tls=True` serves it over
HTTPS with a self-signed certificate. With `bearer_key` every request without that key gets a
plain 401, the way a server keyed by an API key answers (no OAuth metadata).
`mcp_connection(...)` is the admin's connection to it without auth; save it through
`TOOL_SERVERS` after `preserve(TOOL_SERVERS)`. Given FastMCP's `auth` settings and a
`token_verifier`, the SDK guards it the way a real OAuth-protected MCP server is guarded: a
request without a token the verifier accepts gets a 401 pointing at the protected-resource
metadata it also serves. `host` is the address it listens on and `name` the host its URL
carries (the address by default), so a test can reach it by a name.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Iterator

import uvicorn
from mcp.server.fastmcp import Audio, FastMCP, Image

from harness.instance import free_port
from harness.object_storage import self_signed_certificate

ECHO_DESCRIPTION = "Repeat the text back."
CAPSIZE_ERROR = "the boat capsized in the harbour"
PONDERED = "slept on it"
SNAPSHOT_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _silence_wav() -> bytes:
    recording = io.BytesIO()
    with wave.open(recording, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(b"\x00\x00" * 800)
    return recording.getvalue()


CHIME_WAV = _silence_wav()


def _requiring_bearer(app, bearer_key: str):
    expected = f"Bearer {bearer_key}".encode()

    async def guarded(scope, receive, send) -> None:
        if scope["type"] == "http" and dict(scope["headers"]).get(b"authorization") != expected:
            start = {"type": "http.response.start", "status": 401, "headers": []}
            await send(start)
            await send({"type": "http.response.body", "body": b"missing or wrong key"})
            return
        await app(scope, receive, send)

    return guarded


def _echo_server(
    media: bool = False, failing: bool = False, slow: bool = False, **auth: Any
) -> FastMCP:
    server = FastMCP("harness-mcp", log_level="WARNING", **auth)

    @server.tool(description=ECHO_DESCRIPTION)
    def echo(text: str) -> str:
        return text

    if media:

        @server.tool(description="Take a snapshot with the camera.")
        def snapshot() -> Image:
            return Image(data=SNAPSHOT_PNG, format="png")

        @server.tool(description="Play the door chime.")
        def chime() -> Audio:
            return Audio(data=CHIME_WAV, format="wav")

    if failing:

        @server.tool(description="Take the boat out.")
        def capsize() -> str:
            raise RuntimeError(CAPSIZE_ERROR)

    if slow:

        @server.tool(description="Think it over for a while.")
        async def ponder(seconds: float) -> str:
            await asyncio.sleep(seconds)
            return PONDERED

    return server


@contextlib.contextmanager
def serving_mcp(
    port: int | None = None,
    media: bool = False,
    failing: bool = False,
    slow: bool = False,
    tls: bool = False,
    host: str = "127.0.0.1",
    name: str | None = None,
    bearer_key: str | None = None,
    **auth: Any,
) -> Iterator[str]:
    """Serve the echo server on `port` (a free one by default); `auth` goes to FastMCP."""
    port = port or free_port()
    app = _echo_server(media, failing, slow, **auth).streamable_http_app()
    if bearer_key:
        app = _requiring_bearer(app, bearer_key)
    with tempfile.TemporaryDirectory(prefix="owui-mcp-") as certificates:
        certificate = {}
        if tls:
            certificate_path, key_path = self_signed_certificate(Path(certificates))
            certificate = {"ssl_certfile": str(certificate_path), "ssl_keyfile": str(key_path)}
        # log_config=None keeps uvicorn from reconfiguring the test process's logging
        config = uvicorn.Config(
            app, host=host, port=port, log_config=None, log_level="warning", **certificate
        )
        runner = uvicorn.Server(config)
        thread = threading.Thread(target=runner.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 30
        while not runner.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("the MCP server did not start")
            time.sleep(0.05)
        try:
            url_host = name or (f"[{host}]" if ":" in host else host)
            yield f"{'https' if tls else 'http'}://{url_host}:{port}/mcp"
        finally:
            runner.should_exit = True
            thread.join(timeout=10)


TOOL_SERVERS = ("/api/v1/configs/tool_servers", "/api/v1/configs/tool_servers")


def mcp_connection(url: str, server_id: str, access_grants: list[dict]) -> dict:
    """An MCP tool server connection without auth, the way the admin panel saves one."""
    return {
        "url": url,
        "path": "",
        "type": "mcp",
        "auth_type": "none",
        "key": "",
        "config": {"enable": True, "access_grants": access_grants},
        "info": {"id": server_id, "name": server_id},
    }
