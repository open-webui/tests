"""A Model Context Protocol tool server on a local port, built with the `mcp` SDK's own MCPServer.

`serving_mcp()` runs one over Streamable HTTP in a thread of the test process and yields its
URL, the one an admin enters for an MCP tool server connection. It offers a single tool, `echo`,
and stops again when the block ends. With `media=True` it also offers `snapshot`, answering
with `SNAPSHOT_PNG` as an image, and `chime`, answering with `CHIME_WAV` as audio; with
`failing=True`, `capsize`, which fails with `CAPSIZE_ERROR`; with `slow=True`, `ponder`, which
answers `PONDERED` after the number of seconds it is asked to wait; with `structured=True`,
`forecast`, answering `FORECAST_SUMMARY` as text and `FORECAST` as structured content,
`conditions`, answering `CONDITIONS` from a typed return (the same JSON as text and structured
content), `headline`, answering `HEADLINE` as plain text alone, `berths`, answering the list
`BERTHS` from a typed return (one text block per item and the list as structured content), and
`harbours`, answering the list `HARBOURS` the way the standalone `fastmcp` package does (the whole
list as one JSON text block and as structured content under `result`); with `composed=True`,
`find_page`, whose input schema has no top-level properties and offers its arguments in `anyOf`
branches (a page `id`, or a `path` with its `locale`), answering with the arguments it received
as JSON. `tls=True` serves it over
HTTPS with a self-signed certificate. With `bearer_key` every request without that key gets a
plain 401, the way a server keyed by an API key answers (no OAuth metadata). With `moved_from` a
request to that path gets a 307 to `/mcp` on the same server, the way a server that moved within
its own address answers.
`mcp_connection(...)` is the admin's connection to it without auth; save it through
`TOOL_SERVERS` after `preserve(TOOL_SERVERS)`. Given MCPServer's `auth` settings and a
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
import json
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Iterator

import uvicorn
from mcp.server.mcpserver import Audio, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel

from harness.instance import free_port
from harness.object_storage import self_signed_certificate

ECHO_DESCRIPTION = "Repeat the text back."
CAPSIZE_ERROR = "the boat capsized in the harbour"
PONDERED = "slept on it"
FORECAST_SUMMARY = "Mostly dry in Graz, rain on Saturday."
FORECAST = {
    "city": "Graz",
    "days": [
        {"date": "2026-10-09", "high_c": 17.5, "low_c": 6, "rain_mm": 0, "windy": False},
        {"date": "2026-10-10", "high_c": 12, "low_c": 8.5, "rain_mm": 14.2, "windy": True},
    ],
}
CONDITIONS = {"city": "Graz", "temperature_c": 15.5, "humidity": 71, "raining": False}
HEADLINE = "Fog lifting over the Mur by noon."
BERTHS = ["Berth A by the crane", "Berth B past the ferry"]
HARBOURS = [{"name": "Rovinj marina", "berths": 120}, {"name": "Piran marina", "berths": 80}]
FIND_PAGE_SCHEMA = {
    "type": "object",
    "anyOf": [
        {
            "type": "object",
            "properties": {"id": {"type": "integer", "exclusiveMinimum": 0}},
            "required": ["id"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"path": {"type": "string"}, "locale": {"type": "string"}},
            "required": ["path", "locale"],
            "additionalProperties": False,
        },
    ],
}
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


class Conditions(BaseModel):
    city: str
    temperature_c: float
    humidity: int
    raining: bool


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


def _moved(app, old_path: str):
    async def redirecting(scope, receive, send) -> None:
        if scope["type"] == "http" and scope["path"] == old_path:
            headers = [(b"location", b"/mcp"), (b"content-length", b"0")]
            await send({"type": "http.response.start", "status": 307, "headers": headers})
            await send({"type": "http.response.body", "body": b""})
            return
        await app(scope, receive, send)

    return redirecting


def _echo_server(
    media: bool = False,
    failing: bool = False,
    slow: bool = False,
    structured: bool = False,
    composed: bool = False,
    **auth: Any,
) -> MCPServer:
    server = MCPServer("harness-mcp", log_level="WARNING", **auth)

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
            # mcp 2 masks an unexpected crash; ToolError is the anticipated failure
            # whose text still reaches the caller, as a raised error did on mcp 1.
            raise ToolError(CAPSIZE_ERROR)

    if slow:

        @server.tool(description="Think it over for a while.")
        async def ponder(seconds: float) -> str:
            await asyncio.sleep(seconds)
            return PONDERED

    if structured:

        @server.tool(description="The weather forecast for a city.")
        def forecast(city: str) -> CallToolResult:
            return CallToolResult(
                content=[TextContent(type="text", text=FORECAST_SUMMARY)],
                structuredContent=FORECAST,
            )

        @server.tool(description="The current weather in a city.")
        def conditions(city: str) -> Conditions:
            return Conditions(**CONDITIONS)

        @server.tool(description="The weather headline for a city.", structured_output=False)
        def headline(city: str) -> str:
            return HEADLINE

        @server.tool(description="The free berths in a city's harbour.")
        def berths(city: str) -> list[str]:
            return BERTHS

        @server.tool(description="The harbours near a city.")
        def harbours(city: str) -> CallToolResult:
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(HARBOURS))],
                structuredContent={"result": HARBOURS},
            )

    if composed:

        @server.tool(description="Find a wiki page by its id, or by its path and locale.")
        def find_page(
            id: int | None = None, path: str | None = None, locale: str | None = None
        ) -> str:
            given = {"id": id, "path": path, "locale": locale}
            return json.dumps({key: value for key, value in given.items() if value is not None})

        server._tool_manager.get_tool("find_page").parameters = FIND_PAGE_SCHEMA

    return server


@contextlib.contextmanager
def serving_mcp(
    port: int | None = None,
    media: bool = False,
    failing: bool = False,
    slow: bool = False,
    structured: bool = False,
    composed: bool = False,
    tls: bool = False,
    host: str = "127.0.0.1",
    name: str | None = None,
    bearer_key: str | None = None,
    moved_from: str | None = None,
    **auth: Any,
) -> Iterator[str]:
    """Serve the echo server on `port` (a free one by default); `auth` goes to MCPServer."""
    port = port or free_port()
    # mcp 2 moved transport security off the server onto the Streamable HTTP app.
    transport_security = auth.pop("transport_security", None)
    server = _echo_server(media, failing, slow, structured, composed, **auth)
    app = server.streamable_http_app(transport_security=transport_security)
    if bearer_key:
        app = _requiring_bearer(app, bearer_key)
    if moved_from:
        app = _moved(app, moved_from)
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
