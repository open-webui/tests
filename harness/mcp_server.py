"""A Model Context Protocol tool server on a local port, built with the `mcp` SDK's own FastMCP.

`serving_mcp()` runs one over Streamable HTTP in a thread of the test process and yields its
URL, the one an admin enters for an MCP tool server connection. It offers a single tool, `echo`,
and stops again when the block ends.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Iterator

import uvicorn
from mcp.server.fastmcp import FastMCP

from harness.instance import free_port

ECHO_DESCRIPTION = "Repeat the text back."


def _echo_server() -> FastMCP:
    server = FastMCP("harness-mcp", log_level="WARNING")

    @server.tool(description=ECHO_DESCRIPTION)
    def echo(text: str) -> str:
        return text

    return server


@contextlib.contextmanager
def serving_mcp() -> Iterator[str]:
    port = free_port()
    app = _echo_server().streamable_http_app()
    # log_config=None keeps uvicorn from reconfiguring the test process's logging
    runner = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, log_level="warning")
    )
    thread = threading.Thread(target=runner.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not runner.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("the MCP server did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        runner.should_exit = True
        thread.join(timeout=10)
