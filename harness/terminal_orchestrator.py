"""A stand-in for the Terminals orchestrator: one real Open Terminal per terminal context.

An admin connects an orchestrator as a terminal of type `orchestrator`; set to Per chat, Open
WebUI names the chat in an `X-Terminal-Context-Id: chat:<chat id>` header on every call made
for that chat, and the orchestrator gives each context a terminal of its own. Calls without the
header go to the shared terminal. `serving_terminal_orchestrator()` answers on one port and
forwards each call to a real Open Terminal (from `harness.open_terminal`) for its context,
started on its first call, each in a scratch home of its own; `home(context)` names that home
and `contexts()` the contexts that got a terminal. WebSocket sessions are not forwarded.
"""

from __future__ import annotations

import contextlib
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import httpx

from harness.instance import free_port
from harness.open_terminal import API_KEY, OpenTerminal, serving_open_terminal

CONTEXT_HEADER = "X-Terminal-Context-Id"
SHARED = "shared"
# Hop-by-hop headers and the ones httpx sets or undoes itself.
UNFORWARDED = {"host", "connection", "content-length", "transfer-encoding", "content-encoding"}


@dataclass
class TerminalOrchestrator:
    base_url: str
    terminals: dict[str, OpenTerminal] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    stack: contextlib.ExitStack = field(default_factory=contextlib.ExitStack)

    def connection(self, per_chat: bool = True, **fields) -> dict:
        """The orchestrator's connection as the admin form saves it, Chat Uploads on Filesystem."""
        config = {"access_grants": [], "chat_uploads": "filesystem"}
        if per_chat:
            config["contexts"] = {"chat": {"context_id": "chat_id"}}
        return {
            "id": f"orchestrator-{uuid.uuid4().hex[:8]}",
            "name": "Per Chat Terminal",
            "enabled": True,
            "url": self.base_url,
            "path": "/openapi.json",
            "key": API_KEY,
            "auth_type": "bearer",
            "server_type": "orchestrator",
            "config": config,
            **fields,
        }

    def terminal(self, context: str) -> OpenTerminal:
        with self.lock:
            if context not in self.terminals:
                self.terminals[context] = self.stack.enter_context(serving_open_terminal())
            return self.terminals[context]

    def home(self, context: str) -> Path:
        """The home of `context`'s terminal: `SHARED` or `chat:<chat id>`."""
        return self.terminal(context).home

    def contexts(self) -> set[str]:
        with self.lock:
            return set(self.terminals)


@contextlib.contextmanager
def serving_terminal_orchestrator() -> Iterator[TerminalOrchestrator]:
    port = free_port()
    orchestrator = TerminalOrchestrator(base_url=f"http://127.0.0.1:{port}")

    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _forward(self) -> None:
            context = self.headers.get(CONTEXT_HEADER) or SHARED
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            headers = {
                name: value
                for name, value in self.headers.items()
                if name.lower() not in UNFORWARDED
            }
            terminal = orchestrator.terminal(context)
            answer = httpx.request(
                self.command,
                f"{terminal.base_url}{self.path}",
                headers=headers,
                content=body,
                timeout=30,
            )
            self.send_response(answer.status_code)
            for name, value in answer.headers.items():
                if name.lower() not in UNFORWARDED:
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(answer.content)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(answer.content)

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _forward

    # The shared terminal boots here, so a missing open-terminal skips before the server starts.
    orchestrator.terminal(SHARED)
    server = ThreadingHTTPServer(("127.0.0.1", port), RequestHandler)
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    try:
        yield orchestrator
    finally:
        server.shutdown()
        server.server_close()
        orchestrator.stack.close()
