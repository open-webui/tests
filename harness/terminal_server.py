"""A fake terminal server: scripted HTTP routes and interactive sessions on one port.

Stands in for the terminal an admin connects under Settings > Integrations. Every HTTP request
is recorded with its headers (names lower-cased) and cookies; `route` scripts an answer,
`redirect` a redirect, and unrouted paths answer 404. A WebSocket upgrade on any path opens a
`TerminalSession`: its first message is the auth handshake, later ones are recorded and echoed
back the way a shell prints what it is typed, and `ended` is set once either side closes.

`terminal_session` opens the browser's side of that WebSocket against the instance's proxy.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterator

import httpx
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.frames import Opcode
from websockets.http11 import Request
from websockets.protocol import State
from websockets.server import ServerProtocol
from websockets.sync.client import ClientConnection, connect

from harness.instance import free_port
from harness.listener import Answer

# The admin endpoint pair, for `preserve(TERMINAL_SERVERS_CONFIG)`.
TERMINAL_SERVERS_CONFIG = ("/api/v1/configs/terminal_servers", "/api/v1/configs/terminal_servers")


@dataclass
class TerminalRequest:
    method: str
    path: str
    headers: dict[str, str]
    cookies: dict[str, str]
    body: bytes


@dataclass
class TerminalSession:
    path: str
    headers: dict[str, str]
    auth: dict | None = None
    messages: list[str | bytes] = field(default_factory=list)
    ended: threading.Event = field(default_factory=threading.Event)


Handler = Callable[[TerminalRequest], Answer]


@dataclass
class FakeTerminalServer:
    base_url: str
    routes: dict[tuple[str, str], Handler] = field(default_factory=dict)
    received: list[TerminalRequest] = field(default_factory=list)
    sessions: list[TerminalSession] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def route(self, method: str, path: str, handler: Handler | Answer) -> None:
        """Answer `method path` (path without the query string) with a handler or a fixed answer."""
        answer = handler if callable(handler) else (lambda _request, fixed=handler: fixed)
        self.routes[(method.upper(), path)] = answer

    def redirect(self, path: str, location: str, status: int = 302) -> None:
        self.route("GET", path, (status, {"Location": location}, b""))

    def requests_to(self, path: str) -> list[TerminalRequest]:
        with self.lock:
            return [entry for entry in self.received if entry.path.split("?")[0] == path]

    def clear(self) -> None:
        """Forget the traffic so far, such as the spec fetch that saving a connection triggers."""
        with self.lock:
            self.received.clear()
            self.sessions.clear()

    def connection(self, **fields) -> dict:
        """A terminal connection to this server, as the admin form saves it."""
        return {
            "id": f"terminal-{uuid.uuid4().hex[:8]}",
            "name": "Fake terminal",
            "enabled": True,
            "url": self.base_url,
            "path": "/openapi.json",
            "key": "",
            "auth_type": "none",
            "forward_cookies": False,
            "config": {"access_grants": []},
            **fields,
        }


def read_grant(user_id: str) -> dict:
    return {"principal_type": "user", "principal_id": user_id, "permission": "read"}


def configure_terminals(client: httpx.Client, *connections: dict) -> None:
    """Save the terminal connections the way the admin panel does, replacing the current ones."""
    saved = client.post(
        TERMINAL_SERVERS_CONFIG[1], json={"TERMINAL_SERVER_CONNECTIONS": list(connections)}
    )
    assert saved.status_code == 200, f"saving terminal connections failed: {saved.text}"


@contextmanager
def serving_terminal(host: str = "127.0.0.1") -> Iterator[FakeTerminalServer]:
    port = free_port()
    server_state = FakeTerminalServer(base_url=f"http://{host}:{port}")

    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _serve(self) -> None:
            if self.headers.get("Upgrade", "").lower() == "websocket":
                self._serve_session()
                return
            length = int(self.headers.get("Content-Length", 0))
            cookies = SimpleCookie(self.headers.get("Cookie", ""))
            request = TerminalRequest(
                method=self.command,
                path=self.path,
                headers={name.lower(): value for name, value in self.headers.items()},
                cookies={name: morsel.value for name, morsel in cookies.items()},
                body=self.rfile.read(length) if length else b"",
            )
            with server_state.lock:
                server_state.received.append(request)
            handler = server_state.routes.get((self.command, self.path.split("?")[0]))
            status, headers, body = (
                handler(request) if handler else (404, {"Content-Type": "text/plain"}, b"")
            )
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _serve_session(self) -> None:
            session = TerminalSession(
                path=self.path,
                headers={name.lower(): value for name, value in self.headers.items()},
            )
            with server_state.lock:
                server_state.sessions.append(session)
            handshake = ServerProtocol().accept(Request(self.path, Headers(self.headers.items())))
            protocol = ServerProtocol(state=State.OPEN)
            try:
                self.wfile.write(handshake.serialize())
                while not protocol.close_expected():
                    data = self.rfile.read1(65536)
                    if not data:
                        break
                    protocol.receive_data(data)
                    for frame in protocol.events_received():
                        self._answer_frame(protocol, session, frame)
                    self._flush(protocol)
            except OSError:
                pass
            finally:
                session.ended.set()
                self.close_connection = True

        def _answer_frame(self, protocol: ServerProtocol, session: TerminalSession, frame) -> None:
            if frame.opcode not in (Opcode.TEXT, Opcode.BINARY):
                return
            if session.auth is None and frame.opcode == Opcode.TEXT:
                session.auth = json.loads(frame.data)
                return
            session.messages.append(
                frame.data.decode() if frame.opcode == Opcode.TEXT else frame.data
            )
            if frame.opcode == Opcode.TEXT:
                protocol.send_text(frame.data)
            else:
                protocol.send_binary(frame.data)

        def _flush(self, protocol: ServerProtocol) -> None:
            for data in protocol.data_to_send():
                if data:
                    self.wfile.write(data)

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _serve

    server = ThreadingHTTPServer((host, port), RequestHandler)
    # A short poll keeps teardown quick; shutdown waits for one poll.
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    try:
        yield server_state
    finally:
        server.shutdown()
        server.server_close()


@contextmanager
def terminal_session(
    base_url: str, token: str, terminal_id: str, chat_id: str = ""
) -> Iterator[ClientConnection]:
    """The browser's terminal WebSocket through the instance, past its first-message auth."""
    session_id = uuid.uuid4().hex
    url = (
        base_url.replace("http://", "ws://", 1)
        + f"/api/v1/terminals/{terminal_id}/api/terminals/{session_id}"
    )
    with connect(url, proxy=None, open_timeout=10, close_timeout=2) as session:
        session.send(json.dumps({"type": "auth", "token": token, "chat_id": chat_id}))
        yield session


def close_of(session: ClientConnection, timeout: float) -> tuple[int, str] | None:
    """The close code and reason once the instance ends `session`, or None if it stays open."""
    deadline = time.monotonic() + timeout
    try:
        while (remaining := deadline - time.monotonic()) > 0:
            session.recv(timeout=remaining)
    except TimeoutError:
        return None
    except ConnectionClosed as closed:
        return (closed.rcvd.code, closed.rcvd.reason) if closed.rcvd else (1006, "")
    return None
