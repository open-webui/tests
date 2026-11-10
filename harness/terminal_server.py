"""A fake terminal server: scripted HTTP routes and interactive sessions on one port.

Stands in for the terminal an admin connects under Settings > Integrations. Every HTTP request
is recorded with its headers (names lower-cased) and cookies; `route` scripts an answer,
`redirect` a redirect, and unrouted paths answer 404. A WebSocket upgrade on any path opens a
`TerminalSession`: its first message is the auth handshake, later ones are recorded and echoed
back the way a shell prints what it is typed, and `ended` is set once either side closes.
`script_sessions(script)` replaces the echo: the script gets each new session and decides what
the terminal sends, when it closes and with which code, or that it hangs up without closing;
`refuse_sessions(status)` answers the upgrade itself with an HTTP status.
Setting `cors` lets a page of another origin call it, as a terminal a user added in their own
settings is called from their browser: every answer allows any origin and a preflight gets 204.

`terminal_session` opens the browser's side of that WebSocket against the instance's proxy.
"""

from __future__ import annotations

import json
import queue
import socket
import struct
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
from websockets.exceptions import ConnectionClosed, InvalidState
from websockets.frames import Frame, Opcode
from websockets.http11 import Request
from websockets.protocol import State
from websockets.server import ServerProtocol
from websockets.sync.client import ClientConnection, connect

from harness.instance import free_port
from harness.listener import Answer, IPv6HTTPServer, read_body

# The admin endpoint pair, for `preserve(TERMINAL_SERVERS_CONFIG)`.
TERMINAL_SERVERS_CONFIG = ("/api/v1/configs/terminal_servers", "/api/v1/configs/terminal_servers")


@dataclass
class TerminalRequest:
    method: str
    path: str
    headers: dict[str, str]
    cookies: dict[str, str]
    body: bytes


class SessionEnded(Exception):
    """The instance's side of a scripted session has gone."""


@dataclass
class TerminalSession:
    """One WebSocket the instance opened on the terminal.

    A test reads `auth`, `messages`, `ended` and `closed_with`. A script from `script_sessions`
    also drives it: `authenticated`, `receive`, `send`, `close` and `drop`.
    """

    path: str
    headers: dict[str, str]
    auth: dict | None = None
    messages: list[str | bytes] = field(default_factory=list)
    ended: threading.Event = field(default_factory=threading.Event)
    # the close frame the instance sent, as (code, reason); None when it only hung up
    closed_with: tuple[int, str] | None = None
    # what the session's script raised, for the test to assert on
    failure: BaseException | None = None
    _authenticated: threading.Event = field(default_factory=threading.Event, repr=False)
    _inbox: queue.Queue = field(default_factory=queue.Queue, repr=False)
    _write: Callable[[bytes | str], None] | None = field(default=None, repr=False)
    _write_close: Callable[[bytes], None] | None = field(default=None, repr=False)
    _hang_up: Callable[[], None] | None = field(default=None, repr=False)

    def authenticated(self, timeout: float = 10) -> dict:
        """Wait for the first message, the auth handshake, and return it."""
        if not self._authenticated.wait(timeout):
            raise SessionEnded("the session sent no auth message")
        return self.auth

    def receive(self, timeout: float = 10) -> str | bytes:
        """The next message after the auth handshake; SessionEnded once the instance has left."""
        try:
            message = self._inbox.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"the session sent nothing within {timeout}s") from None
        if message is None:
            self._inbox.put(None)
            raise SessionEnded("the instance left the session")
        return message

    def send(self, data: str | bytes) -> None:
        """Send a text or a binary message, as a shell prints output."""
        self._write(data)

    def close(self, code: int | None = 1000, reason: str = "") -> None:
        """Close with any code, reserved ones included; None sends a close without a code."""
        payload = b"" if code is None else struct.pack("!H", code) + reason.encode()
        self._write_close(payload)

    def drop(self) -> None:
        """Hang up without a close frame, as a terminal that died does."""
        self._hang_up()


Handler = Callable[[TerminalRequest], Answer]
SessionScript = Callable[[TerminalSession], None]


@dataclass
class FakeTerminalServer:
    base_url: str
    routes: dict[tuple[str, str], Handler] = field(default_factory=dict)
    received: list[TerminalRequest] = field(default_factory=list)
    sessions: list[TerminalSession] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    cors: bool = False
    session_scripts: dict[str, SessionScript] = field(default_factory=dict)
    session_refusals: dict[str, int] = field(default_factory=dict)

    def script_sessions(self, script: SessionScript, path: str = "*") -> None:
        """Run `script(session)` for each new session on `path` in place of the echo.

        `path` is the upgrade's path without the query string; one ending in `*` takes every
        path under that prefix. The script runs in a thread of its own once the upgrade is
        answered. A session it returns from stays open until one side closes it.
        """
        self.session_scripts[path] = script

    def refuse_sessions(self, status: int, path: str = "*") -> None:
        """Answer the WebSocket upgrade on `path` with an HTTP status; the try is in `received`."""
        self.session_refusals[path] = status

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


def _for_path(scripted: dict, path: str):
    """The entry for `path`: its own, else the longest `prefix*` that covers it."""
    if path in scripted:
        return scripted[path]
    prefixes = [key for key in scripted if key.endswith("*") and path.startswith(key[:-1])]
    return scripted[max(prefixes, key=len)] if prefixes else None


def read_grant(user_id: str) -> dict:
    return {"principal_type": "user", "principal_id": user_id, "permission": "read"}


def write_grant(user_id: str) -> dict:
    """On an orchestrator connection: the user may create terminals of their own on it."""
    return {"principal_type": "user", "principal_id": user_id, "permission": "write"}


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
            cookies = SimpleCookie(self.headers.get("Cookie", ""))
            request = TerminalRequest(
                method=self.command,
                path=self.path,
                headers={name.lower(): value for name, value in self.headers.items()},
                cookies={name: morsel.value for name, morsel in cookies.items()},
                body=read_body(self),
            )
            with server_state.lock:
                server_state.received.append(request)
            handler = server_state.routes.get((self.command, self.path.split("?")[0]))
            status, headers, body = (
                handler(request) if handler else (404, {"Content-Type": "text/plain"}, b"")
            )
            if server_state.cors:
                if self.command == "OPTIONS" and not handler:
                    status, body = 204, b""
                headers = {**headers, **self._cors_headers()}
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _cors_headers(self) -> dict[str, str]:
            return {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, PATCH, OPTIONS",
                "Access-Control-Allow-Headers": self.headers.get(
                    "Access-Control-Request-Headers", "*"
                ),
            }

        def _serve_session(self) -> None:
            headers = {name.lower(): value for name, value in self.headers.items()}
            path = self.path.split("?")[0]
            refusal = _for_path(server_state.session_refusals, path)
            if refusal is not None:
                refused = TerminalRequest(self.command, self.path, headers, {}, b"")
                with server_state.lock:
                    server_state.received.append(refused)
                self.send_response(refusal)
                self.send_header("Content-Length", "0")
                self.end_headers()
                self.close_connection = True
                return
            session = TerminalSession(path=self.path, headers=headers)
            script = _for_path(server_state.session_scripts, path)
            handshake = ServerProtocol().accept(Request(self.path, Headers(self.headers.items())))
            protocol = ServerProtocol(state=State.OPEN)
            # the script's thread and this one both write
            writing = threading.Lock()
            closing = threading.Event()

            def write(data: bytes | str) -> None:
                try:
                    with writing:
                        if isinstance(data, str):
                            protocol.send_text(data.encode())
                        else:
                            protocol.send_binary(data)
                        self._flush(protocol)
                except (OSError, InvalidState) as error:
                    raise SessionEnded("the instance left the session") from error

            def write_close(payload: bytes) -> None:
                closing.set()
                try:
                    with writing:
                        # past the protocol, which refuses the reserved codes
                        self.wfile.write(Frame(Opcode.CLOSE, payload).serialize(mask=False))
                except OSError as error:
                    raise SessionEnded("the instance left the session") from error

            def hang_up() -> None:
                closing.set()
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

            def run_script() -> None:
                try:
                    script(session)
                except SessionEnded:
                    pass
                except BaseException as error:
                    session.failure = error
                    hang_up()

            session._write, session._write_close, session._hang_up = write, write_close, hang_up
            with server_state.lock:
                server_state.sessions.append(session)
            try:
                self.wfile.write(handshake.serialize())
                if script:
                    threading.Thread(target=run_script, daemon=True).start()
                while not protocol.close_expected():
                    data = self.rfile.read1(65536)
                    if not data:
                        break
                    with writing:
                        protocol.receive_data(data)
                        for frame in protocol.events_received():
                            self._take_frame(protocol, session, frame, echo=script is None)
                        # the script's own close frame is the only one the instance gets
                        if not closing.is_set():
                            self._flush(protocol)
            except OSError:
                pass
            finally:
                session._inbox.put(None)
                session.ended.set()
                self.close_connection = True

        def _take_frame(
            self, protocol: ServerProtocol, session: TerminalSession, frame, echo: bool
        ) -> None:
            if frame.opcode == Opcode.CLOSE:
                code = struct.unpack("!H", frame.data[:2])[0] if len(frame.data) >= 2 else 1005
                session.closed_with = (code, frame.data[2:].decode(errors="replace"))
                return
            if frame.opcode not in (Opcode.TEXT, Opcode.BINARY):
                return
            if session.auth is None and frame.opcode == Opcode.TEXT:
                session.auth = json.loads(frame.data)
                session._authenticated.set()
                return
            message = frame.data.decode() if frame.opcode == Opcode.TEXT else bytes(frame.data)
            session.messages.append(message)
            session._inbox.put(message)
            if not echo:
                return
            if frame.opcode == Opcode.TEXT:
                protocol.send_text(frame.data)
            else:
                protocol.send_binary(frame.data)

        def _flush(self, protocol: ServerProtocol) -> None:
            for data in protocol.data_to_send():
                if data:
                    self.wfile.write(data)

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _serve

    server_class = IPv6HTTPServer if ":" in host else ThreadingHTTPServer
    server = server_class((host, port), RequestHandler)
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
