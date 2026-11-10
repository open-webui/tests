"""A fake Tether device: the desktop companion's side of `/api/v1/tether/connect`.

Tether connects out to Open WebUI with a renewable session's access token, says hello and is
then listed for its owner as the terminal `tether:<connection id>`. Open WebUI relays every
call to that terminal over the one socket as binary frames (a 4-byte header length, a JSON
header, the payload): `http-open` or `ws-open` starts a stream, `data` and `end` carry a body,
`window` returns flow-control credit and `cancel` ends the stream from the server's side. The
device answers on the same stream with `http-response` or `ws-ready`, then `data` and `end`,
or `error`. Each side numbers its frames per stream from 0.

`tether_device(base_url, token)` connects one device and serves until the block ends. It is
scripted like the fake terminal of `harness.terminal_server`: `route` answers a request, every
request is recorded in `received` (headers lower-cased), unrouted paths answer 404, and a
handler made for the fake terminal works here unchanged. A WebSocket the server opens becomes a
`DeviceSocket` in `sockets`, echoed unless `script_sockets` says otherwise. `announce` and
`reauthenticate` send the device's own updates, `notifications` holds what it was notified of.

To misbehave, a handler takes the request's `stream` and returns None, which leaves the
answer to it: `stream.send("error", message=...)`, a frame number skipped with
`stream.sequence += 1`, `data` sent past the window with `wait_for_credit=False`, no answer at
all, or `device.drop()` to hang up mid-request. `hello=` overrides fields of the hello for a
handshake the server refuses; `ready` is then None and `wait_closed` gives the close code.

`tether_session` opens the renewable session a device signs in with.
"""

from __future__ import annotations

import json
import queue
import struct
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

import httpx
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection, connect

from harness.actors import Actor, sign_in
from harness.instance import LaunchedInstance
from harness.listener import Answer
from harness.mfa import new_address

# the largest payload of one frame, and the window a device asks for unless told otherwise
CHUNK = 65536
DEFAULT_WINDOW = 256 * 1024
CREDIT_SECONDS = 30


class StreamClosed(Exception):
    """The server cancelled the stream or the device's socket is gone."""


def encode(header: dict, data: bytes = b"") -> bytes:
    raw = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("!I", len(raw)) + raw + data


def decode(frame: bytes) -> tuple[dict, bytes]:
    length = struct.unpack("!I", frame[:4])[0]
    return json.loads(frame[4 : 4 + length]), frame[4 + length :]


@dataclass
class RelayStream:
    """One relayed call. `frames` is every frame the server sent on it, in order."""

    id: str
    device: TetherDevice = field(repr=False)
    opened: dict = field(default_factory=dict)
    frames: list[tuple[dict, bytes]] = field(default_factory=list)
    # the number the next frame from the device carries
    sequence: int = 0
    credit: int = DEFAULT_WINDOW
    cancelled: threading.Event = field(default_factory=threading.Event)
    _inbox: queue.Queue = field(default_factory=queue.Queue, repr=False)
    _credit_changed: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def send(
        self, frame_type: str, data: bytes = b"", wait_for_credit: bool = True, **fields
    ) -> None:
        """Send one frame on the stream; `data` frames wait for the server's credit."""
        with self._credit_changed:
            if frame_type == "data" and wait_for_credit:
                enough = self._credit_changed.wait_for(
                    lambda: self.credit >= len(data) or self.cancelled.is_set(), CREDIT_SECONDS
                )
                if self.cancelled.is_set():
                    raise StreamClosed(self.id)
                if not enough:
                    raise TimeoutError(
                        f"no credit for {len(data)} bytes within {CREDIT_SECONDS}s "
                        f"(left: {self.credit})"
                    )
                self.credit -= len(data)
            header = {**fields, "type": frame_type, "stream": self.id, "sequence": self.sequence}
            self.sequence += 1
            self.device.send_frame(header, data)

    def _take(self, header: dict, data: bytes) -> None:
        self.frames.append((header, data))
        if header["type"] == "window":
            with self._credit_changed:
                self.credit += header.get("bytes", 0)
                self._credit_changed.notify_all()
        elif header["type"] == "cancel":
            self._end()
        else:
            self._inbox.put((header, data))

    def _end(self) -> None:
        self.cancelled.set()
        self._inbox.put(None)
        with self._credit_changed:
            self._credit_changed.notify_all()

    def _next(self, timeout: float) -> tuple[dict, bytes]:
        try:
            frame = self._inbox.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"the server sent nothing on the stream within {timeout}s") from None
        if frame is None:
            self._inbox.put(None)
            raise StreamClosed(self.id)
        if frame[1] and self.device.returns_credit:
            self.send("window", bytes=len(frame[1]))
        return frame


@dataclass
class DeviceRequest:
    """An HTTP call relayed to the device, shaped like the fake terminal's `TerminalRequest`."""

    method: str
    path: str
    headers: dict[str, str]
    cookies: dict[str, str]
    body: bytes
    service: str
    stream: RelayStream = field(repr=False)

    def json(self) -> dict:
        return json.loads(self.body)


@dataclass
class DeviceSocket:
    """A WebSocket the server opened on the device; a script drives it with these methods."""

    path: str
    headers: dict[str, str]
    service: str
    stream: RelayStream = field(repr=False)
    messages: list[str | bytes] = field(default_factory=list)
    ended: threading.Event = field(default_factory=threading.Event)

    def receive(self, timeout: float = 10) -> str | bytes:
        """The next whole message from the server; StreamClosed once it has left."""
        parts = []
        while True:
            header, data = self.stream._next(timeout)
            if header["type"] != "data":
                continue
            parts.append(data)
            if header.get("final"):
                break
        message = b"".join(parts)
        message = message.decode() if header.get("kind") == "text" else message
        self.messages.append(message)
        return message

    def send(self, message: str | bytes) -> None:
        kind = "text" if isinstance(message, str) else "binary"
        raw = message.encode() if isinstance(message, str) else message
        for offset in range(0, max(1, len(raw)), CHUNK):
            final = offset + CHUNK >= len(raw)
            self.stream.send("data", raw[offset : offset + CHUNK], kind=kind, final=final)

    def close(self, code: int = 1000) -> None:
        self.stream.send("end", code=code)


Handler = Callable[[DeviceRequest], Answer | None]
SocketScript = Callable[[DeviceSocket], None]


def _echo(socket: DeviceSocket) -> None:
    while True:
        socket.send(socket.receive(timeout=3600))


@dataclass
class TetherDevice:
    connection_id: str
    name: str
    socket: ClientConnection = field(repr=False)
    # the server's answer to the hello; None when it refused the device
    ready: dict | None = None
    # the session the device signed in with, when `connect_device` opened it
    session: dict | None = None
    # the largest answer body the device keeps in flight, agreed in the hello
    window: int = DEFAULT_WINDOW
    # off: the device consumes request bodies without returning their credit
    returns_credit: bool = True
    routes: dict[tuple[str, str], Handler] = field(default_factory=dict)
    received: list[DeviceRequest] = field(default_factory=list)
    sockets: list[DeviceSocket] = field(default_factory=list)
    streams: dict[str, RelayStream] = field(default_factory=dict)
    notifications: list[dict] = field(default_factory=list)
    # what a handler or a socket script raised, for the test to assert on
    failures: list[BaseException] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    # (code, reason) once the socket is closed
    closed: tuple[int, str] | None = None
    _closed: threading.Event = field(default_factory=threading.Event, repr=False)
    _sending: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _socket_scripts: dict[str, SocketScript] = field(default_factory=dict, repr=False)

    @property
    def terminal_id(self) -> str:
        """The id the device's terminal is listed and called by."""
        return f"tether:{self.connection_id}"

    def route(self, method: str, path: str, handler: Handler | Answer) -> None:
        """Answer `method path` (path without the query string) with a handler or a fixed answer."""
        answer = handler if callable(handler) else (lambda _request, fixed=handler: fixed)
        self.routes[(method.upper(), path)] = answer

    def script_sockets(self, script: SocketScript, path: str = "*") -> None:
        """Run `script(socket)` for each WebSocket opened on `path` in place of the echo."""
        self._socket_scripts[path] = script

    def requests_to(self, path: str) -> list[DeviceRequest]:
        with self.lock:
            return [entry for entry in self.received if entry.path.split("?")[0] == path]

    def clear(self) -> None:
        with self.lock:
            self.received.clear()
            self.sockets.clear()

    def send_frame(self, header: dict, data: bytes = b"") -> None:
        """Send one relay frame as it is, for a frame no stream of the device would send."""
        try:
            with self._sending:
                self.socket.send(encode(header, data))
        except ConnectionClosed as error:
            raise StreamClosed(header.get("stream", "")) from error

    def send_json(self, message: dict) -> None:
        """Send a text message, as the device does for its own updates."""
        with self._sending:
            self.socket.send(json.dumps(message))

    def announce(self, services: Iterable[str], name: str | None = None) -> None:
        """Tell the server which services the device offers now, and its new name if any."""
        update = {"type": "services", "services": list(services)}
        self.send_json(update if name is None else {**update, "name": name})

    def reauthenticate(self, token: str) -> None:
        """Hand the server the access token the device got from a refresh."""
        self.send_json({"type": "auth", "token": token})

    def drop(self) -> None:
        """Hang up at once, as a device that lost its network does."""
        self.socket.close_socket()

    def wait_closed(self, timeout: float = 10) -> tuple[int, str] | None:
        """How the socket ended, (1006, "") without a close frame; None while it is open."""
        return self.closed if self._closed.wait(timeout) else None

    def _read(self) -> None:
        try:
            while True:
                frame = self.socket.recv()
                if isinstance(frame, bytes):
                    self._take(*decode(frame))
        except ConnectionClosed as ended:
            self.closed = (ended.rcvd.code, ended.rcvd.reason) if ended.rcvd else (1006, "")
        finally:
            for stream in list(self.streams.values()):
                stream._end()
            self._closed.set()

    def _take(self, header: dict, data: bytes) -> None:
        if header["type"] == "notification":
            self.notifications.append(json.loads(data))
            return
        stream = self.streams.get(header["stream"])
        if header["type"] in ("http-open", "ws-open"):
            stream = RelayStream(header["stream"], self, opened=header, credit=self.window)
            self.streams[stream.id] = stream
            serve = self._serve_request if header["type"] == "http-open" else self._serve_socket
            threading.Thread(target=self._guarded, args=(serve, stream), daemon=True).start()
        if stream:
            stream._take(header, data)

    def _guarded(self, serve: Callable[[RelayStream], None], stream: RelayStream) -> None:
        try:
            serve(stream)
        except StreamClosed:
            pass
        except BaseException as error:
            self.failures.append(error)

    def _serve_request(self, stream: RelayStream) -> None:
        opened, _ = stream._next(10)
        body = []
        while True:
            header, data = stream._next(60)
            if header["type"] == "end":
                break
            body.append(data)
        path = opened["path"] + (f"?{opened['query']}" if opened.get("query") else "")
        request = DeviceRequest(
            method=opened["method"],
            path=path,
            headers={name.lower(): value for name, value in opened.get("headers", {}).items()},
            cookies={},
            body=b"".join(body),
            service=opened.get("service", ""),
            stream=stream,
        )
        with self.lock:
            self.received.append(request)
        handler = self.routes.get((request.method, opened["path"]))
        answer = handler(request) if handler else (404, {"Content-Type": "text/plain"}, b"")
        if answer is None:
            return
        status, headers, content = answer
        stream.send("http-response", status=status, headers=headers)
        for piece in [content] if isinstance(content, bytes) else content:
            for offset in range(0, len(piece), CHUNK):
                stream.send("data", piece[offset : offset + CHUNK])
        stream.send("end")

    def _serve_socket(self, stream: RelayStream) -> None:
        opened, _ = stream._next(10)
        socket = DeviceSocket(
            path=opened["path"] + (f"?{opened['query']}" if opened.get("query") else ""),
            headers={name.lower(): value for name, value in opened.get("headers", {}).items()},
            service=opened.get("service", ""),
            stream=stream,
        )
        with self.lock:
            self.sockets.append(socket)
        stream.send("ws-ready")
        script = _for_path(self._socket_scripts, opened["path"]) or _echo
        try:
            script(socket)
        finally:
            socket.ended.set()


def _for_path(scripted: dict, path: str):
    """The entry for `path`: its own, else the longest `prefix*` that covers it."""
    if path in scripted:
        return scripted[path]
    prefixes = [key for key in scripted if key.endswith("*") and path.startswith(key[:-1])]
    return scripted[max(prefixes, key=len)] if prefixes else None


def tether_session(instance: LaunchedInstance, account: Actor, name: str | None = None) -> dict:
    """A renewable session of `account`, as Tether holds after its sign-in: `token`,
    `refresh_token` and `session_id`. A device connects with the `token`."""
    person = sign_in(instance, account.email, account.password)
    opened = httpx.post(
        f"{instance.base_url}/api/v1/auths/sessions",
        json={"name": name or f"Tether {uuid.uuid4().hex[:8]}"},
        # the sessions routes allow 30 requests a minute per address
        headers={"Authorization": f"Bearer {person}", "X-Forwarded-For": new_address()},
        timeout=60.0,
    )
    assert opened.status_code == 200, f"opening the device's session failed: {opened.text}"
    return opened.json()


@contextmanager
def tether_device(
    base_url: str,
    token: str,
    name: str | None = None,
    connection_id: str | None = None,
    services: Iterable[str] = ("open_terminal",),
    window: int = DEFAULT_WINDOW,
    notifications: bool = False,
    hello: dict | None = None,
) -> Iterator[TetherDevice]:
    """Connect a device with a session's access `token` and serve until the block ends.

    `hello` overrides fields of the hello message; a value of None leaves the field out.
    """
    connection_id = connection_id or str(uuid.uuid4())
    name = name or f"Device {uuid.uuid4().hex[:8]}"
    message = {
        "type": "hello",
        "version": 2,
        "token": token,
        "connection_id": connection_id,
        "name": name,
        "window": window,
        "services": list(services),
        "notifications": notifications,
        **(hello or {}),
    }
    message = {key: value for key, value in message.items() if value is not None}
    url = base_url.replace("http://", "ws://", 1) + "/api/v1/tether/connect"
    with connect(url, proxy=None, open_timeout=10, close_timeout=2, max_size=None) as socket:
        device = TetherDevice(connection_id=connection_id, name=name, socket=socket)
        socket.send(json.dumps(message))
        try:
            device.ready = json.loads(socket.recv(timeout=15))
            device.window = device.ready.get("window", window)
        except ConnectionClosed:
            pass
        reader = threading.Thread(target=device._read, daemon=True)
        reader.start()
        try:
            yield device
        finally:
            socket.close()
            reader.join(timeout=5)
