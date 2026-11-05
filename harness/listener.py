"""A local HTTP service that records what it is sent and answers from per-path routes.

Stands in for whatever outside service the instance calls: a page to fetch, an image engine,
a tool server, a search provider. `route(method, path, handler)` registers an answer; a handler
gets the recorded request and returns `(status, headers, body)`. Unrouted paths answer 404,
CORS preflights (`OPTIONS`) included. A route path ending in `*` answers every path under that
prefix. A handler may return an iterator of byte chunks as the body: the listener sends them as they
come, and when the client has gone away it closes the iterator, so a handler that finishes work
after its last chunk can tell a client that read to the end from one that did not. `listening(host)`
binds another local address, an IPv6 one included, and `port` a fixed port.
"""

from __future__ import annotations

import json
import socket
import ssl
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Iterable, Iterator

from harness.object_storage import self_signed_certificate


@dataclass
class ReceivedRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    def json(self) -> dict:
        return json.loads(self.body)


Answer = tuple[int, dict[str, str], bytes | Iterable[bytes]]


def read_body(handler: BaseHTTPRequestHandler) -> bytes:
    """The request body, sent with a Content-Length or chunked."""
    if handler.headers.get("Transfer-Encoding", "").lower() != "chunked":
        length = int(handler.headers.get("Content-Length", 0))
        return handler.rfile.read(length) if length else b""
    body = b""
    while size := int(handler.rfile.readline().split(b";")[0], 16):
        body += handler.rfile.read(size)
        handler.rfile.readline()
    while handler.rfile.readline() not in (b"\r\n", b"\n", b""):
        pass  # trailers
    return body


Handler = Callable[[ReceivedRequest], Answer]


def json_answer(payload: dict | list, status: int = 200) -> Answer:
    return status, {"Content-Type": "application/json"}, json.dumps(payload).encode()


def text_answer(text: str, content_type: str = "text/html", status: int = 200) -> Answer:
    return status, {"Content-Type": content_type}, text.encode()


@dataclass
class Listener:
    base_url: str
    port: int
    routes: dict[tuple[str, str], Handler] = field(default_factory=dict)
    received: list[ReceivedRequest] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def route(self, method: str, path: str, handler: Handler | Answer) -> None:
        """Answer `method path` (path without the query string) with a handler or a fixed answer."""
        answer = handler if callable(handler) else (lambda _request, fixed=handler: fixed)
        self.routes[(method.upper(), path)] = answer

    def handler_for(self, method: str, path: str) -> Handler | None:
        exact = self.routes.get((method, path))
        if exact:
            return exact
        for (route_method, route_path), handler in self.routes.items():
            if (
                route_method == method
                and route_path.endswith("*")
                and path.startswith(route_path[:-1])
            ):
                return handler
        return None

    def requests_to(self, path: str) -> list[ReceivedRequest]:
        with self.lock:
            return [entry for entry in self.received if entry.path.split("?")[0] == path]


class IPv6HTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6


@contextmanager
def listening(host: str = "127.0.0.1", tls: bool = False, port: int = 0) -> Iterator[Listener]:
    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _serve(self) -> None:
            request = ReceivedRequest(
                method=self.command,
                path=self.path,
                headers=dict(self.headers.items()),
                body=read_body(self),
            )
            with listener.lock:
                listener.received.append(request)
            handler = listener.handler_for(self.command, self.path.split("?")[0])
            status, headers, body = (
                handler(request) if handler else (404, {"Content-Type": "text/plain"}, b"")
            )
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            if not isinstance(body, bytes):
                self._send_chunks(body)
                return
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_chunks(self, chunks: Iterable[bytes]) -> None:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for chunk in chunks:
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
            except OSError:
                self.close_connection = True
            finally:
                getattr(chunks, "close", lambda: None)()

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _serve

    server_class = IPv6HTTPServer if ":" in host else ThreadingHTTPServer
    server = server_class((host, port), RequestHandler)
    port = server.server_port
    url_host = f"[{host}]" if ":" in host else host
    scheme = "https" if tls else "http"
    listener = Listener(base_url=f"{scheme}://{url_host}:{port}", port=port)
    with tempfile.TemporaryDirectory(prefix="owui-listener-") as certificates:
        if tls:
            certificate, key = self_signed_certificate(Path(certificates))
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certificate, key)
            server.socket = context.wrap_socket(server.socket, server_side=True)
        threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
        try:
            yield listener
        finally:
            server.shutdown()
            server.server_close()
