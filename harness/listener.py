"""A local HTTP service that records what it is sent and answers from per-path routes.

Stands in for whatever outside service the instance calls: a page to fetch, an image engine,
a tool server, a search provider. `route(method, path, handler)` registers an answer; a handler
gets the recorded request and returns `(status, headers, body)`. Unrouted paths answer 404.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterator

from harness.instance import free_port


@dataclass
class ReceivedRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    def json(self) -> dict:
        return json.loads(self.body)


Answer = tuple[int, dict[str, str], bytes]
Handler = Callable[[ReceivedRequest], Answer]


def json_answer(payload: dict | list, status: int = 200) -> Answer:
    return status, {"Content-Type": "application/json"}, json.dumps(payload).encode()


def text_answer(text: str, content_type: str = "text/html", status: int = 200) -> Answer:
    return status, {"Content-Type": content_type}, text.encode()


@dataclass
class Listener:
    base_url: str
    host: str
    port: int
    routes: dict[tuple[str, str], Handler] = field(default_factory=dict)
    received: list[ReceivedRequest] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def route(self, method: str, path: str, handler: Handler | Answer) -> None:
        """Answer `method path` (path without the query string) with a handler or a fixed answer."""
        answer = handler if callable(handler) else (lambda _request, fixed=handler: fixed)
        self.routes[(method.upper(), path)] = answer

    def requests_to(self, path: str) -> list[ReceivedRequest]:
        with self.lock:
            return [entry for entry in self.received if entry.path.split("?")[0] == path]


@contextmanager
def listening(host: str = "127.0.0.1") -> Iterator[Listener]:
    port = free_port()
    listener = Listener(base_url=f"http://{host}:{port}", host=host, port=port)

    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _read_body(self) -> bytes:
            if self.headers.get("Transfer-Encoding", "").lower() != "chunked":
                length = int(self.headers.get("Content-Length", 0))
                return self.rfile.read(length) if length else b""
            body = b""
            while size := int(self.rfile.readline().split(b";")[0], 16):
                body += self.rfile.read(size)
                self.rfile.readline()
            while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                pass  # trailers
            return body

        def _serve(self) -> None:
            request = ReceivedRequest(
                method=self.command,
                path=self.path,
                headers=dict(self.headers.items()),
                body=self._read_body(),
            )
            with listener.lock:
                listener.received.append(request)
            handler = listener.routes.get((self.command, self.path.split("?")[0]))
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

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = _serve

    server = ThreadingHTTPServer((host, port), RequestHandler)
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    try:
        yield listener
    finally:
        server.shutdown()
        server.server_close()
