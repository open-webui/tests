"""A stand-in Redis that fails chosen commands once and records every command it is sent.

Answers like `fake_redis`, as an empty database would, except that the first command each
`fail_once(predicate)` accepts gets an error reply instead, which redis-py raises as a
`ResponseError`: one Redis blip, aimed at one call site. `sent(name)` lists the arguments of
every command of that name so far.
"""

from __future__ import annotations

import threading
from socketserver import StreamRequestHandler, ThreadingTCPServer
from typing import Callable

from integration.fake_redis import _read_command, _reply

Predicate = Callable[[str, list[str]], bool]


class FlakyRedis:
    def __init__(self) -> None:
        self._failures: list[Predicate] = []
        self._sent: list[tuple[str, list[str]]] = []
        self._lock = threading.Lock()
        self._server = ThreadingTCPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"redis://{host}:{port}/0"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def fail_once(self, predicate: Predicate) -> None:
        """Answer the next command `predicate(name, args)` accepts with an error."""
        with self._lock:
            self._failures.append(predicate)

    def sent(self, name: str) -> list[list[str]]:
        with self._lock:
            return [args for sent_name, args in self._sent if sent_name == name]

    def _answer(self, name: str, args: list[str]) -> bytes:
        with self._lock:
            self._sent.append((name, args))
            failure = next((match for match in self._failures if match(name, args)), None)
            if failure is not None:
                self._failures.remove(failure)
        return b"-ERR injected failure\r\n" if failure is not None else _reply(name, args)

    def _handler(self):
        store = self

        class Handler(StreamRequestHandler):
            def handle(self) -> None:
                while (command := _read_command(self.rfile)) is not None:
                    self.wfile.write(store._answer(command[0].upper(), command[1:]))
                    self.wfile.flush()

        return Handler
