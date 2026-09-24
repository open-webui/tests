"""A stand-in Redis that keeps plain keys, for tests that need the instance to read back a write.

GET, SET (with EX, PX, NX and XX), DEL and EXISTS work on real state with real expiry; every
other command gets the empty-database reply of `fake_redis`. A test reads what the instance
stored through `keys()` and `expires_at(key)`.
"""

from __future__ import annotations

import threading
import time
from socketserver import StreamRequestHandler, ThreadingTCPServer

from integration.fake_redis import _read_command, _reply

NULL = b"_\r\n"


class StatefulRedis:
    def __init__(self) -> None:
        self._values: dict[str, tuple[str, float | None]] = {}
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

    def keys(self) -> list[str]:
        with self._lock:
            return [key for key in list(self._values) if self._live(key) is not None]

    def expires_at(self, key: str) -> float | None:
        """Epoch seconds at which `key` expires, None when it never does."""
        with self._lock:
            return self._values[key][1]

    def _live(self, key: str) -> str | None:
        entry = self._values.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and expires_at <= time.time():
            del self._values[key]
            return None
        return value

    def _set(self, key: str, value: str, options: list[str]) -> bytes:
        flags = [option.upper() for option in options]
        exists = self._live(key) is not None
        if ("NX" in flags and exists) or ("XX" in flags and not exists):
            return NULL
        expires_at = None
        if "EX" in flags:
            expires_at = time.time() + int(options[flags.index("EX") + 1])
        elif "PX" in flags:
            expires_at = time.time() + int(options[flags.index("PX") + 1]) / 1000
        self._values[key] = (value, expires_at)
        return b"+OK\r\n"

    def _answer(self, name: str, args: list[str]) -> bytes:
        with self._lock:
            if name == "GET":
                value = self._live(args[0])
                if value is None:
                    return NULL
                encoded = value.encode()
                return b"$%d\r\n%s\r\n" % (len(encoded), encoded)
            if name == "SET":
                return self._set(args[0], args[1], args[2:])
            if name in ("DEL", "UNLINK"):
                removed = [key for key in args if self._live(key) is not None]
                for key in removed:
                    del self._values[key]
                return b":%d\r\n" % len(removed)
            if name == "EXISTS":
                return b":%d\r\n" % sum(self._live(key) is not None for key in args)
        return _reply(name, args)

    def _handler(self):
        store = self

        class Handler(StreamRequestHandler):
            def handle(self) -> None:
                while (command := _read_command(self.rfile)) is not None:
                    self.wfile.write(store._answer(command[0].upper(), command[1:]))
                    self.wfile.flush()

        return Handler
