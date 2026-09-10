"""A stand-in Redis that answers every command with an empty result, slowly on request.

Enough RESP3 for redis-py to connect, subscribe and run the commands the tested paths send:
the reply is whatever an empty database would say. Transactions and pattern subscriptions
are not covered. `delays` maps a command name to seconds to sleep before answering it, which
is how a test makes one Redis operation slow while the rest of the instance stays fast.
"""

from __future__ import annotations

import threading
import time
from socketserver import StreamRequestHandler, ThreadingTCPServer

INTEGER_REPLY = {"INCRBY", "EXPIRE", "SADD", "SREM", "HSET", "HDEL"}
ZERO_REPLY = {"EXISTS", "DEL", "PUBLISH"}
EMPTY_ARRAY_REPLY = {"SMEMBERS", "HKEYS", "KEYS"}
EMPTY_MAP_REPLY = {"HGETALL"}
NULL_REPLY = {"GET", "HGET"}


class FakeRedis:
    def __init__(self) -> None:
        self.delays: dict[str, float] = {}
        self.seen: set[str] = set()
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

    def _handler(self):
        fake = self

        class Handler(StreamRequestHandler):
            def handle(self) -> None:
                while True:
                    command = _read_command(self.rfile)
                    if command is None:
                        return
                    name = command[0].upper()
                    fake.seen.add(name)
                    time.sleep(fake.delays.get(name, 0))
                    self.wfile.write(_reply(name, command[1:]))
                    self.wfile.flush()

        return Handler


def _read_command(stream) -> list[str] | None:
    header = stream.readline()
    if not header:
        return None
    parts = []
    for _ in range(int(header[1:])):
        length = int(stream.readline()[1:])
        parts.append(stream.read(length + 2)[:-2].decode())
    return parts


def _reply(name: str, args: list[str]) -> bytes:
    if name == "PING":
        return b"+PONG\r\n"
    if name == "HELLO":
        return b"%1\r\n$5\r\nproto\r\n:3\r\n"
    if name == "SUBSCRIBE":
        channel = args[0].encode()
        return b">3\r\n$9\r\nsubscribe\r\n$%d\r\n%s\r\n:1\r\n" % (len(channel), channel)
    if name == "MGET":
        return b"*%d\r\n" % len(args) + b"_\r\n" * len(args)
    if name in ("HSCAN", "SCAN"):
        return b"*2\r\n$1\r\n0\r\n*0\r\n"
    if name == "HEXPIRE":
        fields = int(args[args.index("FIELDS") + 1])
        return b"*%d\r\n" % fields + b":1\r\n" * fields
    if name in INTEGER_REPLY:
        return b":1\r\n"
    if name in ZERO_REPLY:
        return b":0\r\n"
    if name in EMPTY_ARRAY_REPLY:
        return b"*0\r\n"
    if name in EMPTY_MAP_REPLY:
        return b"%0\r\n"
    if name in NULL_REPLY:
        return b"_\r\n"
    return b"+OK\r\n"
