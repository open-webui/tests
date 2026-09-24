"""A stand-in Redis that also keeps hashes and records every command it is sent.

Builds on `StatefulRedis` (plain keys with real expiry) and adds HSET, HGET, HGETALL, HKEYS,
HVALS, HEXISTS, HLEN, HDEL and HSCAN on real state, which is what a `RedisDict` needs, so an
instance with `WEBSOCKET_MANAGER=redis` keeps its shared model pool here. `count(command, key)`
is how many times the instance sent that command for that key since `clear_log()`, and
`write_fields` and `write_value` change the state the way another worker of the same deployment
would.
"""

from __future__ import annotations

from integration.stateful_redis import NULL, StatefulRedis

HASH_COMMANDS = {"HSET", "HGET", "HGETALL", "HKEYS", "HVALS", "HEXISTS", "HLEN", "HDEL", "HSCAN"}


def _bulk(value: str) -> bytes:
    encoded = value.encode()
    return b"$%d\r\n%s\r\n" % (len(encoded), encoded)


def _array(values: list[str]) -> bytes:
    return b"*%d\r\n" % len(values) + b"".join(_bulk(value) for value in values)


class HashRedis(StatefulRedis):
    def __init__(self) -> None:
        self._hashes: dict[str, dict[str, str]] = {}
        self._log: list[tuple[str, str | None]] = []
        super().__init__()

    def count(self, command: str, key: str) -> int:
        with self._lock:
            return self._log.count((command, key))

    def clear_log(self) -> None:
        with self._lock:
            self._log.clear()

    def fields(self, key: str) -> dict[str, str]:
        with self._lock:
            return dict(self._hashes.get(key, {}))

    def write_fields(self, key: str, fields: dict[str, str]) -> None:
        with self._lock:
            self._hashes.setdefault(key, {}).update(fields)

    def write_value(self, key: str, value: str | None) -> None:
        """Set a plain key, or delete it with None."""
        with self._lock:
            if value is None:
                self._values.pop(key, None)
            else:
                self._values[key] = (value, None)

    def _answer(self, name: str, args: list[str]) -> bytes:
        with self._lock:
            self._log.append((name, args[0] if args else None))
            if name in HASH_COMMANDS:
                return self._hash_answer(name, args[0], args[1:])
            if name in ("DEL", "UNLINK"):
                removed = [key for key in args if self._hashes.pop(key, None) is not None]
                removed += [key for key in args if self._live(key) is not None]
                for key in args:
                    self._values.pop(key, None)
                return b":%d\r\n" % len(removed)
        return super()._answer(name, args)

    def _hash_answer(self, name: str, key: str, args: list[str]) -> bytes:
        fields = self._hashes.get(key, {})
        if name == "HSET":
            target = self._hashes.setdefault(key, {})
            pairs = dict(zip(args[::2], args[1::2]))
            added = len(pairs.keys() - target.keys())
            target.update(pairs)
            return b":%d\r\n" % added
        if name == "HGET":
            return _bulk(fields[args[0]]) if args[0] in fields else NULL
        if name == "HGETALL":
            return b"%%%d\r\n" % len(fields) + b"".join(
                _bulk(field) + _bulk(value) for field, value in fields.items()
            )
        if name == "HKEYS":
            return _array(list(fields))
        if name == "HVALS":
            return _array(list(fields.values()))
        if name == "HEXISTS":
            return b":%d\r\n" % (args[0] in fields)
        if name == "HLEN":
            return b":%d\r\n" % len(fields)
        if name == "HDEL":
            removed = [field for field in args if fields.pop(field, None) is not None]
            return b":%d\r\n" % len(removed)
        flat = [item for pair in fields.items() for item in pair]
        return b"*2\r\n" + _bulk("0") + _array(flat)  # HSCAN: everything in one page
