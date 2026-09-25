"""Put the scratch instances on a real Postgres and Redis when a run asks for it.

`OWUI_TEST_DATABASE=postgres` gives every instance `launch` boots a database of its own on one
embedded Postgres (`pgserver`) per session, dropped when the instance stops. `OWUI_TEST_REDIS=1`
gives every instance a `redis-server` of its own on a free port, and sets what Open WebUI needs
to keep its sockets, config sync and task tracking there. Without them every instance stays on
SQLite and in-process state, as before.

An instance that brings its own database or Redis keeps it: a `DATABASE_URL`, a `DATA_DIR`
holding a `webui.db` a test prepared, or any Redis setting (the stand-ins under `integration/`).
"""

from __future__ import annotations

import atexit
import contextlib
import os
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Iterator

import pytest

DATABASE = os.getenv("OWUI_TEST_DATABASE", "sqlite").strip().lower() or "sqlite"
REDIS = os.getenv("OWUI_TEST_REDIS", "").strip().lower() in ("1", "true", "yes")

if DATABASE not in ("sqlite", "postgres"):
    raise RuntimeError(f"OWUI_TEST_DATABASE must be sqlite or postgres, not {DATABASE!r}")

REDIS_SETTINGS = ("REDIS_URL", "WEBSOCKET_REDIS_URL", "WEBSOCKET_MANAGER", "REDIS_SENTINEL_HOSTS")

_postgres = None


def on_postgres(instance) -> bool:
    """Whether a launched instance keeps its data in Postgres."""
    return instance.database_url.startswith("postgres")


def _brings_own_database(extra_env: dict[str, str]) -> bool:
    data_dir = extra_env.get("DATA_DIR")
    return "DATABASE_URL" in extra_env or bool(data_dir and (Path(data_dir) / "webui.db").exists())


def _postgres_server():
    global _postgres
    if _postgres is None:
        pgserver = pytest.importorskip(
            "pgserver", reason="OWUI_TEST_DATABASE=postgres needs pgserver (the postgres extra)"
        )
        pgdata = tempfile.mkdtemp(prefix="owui-pg-")
        _postgres = pgserver.get_server(pgdata, cleanup_mode="stop")
        atexit.register(shutil.rmtree, pgdata, ignore_errors=True)
        atexit.register(_postgres.cleanup)
    return _postgres


@contextlib.contextmanager
def _postgres_database() -> Iterator[str]:
    server = _postgres_server()
    name = f"owui_{uuid.uuid4().hex[:12]}"
    server.psql(f"CREATE DATABASE {name};")
    try:
        yield server.get_uri(name)
    finally:
        server.psql(f"DROP DATABASE IF EXISTS {name} WITH (FORCE);")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def _redis_server() -> Iterator[str]:
    binary = shutil.which("redis-server")
    if binary is None:
        pytest.skip("OWUI_TEST_REDIS=1 needs a redis-server binary on PATH")
    port = _free_port()
    command = [binary, "--port", str(port), "--bind", "127.0.0.1", "--save", ""]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 10
        while not _answers_ping(port):
            if process.poll() is not None or time.monotonic() > deadline:
                pytest.fail(f"redis-server did not start on port {port}")
            time.sleep(0.05)
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        process.terminate()
        process.wait(timeout=10)


def _answers_ping(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1) as connection:
            connection.sendall(b"PING\r\n")
            return connection.recv(16).startswith(b"+PONG")
    except OSError:
        return False


@contextlib.contextmanager
def services_for(extra_env: dict[str, str]) -> Iterator[dict[str, str]]:
    """The environment that puts one instance on the backends this run asked for."""
    with contextlib.ExitStack() as stack:
        env: dict[str, str] = {}
        if DATABASE == "postgres" and not _brings_own_database(extra_env):
            env["DATABASE_URL"] = stack.enter_context(_postgres_database())
        if REDIS and not any(name in extra_env for name in REDIS_SETTINGS):
            env["REDIS_URL"] = stack.enter_context(_redis_server())
            env["WEBSOCKET_MANAGER"] = "redis"
        yield env


def write_rows(instance, statement: str, rows: list[dict]) -> None:
    """Run one write statement with `:name` parameters on the instance's own database."""
    import sqlalchemy

    options = {"timeout": 30} if instance.database_url.startswith("sqlite") else {}
    engine = sqlalchemy.create_engine(instance.database_url, connect_args=options)
    try:
        with engine.begin() as connection:
            connection.execute(sqlalchemy.text(statement), rows)
    finally:
        engine.dispose()
