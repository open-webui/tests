"""Boot the checkout on a data directory a test prepared, including one it has to refuse.

`snapshot_database(instance, data_dir)` copies a running instance's SQLite database, migrated to
head, into `data_dir`, for a test to alter before the next boot. `boot_until_settled(data_dir)`
starts the backend there the way `launch` does, with any extra `settings`, and reports how the
boot ended: the process exiting, or `/health` answering. A boot that is expected to work needs
nothing new: `launch` and `instance_with` take `DATA_DIR` as an extra variable, and signing up
the admin works on a data directory without accounts; `booted_again(...)` is such an instance
on the data directory a first boot with other settings left behind. `with_legacy_config(...)`
is one too, shared by the API and browser tests: its data directory starts with a legacy
`config.json`, whose import at boot writes each key as a config row verbatim, ahead of the
boot's repair of old row shapes. `killed_when(data_dir, condition)` kills a boot outright
once `condition()` holds, as a crash or an out-of-memory kill would. `serving(data_dir)` keeps
the backend running on a data directory that already has accounts, with its process id and how
long it took to answer `/health`, and
`restored_postgres(dump)` loads a `pg_dump` into an embedded Postgres for it to use.
`release_data(archive, root)` unpacks a data set an older release wrote
(`integration/migrations/upgrade_data/`), on its own database, ready to serve.
`manual_alembic(data_dir, ...)` runs the `alembic` command an operator runs by hand, which is
also how a test builds a database as a release from before a migration left it.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
import weakref
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import httpx
import pytest
import sqlalchemy

from harness.instance import (
    LAUNCHER,
    LaunchedInstance,
    free_port,
    isolated_env,
    resolve_backend,
    without_colour,
)
from harness.upstream import MOCK_MODEL_ID

_legacy_dirs: weakref.WeakKeyDictionary[Callable, Path] = weakref.WeakKeyDictionary()

# config rows in the shapes older releases stored them
LEGACY_CONFIG_ROWS = {
    "ui.default_models": [MOCK_MODEL_ID, "  second-model  ", ""],
    "ui.default_pinned_models": f"{MOCK_MODEL_ID},second-model",
    "user.permissions.chat.controls": False,
    "user.permissions.workspace.models": True,
}


@dataclass
class BootOutcome:
    healthy: bool  # /health answered 200
    exit_code: int | None  # None while the process still runs
    log: str


def snapshot_database(instance: LaunchedInstance, data_dir: Path) -> Path:
    """A consistent copy of the instance's database as `data_dir/webui.db`."""
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "webui.db"
    with closing(sqlite3.connect(instance.data_dir / "webui.db")) as source:
        with closing(sqlite3.connect(target)) as copy:
            source.backup(copy)
    return target


def with_legacy_config(
    instance_with: Callable[[dict[str, str]], LaunchedInstance], tmp_path_factory
) -> LaunchedInstance:
    """The instance booted on a data directory holding `LEGACY_CONFIG_ROWS` as `config.json`.

    Each module's `instance_with` gets a fresh directory, since the first boot migrates it.
    """
    data_dir = _legacy_dirs.get(instance_with)
    if data_dir is None:
        data_dir = tmp_path_factory.mktemp("legacy-config")
        (data_dir / "config.json").write_text(json.dumps(LEGACY_CONFIG_ROWS), encoding="utf-8")
        _legacy_dirs[instance_with] = data_dir
    return instance_with({"DATA_DIR": str(data_dir)})


def booted_again(
    instance_with: Callable[[dict[str, str]], LaunchedInstance],
    data_dir: Path,
    first: dict[str, str],
    second: dict[str, str],
) -> LaunchedInstance:
    """Boot on a fresh `data_dir` with `first` until healthy, stop, then boot it with `second`."""
    data_dir.mkdir(parents=True)
    outcome = boot_until_settled(data_dir, settings=first)
    assert outcome.healthy, f"the first boot failed:\n{outcome.log[-3000:]}"
    return instance_with({"DATA_DIR": str(data_dir), **second})


def _answers_health(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url}/health", timeout=3.0).status_code == 200
    except httpx.HTTPError:
        return False


def _start(
    data_dir: Path, backend: Path | None, settings: dict[str, str]
) -> tuple[subprocess.Popen, str, Path]:
    """Start the backend on `data_dir`; returns the process, its URL and its scratch directory."""
    backend = backend or resolve_backend()
    if backend is None:
        pytest.skip("open-webui backend source not found (set OPEN_WEBUI_SOURCE_DIR)")
    scratch = Path(tempfile.mkdtemp(prefix="owui-prepared-"))
    (scratch / "static").mkdir()
    port = free_port()
    env = isolated_env(
        {
            "PYTHONUNBUFFERED": "1",
            "WEBUI_SECRET_KEY": "integration-secret-key",
            "DATA_DIR": str(data_dir),
            "STATIC_DIR": str(scratch / "static"),
            "FRONTEND_BUILD_DIR": str(scratch / "build"),
            "OFFLINE_MODE": "true",
            "ENABLE_OLLAMA_API": "false",
            "ENABLE_OPENAI_API": "false",
            **settings,
        }
    )
    with open(scratch / "server.log", "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-c", LAUNCHER, str(backend), str(port)],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    return process, f"http://127.0.0.1:{port}", scratch


def _stop(process: subprocess.Popen, scratch: Path) -> None:
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
    shutil.rmtree(scratch, ignore_errors=True)


def _read_log(scratch: Path) -> str:
    return without_colour((scratch / "server.log").read_text(encoding="utf-8", errors="replace"))


def boot_until_settled(
    data_dir: Path, timeout: float = 180.0, settings: dict[str, str] | None = None
) -> BootOutcome:
    """Start the backend on `data_dir` until it exits or answers `/health`, then stop it."""
    process, base_url, scratch = _start(data_dir, None, settings or {})
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            exit_code = process.poll()
            healthy = exit_code is None and _answers_health(base_url)
            if exit_code is not None or healthy:
                return BootOutcome(healthy=healthy, exit_code=exit_code, log=_read_log(scratch))
            time.sleep(0.5)
        pytest.fail(f"the backend neither exited nor answered /health within {timeout:.0f}s")
    finally:
        _stop(process, scratch)


def killed_when(
    data_dir: Path,
    condition: Callable[[], bool],
    settings: dict[str, str] | None = None,
    timeout: float = 180.0,
) -> str:
    """Start the backend on `data_dir` and kill it outright once `condition()` holds; its log.

    Fails when the backend exits or answers `/health` first, since then nothing was interrupted.
    """
    process, base_url, scratch = _start(data_dir, None, settings or {})
    try:
        deadline = time.monotonic() + timeout
        while not condition():
            if process.poll() is not None or _answers_health(base_url):
                pytest.fail(
                    f"the backend got past the point to kill it at:\n{_read_log(scratch)[-3000:]}"
                )
            if time.monotonic() > deadline:
                pytest.fail(f"the point to kill the backend at never came within {timeout:.0f}s")
            time.sleep(0.05)
        process.kill()
        process.wait(timeout=30)
        return _read_log(scratch)
    finally:
        _stop(process, scratch)


@dataclass
class RunningBackend:
    base_url: str
    scratch: Path
    process_id: int
    seconds_to_start: float  # from launch until /health answered

    def client(self, token: str | None = None) -> httpx.Client:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return httpx.Client(base_url=self.base_url, headers=headers, timeout=120.0)

    def log(self) -> str:
        return _read_log(self.scratch)


@contextlib.contextmanager
def serving(
    data_dir: Path,
    settings: dict[str, str] | None = None,
    backend: Path | None = None,
    timeout: float = 300.0,
) -> Iterator[RunningBackend]:
    """The backend running on `data_dir` as it is, accounts included, until the block ends.

    `backend` defaults to the checkout under test; the seeding script passes an older release.
    """
    launched = time.monotonic()
    process, base_url, scratch = _start(data_dir, backend, settings or {})
    try:
        deadline = launched + timeout
        while not _answers_health(base_url):
            if process.poll() is not None or time.monotonic() > deadline:
                tail = _read_log(scratch)[-4000:]
                pytest.fail(f"the backend did not start on {data_dir}:\n{tail}")
            time.sleep(0.5)
        yield RunningBackend(base_url, scratch, process.pid, time.monotonic() - launched)
    finally:
        _stop(process, scratch)


@contextlib.contextmanager
def restored_postgres(dump: Path, root: Path) -> Iterator[str]:
    """An embedded Postgres (`pgserver`) holding a plain `pg_dump`; yields its SQLAlchemy URL."""
    pgserver = pytest.importorskip("pgserver", reason="pgserver not installed")
    (root / "pgdata").mkdir(parents=True)
    server = pgserver.get_server(str(root / "pgdata"), cleanup_mode="stop")
    try:
        url = server.get_uri()
        psql = Path(pgserver.__file__).parent / "pginstall" / "bin" / "psql"
        restore = [str(psql), "--quiet", "--set", "ON_ERROR_STOP=1", "--file", str(dump), url]
        subprocess.run(restore, check=True, capture_output=True)
        yield url.replace("postgresql://", "postgresql+psycopg2://", 1)
    finally:
        with contextlib.suppress(Exception):  # a server that failed to start has nothing to stop
            server.cleanup()


@dataclass
class ReleaseData:
    data_dir: Path
    manifest: dict  # what the seeding script made, and for whom
    database_url: str
    settings: dict[str, str]  # what `serving` needs to use this data set's database


# uploads are stored by absolute path; a real upgrade keeps its data directory where it was
_RELOCATE_UPLOADS = "UPDATE file SET path = REPLACE(path, :old, :new)"


def run_sql(database_url: str, statement: str, rows: list[dict] | dict) -> None:
    """Run one statement with `:name` parameters on a database no backend is serving yet."""
    engine = sqlalchemy.create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(sqlalchemy.text(statement), rows)
    finally:
        engine.dispose()


@contextlib.contextmanager
def release_data(archive: Path, root: Path) -> Iterator[ReleaseData]:
    """The data set `archive` (`<name>.tar.gz` beside `<name>.json`) unpacked under `root`.

    A Postgres set restores its `pg_dump` into an embedded server that lives for the block.
    """
    manifest = json.loads(
        archive.with_name(archive.name.removesuffix(".tar.gz") + ".json").read_text(
            encoding="utf-8"
        )
    )
    data_dir = root / "data"
    with tarfile.open(archive) as unpacking:
        unpacking.extractall(data_dir, filter="data")
    with contextlib.ExitStack() as stack:
        if manifest["engine"] == "postgres":
            database_url = stack.enter_context(restored_postgres(data_dir / "webui.sql", root))
            settings = {"DATABASE_URL": database_url}
        else:
            database_url = f"sqlite:///{data_dir / 'webui.db'}"
            settings = {}
        relocation = {"old": manifest["data_dir"], "new": str(data_dir)}
        run_sql(database_url, _RELOCATE_UPLOADS, relocation)
        yield ReleaseData(data_dir, manifest, database_url, settings)


def manual_alembic(
    data_dir: Path, *arguments: str, database_url: str | None = None
) -> subprocess.CompletedProcess:
    """`alembic <arguments>` from `backend/open_webui`, as the manual migration guide runs it."""
    backend = resolve_backend()
    if backend is None:
        pytest.skip("open-webui backend source not found (set OPEN_WEBUI_SOURCE_DIR)")
    (data_dir / "static").mkdir(parents=True, exist_ok=True)
    settings = {
        "DATA_DIR": str(data_dir),
        "STATIC_DIR": str(data_dir / "static"),
        "WEBUI_SECRET_KEY": "integration-secret-key",
        "PYTHONPATH": str(backend),
    }
    if database_url:
        settings["DATABASE_URL"] = database_url
    return subprocess.run(
        [sys.executable, "-m", "alembic", *arguments],
        cwd=backend / "open_webui",
        env=isolated_env(settings),
        capture_output=True,
        text=True,
        timeout=300,
    )
