"""Run migration code in a fresh interpreter against a scratch database.

Each run gets an empty `sys.modules`: `open_webui.config` runs the migrations on first import
and caches state in module globals, so a second run inside the test process would see the
first. The code runs with `ALEMBIC` in front of it, which builds the checkout's own Alembic
config as `cfg`, and reports back by printing `RESULT:` and a JSON document; a run that raises
or prints nothing fails the test with its output.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pytest

ALEMBIC = """
import json, os
from alembic import command
from alembic.config import Config as AlembicConfig
from open_webui.env import OPEN_WEBUI_DIR

cfg = AlembicConfig(OPEN_WEBUI_DIR / 'alembic.ini')
cfg.set_main_option('script_location', str(OPEN_WEBUI_DIR / 'migrations'))
"""


@dataclass
class ScratchDatabase:
    backend: Path
    url: str
    data_dir: Path

    def run(self, body: str, what: str, timeout: int = 300, **env: str) -> dict:
        """Run `body` after `ALEMBIC` on this database; returns what it printed as RESULT."""
        environment = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "WEBUI_SECRET_KEY": "test-secret-key",
            "DATABASE_URL": self.url,
            "DATA_DIR": str(self.data_dir),
            "STATIC_DIR": str(self.data_dir / "static"),
            **env,
        }
        script = (
            f"import sys\nsys.path.insert(0, {str(self.backend)!r})\n"
            + ALEMBIC
            + textwrap.dedent(body)
        )
        finished = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            cwd=self.data_dir,
        )
        line = next(
            (line for line in finished.stdout.splitlines() if line.startswith("RESULT:")), None
        )
        if finished.returncode != 0 or line is None:
            pytest.fail(
                f"{what} failed (exit code {finished.returncode}).\n"
                f"--- stderr (tail) ---\n{finished.stderr[-3000:]}\n"
                f"--- stdout (tail) ---\n{finished.stdout[-1500:]}"
            )
        return json.loads(line.removeprefix("RESULT:"))


def _data_dir(root: Path) -> Path:
    (root / "static").mkdir(parents=True, exist_ok=True)
    return root


def sqlite_at(backend: Path, root: Path) -> ScratchDatabase:
    """A fresh SQLite file under `root`, as a new install starts with."""
    return ScratchDatabase(backend, f"sqlite:///{root / 'webui.db'}", _data_dir(root))


@contextlib.contextmanager
def postgres_at(backend: Path, root: Path) -> Iterator[ScratchDatabase]:
    """A fresh embedded Postgres (`pgserver`) under `root`, stopped again afterwards."""
    pgserver = pytest.importorskip("pgserver", reason="pgserver not installed")
    (root / "pgdata").mkdir(parents=True)
    server = pgserver.get_server(str(root / "pgdata"), cleanup_mode=None)
    try:
        url = server.get_uri().replace("postgresql://", "postgresql+psycopg2://", 1)
        yield ScratchDatabase(backend, url, _data_dir(root / "data"))
    finally:
        with contextlib.suppress(Exception):  # a server that failed to start has nothing to stop
            server.cleanup()


@pytest.fixture
def sqlite_database(open_webui_backend: Path, tmp_path: Path) -> ScratchDatabase:
    return sqlite_at(open_webui_backend, tmp_path)
