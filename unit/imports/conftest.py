"""Import one backend module the way a cold boot does, and report what it left behind.

A fresh interpreter has an empty `sys.modules`, like a cold app boot or a fresh alembic run.
Importing a module again inside this pytest session, after another test already imported it,
would hide a reintroduced circular import, since Python caches completed modules. Importing
`open_webui.config` runs the migrations as a side effect, so a clean import also has to leave
the `config` table behind: an import that succeeded while the migrations did not run is the
other half of the failure.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest


@dataclass
class ColdImport:
    module: str
    failure: str | None  # the interpreter's output when the import raised
    tables: set[str]


def import_cold(backend: Path, root: Path, module: str) -> ColdImport:
    """Import `module` in a fresh interpreter on a fresh SQLite database under `root`."""
    database_url = f"sqlite:///{(root / 'webui.db').as_posix()}"
    (root / "static").mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "WEBUI_SECRET_KEY": "test-secret-key",
        "DATABASE_URL": database_url,
        "DATA_DIR": str(root),
        "STATIC_DIR": str(root / "static"),
    }
    script = f"import sys\nsys.path.insert(0, {str(backend)!r})\nimport {module}"
    finished = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        env=environment,
        cwd=root,
    )
    if finished.returncode != 0:
        output = (
            f"--- stderr (tail) ---\n{finished.stderr[-3000:]}\n"
            f"--- stdout (tail) ---\n{finished.stdout[-1000:]}"
        )
        return ColdImport(module, output, set())

    from sqlalchemy import create_engine, inspect

    engine = create_engine(database_url)
    try:
        return ColdImport(module, None, set(inspect(engine).get_table_names()))
    finally:
        engine.dispose()


@pytest.fixture
def cold_import(open_webui_backend: Path, tmp_path: Path) -> Callable[[str], set[str]]:
    """`cold_import("open_webui.x")` imports it cold, fails the test with the interpreter's
    output when that raises, and returns the tables the import left in the database."""

    def run(module: str) -> set[str]:
        imported = import_cold(open_webui_backend, tmp_path, module)
        if imported.failure:
            pytest.fail(f"import {module} failed from a cold interpreter.\n{imported.failure}")
        return imported.tables

    return run
