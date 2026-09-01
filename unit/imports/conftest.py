"""Shared fixture for standalone cyclic-import guards."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest

ImportRunner = Callable[[Path, str, str, Path], "tuple[int, str, str]"]


@pytest.fixture
def run_fresh_import() -> ImportRunner:
    """Import a single dotted module name in a fresh subprocess and report the result.

    A subprocess gives a truly empty sys.modules, matching a cold app boot or a
    fresh alembic invocation. Importing the same module again inside this same
    pytest session, after some other test already imported it, would hide a
    reintroduced circular import, since Python caches completed modules.
    """

    def _run(
        backend: Path, database_url: str, module: str, data_dir: Path
    ) -> "tuple[int, str, str]":
        script = (
            "import os, sys\n"
            "os.environ['WEBUI_SECRET_KEY'] = 'test-secret-key'\n"
            f"os.environ['DATABASE_URL'] = {database_url!r}\n"
            f"os.environ['DATA_DIR'] = {str(data_dir)!r}\n"
            f"sys.path.insert(0, {str(backend)!r})\n"
            f"import {module}\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        return result.returncode, result.stdout, result.stderr

    return _run
