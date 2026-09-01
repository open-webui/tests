"""Shared fixtures for the migrations regression suite."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Callable

import pytest

AlembicUpgradeRunner = Callable[[Path, str, Path], "tuple[int, str, str]"]


@pytest.fixture
def run_alembic_upgrade_head() -> AlembicUpgradeRunner:
    """Run alembic upgrade head as a subprocess against the given DB URL.

    Uses a subprocess so each invocation gets a fresh sys.modules — the
    open_webui.config module caches state in module globals after first
    import. Bypasses open_webui.config's run_migrations() wrapper (which
    silently swallows migration errors with `log.exception(...)`), so the
    alembic exception propagates as a non-zero exit code.
    """

    def _run(backend: Path, database_url: str, data_dir: Path) -> "tuple[int, str, str]":
        script = textwrap.dedent(
            f"""
            import os, sys
            os.environ['DATABASE_URL'] = {database_url!r}
            os.environ['DATA_DIR'] = {str(data_dir)!r}
            sys.path.insert(0, {str(backend)!r})

            from alembic import command
            from alembic.config import Config as AlembicConfig
            from open_webui.env import OPEN_WEBUI_DIR

            cfg = AlembicConfig(OPEN_WEBUI_DIR / 'alembic.ini')
            cfg.set_main_option('script_location', str(OPEN_WEBUI_DIR / 'migrations'))
            command.upgrade(cfg, 'head')
            print('OK')
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        return result.returncode, result.stdout, result.stderr

    return _run
