"""Boot the checkout on a data directory a test prepared, including one it has to refuse.

`snapshot_database(instance, data_dir)` copies a running instance's SQLite database, migrated to
head, into `data_dir`, for a test to alter before the next boot. `boot_until_settled(data_dir)`
starts the backend there the way `launch` does and reports how the boot ended: the process
exiting, or `/health` answering. A boot that is expected to work needs nothing new: `launch`
and `instance_with` take `DATA_DIR` as an extra variable, and signing up the admin works on a
data directory without accounts. `with_legacy_config(...)` is such an instance, shared by the
API and browser tests: its data directory starts with a legacy `config.json`, whose import at
boot writes each key as a config row verbatim, ahead of the boot's repair of old row shapes.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx
import pytest

from harness.instance import (
    INHERITED_ENV_TO_DROP,
    LAUNCHER,
    LaunchedInstance,
    free_port,
    resolve_backend,
)
from harness.upstream import MOCK_MODEL_ID

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
    """The instance booted on a data directory holding `LEGACY_CONFIG_ROWS` as `config.json`."""
    data_dir = tmp_path_factory.getbasetemp() / "legacy-config"
    if not data_dir.exists():
        data_dir.mkdir()
        (data_dir / "config.json").write_text(json.dumps(LEGACY_CONFIG_ROWS), encoding="utf-8")
    return instance_with({"DATA_DIR": str(data_dir)})


def _answers_health(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url}/health", timeout=3.0).status_code == 200
    except httpx.HTTPError:
        return False


def boot_until_settled(data_dir: Path, timeout: float = 180.0) -> BootOutcome:
    """Start the backend on `data_dir` until it exits or answers `/health`, then stop it."""
    backend = resolve_backend()
    if backend is None:
        pytest.skip("open-webui backend source not found (set OPEN_WEBUI_SOURCE_DIR)")
    scratch = Path(tempfile.mkdtemp(prefix="owui-prepared-"))
    (scratch / "static").mkdir()
    port = free_port()
    env = {name: value for name, value in os.environ.items() if name not in INHERITED_ENV_TO_DROP}
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "WEBUI_SECRET_KEY": "integration-secret-key",
            "DATA_DIR": str(data_dir),
            "STATIC_DIR": str(scratch / "static"),
            "FRONTEND_BUILD_DIR": str(scratch / "build"),
            "OFFLINE_MODE": "true",
            "ENABLE_OLLAMA_API": "false",
            "ENABLE_OPENAI_API": "false",
        }
    )
    log_path = scratch / "server.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-c", LAUNCHER, str(backend), str(port)],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            exit_code = process.poll()
            healthy = exit_code is None and _answers_health(f"http://127.0.0.1:{port}")
            if exit_code is not None or healthy:
                log = log_path.read_text(encoding="utf-8", errors="replace")
                return BootOutcome(healthy=healthy, exit_code=exit_code, log=log)
            time.sleep(0.5)
        pytest.fail(f"the backend neither exited nor answered /health within {timeout:.0f}s")
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        shutil.rmtree(scratch, ignore_errors=True)
