"""A real Open Terminal (the `open-terminal` package from PyPI) for the instance to connect to.

`serving_open_terminal()` runs `open-terminal run` on a free port with an API key, in a scratch
home that is also its working directory, and yields it once it answers; it stops with the block.
The server runs from an environment of its own, so its dependencies never meet the backend's:
`OPEN_TERMINAL_BIN` names its executable, else `open-terminal` on `PATH` is used, and without
either the tests that need it skip. `connection(...)` is the admin's connection to it under
Settings > Integrations, for `configure_terminals` in `harness.terminal_server`.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import httpx
import pytest

from harness.instance import free_port

API_KEY = "sk-open-terminal"
# the Files tab is named after the folder it shows, so it opens as this one
HOME_FOLDER = "home"
BOOT_SECONDS = 60


@dataclass
class OpenTerminal:
    base_url: str
    home: Path
    api_key: str = API_KEY

    def connection(self, **fields) -> dict:
        """The terminal connection as the admin form saves it, bearer auth with its key."""
        return {
            "id": f"open-terminal-{uuid.uuid4().hex[:8]}",
            "name": "Open Terminal",
            "enabled": True,
            "url": self.base_url,
            "path": "/openapi.json",
            "key": self.api_key,
            "auth_type": "bearer",
            "config": {"access_grants": []},
            **fields,
        }


def open_terminal_binary() -> str | None:
    return os.getenv("OPEN_TERMINAL_BIN") or shutil.which("open-terminal")


@contextlib.contextmanager
def serving_open_terminal(env: dict[str, str] | None = None) -> Iterator[OpenTerminal]:
    """Run the terminal; `env` adds to its environment, such as an agent command override."""
    binary = open_terminal_binary()
    if binary is None:
        pytest.skip("needs open-terminal (set OPEN_TERMINAL_BIN or put it on PATH)")
    scratch = Path(tempfile.mkdtemp(prefix="open-terminal-"))
    home = scratch / HOME_FOLDER
    home.mkdir()
    port = free_port()
    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(scratch / "config"),
        "XDG_STATE_HOME": str(scratch / "state"),
        "OPEN_TERMINAL_API_KEY": API_KEY,
        **(env or {}),
    }
    command = [binary, "run", "--host", "127.0.0.1", "--port", str(port), "--cwd", str(home)]
    log = open(scratch / "server.log", "w", encoding="utf-8")
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_until_it_answers(process, base_url, scratch / "server.log")
        yield OpenTerminal(base_url=base_url, home=home)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()
        shutil.rmtree(scratch, ignore_errors=True)


def _wait_until_it_answers(process: subprocess.Popen, base_url: str, log_path: Path) -> None:
    deadline = time.monotonic() + BOOT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"open-terminal exited during boot:\n{log_path.read_text()[-2000:]}")
        try:
            if httpx.get(f"{base_url}/openapi.json", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    pytest.fail(f"open-terminal did not answer within {BOOT_SECONDS}s")
