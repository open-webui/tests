"""A real Open Terminal (the `open-terminal` package from PyPI) for the instance to connect to.

`serving_open_terminal()` runs `open-terminal run` on a free port with an API key, in a scratch
home that is also its working directory, and yields it once it answers; it stops with the block.
The server runs from an environment of its own, so its dependencies never meet the backend's:
`OPEN_TERMINAL_BIN` names its executable, else `open-terminal` on `PATH` is used, and without
either the tests that need it skip. `connection(...)` is the admin's connection to it under
Settings > Integrations, for `configure_terminals` in `harness.terminal_server`.

`serving_full_open_terminal()` runs the full image instead, in a container on a loopback port:
the Open Terminal that has Chromium for Chrome mode, a desktop with sound and notebook kernels,
none of which the PyPI package or the slim image bring. `OPEN_TERMINAL_FULL_IMAGE` names the
image, which must already be on the Docker daemon (`harness/open_terminal_full.Dockerfile`
builds it); without it the tests that need it skip and say why, and with
`OWUI_TEST_REQUIRE_FULL_TERMINAL=1` they fail. Its home is inside the container, so files go in
through `write_file`.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import httpx
import pytest

from harness.instance import free_port

API_KEY = "sk-open-terminal"
FULL_IMAGE_ENV = "OPEN_TERMINAL_FULL_IMAGE"
FULL_REQUIRED_ENV = "OWUI_TEST_REQUIRE_FULL_TERMINAL"
CONTAINER_OWNER_LABEL = "owui-tests.open-terminal.owner"
# Chromium, the desktop and a notebook kernel together stay under this
FULL_MEMORY = "2g"
FULL_CPUS = "2"
# the Files tab is named after the folder it shows, but Home in the terminal's home (241c8c0e4)
HOME_TAB = "Home"
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


@dataclass
class ContainedOpenTerminal(OpenTerminal):
    """An Open Terminal in a container; `home` is the path inside it."""

    container: str = field(default="", kw_only=True)

    def client(self) -> httpx.Client:
        """The terminal's own API with its key, for what a test sets up or reads directly."""
        return httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120.0,
        )

    def write_file(self, path: str, content: str) -> None:
        """Put a text file into the terminal; `path` is relative to its home or absolute."""
        with self.client() as client:
            written = client.post("/files/write", json={"path": path, "content": content})
        assert written.status_code == 200, f"writing {path} failed: {written.text}"

    def logs(self) -> str:
        """The container's log, for a failure message."""
        from harness.terminals import _docker

        return _docker("logs", "--tail", "200", self.container, check=False)


def open_terminal_binary() -> str | None:
    return os.getenv("OPEN_TERMINAL_BIN") or shutil.which("open-terminal")


@contextlib.contextmanager
def serving_open_terminal(env: dict[str, str] | None = None) -> Iterator[OpenTerminal]:
    """Run the terminal; `env` adds to its environment, such as an agent command override."""
    binary = open_terminal_binary()
    if binary is None:
        pytest.skip("needs open-terminal (set OPEN_TERMINAL_BIN or put it on PATH)")
    scratch = Path(tempfile.mkdtemp(prefix="open-terminal-"))
    home = scratch / "home"
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


def full_image_unavailable() -> str | None:
    """Why the full Open Terminal cannot run here, or None when it can."""
    from harness.terminals import _docker

    image = os.getenv(FULL_IMAGE_ENV)
    if not image:
        return f"needs the full Open Terminal image (set {FULL_IMAGE_ENV})"
    if shutil.which("docker") is None or not _docker("info", "--format", "{{.ID}}", check=False):
        return "needs a working Docker daemon (`docker info` must succeed for this user)"
    if not _docker("image", "inspect", "--format", "{{.Id}}", image, check=False):
        return f"{FULL_IMAGE_ENV} names {image}, which is not on this Docker daemon (build it)"
    return None


def sweep_containers() -> list[str]:
    """Remove the full terminals whose test process is gone; returns their names."""
    from harness.terminals import _docker, _owner_alive

    listed = _docker(
        *("ps", "-a", "--filter", f"label={CONTAINER_OWNER_LABEL}"),
        *("--format", f'{{{{.Names}}}} {{{{.Label "{CONTAINER_OWNER_LABEL}"}}}}'),
        check=False,
    )
    orphans = [
        name
        for name, _, owner in (line.partition(" ") for line in listed.splitlines())
        if not _owner_alive(owner)
    ]
    if orphans:
        _docker("rm", "-f", "-v", *orphans, check=False)
    return orphans


@contextlib.contextmanager
def serving_full_open_terminal(
    env: dict[str, str] | None = None,
) -> Iterator[ContainedOpenTerminal]:
    """Run the full image; `env` adds to the terminal's settings."""
    from harness.terminals import _docker, _owner

    reason = full_image_unavailable()
    if reason and os.getenv(FULL_REQUIRED_ENV):
        pytest.fail(reason)
    if reason:
        pytest.skip(reason)
    sweep_containers()
    name = f"owui-open-terminal-{uuid.uuid4().hex[:12]}"
    settings = {"OPEN_TERMINAL_API_KEY": API_KEY, **(env or {})}
    setting_options = [option for pair in settings.items() for option in ("-e", "=".join(pair))]
    try:
        _docker(
            *("run", "-d", "--name", name, "--label", f"{CONTAINER_OWNER_LABEL}={_owner()}"),
            *("-p", "127.0.0.1::8000", "--memory", FULL_MEMORY, "--cpus", FULL_CPUS),
            *setting_options,
            os.environ[FULL_IMAGE_ENV],
        )
        address = _docker("port", name, "8000/tcp").splitlines()[0]
        terminal = ContainedOpenTerminal(
            base_url=f"http://{address}", home=Path("/home/user"), container=name
        )
        _wait_until_the_container_answers(terminal)
        yield terminal
    finally:
        _docker("rm", "-f", "-v", name, check=False)


def _wait_until_the_container_answers(terminal: ContainedOpenTerminal) -> None:
    from harness.terminals import _docker

    deadline = time.monotonic() + BOOT_SECONDS
    while time.monotonic() < deadline:
        running = _docker(
            "inspect", "--format", "{{.State.Running}}", terminal.container, check=False
        )
        if running != "true":
            pytest.fail(f"the full Open Terminal exited during boot:\n{terminal.logs()[-2000:]}")
        try:
            if httpx.get(f"{terminal.base_url}/openapi.json", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    pytest.fail(f"the full Open Terminal did not answer within {BOOT_SECONDS}s")
