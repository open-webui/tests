"""Boot a scratch Open WebUI from the checkout and tear it down again.

The instance runs in its own process on a free port against a scratch data directory, with the
mock upstream as its only model provider, so a test can read its log, measure its process and
script every model reply. When the checkout has a built frontend it is served too, which is
what the browser suite drives.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import httpx
import pytest

from harness import backends
from harness.upstream import MockUpstream

ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "adminpassword123"

# `loop="none"` as `open-webui serve` does: uvicorn's default loop leaves Windows sockets unwired
LAUNCHER = """
import os, sys
if os.environ.get("COVERAGE_PROCESS_START"):
    import coverage
    coverage.process_startup()
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])
import uvicorn
import open_webui.main  # noqa: F401
uvicorn.run(
    "open_webui.main:app",
    host="127.0.0.1",
    port=int(sys.argv[2]),
    workers=1,
    loop="none" if sys.platform == "win32" else "auto",
    log_level="warning",
)
"""


# set, not just unset: open_webui.env fills unset names from the checkout's .env
ISOLATED_ENV = {
    "DATABASE_TYPE": "",
    "OPENAI_API_BASE_URLS": "",
    "OPENAI_API_KEYS": "",
    "OPENAI_API_CONFIGS": "",
    "ENABLE_LOGIN_FORM": "true",
    "GLOBAL_LOG_LEVEL": "",
    "STORAGE_PROVIDER": "local",
    "VECTOR_DB": "chroma",
    "CHROMA_HTTP_HOST": "",
    "REDIS_URL": "",
    "REDIS_SENTINEL_HOSTS": "",
    "WEBSOCKET_MANAGER": "",
    "WEBUI_ADMIN_EMAIL": "",
    "WEBUI_ADMIN_PASSWORD": "",
}


def isolated_env(settings: dict[str, str]) -> dict[str, str]:
    """The caller's environment with `settings` and nothing that reaches the caller's services."""
    env = {**os.environ, **ISOLATED_ENV, **settings}
    derived = {
        "DATABASE_URL": f"sqlite:///{env['DATA_DIR']}/webui.db",
        "WEBSOCKET_REDIS_URL": env["REDIS_URL"],
    }
    env.update({name: value for name, value in derived.items() if name not in settings})
    return env


def without_colour(log: str) -> str:
    """The log as plain text: loguru colours it when it sees a CI provider's environment."""
    return re.sub(r"\x1b\[[0-9;]*m", "", log)


def resolve_backend() -> Path | None:
    env = os.getenv("OPEN_WEBUI_SOURCE_DIR")
    if env:
        candidate = Path(env).expanduser()
        return candidate if candidate.is_dir() else None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "open-webui" / "backend"
        if (candidate / "open_webui" / "retrieval" / "web" / "utils.py").is_file():
            return candidate
    return None


def resolve_frontend_build(backend: Path) -> Path | None:
    """The built frontend: `OPEN_WEBUI_BUILD_DIR`, else `build/` next to the backend."""
    env = os.getenv("OPEN_WEBUI_BUILD_DIR")
    candidate = Path(env).expanduser() if env else backend.parent / "build"
    return candidate if (candidate / "index.html").is_file() else None


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class LaunchedInstance:
    base_url: str
    pid: int
    log_path: Path
    data_dir: Path
    admin_token: str
    upstream: MockUpstream
    serves_frontend: bool
    database_url: str = ""
    redis_url: str = ""

    def client(self, token: str | None = None) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {token or self.admin_token}"},
            timeout=120.0,
        )

    def log_size(self) -> int:
        return self.log_path.stat().st_size

    def log_since(self, offset: int) -> str:
        with open(self.log_path, "rb") as handle:
            handle.seek(offset)
            return without_colour(handle.read().decode("utf-8", errors="replace"))

    def rss_bytes(self) -> int:
        """Resident set size of the server, without a third-party dependency.

        On Windows a venv's python.exe is a launcher whose child is the interpreter, so the
        working sets of the pid and its children are summed.
        """
        if sys.platform == "win32":
            query = (
                f"Get-CimInstance Win32_Process | Where-Object {{ $_.ProcessId -eq {self.pid} "
                f"-or $_.ParentProcessId -eq {self.pid} }} "
                "| Measure-Object WorkingSetSize -Sum | Select-Object -ExpandProperty Sum"
            )
            command = ["powershell", "-NoProfile", "-Command", query]
            return int(subprocess.check_output(command, text=True).strip())
        status = Path(f"/proc/{self.pid}/status").read_text().splitlines()
        rss_line = next(line for line in status if line.startswith("VmRSS:"))
        return int(rss_line.split()[1]) * 1024

    def cpu_seconds(self) -> float:
        """User plus system time the server has burned, read the same way as `rss_bytes`."""
        if sys.platform == "win32":
            query = (
                f"Get-CimInstance Win32_Process | Where-Object {{ $_.ProcessId -eq {self.pid} "
                f"-or $_.ParentProcessId -eq {self.pid} }} "
                "| Measure-Object -Property UserModeTime, KernelModeTime -Sum "
                "| Measure-Object -Property Sum -Sum | Select-Object -ExpandProperty Sum"
            )
            command = ["powershell", "-NoProfile", "-Command", query]
            return int(subprocess.check_output(command, text=True).strip()) / 1e7
        fields = Path(f"/proc/{self.pid}/stat").read_text().rsplit(") ", 1)[1].split()
        utime, stime = int(fields[11]), int(fields[12])
        return (utime + stime) / os.sysconf("SC_CLK_TCK")


def launch(upstream: MockUpstream, extra_env: dict[str, str]) -> Iterator[LaunchedInstance]:
    """Boot, seed the admin, yield, tear down. Use it as the body of a generator fixture."""
    backend = resolve_backend()
    if backend is None:
        pytest.skip("open-webui backend source not found (set OPEN_WEBUI_SOURCE_DIR)")
    build = resolve_frontend_build(backend)
    services = contextlib.ExitStack()
    service_env = services.enter_context(backends.services_for(extra_env))

    scratch = Path(tempfile.mkdtemp(prefix="owui-integration-"))
    for name in ("data", "static"):
        (scratch / name).mkdir()
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = isolated_env(
        {
            "PYTHONUNBUFFERED": "1",
            "WEBUI_SECRET_KEY": "integration-secret-key",
            "WEBUI_AUTH": "true",
            "WEBUI_URL": base_url,
            # socket.io derives its allowed origins from this; a mismatch is a 403 on the handshake
            "CORS_ALLOW_ORIGIN": "*",
            "DATA_DIR": str(scratch / "data"),
            "STATIC_DIR": str(scratch / "static"),
            "FRONTEND_BUILD_DIR": str(build or scratch / "build"),
            "OFFLINE_MODE": "true",
            "RAG_EMBEDDING_ENGINE": "openai",  # served by the mock, so nothing is downloaded
            # its default is the hardcoded api.openai.com, not OPENAI_API_BASE_URL
            "RAG_OPENAI_API_BASE_URL": upstream.base_url,
            "RAG_OPENAI_API_KEY": "sk-mock",
            "ENABLE_OLLAMA_API": "false",
            "ENABLE_OPENAI_API": "true",
            "OPENAI_API_BASE_URL": upstream.base_url,
            "OPENAI_API_KEY": "sk-mock",
            "OPENAI_API_BASE_URLS": upstream.base_url,
            "OPENAI_API_KEYS": "sk-mock",
            **service_env,
            **extra_env,
        }
    )
    log_path = scratch / "server.log"
    # an undrained pipe wedges the child once startup output fills it
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-c", LAUNCHER, str(backend), str(port)],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    try:
        _wait_for_health(proc, base_url, log_path)
        if env["WEBUI_AUTH"].lower() == "false":
            # the web client's empty sign-in, which makes the first visitor the admin
            signup = httpx.post(
                f"{base_url}/api/v1/auths/signin",
                json={"email": "", "password": ""},
                timeout=60.0,
            )
        else:
            signup = httpx.post(
                f"{base_url}/api/v1/auths/signup",
                json={"name": "Admin", "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
                timeout=60.0,
            )
        if signup.status_code != 200:
            pytest.fail(f"admin signup failed: HTTP {signup.status_code} {signup.text}")
        yield LaunchedInstance(
            base_url=base_url,
            pid=proc.pid,
            log_path=log_path,
            data_dir=scratch / "data",
            admin_token=signup.json()["token"],
            upstream=upstream,
            serves_frontend=build is not None,
            database_url=env["DATABASE_URL"],
            redis_url=env["REDIS_URL"],
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        keep_logs_in = os.getenv("OPEN_WEBUI_LOG_DIR")
        if keep_logs_in:
            Path(keep_logs_in).mkdir(parents=True, exist_ok=True)
            shutil.copy(log_path, Path(keep_logs_in) / f"server-{port}.log")
        shutil.rmtree(scratch, ignore_errors=True)
        services.close()


def _wait_for_health(proc: subprocess.Popen, base_url: str, log_path: Path) -> None:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            pytest.fail(f"backend exited during boot (code {proc.returncode}):\n{tail}")
        try:
            if httpx.get(f"{base_url}/health", timeout=3.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    pytest.fail(f"backend did not become healthy within 300s:\n{tail}")
