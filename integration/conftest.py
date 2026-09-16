"""Fixtures for tests that need an instance they control: its log, its process, its upstream.

The httpx tests in this directory run against whatever `OPEN_WEBUI_URL` points at. The
footprint and resilience tests cannot: they read the server log, measure the server process
and swap the model provider, Redis or vector DB for something they control. `launched_instance`
boots a scratch backend from the checkout (`OPEN_WEBUI_SOURCE_DIR`, or the sibling walk
`unit/conftest.py` uses) with a mock OpenAI upstream, seeds an admin and tears it down at the
end of the session; `degraded_instance` is a second one on a fake Redis and a dead vector DB.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Generator

import httpx
import pytest

from integration.fake_redis import FakeRedis

MOCK_MODEL_ID = "mock-model"
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "adminpassword123"

# Mirrors scripts/e2e_instance.py: `loop="none"` is what `open-webui serve` does, a bare
# uvicorn default loop on Windows leaves the websocket transport unwired.
LAUNCHER = """
import os, sys
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


def _resolve_backend() -> Path | None:
    env = os.getenv("OPEN_WEBUI_SOURCE_DIR")
    if env:
        candidate = Path(env).expanduser()
        return candidate if candidate.is_dir() else None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "open-webui" / "backend"
        if (candidate / "open_webui" / "retrieval" / "web" / "utils.py").is_file():
            return candidate
    return None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class MockUpstream:
    """An OpenAI-shaped provider whose /chat/completions behaviour a test sets per call."""

    base_url: str
    behaviour: dict

    def reset(self, mode: str, **options) -> None:
        self.behaviour = {"mode": mode, **options}


def _mock_upstream_handler(upstream: MockUpstream):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # chunked framing needs it

        def log_message(self, *args) -> None:
            pass

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.endswith("/models"):
                self._send_json(200, {"object": "list", "data": [{"id": MOCK_MODEL_ID}]})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path.endswith("/embeddings"):
                inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
                vectors = [{"embedding": [0.1, 0.2, 0.3]} for _ in inputs]
                self._send_json(200, {"object": "list", "data": vectors})
                return
            mode = upstream.behaviour["mode"]
            if mode == "error":
                self._send_json(500, {"error": {"message": "upstream failed"}})
            elif mode == "stream":
                self._stream(upstream.behaviour["chunks"], upstream.behaviour["chunk_text"])
            else:
                self._send_json(200, _completion(upstream.behaviour.get("text", "ok")))

        def _stream(self, chunks: int, chunk_text: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(chunks):
                delta = {"content": chunk_text} if i else {"role": "assistant", "content": ""}
                event = {
                    "id": "mock",
                    "object": "chat.completion.chunk",
                    "model": MOCK_MODEL_ID,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                self._chunk(f"data: {json.dumps(event)}\n\n")
            self._chunk("data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def _chunk(self, text: str) -> None:
            data = text.encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")

    return Handler


def _completion(text: str) -> dict:
    return {
        "id": "mock",
        "object": "chat.completion",
        "model": MOCK_MODEL_ID,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture(scope="session")
def mock_upstream() -> Generator[MockUpstream, None, None]:
    port = _free_port()
    upstream = MockUpstream(base_url=f"http://127.0.0.1:{port}/v1", behaviour={"mode": "ok"})
    server = ThreadingHTTPServer(("127.0.0.1", port), _mock_upstream_handler(upstream))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield upstream
    server.shutdown()
    server.server_close()


@dataclass
class LaunchedInstance:
    base_url: str
    pid: int
    log_path: Path
    admin_token: str
    upstream: MockUpstream

    def client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.admin_token}"},
            timeout=120.0,
        )

    def log_size(self) -> int:
        return self.log_path.stat().st_size

    def log_since(self, offset: int) -> str:
        with open(self.log_path, "rb") as handle:
            handle.seek(offset)
            return handle.read().decode("utf-8", errors="replace")

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


@pytest.fixture(scope="session")
def launched_instance(mock_upstream: MockUpstream) -> Generator[LaunchedInstance, None, None]:
    """A scratch instance with the mock upstream as its only model provider."""
    yield from _launch(mock_upstream, {})


@pytest.fixture(scope="session")
def fake_redis() -> Generator[FakeRedis, None, None]:
    redis = FakeRedis()
    yield redis
    redis.close()


@pytest.fixture(scope="session")
def degraded_instance(
    mock_upstream: MockUpstream, fake_redis: FakeRedis
) -> Generator[LaunchedInstance, None, None]:
    """Same, on the fake Redis and a vector DB that refuses every connection."""
    yield from _launch(
        mock_upstream,
        {"REDIS_URL": fake_redis.url, "VECTOR_DB": "qdrant", "QDRANT_URI": "http://127.0.0.1:9"},
    )


def _launch(
    upstream: MockUpstream, extra_env: dict[str, str]
) -> Generator[LaunchedInstance, None, None]:
    backend = _resolve_backend()
    if backend is None:
        pytest.skip("open-webui backend source not found (set OPEN_WEBUI_SOURCE_DIR)")

    scratch = Path(tempfile.mkdtemp(prefix="owui-integration-"))
    for name in ("data", "static"):
        (scratch / name).mkdir()
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "WEBUI_SECRET_KEY": "integration-secret-key",
        "WEBUI_AUTH": "true",
        "DATA_DIR": str(scratch / "data"),
        "STATIC_DIR": str(scratch / "static"),
        "FRONTEND_BUILD_DIR": str(scratch / "build"),
        "OFFLINE_MODE": "true",
        "RAG_EMBEDDING_ENGINE": "openai",  # served by the mock, so nothing is downloaded
        "ENABLE_OLLAMA_API": "false",
        "ENABLE_OPENAI_API": "true",
        "OPENAI_API_BASE_URL": upstream.base_url,
        "OPENAI_API_KEY": "sk-mock",
    }
    for name in (
        "DATABASE_URL",
        "DATABASE_TYPE",
        "OPENAI_API_BASE_URLS",
        "OPENAI_API_KEYS",
        "OPENAI_API_CONFIGS",
        "ENABLE_LOGIN_FORM",
        "GLOBAL_LOG_LEVEL",
        "RAG_OPENAI_API_BASE_URL",
        "RAG_OPENAI_API_KEY",
        "STORAGE_PROVIDER",
        "REDIS_URL",
        "REDIS_SENTINEL_HOSTS",
        "CHROMA_HTTP_HOST",
        "WEBSOCKET_REDIS_URL",
        "WEBSOCKET_MANAGER",
        "VECTOR_DB",
    ):
        env.pop(name, None)
    env.update(extra_env)
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
            admin_token=signup.json()["token"],
            upstream=upstream,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(scratch, ignore_errors=True)


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
        time.sleep(2)
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    pytest.fail(f"backend did not become healthy within 300s:\n{tail}")
