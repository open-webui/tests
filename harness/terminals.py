"""The real Terminals orchestrator (open-webui/terminals) with real Open Terminal containers.

`serving_terminals()` runs the Terminals service image on Docker with an API key and a scratch
data folder, on a Docker network of its own and a loopback port Docker picks, and yields it
once it answers. The service starts an Open Terminal container for every terminal a template
asks for. `TERMINALS_SERVICE_IMAGE` names the service image and `OPEN_TERMINAL_IMAGE` the image
of the terminals it starts; both must already be on the Docker daemon, since nothing here
pulls. Without them, or without a working Docker daemon, the tests that need the stack skip
and say why; with `OWUI_TEST_REQUIRE_TERMINALS=1` they fail.

Every stack is one run. Its network and each of its containers carry the run's label, so two
runs on one machine never see each other's: the service reaches Docker through a socket of the
run's own that labels every container it creates and lists only those. Leaving the block
removes the run's containers, network and files, also after a failure or an interrupt, and
`sweep()` removes the runs whose test process died before it got that far.

`connection(...)` is the admin's connection to the service under Settings > Integrations, for
`configure_terminals` in `harness.terminal_server`. On that connection a terminal is a
template (`create_template`), and each user, chat or model it is scoped to gets a machine of its
own on first use (`start_instance`, `execute`).
"""

from __future__ import annotations

import contextlib
import functools
import http.client
import json
import os
import re
import shutil
import socket
import socketserver
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit

import httpx
import psutil
import pytest

from harness.listener import read_body

SERVICE_IMAGE_ENV = "TERMINALS_SERVICE_IMAGE"
OPEN_TERMINAL_IMAGE_ENV = "OPEN_TERMINAL_IMAGE"
REQUIRED_ENV = "OWUI_TEST_REQUIRE_TERMINALS"
RUN_LABEL = "owui-tests.terminals.run"
OWNER_LABEL = "owui-tests.terminals.owner"
SCRATCH_LABEL = "owui-tests.terminals.scratch"
BOOT_SECONDS = 60
# What a terminal of `create_template` may use; the service refuses more than the caps.
CPU_LIMIT = "1"
MEMORY_LIMIT = "512Mi"
MAX_CPU = "2"
MAX_MEMORY = "1Gi"
# Hop-by-hop headers and the ones http.client sets itself.
UNFORWARDED = {"host", "connection", "content-length", "transfer-encoding"}


def _docker(*arguments: str, check: bool = True) -> str:
    ran = subprocess.run(["docker", *arguments], capture_output=True, text=True, timeout=120)
    if check and ran.returncode != 0:
        raise RuntimeError(f"docker {' '.join(arguments)} failed: {ran.stderr.strip()}")
    return ran.stdout.strip()


@functools.cache
def _daemon_socket() -> str | None:
    """The unix socket of the Docker daemon the CLI talks to, or None without a local one."""
    if shutil.which("docker") is None:
        return None
    try:
        host = _docker("context", "inspect", "--format", "{{.Endpoints.docker.Host}}", check=False)
        answers = (
            subprocess.run(["docker", "info"], capture_output=True, timeout=60).returncode == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return host.removeprefix("unix://") if answers and host.startswith("unix://") else None


@functools.cache
def unavailable() -> str | None:
    """Why the stack cannot run here, or None when it can."""
    images = {name: os.getenv(name) for name in (SERVICE_IMAGE_ENV, OPEN_TERMINAL_IMAGE_ENV)}
    unset = [name for name, image in images.items() if not image]
    if unset:
        return f"needs the Terminals stack (set {' and '.join(unset)})"
    if _daemon_socket() is None:
        return "needs a working local Docker daemon (`docker info` must succeed for this user)"
    for name, image in images.items():
        if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode:
            return f"{name} names {image}, which is not on this Docker daemon (pull or build it)"
    return None


@dataclass
class Terminals:
    base_url: str
    api_key: str
    run_id: str
    data_dir: Path

    @property
    def name(self) -> str:
        """The name of the service's container and of the run's network."""
        return _run_name(self.run_id)

    def connection(self, **fields) -> dict:
        """The orchestrator's connection as the admin form saves it, open to admins only."""
        return {
            "id": f"terminals-{uuid.uuid4().hex[:8]}",
            "name": "Terminals",
            "enabled": True,
            "url": self.base_url,
            "path": "/openapi.json",
            "key": self.api_key,
            "auth_type": "bearer",
            "server_type": "orchestrator",
            "config": {"access_grants": []},
            **fields,
        }

    def client(self) -> httpx.Client:
        """The service's own admin API, with its key."""
        return httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120.0,
        )

    def templates(self) -> list[dict]:
        with self.client() as client:
            listed = client.get("/api/v1/terminals/templates")
        listed.raise_for_status()
        return listed.json()

    def instances(self) -> list[dict]:
        """Every machine the service tracks: its id, template, context (`user:<id>`) and status."""
        with self.client() as client:
            listed = client.get("/api/v1/terminals")
        listed.raise_for_status()
        return listed.json()

    def containers(self, running_only: bool = False) -> list[str]:
        """The names of the run's terminal containers, `terminal-<instance id>` each."""
        scope = [] if running_only else ["-a"]
        listed = _docker(
            "ps", *scope, "--filter", f"label={RUN_LABEL}={self.run_id}", "--format", "{{.Names}}"
        )
        return sorted(name for name in listed.split() if name != self.name)

    def home(self, instance_id: str) -> Path:
        """The folder on this machine that the instance's container mounts as its home."""
        return self.data_dir / "terminals" / ".terminals" / instance_id

    def logs(self) -> str:
        """The service's log and each terminal container's, for a failure message."""
        parts = []
        for container in [self.name, *self.containers()]:
            ran = subprocess.run(
                ["docker", "logs", "--tail", "200", container], capture_output=True, text=True
            )
            parts.append(f"--- {container} ---\n{ran.stdout}{ran.stderr}")
        return "\n".join(parts)

    def reset(self) -> None:
        """Delete every template with its machines and files, so the next test starts empty."""
        with self.client() as client:
            for template in self.templates():
                client.delete(f"/api/v1/terminals/templates/{template['id']}")
        leftover = self.containers()
        if leftover:
            _docker("rm", "-f", "-v", *leftover, check=False)
        assert not self.templates(), f"the service kept templates after the test:\n{self.logs()}"


def _run_name(run_id: str) -> str:
    return f"owui-terminals-{run_id}"


def _owner() -> str:
    return f"{os.getpid()}:{psutil.Process().create_time():.0f}"


def _owner_alive(owner: str) -> bool:
    pid, _, started = owner.partition(":")
    try:
        return f"{psutil.Process(int(pid)).create_time():.0f}" == started
    except (psutil.Error, ValueError):
        return False


class _DaemonConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost", timeout=120)
        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


class _UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


@contextlib.contextmanager
def _run_docker_socket(path: Path, run_id: str) -> Iterator[None]:
    """A Docker socket that labels the containers created through it and lists only those.

    Terminals 0.4.1 adopts every running terminal container on the daemon when it starts and
    fails on one whose template it does not know, so a second run could not start beside the first.
    """
    label = f"{RUN_LABEL}={run_id}"

    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _forward(self) -> None:
            body = read_body(self)
            target = urlsplit(self.path)
            endpoint = re.sub(r"^/v[\d.]+", "", target.path)
            query = parse_qsl(target.query, keep_blank_values=True)
            if (self.command, endpoint) == ("POST", "/containers/create"):
                config = json.loads(body)
                config["Labels"] = {**(config.get("Labels") or {}), RUN_LABEL: run_id}
                body = json.dumps(config).encode()
            if (self.command, endpoint) == ("GET", "/containers/json"):
                filters = json.loads(dict(query).get("filters") or "{}")
                filters["label"] = [*(filters.get("label") or []), label]
                query = [pair for pair in query if pair[0] != "filters"]
                query.append(("filters", json.dumps(filters)))
            headers = {
                name: value
                for name, value in self.headers.items()
                if name.lower() not in UNFORWARDED
            }
            daemon = _DaemonConnection(_daemon_socket())
            try:
                daemon.request(
                    self.command, f"{target.path}?{urlencode(query)}", body=body, headers=headers
                )
                answer = daemon.getresponse()
                content = answer.read()
            finally:
                daemon.close()
            self.send_response(answer.status)
            for name, value in answer.getheaders():
                if name.lower() not in UNFORWARDED:
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(content)

        do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = _forward

    server = _UnixHTTPServer(str(path), RequestHandler)
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()


def _remove_run(run_id: str, scratch: Path) -> None:
    """Remove the run's containers, files and network; the service wrote its files as root."""
    name = _run_name(run_id)
    image = _docker("inspect", "--format", "{{.Image}}", name, check=False)
    image = image or os.getenv(SERVICE_IMAGE_ENV)
    _docker("rm", "-f", "-v", name, check=False)
    containers = _docker("ps", "-aq", "--filter", f"label={RUN_LABEL}={run_id}", check=False)
    if containers:
        _docker("rm", "-f", "-v", *containers.split(), check=False)
    if image and (scratch / "data").exists():
        mount = f"{scratch}:/scratch"
        _docker(
            *("run", "--rm", "--network", "none", "--entrypoint", "rm", "-v", mount, image),
            *("-rf", "/scratch/data"),
            check=False,
        )
    _docker("network", "rm", name, check=False)
    shutil.rmtree(scratch, ignore_errors=True)


def sweep() -> list[str]:
    """Remove the runs whose test process is gone; returns their ids."""
    removed = []
    networks = _docker("network", "ls", "--filter", f"label={RUN_LABEL}", "--format", "{{.ID}}")
    for network in networks.split():
        described = _docker(
            "network", "inspect", "--format", "{{json .Labels}}", network, check=False
        )
        labels = json.loads(described or "{}")
        if RUN_LABEL not in labels or _owner_alive(labels.get(OWNER_LABEL, "")):
            continue
        _remove_run(labels[RUN_LABEL], Path(labels[SCRATCH_LABEL]))
        removed.append(labels[RUN_LABEL])
    return removed


def _keep_logs(terminals: Terminals) -> None:
    keep_logs_in = os.getenv("OPEN_WEBUI_LOG_DIR")
    if keep_logs_in:
        Path(keep_logs_in).mkdir(parents=True, exist_ok=True)
        (Path(keep_logs_in) / f"terminals-{terminals.run_id}.log").write_text(terminals.logs())


def _wait_until_it_answers(terminals: Terminals) -> None:
    deadline = time.monotonic() + BOOT_SECONDS
    while time.monotonic() < deadline:
        state = _docker("inspect", "--format", "{{.State.Running}}", terminals.name, check=False)
        if state != "true":
            pytest.fail(f"the Terminals service exited during boot:\n{terminals.logs()[-3000:]}")
        try:
            if httpx.get(f"{terminals.base_url}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    pytest.fail(f"the Terminals service did not answer within {BOOT_SECONDS}s")


@contextlib.contextmanager
def serving_terminals(env: dict[str, str] | None = None) -> Iterator[Terminals]:
    """Run the service; `env` adds to its settings, such as `TERMINALS_IDLE_TIMEOUT_MINUTES`."""
    reason = unavailable()
    if reason and os.getenv(REQUIRED_ENV):
        pytest.fail(reason)
    if reason:
        pytest.skip(reason)
    sweep()
    run_id = uuid.uuid4().hex[:12]
    name = _run_name(run_id)
    scratch = Path(tempfile.mkdtemp(prefix="owui-terminals-"))
    data_dir = scratch / "data"
    data_dir.mkdir()
    api_key = f"sk-terminals-{uuid.uuid4().hex}"
    settings = {
        "TERMINALS_API_KEY": api_key,
        "TERMINALS_DOCKER_NETWORK": name,
        "TERMINALS_DOCKER_HOST_DATA_DIR": str(data_dir / "terminals"),
        "TERMINALS_IMAGE": os.environ[OPEN_TERMINAL_IMAGE_ENV],
        "TERMINALS_IMAGE_PULL_POLICY": "Never",
        "TERMINALS_MAX_CPU": MAX_CPU,
        "TERMINALS_MAX_MEMORY": MAX_MEMORY,
        **(env or {}),
    }
    labels = {RUN_LABEL: run_id, OWNER_LABEL: _owner(), SCRATCH_LABEL: str(scratch)}
    label_options = [option for pair in labels.items() for option in ("--label", "=".join(pair))]
    setting_options = [option for pair in settings.items() for option in ("-e", "=".join(pair))]
    try:
        with _run_docker_socket(scratch / "docker.sock", run_id):
            _docker("network", "create", *label_options, name)
            _docker(
                *("run", "-d", "--name", name, "--network", name, *label_options),
                *("-p", "127.0.0.1::3000", "--memory", "512m", "--cpus", "1"),
                *("-v", f"{scratch / 'docker.sock'}:/var/run/docker.sock"),
                *("-v", f"{data_dir}:/app/data", *setting_options),
                os.environ[SERVICE_IMAGE_ENV],
            )
            address = _docker("port", name, "3000/tcp").splitlines()[0]
            terminals = Terminals(f"http://{address}", api_key, run_id, data_dir)
            _wait_until_it_answers(terminals)
            try:
                yield terminals
            finally:
                _keep_logs(terminals)
    finally:
        _remove_run(run_id, scratch)


def create_template(client: httpx.Client, connection: dict, **data) -> dict:
    """Create a terminal on the orchestrator connection the way New terminal does.

    `client` is its creator's and needs write access to the connection. One machine per user
    unless `scope` says otherwise (None for one shared by all, "chat", "model"). The answer
    carries the `terminal_id` the terminal is picked and called by.
    """
    template = {
        "name": f"Terminal {uuid.uuid4().hex[:8]}",
        "scope": "user",
        "cpu_limit": CPU_LIMIT,
        "memory_limit": MEMORY_LIMIT,
        **data,
    }
    created = client.post(
        "/api/v1/terminals/templates",
        json={"id": str(uuid.uuid4()), "connection_id": connection["id"], "data": template},
    )
    assert created.status_code == 201, f"creating the terminal failed: {created.text}"
    return created.json()


def share_template(client: httpx.Client, terminal_id: str, *grants: dict) -> None:
    """Share the terminal the way its Access dialog does; replaces the grants it had."""
    shared = client.post(
        f"/api/v1/terminals/templates/{terminal_id}/access", json={"access_grants": list(grants)}
    )
    assert shared.status_code == 200, f"sharing the terminal failed: {shared.text}"


def start_instance(client: httpx.Client, terminal_id: str, **context) -> dict:
    """Start the machine the terminal gives `client`'s user and return its state."""
    started = client.post(f"/api/v1/terminals/templates/{terminal_id}/start", json=context)
    assert started.status_code == 200, f"starting the terminal failed: {started.text}"
    return instance_of(client, terminal_id)


def instance_of(client: httpx.Client, terminal_id: str, **context) -> dict:
    """The machine the terminal gives `client`'s user: its id and status."""
    found = client.get(f"/api/v1/terminals/templates/{terminal_id}/instance", params=context)
    assert found.status_code == 200, f"reading the terminal's machine failed: {found.text}"
    return found.json()


def execute(
    client: httpx.Client, terminal_id: str, command: str, chat_id: str | None = None
) -> httpx.Response:
    """Run `command` in the terminal through Open WebUI and wait for it to finish."""
    return client.post(
        f"/api/v1/terminals/{terminal_id}/execute",
        params={"wait": 30},
        headers={"X-Session-Id": chat_id} if chat_id else {},
        json={"command": command},
    )


def output_of(answer: httpx.Response) -> str:
    """What a finished `execute` printed."""
    assert answer.status_code == 200, f"the command failed: {answer.status_code} {answer.text}"
    return "".join(entry["data"] for entry in answer.json()["output"])


if __name__ == "__main__":
    print("\n".join(sweep()) or "nothing left behind")
