"""A Playwright browser server on a local port, the one the playwright package ships.

A deployment points `PLAYWRIGHT_WS_URL` at a browser it runs apart from Open WebUI (the
`playwright run-server` container). `serving_playwright()` starts that server from the installed
package behind a relay that counts the connections it passes on, and yields a `BrowserServer`:
`ws_url` for the setting and `connections`, how many times something connected. It skips when
no Chromium is installed for the package.
"""

from __future__ import annotations

import contextlib
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pytest

from harness.instance import free_port


@dataclass
class BrowserServer:
    ws_url: str = ""
    connections: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


def _chromium_path() -> Path:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        return Path(playwright.chromium.executable_path)


def chromium_installed() -> bool:
    # a thread of its own: the sync API refuses a thread whose event loop a browser test runs
    with ThreadPoolExecutor(max_workers=1) as worker:
        return worker.submit(_chromium_path).result().exists()


def _wait_for_port(process: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"playwright run-server exited with {process.returncode}")
        with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), 1):
            return
        time.sleep(0.1)
    raise AssertionError("playwright run-server never listened")


def _pipe(source: socket.socket, target: socket.socket) -> None:
    with contextlib.suppress(OSError):
        while chunk := source.recv(65536):
            target.sendall(chunk)
    with contextlib.suppress(OSError):
        target.shutdown(socket.SHUT_WR)


@contextlib.contextmanager
def _relay(server: BrowserServer, target_port: int) -> Iterator[int]:
    listening = socket.create_server(("127.0.0.1", 0))
    listening.settimeout(0.2)
    stopped = threading.Event()

    def accept() -> None:
        while not stopped.is_set():
            try:
                client, _ = listening.accept()
            except (TimeoutError, OSError):
                continue
            with server.lock:
                server.connections += 1
            upstream = socket.create_connection(("127.0.0.1", target_port))
            for source, target in ((client, upstream), (upstream, client)):
                threading.Thread(target=_pipe, args=(source, target), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    try:
        yield listening.getsockname()[1]
    finally:
        stopped.set()
        listening.close()


@contextlib.contextmanager
def serving_playwright() -> Iterator[BrowserServer]:
    if not chromium_installed():
        pytest.skip("no Chromium for the playwright package (playwright install chromium)")
    from playwright._impl._driver import compute_driver_executable, get_driver_env

    port = free_port()
    # the driver itself, so stopping it takes the browsers it started along
    command = [*compute_driver_executable(), "run-server", "--host", "127.0.0.1"]
    process = subprocess.Popen(
        [*command, "--port", str(port)],
        env=get_driver_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    server = BrowserServer()
    try:
        _wait_for_port(process, port)
        with _relay(server, port) as relay_port:
            server.ws_url = f"ws://127.0.0.1:{relay_port}/"
            yield server
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
