"""A stand-in Jupyter server whose kernel runs every cell it is sent in a fresh Python process.

Just enough of the Jupyter protocol for Open WebUI's Jupyter engine: `POST /api/kernels`
starts a kernel, its `channels` websocket takes one `execute_request` and answers with the
cell's stdout and stderr as `stream` messages and then an `idle` status; a cell whose process
crashed says so on stderr. `executed_cells()` is the code of every cell, exactly as Open WebUI
sent it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import BinaryIO, Iterator

from harness.instance import free_port

KERNEL_ID = "scratch-kernel"
CELL_TIMEOUT_SECONDS = 30
WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
TEXT_FRAME, CLOSE_FRAME = 0x1, 0x8


@dataclass
class FakeJupyter:
    base_url: str
    cells: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def executed_cells(self) -> list[str]:
        with self.lock:
            return list(self.cells)


def _read_frame(stream: BinaryIO) -> tuple[int, bytes]:
    first, second = stream.read(2)
    length = second & 0x7F
    if length == 126:
        (length,) = struct.unpack("!H", stream.read(2))
    elif length == 127:
        (length,) = struct.unpack("!Q", stream.read(8))
    mask = stream.read(4) if second & 0x80 else bytes(4)
    payload = stream.read(length)
    return first & 0x0F, bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))


def _write_frame(stream: BinaryIO, opcode: int, payload: bytes) -> None:
    length = len(payload)
    if length < 126:
        header = struct.pack("!BB", 0x80 | opcode, length)
    elif length < 1 << 16:
        header = struct.pack("!BBH", 0x80 | opcode, 126, length)
    else:
        header = struct.pack("!BBQ", 0x80 | opcode, 127, length)
    stream.write(header + payload)
    stream.flush()


def _run_cell(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        timeout=CELL_TIMEOUT_SECONDS,
    )


@contextmanager
def fake_jupyter() -> Iterator[FakeJupyter]:
    port = free_port()
    jupyter = FakeJupyter(base_url=f"http://127.0.0.1:{port}")

    class KernelHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _answer(self, status: int, payload: dict | None = None) -> None:
            body = json.dumps(payload).encode() if payload is not None else b""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            self._answer(201, {"id": KERNEL_ID, "name": "python3"})

        def do_DELETE(self) -> None:
            self._answer(204)

        def do_GET(self) -> None:
            key = self.headers.get("Sec-WebSocket-Key")
            if not key:
                self._answer(404, {"message": "not found"})
                return
            accept = hashlib.sha1((key + WEBSOCKET_GUID).encode()).digest()
            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", base64.b64encode(accept).decode())
            self.end_headers()

            _, payload = _read_frame(self.rfile)
            request = json.loads(payload)
            code = request["content"]["code"]
            with jupyter.lock:
                jupyter.cells.append(code)
            finished = _run_cell(code)

            stderr = finished.stderr
            if finished.returncode:
                stderr += f"[the cell's process exited with code {finished.returncode}]"
            parent = {"msg_id": request["header"]["msg_id"]}
            for name, text in (("stdout", finished.stdout), ("stderr", stderr)):
                if text:
                    self._send(parent, "stream", {"name": name, "text": text})
            self._send(parent, "status", {"execution_state": "idle"})

            opcode, closing = _read_frame(self.rfile)
            if opcode == CLOSE_FRAME:
                _write_frame(self.wfile, CLOSE_FRAME, closing[:2])
            self.close_connection = True

        def _send(self, parent: dict, msg_type: str, content: dict) -> None:
            message = {"parent_header": parent, "msg_type": msg_type, "content": content}
            _write_frame(self.wfile, TEXT_FRAME, json.dumps(message).encode())

    server = ThreadingHTTPServer(("127.0.0.1", port), KernelHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield jupyter
    finally:
        server.shutdown()
        server.server_close()
