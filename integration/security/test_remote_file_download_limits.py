"""Regression: a document fetched from a link must be streamed to disk under the file size limit.

open-webui 0.11.1 fix `6b438f1a7` (PR #28945): when `POST /api/v1/retrieval/process/web` got a
link to a binary document, it wrote `response.content` to a temp file, so the whole body was
read into memory first and no size limit applied; the temp file was also created outside the
`try`, so a download that died partway stayed on disk. The fix streams the body in 64 KiB blocks,
refuses it once it passes the admin's `FILE_MAX_SIZE` and removes the temp file either way.

The document host is a local server that writes its body in paced 64 KiB blocks and counts what
it delivered before the instance hung up. The instance names its temp file after the link's
extension, so a unique extension per document shows whether that file outlived the request.

Twin of unit/security/test_remote_file_download_limits.py.

Discriminates: passes on dev bbfa876af, fails with `6b438f1a7` reverted (the 16 MB download runs
to its end and is extracted with HTTP 200, and a dropped download leaves its temp file behind).
"""

from __future__ import annotations

import glob
import os
import socket
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from harness.actors import create_user

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

LOCAL_FETCH = {"ENABLE_LOCAL_WEB_FETCH": "true"}
RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
KB = 1024
MB = 1024 * KB
BLOCK = 64 * KB
# Slower than the instance reads, so the socket buffers stay near empty and the count is honest.
PACE_SECONDS = 0.002
MARKER = b"the first line of the linked document\n"


@dataclass
class HostedDocument:
    url: str
    extension: str
    body: bytes
    content_type: str
    drop_after: int | None
    expect_temp_file: bool
    delivered: int = 0
    temp_file_seen: bool = False
    finished: threading.Event = field(default_factory=threading.Event)

    def temp_files(self) -> list[str]:
        return glob.glob(os.path.join(tempfile.gettempdir(), f"*.{self.extension}"))

    def bytes_delivered(self) -> int:
        assert self.finished.wait(10), "the download was still running 10 s after the reply"
        return self.delivered


def _wait_for(condition, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


class DocumentHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def setup(self) -> None:
        self.request.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, BLOCK)
        super().setup()

    def do_GET(self) -> None:
        document: HostedDocument = self.server.documents[self.path]
        self.send_response(200)
        self.send_header("Content-Type", document.content_type)
        self.send_header("Content-Length", str(len(document.body)))
        self.end_headers()
        # the instance creates its temp file between reading the headers and the body
        if document.expect_temp_file:
            document.temp_file_seen = _wait_for(document.temp_files)
        end = len(document.body) if document.drop_after is None else document.drop_after
        try:
            while document.delivered < end:
                block_end = min(document.delivered + BLOCK, end)
                self.wfile.write(document.body[document.delivered : block_end])
                document.delivered = block_end
                time.sleep(PACE_SECONDS)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.close_connection = True
            document.finished.set()


class DocumentHost:
    def __init__(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), DocumentHandler)
        self.server.documents = {}
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def add(
        self,
        body: bytes,
        content_type: str = "application/octet-stream",
        *,
        drop_after: int | None = None,
        expect_temp_file: bool = False,
    ) -> HostedDocument:
        extension = f"owui{uuid.uuid4().hex[:10]}"
        path = f"/files/report.{extension}"
        document = HostedDocument(
            f"{self.base_url}{path}", extension, body, content_type, drop_after, expect_temp_file
        )
        self.server.documents[path] = document
        return document


@pytest.fixture
def local(instance_with):
    return instance_with(LOCAL_FETCH)


@pytest.fixture
def reader(local):
    return create_user(local)


@pytest.fixture
def file_size_limit(local):
    """`file_size_limit(megabytes)` sets the admin's file size limit (None lifts it) for a test.

    Restored by hand: `preserve` acts on the shared instance, and posting back a snapshot whose
    limit is None leaves the current limit in place.
    """
    with local.client() as client:
        original = client.get(RETRIEVAL_CONFIG[0]).json()["FILE_MAX_SIZE"]

        def apply(megabytes: int | None) -> None:
            # the update route reads None as "unchanged" and "" as "no limit"
            limit = "" if megabytes is None else megabytes
            client.post(RETRIEVAL_CONFIG[1], json={"FILE_MAX_SIZE": limit}).raise_for_status()

        yield apply
        apply(original)


@pytest.fixture
def document_host():
    host = DocumentHost()
    threading.Thread(target=host.server.serve_forever, args=(0.05,), daemon=True).start()
    yield host
    host.server.shutdown()
    host.server.server_close()
    for document in host.server.documents.values():
        for leftover in document.temp_files():
            os.remove(leftover)


def document_text(size: int) -> bytes:
    return MARKER + b"a" * (size - len(MARKER))


def fetch(actor, route: str, url: str):
    with actor.client() as client:
        return client.post(f"/api/v1/retrieval/{route}?process=false", json={"url": url})


@pytest.mark.parametrize(
    "content_type", ["application/pdf", "application/zip", "image/png", "application/octet-stream"]
)
def test_an_oversized_download_is_refused_while_it_streams(
    local, reader, file_size_limit, document_host, content_type
):
    file_size_limit(1)
    document = document_host.add(document_text(16 * MB), content_type)
    log_offset = local.log_size()

    fetched = fetch(reader, "process/web", document.url)

    assert fetched.status_code == 400, (
        f"a 16 MB {content_type} link was read past the 1 MB limit (#28945): "
        f"HTTP {fetched.status_code} {fetched.text[:200]}"
    )
    assert "too large" in local.log_since(log_offset), "the refusal was not the size limit"
    delivered = document.bytes_delivered()
    assert delivered < 4 * MB, (
        f"the instance took {delivered / MB:.1f} MB of a 16 MB download against a 1 MB limit; "
        "the body was buffered whole instead of streamed (#28945)"
    )


def test_the_sibling_link_route_stops_at_the_limit_too(reader, file_size_limit, document_host):
    file_size_limit(1)
    document = document_host.add(document_text(16 * MB))

    fetched = fetch(reader, "process/url", document.url)

    assert fetched.status_code == 413, fetched.text
    assert document.bytes_delivered() < 4 * MB


def test_a_download_that_drops_midway_leaves_no_temp_file(reader, document_host):
    document = document_host.add(document_text(2 * MB), drop_after=256 * KB, expect_temp_file=True)

    fetched = fetch(reader, "process/web", document.url)

    assert fetched.status_code == 400, fetched.text
    assert document.temp_file_seen, (
        f"no temp file named *.{document.extension} appeared during the download; the instance "
        "names its temp files differently now, retarget this test"
    )
    assert document.temp_files() == [], (
        "a download that died partway left its temp file on the server's disk (#28945)"
    )


def test_a_document_exactly_at_the_limit_is_read_and_its_temp_file_removed(
    reader, file_size_limit, document_host
):
    file_size_limit(1)
    document = document_host.add(document_text(1 * MB), expect_temp_file=True)

    fetched = fetch(reader, "process/web", document.url)

    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["content"].startswith(MARKER.decode().strip())
    assert document.temp_file_seen
    assert document.temp_files() == []


def test_without_a_limit_a_large_document_is_read_whole(reader, file_size_limit, document_host):
    file_size_limit(None)
    body = document_text(3 * MB)
    document = document_host.add(body)

    fetched = fetch(reader, "process/web", document.url)

    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["content"] == body.decode()
    assert document.bytes_delivered() == len(body)
