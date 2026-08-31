"""Regression: a remote image URL must not be buffered into memory without a limit.

open-webui 0.11.2 fix `756241b34` (withheld security advisory):
`get_image_base64_from_url` in `open_webui/utils/files.py` did
`await response.read()`, pulling the whole body of any URL a user or a model
could steer the server at into RAM before encoding it.

The fix reads the body in 64 KiB chunks, charges each chunk against the
configured `rag.file.max_size` (megabytes, 0 or unset meaning no limit) and
returns None the moment the running total passes it.

Discriminates: passes on v0.11.2, fails on v0.11.1 (no limit exists there, so
the oversized body is read whole and encoded into a data URL instead of being
refused).

Everything here drives a fake aiohttp response; no network, and the fake bodies
are 2 MB at most so the buggy checkout cannot exhaust anything either.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression

KB = 1024
MB = 1024 * 1024

# The limit is read in whole megabytes, so 1 MB is the smallest real cap.
MAX_SIZE_MB = 1
CHUNK = 256 * KB

URL = "https://example.com/image.png"


class FakeContent:
    def __init__(self, response: "FakeResponse") -> None:
        self._response = response

    async def iter_chunked(self, chunk_size: int):
        self._response.requested_chunk_sizes.append(chunk_size)
        for chunk in self._response.chunks:
            self._response.chunks_yielded += 1
            yield chunk


class FakeResponse:
    """Fake aiohttp response, streamable via `content.iter_chunked` or buffered via `read`.

    Records which of the two the caller used, and how far it got into the body.
    """

    def __init__(self, chunks: list[bytes], content_type: str = "image/png") -> None:
        self.chunks = chunks
        self.headers = {"Content-Type": content_type}
        self.content = FakeContent(self)
        self.chunks_yielded = 0
        self.read_called = False
        self.requested_chunk_sizes: list[int] = []

    def raise_for_status(self) -> None:
        return None

    async def read(self) -> bytes:
        self.read_called = True
        return b"".join(self.chunks)

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.requested_urls: list[str] = []

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.requested_urls.append(url)
        return self.response

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


@pytest.fixture(scope="session")
def files_utils_module(owui_module):
    """`open_webui.utils.files` (get_image_base64_from_url)."""
    return owui_module("open_webui.utils.files")


@pytest.fixture(scope="session")
def config_store(owui_module):
    """`open_webui.models.config.Config`, whose `get` carries the size limit."""
    owui_module("open_webui.config")
    return owui_module("open_webui.models.config").Config


@pytest.fixture
def fetch(files_utils_module, config_store, monkeypatch):
    """Drive the real fetch against a fake response, with the limit patched in."""
    validated: list[str] = []
    monkeypatch.setattr(files_utils_module, "validate_url", validated.append)

    async def fake_config_get(key, default=None):
        return limit["rag.file.max_size"] if key == "rag.file.max_size" else default

    limit = {"rag.file.max_size": MAX_SIZE_MB}
    monkeypatch.setattr(config_store, "get", staticmethod(fake_config_get))

    async def _fetch(chunks, content_type="image/png", max_size_mb=MAX_SIZE_MB, url=URL):
        limit["rag.file.max_size"] = max_size_mb
        response = FakeResponse(chunks, content_type)
        session = FakeSession(response)
        monkeypatch.setattr(files_utils_module, "get_ssrf_safe_session", lambda: session)
        result = await files_utils_module.get_image_base64_from_url(url)
        return SimpleNamespace(
            result=result, response=response, session=session, validated=validated
        )

    return _fetch


def body(total_bytes: int) -> list[bytes]:
    chunks = [b"x" * CHUNK] * (total_bytes // CHUNK)
    remainder = total_bytes % CHUNK
    return chunks + ([b"x" * remainder] if remainder else [])


# ---------------------------------------------------------------------------
# narrow: the fix itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oversized_image_is_refused(fetch):
    outcome = await fetch(body(2 * MB))

    assert outcome.result is None


@pytest.mark.asyncio
async def test_oversized_image_is_cut_off_mid_stream(fetch):
    """The limit trips while the body is still arriving, not after it all landed."""
    outcome = await fetch(body(2 * MB))

    assert outcome.response.read_called is False
    assert outcome.response.chunks_yielded < len(outcome.response.chunks)


@pytest.mark.asyncio
async def test_body_is_requested_in_bounded_chunks(fetch):
    """The per-read size is what bounds peak memory, so pin it, not just the total."""
    outcome = await fetch(body(128 * KB))

    assert outcome.response.requested_chunk_sizes
    assert max(outcome.response.requested_chunk_sizes) <= MB


@pytest.mark.asyncio
async def test_image_under_the_limit_still_encodes(fetch):
    outcome = await fetch(body(128 * KB))

    assert outcome.result == f"data:image/png;base64,{base64.b64encode(b'x' * 128 * KB).decode()}"


# ---------------------------------------------------------------------------
# broad: the invariant the bug was an instance of
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content_type",
    ["image/png", "image/svg+xml", "application/octet-stream", "text/html"],
)
@pytest.mark.asyncio
async def test_limit_applies_regardless_of_content_type(fetch, content_type):
    outcome = await fetch(body(2 * MB), content_type=content_type)

    assert outcome.result is None


@pytest.mark.asyncio
async def test_body_one_byte_over_the_limit_is_refused(fetch):
    outcome = await fetch(body(MAX_SIZE_MB * MB) + [b"x"])

    assert outcome.result is None


@pytest.mark.asyncio
async def test_larger_limit_admits_a_body_the_smaller_one_refused(fetch):
    refused = await fetch(body(2 * MB), max_size_mb=1)
    admitted = await fetch(body(2 * MB), max_size_mb=8)

    assert refused.result is None
    assert admitted.result is not None


# ---------------------------------------------------------------------------
# nearby: behaviour that was already correct
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_body_exactly_at_the_limit_is_accepted(fetch):
    """The cap trips on strictly-greater, so a body of exactly the cap still encodes."""
    outcome = await fetch(body(MAX_SIZE_MB * MB))

    assert outcome.result is not None
    assert outcome.result.startswith("data:image/png;base64,")


@pytest.mark.parametrize("max_size_mb", [0, None])
@pytest.mark.asyncio
async def test_unset_limit_means_no_limit(fetch, max_size_mb):
    outcome = await fetch(body(2 * MB), max_size_mb=max_size_mb)

    assert outcome.result is not None
    assert outcome.result.startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_external_url_is_still_ssrf_validated_and_fetched(fetch):
    outcome = await fetch(body(64 * KB))

    assert outcome.validated == [URL]
    assert outcome.session.requested_urls == [URL]


@pytest.mark.asyncio
async def test_content_type_header_is_carried_into_the_data_url(fetch):
    outcome = await fetch(body(64 * KB), content_type="image/webp")

    assert outcome.result.startswith("data:image/webp;base64,")


@pytest.mark.asyncio
async def test_non_http_string_never_touches_the_http_client(
    files_utils_module, monkeypatch
):
    """A data: URL or bare id is a file-id lookup, gated by ownership elsewhere."""
    session = FakeSession(FakeResponse([b"x"]))
    monkeypatch.setattr(files_utils_module, "get_ssrf_safe_session", lambda: session)

    async def missing_file(file_id, db=None):
        return None

    monkeypatch.setattr(files_utils_module.Files, "get_file_by_id", missing_file)

    for value in ("data:image/png;base64,AAAA", "some-file-id", "ftp://host/x.png", ""):
        assert await files_utils_module.get_image_base64_from_url(value) is None

    assert session.requested_urls == []
