"""Regression: a chat image whose host forces a Content-Encoding must not be inlined.

open-webui 0.11.4 fix `6fb68e43c` (PR #29623): `get_image_base64_from_url` in
`open_webui/utils/files.py` fetched a remote chat image with the client's default
Accept-Encoding and never looked at the response's Content-Encoding. A host that
stores and echoes a Content-Encoding no matter what the client asked for (an S3 or
MinIO object uploaded with that metadata, or any server behind a compressing proxy)
then had its body inlined as if it were the raw image bytes, and the reply it sat
in broke. The fetch now asks for an identity-encoded response and skips (returns
None for) any image that still comes back content-encoded, the same way an
unreachable image already behaved: the message keeps its original link.

Everything here drives the real function against a fake aiohttp session; no
network.

Discriminates: passes on dev 344ea5306, fails on `6fb68e43c^` (the request carries
no Accept-Encoding and a gzip-encoded response is read whole and inlined into a
data URL instead of being refused).
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression

URL = "https://example.com/photo.png"
PNG_BYTES = b"fake-png-body"
PNG_DATA_URL = f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"


class FakeHeaders:
    """Stands in for aiohttp's CIMultiDict: supports .get and .getall."""

    def __init__(self, mapping: dict[str, str], multi: dict[str, list[str]] | None = None) -> None:
        self._mapping = mapping
        self._multi = multi or {}

    def get(self, name: str, default=None):
        return self._mapping.get(name, default)

    def getall(self, name: str, default=()):
        return tuple(self._multi.get(name, default))


class FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunked(self, chunk_size: int):
        for chunk in self._chunks:
            yield chunk


class FakeResponse:
    def __init__(self, chunks: list[bytes], headers: FakeHeaders) -> None:
        self.headers = headers
        self.content = FakeContent(chunks)

    def raise_for_status(self) -> None:
        return None

    async def read(self) -> bytes:
        return b"".join(self.content._chunks)

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class FakeSession:
    """Records the request kwargs and hands back the prepared response."""

    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.request_kwargs: dict | None = None

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.request_kwargs = {"url": url, **kwargs}
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
    """Drive the real fetch against a fake session, with size limit and SSRF off."""

    async def fake_config_get(key, default=None):
        if key == "rag.file.max_size":
            return 0  # unset: these tests are about encoding, not size
        return default

    monkeypatch.setattr(config_store, "get", staticmethod(fake_config_get))
    monkeypatch.setattr(files_utils_module, "validate_url", lambda url: None)

    async def _fetch(
        encodings: tuple[str, ...] = (),
        content_type: str = "image/png",
    ) -> SimpleNamespace:
        headers = FakeHeaders(
            {"Content-Type": content_type},
            {"Content-Encoding": list(encodings)} if encodings else None,
        )
        response = FakeResponse([PNG_BYTES], headers)
        session = FakeSession(response)
        monkeypatch.setattr(files_utils_module, "get_ssrf_safe_session", lambda: session)
        result = await files_utils_module.get_image_base64_from_url(URL)
        return SimpleNamespace(result=result, session=session)

    return _fetch


# ---------------------------------------------------------------------------
# narrow: the fix itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_encoded_image_is_refused(fetch):
    outcome = await fetch(encodings=("gzip",))

    assert outcome.result is None, (
        "a Content-encoded image was inlined; the reply it sits in breaks because the "
        "body was already decoded before it reached us"
    )


@pytest.mark.asyncio
async def test_the_fetch_asks_for_an_unencoded_body(fetch):
    outcome = await fetch(encodings=("gzip",))

    assert outcome.session.request_kwargs is not None
    assert outcome.session.request_kwargs.get("headers") == {"Accept-Encoding": "identity"}, (
        "the request did not ask for an identity-encoded response"
    )


# ---------------------------------------------------------------------------
# broad: the invariant the bug was an instance of
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["gzip", "br", "deflate", "zstd", "GZIP", "Br"])
async def test_every_real_compression_scheme_is_refused(fetch, encoding):
    outcome = await fetch(encodings=(encoding,))

    assert outcome.result is None, f"Content-Encoding: {encoding} was inlined"


@pytest.mark.asyncio
async def test_one_real_encoding_among_identity_values_still_refuses(fetch):
    outcome = await fetch(encodings=("identity", "gzip"))

    assert outcome.result is None


# ---------------------------------------------------------------------------
# nearby: behaviour that was already correct
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("encodings", [(), ("",), ("identity",), ("identity", "IDENTITY")])
async def test_an_unencoded_image_still_inlines(fetch, encodings):
    outcome = await fetch(encodings=encodings)

    assert outcome.result == PNG_DATA_URL, (
        f"an image with Content-Encoding {encodings!r} must still be inlined"
    )


@pytest.mark.asyncio
async def test_the_content_type_still_reaches_the_data_url(fetch):
    outcome = await fetch(content_type="image/webp")

    assert outcome.result is not None
    assert outcome.result.startswith("data:image/webp;base64,")
