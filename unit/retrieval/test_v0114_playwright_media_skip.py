"""Regression: the Playwright web loader must abort media requests before fetching them.

open-webui 0.11.4 fix `ae01ef9c9` (PR #29742, issue #29741): SafePlaywrightURLLoader
routes every request a page makes through the backend HTTP client and buffers the
full body before the browser sees any of it, and a browser cancelling a media
request once it has enough never reaches that download. A page with a few dozen
audio players therefore pulled every media file in full for a text extraction that
never reads it, so loading was extremely slow or timed out. Image, media and font
requests are now aborted in the interceptor before any fetch is made.

These tests drive both interceptors (`_intercept_navigation` and
`_intercept_navigation_sync`) directly with fake Playwright routes and fake HTTP
sessions, so no browser and no network are involved.

Discriminates: passes on dev 344ea5306, fails on `ae01ef9c9^` (the interceptor has
no resource-type check, so a font/image/media route is fetched through the session
like any other request and the fake session records the fetch).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression

MEDIA_RESOURCE_TYPES = ("font", "image", "media")
PAGE_RESOURCE_TYPES = ("document", "script", "stylesheet", "xhr", "fetch")

PAGE_URL = "https://example.com/page"
PAGE_BODY = b"<html>text</html>"


@pytest.fixture(scope="session")
def playwright_loader(retrieval_web_utils_module):
    """A real SafePlaywrightURLLoader; no browser is launched at construction."""
    return retrieval_web_utils_module.SafePlaywrightURLLoader(web_paths=["https://example.com"])


# ---------------------------------------------------------------------------
# fakes for the async interceptor (aiohttp session)
# ---------------------------------------------------------------------------


class FakeAsyncRequest:
    def __init__(self, resource_type: str) -> None:
        self.resource_type = resource_type
        self.method = "GET"
        self.url = PAGE_URL
        self.post_data_buffer = None

    async def all_headers(self) -> dict:
        return {}


class FakeAsyncRoute:
    def __init__(self, resource_type: str) -> None:
        self.request = FakeAsyncRequest(resource_type)
        self.aborted = False
        self.fulfilled = None

    async def abort(self) -> None:
        self.aborted = True

    async def fulfill(self, **kwargs) -> None:
        self.fulfilled = kwargs


class FakeAsyncResponse:
    def __init__(self, url: str, body: bytes) -> None:
        self.status = 200
        self.headers = {"content-type": "text/html"}
        self.url = url
        self._body = body

    async def read(self) -> bytes:
        return self._body


class FakeAsyncSession:
    """Records every fetch; raises when asked to, otherwise serves a 200."""

    def __init__(self, error: Exception | None = None) -> None:
        self.fetches: list[tuple[str, str]] = []
        self._error = error

    async def request(self, method: str, url: str, **kwargs):
        self.fetches.append((method, url))
        if self._error is not None:
            raise self._error
        return FakeAsyncResponse(url, PAGE_BODY)


# ---------------------------------------------------------------------------
# fakes for the sync interceptor (requests session)
# ---------------------------------------------------------------------------


class FakeSyncRequest:
    def __init__(self, resource_type: str) -> None:
        self.resource_type = resource_type
        self.method = "GET"
        self.url = PAGE_URL
        self.post_data_buffer = None

    def all_headers(self) -> dict:
        return {}


class FakeSyncRoute:
    def __init__(self, resource_type: str) -> None:
        self.request = FakeSyncRequest(resource_type)
        self.aborted = False
        self.fulfilled = None

    def abort(self) -> None:
        self.aborted = True

    def fulfill(self, **kwargs) -> None:
        self.fulfilled = kwargs


class FakeSyncSession:
    def __init__(self, error: Exception | None = None) -> None:
        self.fetches: list[tuple[str, str]] = []
        self._error = error

    def request(self, method: str, url: str, **kwargs):
        self.fetches.append((method, url))
        if self._error is not None:
            raise self._error
        return SimpleNamespace(
            status_code=200,
            raw=SimpleNamespace(headers={"Content-Type": "text/html"}),
            content=PAGE_BODY,
        )


# ---------------------------------------------------------------------------
# narrow: the fix itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_type", MEDIA_RESOURCE_TYPES)
async def test_async_interceptor_aborts_media_before_any_fetch(
    playwright_loader, resource_type
):
    session = FakeAsyncSession()
    route = FakeAsyncRoute(resource_type)

    await playwright_loader._intercept_navigation(route, session)

    assert route.aborted, f"a {resource_type} request was not aborted"
    assert session.fetches == [], f"a {resource_type} request was fetched through the server"
    assert route.fulfilled is None


def test_sync_interceptor_aborts_media_before_any_fetch(playwright_loader):
    session = FakeSyncSession()
    route = FakeSyncRoute("image")

    playwright_loader._intercept_navigation_sync(route, session)

    assert route.aborted, "an image request was not aborted"
    assert session.fetches == [], "an image request was fetched through the server"
    assert route.fulfilled is None


# ---------------------------------------------------------------------------
# broad: the invariant the bug was an instance of
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_type", PAGE_RESOURCE_TYPES)
async def test_async_interceptor_still_fetches_page_resources(
    retrieval_web_utils_module, playwright_loader, monkeypatch, resource_type
):
    monkeypatch.setattr(retrieval_web_utils_module, "validate_url", lambda url: None)
    session = FakeAsyncSession(error=RuntimeError("fetch attempted"))
    route = FakeAsyncRoute(resource_type)

    await playwright_loader._intercept_navigation(route, session)

    assert session.fetches == [("GET", PAGE_URL)], (
        f"a {resource_type} request never reached the server-side fetch"
    )
    assert route.aborted, "a failed fetch must still drop the route, not leave it hanging"


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_type", PAGE_RESOURCE_TYPES)
async def test_sync_interceptor_still_fetches_page_resources(
    retrieval_web_utils_module, playwright_loader, monkeypatch, resource_type
):
    monkeypatch.setattr(retrieval_web_utils_module, "validate_url", lambda url: None)
    session = FakeSyncSession(error=RuntimeError("fetch attempted"))
    route = FakeSyncRoute(resource_type)

    playwright_loader._intercept_navigation_sync(route, session)

    assert session.fetches == [("GET", PAGE_URL)], (
        f"a {resource_type} request never reached the server-side fetch"
    )
    assert route.aborted, "a failed fetch must still drop the route, not leave it hanging"


# ---------------------------------------------------------------------------
# nearby: behaviour that was already correct
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_fetched_document_is_fulfilled_to_the_browser(
    retrieval_web_utils_module, playwright_loader, monkeypatch
):
    monkeypatch.setattr(retrieval_web_utils_module, "validate_url", lambda url: None)
    session = FakeAsyncSession()
    route = FakeAsyncRoute("document")

    await playwright_loader._intercept_navigation(route, session)

    assert route.fulfilled is not None, "the document request was not fulfilled"
    assert route.fulfilled["status"] == 200
    assert route.fulfilled["body"] == PAGE_BODY
    assert route.aborted is False


def test_a_fetched_document_is_fulfilled_to_the_browser_sync(
    retrieval_web_utils_module, playwright_loader, monkeypatch
):
    monkeypatch.setattr(retrieval_web_utils_module, "validate_url", lambda url: None)
    session = FakeSyncSession()
    route = FakeSyncRoute("document")

    playwright_loader._intercept_navigation_sync(route, session)

    assert route.fulfilled is not None, "the document request was not fulfilled"
    assert route.fulfilled["status"] == 200
    assert route.fulfilled["body"] == PAGE_BODY
    assert route.aborted is False
