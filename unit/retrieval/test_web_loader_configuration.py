"""Regression: two v0.11.0 web loader fixes that no HTTP route or browser-free lane can reach.

- urllib3-future socket options (PR #26796, commit 7ef0530b2, issue #26791): `_ssrf_safe_new_conn`
  unpacked every socket option straight into `setsockopt()`, which takes three arguments.
  urllib3-future, which shadows urllib3 once a tool or function installs it, uses a four-element
  per-protocol form, so every synchronous web fetch failed on connect. No route sets socket
  options, so the SSRF-safe session is given them the way a requests adapter is, and fetches a
  local page.
- Leaked Playwright sessions (PR #27526, commit 94b1b7e6b, issue #25880): `SafePlaywrightURLLoader`
  never closed a page and closed the browser only after the URL loop finished, so a page timeout
  or an abandoned search left the remote browser session open until a restart. The unit lane has
  no browser, so the loader runs against specced Playwright classes (`unit/specced_playwright.py`).

The admin loader settings (#26747), embedding failures (#26883) and per-URL parser (#27367) moved
to integration/retrieval/test_web_loader_configuration.py.

Discriminates: passes on dev bbfa876af; passing every option to `setsockopt` whole fails the
four-element cases, and opening the page outside a `with` block while closing the browser only
after the loop fails all five cleanup cases.
"""

from __future__ import annotations

import socket

import playwright.async_api
import playwright.sync_api
import pytest

from harness.listener import listening, text_answer
from unit.specced_playwright import released, specced_browser

pytestmark = pytest.mark.regression


@pytest.fixture
def pages(retrieval_web_utils_module, monkeypatch):
    """Two local pages the loader may fetch."""
    monkeypatch.setattr(retrieval_web_utils_module, "ENABLE_LOCAL_WEB_FETCH", True)
    with listening() as page_host:
        for name in ("a", "b"):
            page_host.route("GET", f"/{name}", text_answer(f"<p>page {name}</p>"))
        yield [f"{page_host.base_url}/a", f"{page_host.base_url}/b"]


@pytest.mark.parametrize(
    "socket_options",
    [
        [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1, "tcp")],  # urllib3-future's form
        [(socket.IPPROTO_UDP, 1, 1, "udp")],  # meant for a udp socket, so skipped
        [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)],  # stock urllib3's form
        None,
    ],
)
def test_a_fetch_connects_with_every_socket_option_form(
    retrieval_web_utils_module, pages, socket_options
):
    session = retrieval_web_utils_module.get_ssrf_safe_requests_session()
    session.get_adapter(pages[0]).init_poolmanager(1, 1, socket_options=socket_options)

    response = session.get(pages[0], timeout=10)

    assert response.text == "<p>page a</p>"


def playwright_loader(module, urls, continue_on_failure=False):
    return module.SafePlaywrightURLLoader(
        web_paths=urls, verify_ssl=False, continue_on_failure=continue_on_failure
    )


async def load_async(documents):
    return [document async for document in documents]


def test_a_page_timeout_releases_the_page_and_the_browser(retrieval_web_utils_module, pages):
    loader = playwright_loader(retrieval_web_utils_module, pages[:1])

    with specced_browser(failing=(pages[0],)) as browsing:
        with pytest.raises(playwright.sync_api.TimeoutError):
            list(loader.lazy_load())

    assert [released(page) for page in browsing.pages] == [True]
    assert released(browsing.browser)


def test_every_page_is_released_as_soon_as_it_is_loaded(retrieval_web_utils_module, pages):
    loader = playwright_loader(retrieval_web_utils_module, pages)

    with specced_browser() as browsing:
        documents = list(loader.lazy_load())

    assert [document.metadata["source"] for document in documents] == pages
    assert [released(page) for page in browsing.pages] == [True, True]
    assert released(browsing.browser)


@pytest.mark.asyncio
async def test_an_abandoned_search_releases_the_browser(retrieval_web_utils_module, pages):
    documents = playwright_loader(retrieval_web_utils_module, pages).alazy_load()

    with specced_browser(asynchronous=True) as browsing:
        first = await documents.__anext__()
        await documents.aclose()

    assert first.metadata["source"] == pages[0]
    assert [released(page) for page in browsing.pages] == [True]
    assert released(browsing.browser)


@pytest.mark.asyncio
async def test_an_async_page_timeout_releases_the_page_and_the_browser(
    retrieval_web_utils_module, pages
):
    loader = playwright_loader(retrieval_web_utils_module, pages[:1])

    with specced_browser(asynchronous=True, failing=(pages[0],)) as browsing:
        with pytest.raises(playwright.async_api.TimeoutError):
            await load_async(loader.alazy_load())

    assert [released(page) for page in browsing.pages] == [True]
    assert released(browsing.browser)


@pytest.mark.asyncio
async def test_a_tolerated_failure_still_releases_everything(retrieval_web_utils_module, pages):
    loader = playwright_loader(retrieval_web_utils_module, pages, continue_on_failure=True)

    with specced_browser(asynchronous=True, failing=(pages[0],)) as browsing:
        documents = await load_async(loader.alazy_load())

    assert [document.metadata["source"] for document in documents] == pages[1:]
    assert [released(page) for page in browsing.pages] == [True, True]
    assert released(browsing.browser)
