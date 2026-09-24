"""Regression: the Playwright web loader checks every request and every redirect hop it routes.

`bef63a2` (v0.11.0) rewrote the Playwright route hooks in `open_webui/retrieval/web/utils.py`.
v0.10.2 waved through every request whose `resource_type` was not `document`, and with
`AIOHTTP_CLIENT_ALLOW_REDIRECTS` on it let the browser follow a redirect chain unchecked, so a
public page could pull an internal target as a sub-resource or redirect to one. v0.11.0 fetches
every request and every hop itself through the address checks and blocks service workers and
websockets when it opens the page.

The unit lane has no browser, so `lazy_load` and `alazy_load` run against specced Playwright
classes (`unit/specced_playwright.py`) while the loader's own SSRF-safe sessions make real
requests: the page host is a local service the loader may fetch, the internal host a second one
on 127.0.0.2 that the fetch filter list blocks.

The address unwrapping (`1717b49`) and the pacing fix (`1e0ab8471`) moved to
integration/security/test_ssrf_address_and_fetch_scope.py; the positive path for public
addresses stays here because showing it over HTTP means fetching a public host.

Discriminates: passes on dev bbfa876af; continuing sub-resource routes, continuing a redirected
document, opening the page without `service_workers='block'` or dropping the websocket route each
fail their tests, and blocking every address that embeds an IPv4 one fails the public addresses.
"""

from __future__ import annotations

import pytest

from harness.listener import listening, text_answer
from unit.specced_playwright import released, specced_browser, websocket_opened

pytestmark = pytest.mark.regression

INTERNAL_HOST = "127.0.0.2"


@pytest.mark.parametrize(
    "address",
    ["93.184.216.34", "[2606:4700:4700::1111]", "[::1.1.1.1]", "[64:ff9b::8.8.8.8]"],
)
def test_public_addresses_stay_allowed(retrieval_web_utils_module, monkeypatch, address):
    monkeypatch.setattr(retrieval_web_utils_module, "ENABLE_LOCAL_WEB_FETCH", False)
    monkeypatch.setattr(retrieval_web_utils_module, "WEB_FETCH_FILTER_LIST", [])

    assert retrieval_web_utils_module.validate_url(f"http://{address}/") is True


@pytest.fixture
def hosts(retrieval_web_utils_module, monkeypatch):
    """A page host the loader may fetch, and an internal host the filter list blocks."""
    monkeypatch.setattr(retrieval_web_utils_module, "ENABLE_LOCAL_WEB_FETCH", True)
    monkeypatch.setattr(retrieval_web_utils_module, "WEB_FETCH_FILTER_LIST", [f"!{INTERNAL_HOST}"])
    with listening() as page_host, listening(host=INTERNAL_HOST) as internal_host:
        page_host.route("GET", "/page", text_answer("<p>public page</p>"))
        page_host.route("GET", "/app.js", text_answer("run()", content_type="text/javascript"))
        internal_host.route("GET", "/admin", text_answer("internal only"))
        yield page_host, internal_host


def playwright_loader(module, url, continue_on_failure=False):
    return module.SafePlaywrightURLLoader(
        web_paths=[url], verify_ssl=False, continue_on_failure=continue_on_failure
    )


async def load_async(loader):
    return [document async for document in loader.alazy_load()]


def assert_refused(route):
    assert route.abort.called, f"{route.request.url} was not aborted"
    assert not route.continue_.called, f"{route.request.url} was left to the browser"
    assert not route.fulfill.called


@pytest.mark.parametrize("resource_type", ["script", "stylesheet", "xhr", "fetch", "image"])
def test_sub_resource_on_an_internal_host_is_refused(
    retrieval_web_utils_module, hosts, resource_type
):
    page_host, internal_host = hosts
    internal = f"{internal_host.base_url}/admin"

    with specced_browser(subresources=((internal, resource_type),)) as browsing:
        list(
            playwright_loader(retrieval_web_utils_module, f"{page_host.base_url}/page").lazy_load()
        )

    assert_refused(browsing.routes_to(internal)[0])
    assert internal_host.received == []


@pytest.mark.asyncio
async def test_async_sub_resource_on_an_internal_host_is_refused(retrieval_web_utils_module, hosts):
    page_host, internal_host = hosts
    internal = f"{internal_host.base_url}/admin"

    with specced_browser(asynchronous=True, subresources=((internal, "xhr"),)) as browsing:
        await load_async(
            playwright_loader(retrieval_web_utils_module, f"{page_host.base_url}/page")
        )

    assert_refused(browsing.routes_to(internal)[0])
    assert internal_host.received == []


def test_public_sub_resource_is_fetched_by_the_loader_not_the_browser(
    retrieval_web_utils_module, hosts
):
    page_host, _ = hosts
    script = f"{page_host.base_url}/app.js"

    with specced_browser(subresources=((script, "script"),)) as browsing:
        list(
            playwright_loader(retrieval_web_utils_module, f"{page_host.base_url}/page").lazy_load()
        )

    [route] = browsing.routes_to(script)
    assert not route.continue_.called
    assert route.fulfill.call_args.kwargs["body"] == b"run()"


@pytest.fixture
def redirects_allowed(retrieval_web_utils_module, monkeypatch):
    monkeypatch.setattr(retrieval_web_utils_module, "AIOHTTP_CLIENT_ALLOW_REDIRECTS", True)


def redirect_to(location):
    return 302, {"Location": location}, b""


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_redirect_hop_to_an_internal_host_is_refused(
    retrieval_web_utils_module, hosts, redirects_allowed, asynchronous
):
    page_host, internal_host = hosts
    start = f"{page_host.base_url}/go"
    page_host.route("GET", "/go", redirect_to(f"{internal_host.base_url}/admin"))
    loader = playwright_loader(retrieval_web_utils_module, start, continue_on_failure=True)

    with specced_browser(asynchronous=asynchronous) as browsing:
        documents = await load_async(loader) if asynchronous else list(loader.lazy_load())

    assert documents == []
    assert_refused(browsing.routes_to(start)[0])
    assert internal_host.received == []


def test_redirect_chain_is_followed_hop_by_hop(
    retrieval_web_utils_module, hosts, redirects_allowed
):
    page_host, _ = hosts
    start = f"{page_host.base_url}/go"
    page_host.route("GET", "/go", redirect_to("/page"))

    with specced_browser() as browsing:
        list(playwright_loader(retrieval_web_utils_module, start).lazy_load())

    [route] = browsing.routes_to(start)
    assert route.fulfill.call_args.kwargs["body"] == b"<p>public page</p>"
    assert [request.path for request in page_host.received] == ["/go", "/page"]


def test_endless_redirect_chain_is_abandoned(retrieval_web_utils_module, hosts, redirects_allowed):
    page_host, _ = hosts
    start = f"{page_host.base_url}/loop"
    page_host.route("GET", "/loop", redirect_to("/loop"))
    loader = playwright_loader(retrieval_web_utils_module, start, continue_on_failure=True)

    with specced_browser() as browsing:
        assert list(loader.lazy_load()) == []

    assert_refused(browsing.routes_to(start)[0])
    assert len(page_host.requests_to("/loop")) <= 25


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_every_page_blocks_service_workers_and_websockets(
    retrieval_web_utils_module, hosts, asynchronous
):
    page_host, _ = hosts
    loader = playwright_loader(retrieval_web_utils_module, f"{page_host.base_url}/page")

    with specced_browser(asynchronous=asynchronous) as browsing:
        documents = await load_async(loader) if asynchronous else list(loader.lazy_load())
    websocket = await websocket_opened(browsing.pages[0], asynchronous=asynchronous)

    assert len(documents) == 1
    assert browsing.browser.new_page.call_args.kwargs.get("service_workers") == "block"
    assert not websocket.connect_to_server.called, "the page's websocket may dial any host"
    if not asynchronous:
        assert not websocket.close.called, "sync close() inside the handler spins a core (#30024)"
    assert released(browsing.pages[0])
