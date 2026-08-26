"""Regression: three 0.11.0 fixes in `open_webui/retrieval/web/utils.py`.

1. `1717b49` added address unwrapping, which pulls out the IPv4 address an IPv6 answer can
   carry (IPv4-mapped, IPv4-compatible `::a.b.c.d`, 6to4, teredo, NAT64) and re-checks the
   embedded address. v0.10.2 asked `ipaddress.ip_address(ip).is_global` and nothing else, so
   any encoding CPython reports as global reached an internal host. On CPython 3.12 the
   encodings that actually slipped through are `::a.b.c.d` and the NAT64 well-known prefix
   `64:ff9b::/96`; mapped/6to4/teredo are already non-global there and are covered as the
   broad layer.

2. `bef63a2` rewrote the Playwright route hooks. v0.10.2 waved through every request whose
   `resource_type` was not `document`, and when `AIOHTTP_CLIENT_ALLOW_REDIRECTS` was on it
   let Playwright follow the redirect chain unvalidated. So a public page could pull an
   internal target as a sub-resource, or redirect to one. v0.11.0 validates every request and
   every hop, and blocks service workers and websockets at page creation.

3. `1e0ab8471` (#27528, issue #26079) unshadowed the `time` module. `from datetime import
   datetime, time, timedelta` made `_sync_wait_for_rate_limit` call `datetime.time.sleep`,
   which raises `AttributeError`. The exception surfaced inside each loader's per-URL `try`,
   so with `continue_on_failure=True` the page in flight was dropped, and Tavily logged the
   loss as an SSL verification failure.

v0.11.1 (`e3e4bd87d`, #27823) kept all three verdicts and moved the machinery: the route hooks
take the SSRF-safe session as a second argument and fetch each hop through it rather than
through Playwright's `route.fetch`, fulfilling with an explicit status/headers/body, and
`SafeTavilyLoader` lost its `api_base_url` argument. The hooks are therefore driven through
whichever shape the checkout declares.

Discriminates: passes on v0.11.0 and v0.11.1, fails on v0.10.2 (no address unwrapping,
sub-resources and redirect hops unchecked, no worker/socket block, and the rate limiter raising
instead of sleeping).

No network is touched: DNS is replaced with a lookup table, the Playwright driver and the
per-hop transport are fakes, and the pacing path runs on a frozen clock so the one real sleep
is a single millisecond.
"""

from __future__ import annotations

import inspect
from datetime import datetime
from types import SimpleNamespace

import pytest
from multidict import CIMultiDict

pytestmark = pytest.mark.regression


# --- DNS lookup table standing in for resolve_hostname ---

FAKE_DNS = {
    "public.example": (["93.184.216.34"], []),
    "internal.example": (["10.0.0.5"], []),
    "127.0.0.1": (["127.0.0.1"], []),
    "firecrawl.test": (["93.184.216.34"], []),
}


def _install_address(monkeypatch, mod, address):
    """Point validate_url at a single resolved address, with the filters neutral."""
    monkeypatch.setattr(mod, "ENABLE_LOCAL_WEB_FETCH", False)
    monkeypatch.setattr(mod, "WEB_FETCH_FILTER_LIST", [])
    is_v6 = ":" in address
    monkeypatch.setattr(
        mod,
        "resolve_hostname",
        lambda hostname: ([] if is_v6 else [address], [address] if is_v6 else []),
    )


def _install_fake_dns(monkeypatch, mod):
    monkeypatch.setattr(mod, "ENABLE_LOCAL_WEB_FETCH", False)
    monkeypatch.setattr(mod, "WEB_FETCH_FILTER_LIST", [])
    monkeypatch.setattr(mod, "resolve_hostname", lambda hostname: FAKE_DNS[hostname])


# --- narrow: IPv6 encodings that CPython calls global but that carry an internal IPv4 ---


@pytest.mark.parametrize(
    "address",
    [
        "::7f00:1",  # IPv4-compatible 127.0.0.1
        "::a00:1",  # IPv4-compatible 10.0.0.1
        "::a9fe:a9fe",  # IPv4-compatible 169.254.169.254
        "64:ff9b::127.0.0.1",  # NAT64 well-known prefix
        "64:ff9b::10.0.0.1",
        "64:ff9b::169.254.169.254",
    ],
)
def test_ipv6_wrapping_an_internal_ipv4_is_refused(
    retrieval_web_utils_module, monkeypatch, address
):
    mod = retrieval_web_utils_module
    _install_address(monkeypatch, mod, address)

    with pytest.raises(ValueError):
        mod.validate_url("http://target.example")


# --- broad: every embedding form the fix knows about resolves to the same verdict ---


@pytest.mark.parametrize(
    "address",
    [
        "::ffff:127.0.0.1",
        "::ffff:10.0.0.1",
        "::ffff:169.254.169.254",
        "2002:7f00:0001::",  # 6to4 wrapping 127.0.0.1
        "2002:0a00:0001::",  # 6to4 wrapping 10.0.0.1
        "2002:a9fe:a9fe::",  # 6to4 wrapping 169.254.169.254
        "2001:0:4136:e378:8000:63bf:3fff:fdd2",  # teredo
        "64:ff9b:1::7f00:1",  # NAT64 local-use prefix
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "::1",
    ],
)
def test_every_internal_address_encoding_is_refused(
    retrieval_web_utils_module, monkeypatch, address
):
    mod = retrieval_web_utils_module
    _install_address(monkeypatch, mod, address)

    with pytest.raises(ValueError):
        mod.validate_url("http://target.example")


# --- nearby: genuinely public addresses, wrapped or not, stay reachable ---


@pytest.mark.parametrize(
    "address",
    [
        "93.184.216.34",
        "2606:4700:4700::1111",
        "::1.1.1.1",  # IPv4-compatible, public payload
        "64:ff9b::8.8.8.8",  # NAT64, public payload
    ],
)
def test_public_addresses_are_still_allowed(retrieval_web_utils_module, monkeypatch, address):
    mod = retrieval_web_utils_module
    _install_address(monkeypatch, mod, address)

    assert mod.validate_url("http://target.example") is True


# --- fake Playwright route objects and the per-hop transport behind them ---


class CannedResponse:
    def __init__(self, url, status=200, location=None):
        self.url = url
        self.status = status
        self.headers = {"location": location} if location else {}


class Transport:
    """Serves one canned response per URL and records every hop that was fetched."""

    def __init__(self, responses):
        self.responses = responses
        self.fetched = []

    def serve(self, url):
        self.fetched.append(url)
        return self.responses[url]


class _RawHeaders:
    def __init__(self, headers):
        self._headers = dict(headers)

    def getlist(self, name):
        value = self._headers.get(name)
        return [value] if value else []

    def items(self):
        return self._headers.items()


class _RequestsShapedResponse:
    def __init__(self, canned):
        self.url = canned.url
        self.status_code = canned.status
        self.headers = dict(canned.headers)
        self.content = canned.url.encode()  # identifies which hop was fulfilled
        self.raw = SimpleNamespace(headers=_RawHeaders(canned.headers))


class _AiohttpShapedResponse:
    def __init__(self, canned):
        self.url = canned.url
        self.status = canned.status
        self.headers = CIMultiDict(canned.headers)
        self._body = canned.url.encode()

    async def read(self):
        return self._body


class FakeRequestsSession:
    def __init__(self, transport):
        self.transport = transport

    def request(self, method, url, **kwargs):
        return _RequestsShapedResponse(self.transport.serve(url))


class FakeAiohttpSession:
    def __init__(self, transport):
        self.transport = transport

    async def request(self, method, url, **kwargs):
        return _AiohttpShapedResponse(self.transport.serve(url))


class FakeRequest:
    def __init__(self, url, resource_type):
        self.url = url
        self.resource_type = resource_type
        self.method = "GET"
        self.post_data_buffer = None

    def all_headers(self):
        return {"user-agent": "test", "host": "public.example"}


class AsyncFakeRequest(FakeRequest):
    async def all_headers(self):
        return FakeRequest.all_headers(self)


class FakeRoute:
    """Records what the hook decided; `fetch` serves the same canned responses the
    session does, so either transport shape drives the same scenario."""

    request_cls = FakeRequest

    def __init__(self, url, resource_type="document", responses=None):
        self.transport = Transport(responses or {url: CannedResponse(url)})
        self.request = self.request_cls(url, resource_type)
        self.actions = []
        self.fulfilled_url = None

    @property
    def fetched(self):
        return self.transport.fetched

    def fetch(self, url=None, max_redirects=None):
        return self.transport.serve(url or self.request.url)

    def continue_(self):
        self.actions.append("continue")

    def abort(self):
        self.actions.append("abort")

    def fulfill(self, response=None, status=None, headers=None, body=None):
        self.actions.append("fulfill")
        self.fulfilled_url = response.url if response is not None else body.decode()


class AsyncFakeRoute(FakeRoute):
    request_cls = AsyncFakeRequest

    async def fetch(self, url=None, max_redirects=None):
        return FakeRoute.fetch(self, url=url, max_redirects=max_redirects)

    async def continue_(self):
        FakeRoute.continue_(self)

    async def abort(self):
        FakeRoute.abort(self)

    async def fulfill(self, response=None, status=None, headers=None, body=None):
        FakeRoute.fulfill(self, response=response, status=status, headers=headers, body=body)


def _playwright_loader(mod, **kwargs):
    loader = mod.SafePlaywrightURLLoader(
        web_paths=kwargs.pop("web_paths", ["http://public.example/"]),
        verify_ssl=False,
        **kwargs,
    )
    loader.evaluator = SimpleNamespace(evaluate=lambda page, browser, response: "text")
    return loader


def _takes_session(hook):
    """0.11.1 fetches each hop through a session handed to the hook; 0.11.0 uses route.fetch."""
    return "session" in inspect.signature(hook).parameters


def _drive_sync_hook(mod, route):
    hook = _playwright_loader(mod)._intercept_navigation_sync
    if _takes_session(hook):
        hook(route, FakeRequestsSession(route.transport))
    else:
        hook(route)


async def _drive_async_hook(mod, route):
    hook = _playwright_loader(mod)._intercept_navigation
    if _takes_session(hook):
        await hook(route, FakeAiohttpSession(route.transport))
    else:
        await hook(route)


# --- narrow: sub-resource requests go through the address rules ---


@pytest.mark.parametrize("resource_type", ["image", "xhr", "fetch", "script"])
def test_sub_resource_request_to_an_internal_host_is_refused(
    retrieval_web_utils_module, monkeypatch, resource_type
):
    mod = retrieval_web_utils_module
    _install_fake_dns(monkeypatch, mod)
    route = FakeRoute("http://internal.example/admin", resource_type=resource_type)

    _drive_sync_hook(mod, route)

    assert route.actions == ["abort"], (
        f"a {resource_type} sub-resource pointing at an internal host was not refused: "
        f"{route.actions}"
    )
    assert route.fetched == [], "the internal sub-resource was fetched anyway"


def test_sub_resource_request_is_not_waved_through_unvalidated(
    retrieval_web_utils_module, monkeypatch
):
    mod = retrieval_web_utils_module
    _install_fake_dns(monkeypatch, mod)
    route = FakeRoute("http://public.example/logo.png", resource_type="image")

    _drive_sync_hook(mod, route)

    assert "continue" not in route.actions, "sub-resource bypassed the fetch hook"
    assert route.actions == ["fulfill"]


@pytest.mark.asyncio
async def test_async_sub_resource_request_to_an_internal_host_is_refused(
    retrieval_web_utils_module, monkeypatch
):
    mod = retrieval_web_utils_module
    _install_fake_dns(monkeypatch, mod)
    route = AsyncFakeRoute("http://internal.example/admin", resource_type="xhr")

    await _drive_async_hook(mod, route)

    assert route.actions == ["abort"]
    assert route.fetched == []


# --- narrow: each redirect hop is validated in turn ---


def test_redirect_hop_pointing_at_loopback_is_refused(retrieval_web_utils_module, monkeypatch):
    mod = retrieval_web_utils_module
    _install_fake_dns(monkeypatch, mod)
    monkeypatch.setattr(mod, "AIOHTTP_CLIENT_ALLOW_REDIRECTS", True)
    start = "http://public.example/go"
    route = FakeRoute(
        start,
        responses={
            start: CannedResponse(start, status=302, location="http://127.0.0.1/admin"),
            "http://127.0.0.1/admin": CannedResponse("http://127.0.0.1/admin"),
        },
    )

    _drive_sync_hook(mod, route)

    assert route.actions == ["abort"], "redirect hop to loopback was not refused"
    assert "http://127.0.0.1/admin" not in route.fetched


def test_redirect_chain_is_followed_hop_by_hop_to_the_final_response(
    retrieval_web_utils_module, monkeypatch
):
    mod = retrieval_web_utils_module
    _install_fake_dns(monkeypatch, mod)
    monkeypatch.setattr(mod, "AIOHTTP_CLIENT_ALLOW_REDIRECTS", True)
    start = "http://public.example/go"
    hop = "http://public.example/next"
    route = FakeRoute(
        start,
        responses={
            start: CannedResponse(start, status=302, location="/next"),
            hop: CannedResponse(hop, status=200),
        },
    )

    _drive_sync_hook(mod, route)

    assert route.actions == ["fulfill"]
    assert route.fulfilled_url == hop, "the hop was not fetched separately"
    assert route.fetched == [start, hop]


@pytest.mark.asyncio
async def test_async_redirect_hop_pointing_at_loopback_is_refused(
    retrieval_web_utils_module, monkeypatch
):
    mod = retrieval_web_utils_module
    _install_fake_dns(monkeypatch, mod)
    monkeypatch.setattr(mod, "AIOHTTP_CLIENT_ALLOW_REDIRECTS", True)
    start = "http://public.example/go"
    route = AsyncFakeRoute(
        start,
        responses={
            start: CannedResponse(start, status=302, location="http://127.0.0.1/admin"),
            "http://127.0.0.1/admin": CannedResponse("http://127.0.0.1/admin"),
        },
    )

    await _drive_async_hook(mod, route)

    assert route.actions == ["abort"]
    assert "http://127.0.0.1/admin" not in route.fetched


def test_redirect_chain_is_bounded(retrieval_web_utils_module, monkeypatch):
    mod = retrieval_web_utils_module
    _install_fake_dns(monkeypatch, mod)
    monkeypatch.setattr(mod, "AIOHTTP_CLIENT_ALLOW_REDIRECTS", True)
    start = "http://public.example/loop"
    route = FakeRoute(
        start, responses={start: CannedResponse(start, status=302, location="/loop")}
    )

    _drive_sync_hook(mod, route)

    assert route.actions == ["abort"], "an endless redirect chain was not aborted"
    assert len(route.fetched) <= 25


# --- narrow: service workers and websockets are blocked at page creation ---


class FakePage:
    def __init__(self):
        self.ws_handlers = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def route(self, pattern, handler):
        pass

    def route_web_socket(self, pattern, handler):
        self.ws_handlers.append(handler)

    def goto(self, url, timeout=None):
        return CannedResponse(url)


class FakeBrowser:
    def __init__(self):
        self.new_page_kwargs = []
        self.pages = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def new_page(self, **kwargs):
        self.new_page_kwargs.append(kwargs)
        page = FakePage()
        self.pages.append(page)
        return page

    def close(self):
        """0.10.2 closes the browser explicitly; 0.11.x uses `with browser:`."""
        pass


class FakePlaywrightContext:
    def __init__(self, browser):
        self.chromium = SimpleNamespace(
            launch=lambda **kwargs: browser,
            connect=lambda *args, **kwargs: browser,
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run_playwright_loader(mod, monkeypatch):
    import playwright.sync_api

    browser = FakeBrowser()
    monkeypatch.setattr(
        playwright.sync_api,
        "sync_playwright",
        lambda: FakePlaywrightContext(browser),
    )
    docs = list(_playwright_loader(mod, continue_on_failure=False).lazy_load())
    return browser, docs


def test_service_workers_are_blocked_on_every_fetched_page(retrieval_web_utils_module, monkeypatch):
    mod = retrieval_web_utils_module
    browser, docs = _run_playwright_loader(mod, monkeypatch)

    assert len(docs) == 1
    assert browser.new_page_kwargs == [{"service_workers": "block"}], (
        f"page was created without the service-worker block: {browser.new_page_kwargs}"
    )


def test_websocket_connections_the_page_opens_are_closed(retrieval_web_utils_module, monkeypatch):
    mod = retrieval_web_utils_module
    browser, _ = _run_playwright_loader(mod, monkeypatch)

    handlers = browser.pages[0].ws_handlers
    assert handlers, "no websocket route was registered, the page can dial any host"

    closed = []
    handlers[0](SimpleNamespace(close=lambda: closed.append(True)))
    assert closed == [True], "the websocket route handler did not close the connection"


# --- narrow: the paced loaders must not drop the page they are pacing ---


@pytest.fixture
def frozen_clock(retrieval_web_utils_module, monkeypatch):
    """Freeze the loader's clock so pacing always takes the sleep branch."""
    instant = datetime.now()
    monkeypatch.setattr(
        retrieval_web_utils_module,
        "datetime",
        SimpleNamespace(now=staticmethod(lambda: instant)),
    )
    return instant


def test_paced_firecrawl_fetch_returns_every_page(
    retrieval_web_utils_module, monkeypatch, frozen_clock
):
    mod = retrieval_web_utils_module
    from langchain_core.documents import Document

    scraped = []

    def fake_scrape(api_url, api_key, url, **kwargs):
        scraped.append(url)
        return Document(page_content=f"body of {url}", metadata={"source": url})

    monkeypatch.setattr(mod, "scrape_firecrawl_url", fake_scrape)
    urls = ["http://public.example/a", "http://public.example/b", "http://public.example/c"]
    loader = mod.SafeFireCrawlLoader(
        web_paths=urls,
        verify_ssl=False,
        requests_per_second=1000,  # 1 ms interval, so the real sleep stays bounded
        api_key="k",
        api_url="http://firecrawl.test",
    )

    docs = list(loader.lazy_load())

    assert scraped == urls, f"pacing dropped a page before it was fetched: {scraped}"
    assert [d.metadata["source"] for d in docs] == urls


def test_paced_url_check_succeeds_instead_of_raising(
    retrieval_web_utils_module, monkeypatch, frozen_clock
):
    mod = retrieval_web_utils_module
    loader = mod.SafeFireCrawlLoader(
        web_paths=[], verify_ssl=False, requests_per_second=1000, api_key="k"
    )
    loader._sync_wait_for_rate_limit()

    assert loader._safe_process_url_sync("http://public.example/a") is True


def test_paced_tavily_loss_is_not_reported_as_a_security_check_failure(
    retrieval_web_utils_module, monkeypatch, frozen_clock, caplog
):
    mod = retrieval_web_utils_module
    from langchain_core.documents import Document

    urls = ["http://public.example/a", "http://public.example/b"]

    class FakeTavilyLoader:
        def __init__(self, urls, **kwargs):
            self.urls = urls

        def lazy_load(self):
            for url in self.urls:
                yield Document(page_content="body", metadata={"source": url})

    monkeypatch.setattr(mod, "TavilyLoader", FakeTavilyLoader)
    kwargs = {
        "web_paths": urls,
        "verify_ssl": False,
        "requests_per_second": 1000,
        "api_key": "k",
    }
    # 0.11.1 dropped api_base_url from the constructor.
    if "api_base_url" in inspect.signature(mod.SafeTavilyLoader.__init__).parameters:
        kwargs["api_base_url"] = "https://api.tavily.test"
    loader = mod.SafeTavilyLoader(**kwargs)

    with caplog.at_level("WARNING", logger=mod.log.name):
        docs = list(loader.lazy_load())

    assert [d.metadata["source"] for d in docs] == urls
    assert "SSL verification failed" not in caplog.text


# --- nearby: pacing is skipped entirely when no rate is configured ---


def test_unpaced_fetch_returns_every_page(retrieval_web_utils_module, monkeypatch):
    mod = retrieval_web_utils_module
    from langchain_core.documents import Document

    monkeypatch.setattr(
        mod,
        "scrape_firecrawl_url",
        lambda api_url, api_key, url, **kwargs: Document(
            page_content="body", metadata={"source": url}
        ),
    )
    urls = ["http://public.example/a", "http://public.example/b"]
    loader = mod.SafeFireCrawlLoader(
        web_paths=urls, verify_ssl=False, requests_per_second=None, api_key="k"
    )

    assert [d.metadata["source"] for d in loader.lazy_load()] == urls
