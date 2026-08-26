"""Five web-loader regressions fixed in v0.11.0.

- Admin web loader settings ignored (PR #26749, commit f7e7f3210, issue #26747,
  plus commit 304cbe456 for the last call site): `get_web_loader` dispatched on
  the module-level `WEB_LOADER_ENGINE` constant read from the environment at
  import time, so the engine and per-engine settings saved in the Admin UI were
  never consulted and pages were always fetched directly by the built-in loader.
  Now the resolved DB config is passed in as `loader_config` and wins over the
  env constants, and `process_web_search` reads SSL verification, pacing and
  trust_env from that same live config instead of the startup config object.
- urllib3-future socket options (PR #26796, commit 7ef0530b2, issue #26791):
  `_ssrf_safe_new_conn` unpacked every socket option straight into
  `setsockopt()`, which takes exactly three arguments. urllib3-future, which
  shadows urllib3 when a tool or function pulls it in, uses a four-element
  per-protocol form, so every synchronous web fetch failed on connect.
- Silent web search embedding failures (PR #26883, commit 6c7478c1c): when the
  pages were fetched but the vector write failed, `process_web_search` logged at
  debug level and still returned `status: True` with a collection name, so the
  chat said "searched N sites" and then found no sources. It now raises a 500
  naming the embedding configuration.
- Batch parser locked to the first URL (PR #27367, commit acf586c00):
  `_unpack_fetch_results` assigned the resolved parser back to the `parser`
  parameter, so the None check only ran for the first URL and every later
  document in a mixed batch was parsed with the first URL's parser.
- Leaked Playwright sessions (PR #27526, commit 94b1b7e6b, issue #25880):
  `SafePlaywrightURLLoader` closed the page never and the browser only after the
  URL loop finished normally, so a page timeout or an abandoned search left
  pages and the remote browser session open until the server was restarted.

Discriminates: passes on v0.11.0, fails on v0.10.2 (pre-fix `get_loader` builds
the built-in loader whatever engine the admin picked, `process_web_search`
passes the startup config values, a four-element socket option reaches
`setsockopt` intact, an embedding failure returns a healthy-looking collection,
the second document of a mixed batch is parsed with the first URL's parser, and
neither the page nor the browser is closed when `goto` times out or the caller
abandons the loader).
"""

from __future__ import annotations

import asyncio
import socket
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import playwright.async_api
import playwright.sync_api
import pytest

pytestmark = pytest.mark.regression

PUBLIC_IP = "93.184.216.34"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def retrieval_router_module(owui_module):
    """`open_webui.routers.retrieval` (process_web_search)."""
    return owui_module("open_webui.routers.retrieval")


# =============================================================================
# Admin web loader settings (PR #26749 / commit 304cbe456)
# =============================================================================


def _build_via_get_loader(utils_module, web_utils_module, config: dict):
    """Run the real `get_loader`, skipping only the DNS/SSRF check."""
    with patch.object(web_utils_module, "safe_validate_urls", lambda urls: list(urls)):
        return utils_module.get_loader(None, "https://example.com/page", config)


def test_get_loader_honors_admin_selected_engine(
    retrieval_utils_module, retrieval_web_utils_module
) -> None:
    """Regression for issue #26747: the Admin UI loader engine was ignored.

    An egress-restricted instance configured for an external loader kept
    fetching pages directly with the built-in loader.
    """
    from open_webui.retrieval.loaders.external_web import ExternalWebLoader

    loader = _build_via_get_loader(
        retrieval_utils_module,
        retrieval_web_utils_module,
        {
            "web_loader_engine": "external",
            "external_web_loader_url": "https://loader.internal/fetch",
            "external_web_loader_api_key": "loader-key",
            "web_loader_ssl_verification": False,
            "web_loader_concurrent_requests": 5,
            "web_search_trust_env": True,
        },
    )

    assert isinstance(loader, ExternalWebLoader)
    assert loader.external_url == "https://loader.internal/fetch"
    assert loader.external_api_key == "loader-key"


@pytest.mark.parametrize(
    ("engine", "extra", "class_name"),
    [
        (
            "playwright",
            {"playwright_ws_url": "ws://playwright.internal/ws", "playwright_timeout": 5000},
            "SafePlaywrightURLLoader",
        ),
        (
            "firecrawl",
            {"firecrawl_api_key": "fc-key", "firecrawl_api_url": "https://fc.internal"},
            "SafeFireCrawlLoader",
        ),
        (
            "external",
            {
                "external_web_loader_url": "https://loader.internal/fetch",
                "external_web_loader_api_key": "loader-key",
            },
            "ExternalWebLoader",
        ),
    ],
)
def test_every_admin_selectable_engine_reaches_the_loader(
    retrieval_utils_module,
    retrieval_web_utils_module,
    engine: str,
    extra: dict,
    class_name: str,
) -> None:
    """Broad: the engine saved by the admin decides the loader, for every engine.

    Tavily and Microsoft Web IQ are left out: their constructors are broken on
    v0.11.0 itself and are covered by test_web_loaders.py.
    """
    config = {"web_loader_engine": engine}
    config.update(extra)

    loader = _build_via_get_loader(retrieval_utils_module, retrieval_web_utils_module, config)

    assert type(loader).__name__ == class_name


def test_playwright_engine_gets_admin_saved_endpoint_and_timeout(
    retrieval_utils_module, retrieval_web_utils_module
) -> None:
    """Per-engine settings come from the admin config, not the boot environment."""
    loader = _build_via_get_loader(
        retrieval_utils_module,
        retrieval_web_utils_module,
        {
            "web_loader_engine": "playwright",
            "playwright_ws_url": "ws://playwright.internal/ws",
            "playwright_timeout": 4321,
        },
    )

    assert loader.playwright_ws_url == "ws://playwright.internal/ws"
    assert loader.playwright_timeout == 4321


def test_get_loader_passes_admin_ssl_and_pacing(
    retrieval_utils_module, retrieval_web_utils_module
) -> None:
    """Nearby: the keys that were already live keep working."""
    loader = _build_via_get_loader(
        retrieval_utils_module,
        retrieval_web_utils_module,
        {
            "web_loader_engine": "safe_web",
            "web_loader_ssl_verification": False,
            "web_loader_concurrent_requests": 7,
            "web_search_trust_env": True,
        },
    )

    assert loader.session.verify is False
    assert loader.requests_per_second == 7
    assert loader.trust_env is True


# =============================================================================
# process_web_search: live loader config (commit 304cbe456) and embedding
# failures (PR #26883)
# =============================================================================


class _SearchItem:
    """Stands in for a SearchResult: attribute access plus `dict(item)`."""

    def __init__(self, link: str, title: str, snippet: str) -> None:
        self.link = link
        self.title = title
        self.snippet = snippet

    def keys(self):
        return ("link", "title", "snippet")

    def __getitem__(self, key):
        return getattr(self, key)


def _retrieval_config(**overrides):
    config = SimpleNamespace(
        ENABLE_WEB_SEARCH=True,
        USER_PERMISSIONS={},
        WEB_SEARCH_ENGINE="duckduckgo",
        WEB_SEARCH_CONCURRENT_REQUESTS=0,
        BYPASS_WEB_SEARCH_WEB_LOADER=True,
        BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=False,
        # Startup-config values the pre-fix call site read instead of the live ones.
        ENABLE_WEB_LOADER_SSL_VERIFICATION=True,
        WEB_LOADER_CONCURRENT_REQUESTS=1,
        WEB_SEARCH_TRUST_ENV=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


async def _run_web_search(module, config, save_docs, loader_config=None, get_web_loader=None):
    """Drive the real `process_web_search` with every I/O boundary stubbed."""

    async def fake_get_retrieval_config():
        return config

    async def fake_search_web(request, engine, query, user=None):
        return [_SearchItem("https://example.com/a", "A", "snippet a")]

    async def fake_get_loader_config():
        return dict(loader_config or {})

    patches = [
        patch.object(module, "get_retrieval_config", fake_get_retrieval_config),
        patch.object(module, "search_web", fake_search_web),
        patch.object(module, "get_loader_config", fake_get_loader_config),
        patch.object(module, "save_docs_to_vector_db", save_docs),
    ]
    if get_web_loader is not None:
        patches.append(patch.object(module, "get_web_loader", get_web_loader))

    user = SimpleNamespace(id="user-1", role="admin", name="tester", email="t@example.com")
    form_data = module.SearchForm(queries=["what is open webui"])

    with ExitStack() as stack:
        for entry in patches:
            stack.enter_context(entry)
        return await asyncio.wait_for(
            module.process_web_search(SimpleNamespace(), form_data, user=user), timeout=30
        )


@pytest.mark.asyncio
async def test_web_search_reports_embedding_failure(retrieval_router_module) -> None:
    """Regression for PR #26883: a failed vector write looked like a clean search.

    The chat showed "searched N sites" and then "no sources found" with no hint
    that the embedding endpoint was misconfigured.
    """
    from fastapi import HTTPException

    def save_docs(*args, **kwargs):
        raise RuntimeError("embedding endpoint unreachable")

    with pytest.raises(HTTPException) as excinfo:
        await _run_web_search(retrieval_router_module, _retrieval_config(), save_docs)

    assert excinfo.value.status_code == 500
    assert "embed" in str(excinfo.value.detail).lower()
    assert "Documents" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_web_search_returns_collection_when_embedding_succeeds(
    retrieval_router_module,
) -> None:
    """Nearby: the healthy path still returns the collection it wrote."""
    calls = []

    def save_docs(*args, **kwargs):
        calls.append(args)

    result = await _run_web_search(retrieval_router_module, _retrieval_config(), save_docs)

    assert result["status"] is True
    assert len(result["collection_names"]) == 1
    assert result["filenames"] == ["https://example.com/a"]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_web_search_builds_loader_from_live_config(retrieval_router_module) -> None:
    """Regression for commit 304cbe456: the search call site mixed stale and live.

    SSL verification, pacing and trust_env came from the startup config object
    while only `loader_config` was resolved from the database.
    """
    recorded = {}

    class _StubLoader:
        async def aload(self):
            from langchain_core.documents import Document

            return [Document(page_content="a", metadata={"source": "https://example.com/a"})]

    def spy_get_web_loader(urls, **kwargs):
        recorded.update(kwargs)
        recorded["urls"] = urls
        return _StubLoader()

    live_config = {
        "web_loader_engine": "safe_web",
        "web_loader_ssl_verification": False,
        "web_loader_concurrent_requests": 7,
        "web_search_trust_env": True,
    }

    def save_docs(*args, **kwargs):
        return None

    await _run_web_search(
        retrieval_router_module,
        _retrieval_config(BYPASS_WEB_SEARCH_WEB_LOADER=False),
        save_docs,
        loader_config=live_config,
        get_web_loader=spy_get_web_loader,
    )

    assert recorded["verify_ssl"] is False
    assert recorded["requests_per_second"] == 7
    assert recorded["trust_env"] is True
    assert recorded["loader_config"] == live_config


# =============================================================================
# urllib3-future socket options (PR #26796)
# =============================================================================


class _RecordingSocket:
    def __init__(self) -> None:
        self.options: list[tuple] = []
        self.connected = None

    def settimeout(self, timeout) -> None:
        return None

    def setsockopt(self, *args) -> None:
        self.options.append(args)

    def connect(self, address) -> None:
        self.connected = address

    def close(self) -> None:
        return None


def _connect_with_socket_options(module, socket_options):
    """Run the real `_ssrf_safe_new_conn` against a recording socket."""
    conn = SimpleNamespace(
        host="example.com",
        port=443,
        timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
        source_address=None,
        socket_options=socket_options,
    )
    recorder = _RecordingSocket()
    addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 443))]

    with patch.object(socket, "getaddrinfo", lambda *a, **k: addrinfo):
        with patch.object(socket, "socket", lambda *a, **k: recorder):
            assert module._ssrf_safe_new_conn(conn) is recorder

    assert recorder.connected == (PUBLIC_IP, 443)
    return recorder


def test_four_element_tcp_socket_option_is_truncated(retrieval_web_utils_module) -> None:
    """Regression for issue #26791: urllib3-future broke every sync web fetch.

    Its per-protocol option tuples carry a fourth element, and `setsockopt`
    takes exactly three arguments.
    """
    recorder = _connect_with_socket_options(
        retrieval_web_utils_module,
        [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1, "tcp")],
    )

    assert recorder.options == [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]


def test_four_element_udp_socket_option_is_skipped(retrieval_web_utils_module) -> None:
    """Every socket built here is SOCK_STREAM, so a udp-only option is dropped."""
    recorder = _connect_with_socket_options(
        retrieval_web_utils_module,
        [(socket.IPPROTO_UDP, 1, 1, "udp")],
    )

    assert recorder.options == []


def test_three_element_socket_option_still_applied(retrieval_web_utils_module) -> None:
    """Nearby: stock urllib3's three-element form is unchanged."""
    recorder = _connect_with_socket_options(
        retrieval_web_utils_module,
        [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)],
    )

    assert recorder.options == [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]


def test_no_socket_options_still_connects(retrieval_web_utils_module) -> None:
    """Nearby: the empty and None cases are not special-cased away."""
    assert _connect_with_socket_options(retrieval_web_utils_module, None).options == []
    assert _connect_with_socket_options(retrieval_web_utils_module, ()).options == []


# =============================================================================
# Per-URL parser (PR #27367)
# =============================================================================


XML_BODY = "<feed><entry><title>a</title></entry></feed>"
HTML_BODY = "<html><body><p>b</p></body></html>"


def _web_base_loader(module):
    return module.SafeWebBaseLoader(
        web_paths=["https://example.com/page"],
        verify_ssl=False,
        continue_on_failure=True,
        trust_env=False,
    )


def test_html_after_xml_is_not_parsed_as_xml(retrieval_web_utils_module) -> None:
    """Regression for PR #27367: the first URL's parser was reused for the batch."""
    loader = _web_base_loader(retrieval_web_utils_module)

    soups = loader._unpack_fetch_results(
        [XML_BODY, HTML_BODY],
        ["https://example.com/feed.xml", "https://example.com/page.html"],
    )

    assert soups[0].builder.is_xml is True
    assert soups[1].builder.is_xml is False


def test_xml_after_html_is_still_parsed_as_xml(retrieval_web_utils_module) -> None:
    """Regression for PR #27367, the other batch order."""
    loader = _web_base_loader(retrieval_web_utils_module)

    soups = loader._unpack_fetch_results(
        [HTML_BODY, XML_BODY],
        ["https://example.com/page.html", "https://example.com/feed.xml"],
    )

    assert soups[0].builder.is_xml is False
    assert soups[1].builder.is_xml is True


def test_explicit_parser_applies_to_whole_batch(retrieval_web_utils_module) -> None:
    """Nearby: an explicitly passed parser still overrides the per-URL guess."""
    loader = _web_base_loader(retrieval_web_utils_module)

    soups = loader._unpack_fetch_results(
        [XML_BODY, HTML_BODY],
        ["https://example.com/feed.xml", "https://example.com/page.html"],
        parser="html.parser",
    )

    assert [soup.builder.NAME for soup in soups] == ["html.parser", "html.parser"]


# =============================================================================
# Playwright session leak (PR #27526)
# =============================================================================


class _FakeEvaluator:
    def evaluate(self, page, browser, response) -> str:
        return "page text"

    async def evaluate_async(self, page, browser, response) -> str:
        return "page text"


class _FakeSyncPage:
    def __init__(self, browser) -> None:
        self.browser = browser
        self.close_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False

    def close(self) -> None:
        self.close_count += 1

    def route(self, *args, **kwargs) -> None:
        return None

    def route_web_socket(self, *args, **kwargs) -> None:
        return None

    def goto(self, url, timeout=None):
        return self.browser.goto(url)


class _FakeSyncBrowser:
    def __init__(self, goto) -> None:
        self._goto = goto
        self.pages: list[_FakeSyncPage] = []
        self.close_count = 0

    def goto(self, url):
        return self._goto(url)

    def new_page(self, **kwargs) -> _FakeSyncPage:
        page = _FakeSyncPage(self)
        self.pages.append(page)
        return page

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False

    def close(self) -> None:
        self.close_count += 1


class _FakeSyncPlaywright:
    def __init__(self, browser) -> None:
        self.chromium = SimpleNamespace(
            connect=lambda *a, **k: browser,
            launch=lambda *a, **k: browser,
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


class _FakeAsyncPage:
    def __init__(self, browser) -> None:
        self.browser = browser
        self.close_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info) -> bool:
        await self.close()
        return False

    async def close(self) -> None:
        self.close_count += 1

    async def route(self, *args, **kwargs) -> None:
        return None

    async def route_web_socket(self, *args, **kwargs) -> None:
        return None

    async def goto(self, url, timeout=None):
        return self.browser.goto(url)


class _FakeAsyncBrowser:
    def __init__(self, goto) -> None:
        self._goto = goto
        self.pages: list[_FakeAsyncPage] = []
        self.close_count = 0

    def goto(self, url):
        return self._goto(url)

    async def new_page(self, **kwargs) -> _FakeAsyncPage:
        page = _FakeAsyncPage(self)
        self.pages.append(page)
        return page

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info) -> bool:
        await self.close()
        return False

    async def close(self) -> None:
        self.close_count += 1


class _FakeAsyncPlaywright:
    def __init__(self, browser) -> None:
        async def _connect(*args, **kwargs):
            return browser

        self.chromium = SimpleNamespace(connect=_connect, launch=_connect)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False


def _playwright_loader(module, urls, continue_on_failure=False):
    loader = module.SafePlaywrightURLLoader(
        web_paths=urls,
        verify_ssl=False,
        continue_on_failure=continue_on_failure,
        playwright_ws_url="ws://playwright.internal/ws",
        playwright_timeout=1000,
    )
    loader.evaluator = _FakeEvaluator()
    return loader


def test_playwright_closes_page_and_browser_on_goto_timeout(retrieval_web_utils_module) -> None:
    """Regression for issue #25880: a page timeout leaked the remote session.

    The timeout is raised synchronously by the stub, so nothing here waits.
    """

    def goto(url):
        raise playwright.sync_api.TimeoutError(f"Timeout 1000ms exceeded navigating to {url}")

    browser = _FakeSyncBrowser(goto)
    loader = _playwright_loader(retrieval_web_utils_module, ["https://example.com/a"])

    with patch.object(
        playwright.sync_api, "sync_playwright", lambda: _FakeSyncPlaywright(browser)
    ):
        with pytest.raises(playwright.sync_api.TimeoutError):
            list(loader.lazy_load())

    assert [page.close_count for page in browser.pages] == [1]
    assert browser.close_count == 1


def test_playwright_closes_every_page_on_success(retrieval_web_utils_module) -> None:
    """Each URL's page is released as soon as that URL is done."""
    browser = _FakeSyncBrowser(lambda url: SimpleNamespace(url=url, status=200))
    loader = _playwright_loader(
        retrieval_web_utils_module, ["https://example.com/a", "https://example.com/b"]
    )

    with patch.object(
        playwright.sync_api, "sync_playwright", lambda: _FakeSyncPlaywright(browser)
    ):
        docs = list(loader.lazy_load())

    assert [doc.metadata["source"] for doc in docs] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert [page.close_count for page in browser.pages] == [1, 1]
    assert browser.close_count == 1


@pytest.mark.asyncio
async def test_playwright_closes_browser_when_search_is_abandoned(
    retrieval_web_utils_module,
) -> None:
    """Regression for issue #25880: abandoning the loader mid-batch leaked too.

    A search cancelled after the first result never reached the post-loop close.
    """
    browser = _FakeAsyncBrowser(lambda url: SimpleNamespace(url=url, status=200))
    loader = _playwright_loader(
        retrieval_web_utils_module, ["https://example.com/a", "https://example.com/b"]
    )

    documents = loader.alazy_load()
    with patch.object(
        playwright.async_api, "async_playwright", lambda: _FakeAsyncPlaywright(browser)
    ):
        first = await asyncio.wait_for(documents.__anext__(), timeout=30)
        await asyncio.wait_for(documents.aclose(), timeout=30)

    assert first.metadata["source"] == "https://example.com/a"
    assert [page.close_count for page in browser.pages] == [1]
    assert browser.close_count == 1


@pytest.mark.asyncio
async def test_playwright_async_closes_browser_on_failure(retrieval_web_utils_module) -> None:
    """Regression for issue #25880 on the async path used by web search."""

    def goto(url):
        raise playwright.async_api.TimeoutError(f"Timeout 1000ms exceeded navigating to {url}")

    browser = _FakeAsyncBrowser(goto)
    loader = _playwright_loader(retrieval_web_utils_module, ["https://example.com/a"])

    with patch.object(
        playwright.async_api, "async_playwright", lambda: _FakeAsyncPlaywright(browser)
    ):
        with pytest.raises(playwright.async_api.TimeoutError):
            await asyncio.wait_for(
                _drain(loader.alazy_load()),
                timeout=30,
            )

    assert [page.close_count for page in browser.pages] == [1]
    assert browser.close_count == 1


async def _drain(async_iterator) -> list:
    return [item async for item in async_iterator]


@pytest.mark.asyncio
async def test_playwright_continue_on_failure_still_closes_everything(
    retrieval_web_utils_module,
) -> None:
    """A tolerated failure closes its page and keeps going."""

    def goto(url):
        if url.endswith("/a"):
            raise playwright.async_api.TimeoutError("Timeout 1000ms exceeded")
        return SimpleNamespace(url=url, status=200)

    browser = _FakeAsyncBrowser(goto)
    loader = _playwright_loader(
        retrieval_web_utils_module,
        ["https://example.com/a", "https://example.com/b"],
        continue_on_failure=True,
    )

    with patch.object(
        playwright.async_api, "async_playwright", lambda: _FakeAsyncPlaywright(browser)
    ):
        docs = await asyncio.wait_for(_drain(loader.alazy_load()), timeout=30)

    assert [doc.metadata["source"] for doc in docs] == ["https://example.com/b"]
    assert [page.close_count for page in browser.pages] == [1, 1]
    assert browser.close_count == 1
