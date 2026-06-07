"""Dependency contract: playwright (import name ``playwright``).

Open WebUI's ``playwright`` web-loader engine (``retrieval/web/utils.py``)
drives a real browser to render JS-heavy pages for RAG. Both the sync and
async code paths are present and use a specific call chain:

  sync (``lazy_load``)::

      from playwright.sync_api import sync_playwright
      with sync_playwright() as p:
          browser = p.chromium.connect(ws_url)          # remote, OR
          browser = p.chromium.launch(headless=, proxy=) # local
          page = browser.new_page()
          page.route("**/*", handler)
          response = page.goto(url, timeout=...)
          ...
          browser.close()

  async (``alazy_load``) is the awaited mirror via
  ``from playwright.async_api import async_playwright``.

This module pins exactly that surface for BOTH apis and the error types,
WITHOUT launching or connecting to any browser (no driver subprocess, no
network): it asserts the entry points exist and are callable, the
``chromium`` BrowserType exposes ``connect`` / ``launch`` with the keyword
arguments the backend passes (``ws_endpoint``, ``headless``, ``proxy``,
``timeout``), the ``Browser`` / ``Page`` / ``Response`` classes expose the
methods/attributes the loader calls (``new_page``, ``route``, ``goto``,
``close``, ``status``), ``page.goto`` keeps its ``url`` + ``timeout``
contract, and the async variants of those methods are coroutine functions.
It also pins the ``Error`` / ``TimeoutError`` hierarchy the loader's broad
``except Exception`` ultimately relies on.

IMPORTANT: nothing here calls ``sync_playwright().__enter__()`` /
``.start()`` — doing so spawns the Node driver. We only introspect symbols,
signatures, and the context-manager protocol shape.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "playwright"
DIST_NAME = "playwright"

# Symbols the backend imports from playwright.sync_api / playwright.async_api.
SYNC_API_SYMBOLS = [
    "sync_playwright",
    "Playwright",
    "Browser",
    "BrowserType",
    "BrowserContext",
    "Page",
    "Response",
    "Route",
    "Error",
    "TimeoutError",
]
ASYNC_API_SYMBOLS = [
    "async_playwright",
    "Playwright",
    "Browser",
    "BrowserType",
    "Page",
    "Response",
    "Error",
    "TimeoutError",
]


def _sync_api(depcheck):
    depcheck.load(IMPORT_NAME)
    return depcheck.resolve(depcheck.load(IMPORT_NAME), "sync_api")


def _async_api(depcheck):
    return depcheck.resolve(depcheck.load(IMPORT_NAME), "async_api")


# --------------------------------------------------------------------------- #
# Import / version
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "playwright"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_sync_and_async_api_importable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert depcheck.has(mod, "sync_api")
    assert depcheck.has(mod, "async_api")


# --------------------------------------------------------------------------- #
# Symbol existence (API surface)
# --------------------------------------------------------------------------- #


def test_sync_api_symbols_exist(depcheck):
    depcheck.assert_symbols(_sync_api(depcheck), SYNC_API_SYMBOLS)


def test_async_api_symbols_exist(depcheck):
    depcheck.assert_symbols(_async_api(depcheck), ASYNC_API_SYMBOLS)


def test_sync_playwright_callable(depcheck):
    depcheck.assert_callable(_sync_api(depcheck), "sync_playwright")


def test_async_playwright_callable(depcheck):
    depcheck.assert_callable(_async_api(depcheck), "async_playwright")


# --------------------------------------------------------------------------- #
# Context-manager protocol (without entering it — no driver spawned)
# --------------------------------------------------------------------------- #


def test_sync_playwright_returns_context_manager(depcheck):
    """`with sync_playwright() as p:` requires the returned object to support
    the context-manager protocol. We check the protocol shape only and do NOT
    enter it (entering would start the Node driver subprocess)."""
    sync_api = _sync_api(depcheck)
    cm = sync_api.sync_playwright()
    try:
        assert hasattr(cm, "__enter__") and hasattr(cm, "__exit__"), (
            "sync_playwright() result is not a context manager"
        )
        # Playwright's manager also exposes start()/stop(); pin start exists.
        assert hasattr(cm, "start")
    finally:
        # Do not call __enter__/start; nothing to tear down.
        del cm


def test_async_playwright_returns_async_context_manager(depcheck):
    """`async with async_playwright() as p:` needs the async CM protocol."""
    async_api = _async_api(depcheck)
    cm = async_api.async_playwright()
    try:
        assert hasattr(cm, "__aenter__") and hasattr(cm, "__aexit__"), (
            "async_playwright() result is not an async context manager"
        )
    finally:
        del cm


# --------------------------------------------------------------------------- #
# BrowserType.connect / launch (the p.chromium entry points)
# --------------------------------------------------------------------------- #


def test_browsertype_has_connect_and_launch(depcheck):
    """p.chromium is a BrowserType; the loader calls .connect(ws_url) and
    .launch(headless=, proxy=). Pin both methods exist on BrowserType."""
    sync_api = _sync_api(depcheck)
    names = set(dir(sync_api.BrowserType))
    for meth in ("connect", "launch"):
        assert meth in names, f"BrowserType.{meth} missing"
        assert callable(getattr(sync_api.BrowserType, meth))


def test_launch_accepts_headless_and_proxy(depcheck):
    """The loader calls p.chromium.launch(headless=self.headless,
    proxy=self.proxy). Pin those keyword names remain on launch."""
    sync_api = _sync_api(depcheck)
    depcheck.assert_params(sync_api.BrowserType.launch, ["headless", "proxy"])


def test_connect_accepts_ws_endpoint_positionally(depcheck):
    """SafePlaywrightURLLoader calls `p.chromium.connect(self.playwright_ws_url)`
    with the ws endpoint as the FIRST POSITIONAL arg, so the contract is that
    connect accepts a positional endpoint. (Playwright renamed the parameter
    ws_endpoint -> endpoint; the backend's positional call is unaffected, so we
    pin positionality + accept either name rather than the exact old name.)"""
    sync_api = _sync_api(depcheck)
    sig = inspect.signature(sync_api.BrowserType.connect)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    assert params, "BrowserType.connect takes no params besides self"
    first = params[0]
    assert first.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ), f"connect's first param {first.name!r} is not positional ({first.kind})"
    assert first.name in {"ws_endpoint", "endpoint", "endpoint_url"}, (
        f"unexpected connect endpoint param name: {first.name!r}"
    )


# --------------------------------------------------------------------------- #
# Browser / Page / Response method surface (loader call sites)
# --------------------------------------------------------------------------- #


def test_browser_methods_exist(depcheck):
    """browser.new_page() and browser.close() are called by the loader."""
    sync_api = _sync_api(depcheck)
    names = set(dir(sync_api.Browser))
    for meth in ("new_page", "close"):
        assert meth in names, f"Browser.{meth} missing"
        assert callable(getattr(sync_api.Browser, meth))


def test_page_methods_exist(depcheck):
    """page.route(pattern, handler) and page.goto(url, timeout=) are called."""
    sync_api = _sync_api(depcheck)
    names = set(dir(sync_api.Page))
    for meth in ("route", "goto", "content"):
        assert meth in names, f"Page.{meth} missing"
        assert callable(getattr(sync_api.Page, meth))


def test_page_goto_signature(depcheck):
    """page.goto(url, timeout=self.playwright_timeout). Pin url (first
    positional) and timeout (kwarg the loader passes)."""
    sync_api = _sync_api(depcheck)
    sig = inspect.signature(sync_api.Page.goto)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    assert params and params[0].name == "url"
    depcheck.assert_params(sync_api.Page.goto, ["url", "timeout"])


def test_page_route_signature(depcheck):
    """page.route('**/*', handler). Pin the url (pattern) + handler params."""
    sync_api = _sync_api(depcheck)
    depcheck.assert_params(sync_api.Page.route, ["url", "handler"])


def test_response_has_status(depcheck):
    """The loader inspects the response object (page.goto returns a Response or
    None). Pin Response exposes status."""
    sync_api = _sync_api(depcheck)
    names = set(dir(sync_api.Response))
    assert "status" in names


# --------------------------------------------------------------------------- #
# Async variants are coroutine functions
# --------------------------------------------------------------------------- #


def test_async_methods_are_coroutines(depcheck):
    """alazy_load awaits p.chromium.launch/connect, browser.new_page,
    page.route, page.goto, browser.close. The async-api versions of these must
    be coroutine functions for `await` to be valid."""
    async_api = _async_api(depcheck)
    assert inspect.iscoroutinefunction(async_api.BrowserType.launch)
    assert inspect.iscoroutinefunction(async_api.BrowserType.connect)
    assert inspect.iscoroutinefunction(async_api.Browser.new_page)
    assert inspect.iscoroutinefunction(async_api.Page.goto)
    assert inspect.iscoroutinefunction(async_api.Browser.close)


def test_sync_methods_are_not_coroutines(depcheck):
    """The sync-api equivalents must NOT be coroutine functions (the lazy_load
    path calls them directly without await)."""
    sync_api = _sync_api(depcheck)
    assert not inspect.iscoroutinefunction(sync_api.Page.goto)
    assert not inspect.iscoroutinefunction(sync_api.BrowserType.launch)


# --------------------------------------------------------------------------- #
# Error hierarchy
# --------------------------------------------------------------------------- #


def test_error_hierarchy(depcheck):
    """playwright.sync_api.Error is the base playwright exception and
    TimeoutError (raised by goto timeouts) subclasses it; both subclass
    Exception so the loader's `except Exception` catches them."""
    sync_api = _sync_api(depcheck)
    assert issubclass(sync_api.Error, Exception)
    assert issubclass(sync_api.TimeoutError, sync_api.Error)


def test_async_error_hierarchy(depcheck):
    async_api = _async_api(depcheck)
    assert issubclass(async_api.Error, Exception)
    assert issubclass(async_api.TimeoutError, async_api.Error)
