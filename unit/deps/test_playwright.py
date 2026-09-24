"""Dependency contract: playwright (import name ``playwright``).

Open WebUI's ``playwright`` web-loader engine (``SafePlaywrightURLLoader`` in
``retrieval/web/utils.py``) renders JS-heavy pages for RAG, synchronously in ``lazy_load`` and
asynchronously in ``alazy_load``, with the same call chain on both APIs::

    with sync_playwright() as p:
        browser = p.chromium.connect(ws_url)             # remote, or
        browser = p.chromium.launch(headless=, proxy=)  # local
        with browser, browser.new_page(service_workers="block") as page:
            page.route("**/*", handler)                  # route.abort() / route.fulfill(...)
            page.route_web_socket("**/*", handler)
            page.goto(url, timeout=...)
            page.locator(selector).all()                 # .is_visible() / .evaluate(js)
            page.content()
            page.unroute_all(behavior="ignoreErrors")

Each call is pinned the way the loader makes it: the positional arguments it passes and the
keywords it names, bound against the method's signature. A parameter the loader passes
positionally is never pinned by name, which is what broke this file when Playwright renamed
``connect``'s ``ws_endpoint`` to ``endpoint``. Nothing launches a browser or starts the driver.

Discriminates: a playwright whose ``connect`` takes its endpoint by keyword only, whose
``new_page`` drops ``service_workers`` or whose ``Page`` stops being a context manager fails
here (each patched into the installed package in process).
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "playwright"
DIST_NAME = "playwright"
APIS = ["sync_api", "async_api"]


def _ignore(route) -> None:
    """Stands in for the loader's route handlers."""


# (class, method, positional arguments, keywords) exactly as the loader calls them.
LOADER_CALLS = [
    ("BrowserType", "connect", ("ws://127.0.0.1:3000",), {}),
    ("BrowserType", "launch", (), {"headless": True, "proxy": None}),
    ("Browser", "new_page", (), {"service_workers": "block"}),
    ("Page", "route", ("**/*", _ignore), {}),
    ("Page", "route_web_socket", ("**/*", _ignore), {}),
    ("Page", "goto", ("https://example.com",), {"timeout": 1000}),
    ("Page", "locator", ("nav",), {}),
    ("Page", "content", (), {}),
    ("Page", "unroute_all", (), {"behavior": "ignoreErrors"}),
    ("Locator", "all", (), {}),
    ("Locator", "is_visible", (), {}),
    ("Locator", "evaluate", ("element => element.remove()",), {}),
    ("Route", "abort", (), {}),
    ("Route", "fulfill", (), {"status": 200, "headers": {}, "body": b""}),
    ("Request", "all_headers", (), {}),
]
# Awaited in alazy_load, called directly in lazy_load.
AWAITED = {"connect", "launch", "new_page", "goto", "content", "all", "is_visible", "evaluate"}
# What the route handlers read off the intercepted request.
REQUEST_PROPERTIES = ["url", "method", "resource_type", "post_data_buffer"]


def _api(depcheck, name: str):
    return depcheck.resolve(depcheck.load(IMPORT_NAME), name)


def test_import(depcheck):
    assert depcheck.load(IMPORT_NAME).__name__ == "playwright"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize(
    ("owner", "method", "args", "kwargs"),
    LOADER_CALLS,
    ids=[f"{owner}.{method}" for owner, method, _, _ in LOADER_CALLS],
)
def test_the_loader_calls_still_bind(depcheck, api, owner, method, args, kwargs):
    target = getattr(getattr(_api(depcheck, api), owner), method, None)
    assert callable(target), f"{api}.{owner}.{method} is gone"
    try:
        inspect.signature(target).bind(None, *args, **kwargs)
    except TypeError as error:
        pytest.fail(f"the loader's {owner}.{method}(...) call no longer binds on {api}: {error}")
    if method in AWAITED:
        assert inspect.iscoroutinefunction(target) == (api == "async_api"), (
            f"{api}.{owner}.{method} is awaited in alazy_load and called directly in lazy_load"
        )


@pytest.mark.parametrize("api", APIS)
def test_browser_and_page_open_as_context_managers(depcheck, api):
    """`with browser, browser.new_page(...) as page` and its `async with` twin."""
    module = _api(depcheck, api)
    protocol = ("__enter__", "__exit__") if api == "sync_api" else ("__aenter__", "__aexit__")
    for owner in ("Browser", "Page"):
        for dunder in protocol:
            assert callable(getattr(getattr(module, owner), dunder, None)), f"{owner}.{dunder}"


@pytest.mark.parametrize("api", APIS)
def test_the_entry_points_return_context_managers_without_starting_the_driver(depcheck, api):
    module = _api(depcheck, api)
    manager = module.sync_playwright() if api == "sync_api" else module.async_playwright()
    protocol = ("__enter__", "__exit__") if api == "sync_api" else ("__aenter__", "__aexit__")
    assert all(hasattr(manager, dunder) for dunder in protocol)


@pytest.mark.parametrize("api", APIS)
def test_the_route_handlers_find_what_they_read(depcheck, api):
    module = _api(depcheck, api)
    assert isinstance(getattr(module.Route, "request", None), property), "Route.request"
    for name in REQUEST_PROPERTIES:
        assert isinstance(getattr(module.Request, name, None), property), f"Request.{name}"
    assert callable(getattr(module.WebSocketRoute, "close", None)), "WebSocketRoute.close"
