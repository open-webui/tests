"""Playwright's browser, page and route classes, specced, to drive the web loader without a browser.

`specced_browser` patches `sync_playwright` or `async_playwright` so the loader's own
`lazy_load` or `alazy_load` runs against `create_autospec` stand-ins of the real classes: the
loader may call any method Playwright has, and a call Playwright would reject fails here too.
`goto` sends the page, then any sub-resources the test names, through the handler the loader
registered with `page.route`, the way the browser does. Like the browser, it fails when the
document's own route is aborted, and it raises Playwright's `TimeoutError` for the URLs the test
lists as failing.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator
from unittest.mock import MagicMock, create_autospec, patch

import playwright.async_api
import playwright.sync_api

PAGE_HTML = "<html><body><p>page text</p></body></html>"


@dataclass
class Browsing:
    browser: MagicMock
    pages: list[MagicMock] = field(default_factory=list)
    routes: list[MagicMock] = field(default_factory=list)

    def routes_to(self, url: str) -> list[MagicMock]:
        return [route for route in self.routes if route.request.url == url]


def _enters_as(stand_in: MagicMock, entered, asynchronous: bool) -> None:
    # a truthy __exit__ result would swallow the exception the loader is leaving with
    prefix = "__a" if asynchronous else "__"
    getattr(stand_in, f"{prefix}enter__").return_value = entered
    getattr(stand_in, f"{prefix}exit__").return_value = False


def _route(api, url: str, resource_type: str) -> MagicMock:
    request = create_autospec(api.Request, instance=True)
    request.url, request.resource_type, request.method = url, resource_type, "GET"
    request.post_data_buffer = None
    request.all_headers.return_value = {"user-agent": "specced-browser"}
    route = create_autospec(api.Route, instance=True)
    route.request = request
    return route


def _handler(api, page: MagicMock, method: str):
    """The handler the loader registered with `page.<method>`, however it passed it."""
    call = getattr(page, method).call_args
    assert call is not None, f"the loader never called page.{method}"
    arguments = inspect.signature(getattr(api.Page, method)).bind(page, *call.args, **call.kwargs)
    return arguments.arguments["handler"]


@contextmanager
def specced_browser(
    asynchronous: bool = False,
    subresources: tuple[tuple[str, str], ...] = (),
    failing: tuple[str, ...] = (),
) -> Iterator[Browsing]:
    """`subresources` are `(url, resource_type)` pairs every page requests after itself."""
    api = playwright.async_api if asynchronous else playwright.sync_api
    browsing = Browsing(browser=create_autospec(api.Browser, instance=True))
    _enters_as(browsing.browser, browsing.browser, asynchronous)

    def routes_for(page, url):
        if url in failing:
            raise api.TimeoutError(f"Timeout exceeded navigating to {url}")
        routes = [_route(api, url, "document")]
        routes += [_route(api, sub_url, resource_type) for sub_url, resource_type in subresources]
        browsing.routes.extend(routes)
        return _handler(api, page, "route"), routes

    def navigated(document_route, response):
        if document_route.abort.called:
            raise api.Error(f"net::ERR_FAILED at {document_route.request.url}")
        return response

    def open_page(**options):
        page = create_autospec(api.Page, instance=True)
        _enters_as(page, page, asynchronous)
        page.content.return_value = PAGE_HTML
        response = create_autospec(api.Response, instance=True)
        if asynchronous:

            async def goto(url, **options):
                handler, routes = routes_for(page, url)
                for route in routes:
                    await handler(route)
                return navigated(routes[0], response)

        else:

            def goto(url, **options):
                handler, routes = routes_for(page, url)
                for route in routes:
                    handler(route)
                return navigated(routes[0], response)

        page.goto.side_effect = goto
        browsing.pages.append(page)
        return page

    browsing.browser.new_page.side_effect = open_page
    driver = create_autospec(api.Playwright, instance=True)
    driver.chromium = create_autospec(api.BrowserType, instance=True)
    driver.chromium.launch.return_value = browsing.browser
    driver.chromium.connect.return_value = browsing.browser
    manager = MagicMock()
    _enters_as(manager, driver, asynchronous)
    entrypoint = "async_playwright" if asynchronous else "sync_playwright"
    with patch.object(api, entrypoint, return_value=manager):
        yield browsing


async def websocket_opened(page: MagicMock, asynchronous: bool = False) -> MagicMock:
    """A websocket the page opens, once the loader's `route_web_socket` handler has had it."""
    api = playwright.async_api if asynchronous else playwright.sync_api
    websocket = create_autospec(api.WebSocketRoute, instance=True)
    handled = _handler(api, page, "route_web_socket")(websocket)
    if inspect.isawaitable(handled):
        await handled
    return websocket


def released(stand_in: MagicMock) -> bool:
    """Closed, or left through its context manager, which is how Playwright closes it."""
    exits = [
        stand_in.close,
        getattr(stand_in, "__exit__", None),
        getattr(stand_in, "__aexit__", None),
    ]
    return any(method is not None and method.called for method in exits)
