"""Every browser page of the suite opens Open WebUI over plain HTTP, outside a secure context.

The instances listen on loopback, which the browser treats as a secure context, so it offers APIs
that a user who opens Open WebUI over plain HTTP by address (a LAN address, Docker without TLS)
never gets: `crypto.randomUUID`, `navigator.clipboard`, `crypto.subtle` and more. Pages that
relied on them broke for those users (open-webui/open-webui#33847, #32238, #33919, #33927).
Every Chromium the suite launches resolves `LAN_HOST` to loopback, and every browser context and
page navigation that names a loopback address goes to `LAN_HOST` instead, as a phone on the LAN
does.

A test of a feature that the browser itself offers only in a secure context (the microphone, the
location, encrypted chat, which checks for it) carries `pytest.mark.secure_context` and keeps the
loopback address.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import pytest
from playwright.sync_api import Browser, BrowserContext, BrowserType, Page

from harness.instance import LaunchedInstance

# a host name that is neither localhost nor a loopback address
LAN_HOST = "owui-lan.test"
# the instances listen here; a test that names `localhost` means it
LOOPBACK_HOST = "127.0.0.1"
RESOLVE_LAN_HOST = f"--host-resolver-rules=MAP {LAN_HOST} {LOOPBACK_HOST}"
ADMIN_CONFIG = "/api/v1/auths/admin/config"

_secure_context_test = False
_navigate = Page.goto


def follow(item: pytest.Item) -> None:
    """Keep the loopback address for the browsers of a test marked `secure_context`."""
    global _secure_context_test
    _secure_context_test = item.get_closest_marker("secure_context") is not None


def is_secure_context_test() -> bool:
    return _secure_context_test


def lan_url(url: str) -> str:
    """`url` with the loopback host replaced by `LAN_HOST`; any other URL as it is."""
    parts = urlsplit(url)
    if parts.hostname != LOOPBACK_HOST:
        return url
    netloc = f"{LAN_HOST}:{parts.port}" if parts.port else LAN_HOST
    return urlunsplit(parts._replace(netloc=netloc))


def serve_at_lan_host(instance: LaunchedInstance) -> None:
    """Set the instance's WebUI URL to its `LAN_HOST` address, as an admin of a LAN install does.

    Open WebUI sends the browser back to that URL after an SSO sign-in, and the sign-in cookies
    belong to the address the browser came from.
    """
    with instance.client() as admin:
        config = admin.get(ADMIN_CONFIG)
        config.raise_for_status()
        saved = admin.post(
            ADMIN_CONFIG, json={**config.json(), "WEBUI_URL": lan_url(instance.base_url)}
        )
    assert saved.status_code == 200, f"setting the WebUI URL failed: {saved.text}"


def loopback_url(page_url: str, path: str) -> str:
    """`path` on the loopback address of the server `page_url` is on, for a request from Python."""
    server = urlsplit(page_url)
    return f"{server.scheme}://{LOOPBACK_HOST}:{server.port}{path}"


def loopback_page(page: Page) -> Page:
    """A new page beside `page` on its server's loopback address, which is a secure context.

    Test code reaches the clipboard through it: a plain HTTP page has no `navigator.clipboard`,
    and the pages of one browser context share one clipboard.
    """
    helper = page.context.new_page()
    _navigate(helper, loopback_url(page.url, "/health"))
    return helper


def install() -> None:
    """Send every browser context and page navigation of the suite to `LAN_HOST`."""
    launch = BrowserType.launch
    open_context = Browser.new_context

    def launch_resolving_lan_host(self: BrowserType, *args, **kwargs) -> Browser:
        kwargs["args"] = [*(kwargs.get("args") or []), RESOLVE_LAN_HOST]
        return launch(self, *args, **kwargs)

    def new_context(self: Browser, *args, **kwargs) -> BrowserContext:
        if kwargs.get("base_url") and not _secure_context_test:
            kwargs["base_url"] = lan_url(kwargs["base_url"])
        return open_context(self, *args, **kwargs)

    def goto(self: Page, url: str, **kwargs):
        return _navigate(self, url if _secure_context_test else lan_url(url), **kwargs)

    BrowserType.launch = launch_resolving_lan_host
    Browser.new_context = new_context
    Page.goto = goto
