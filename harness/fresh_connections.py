"""Every Playwright API request goes out on a fresh connection.

Playwright sends `page.request`, `context.request` and `route.fetch` from one process-wide pool
of kept connections, shared by every test. Open WebUI closes a connection after 5 s idle, and a
request sent on a kept connection just as the server closes it fails with "socket hang up"
before the server sees it. With `Connection: close` no connection stays in the pool.

The harness's own httpx clients race the same way: httpx keeps idle connections for 5 s, as
the server does. They expire idle connections after 2 s instead.
"""

from __future__ import annotations

import functools

import httpx
from playwright._impl._fetch import APIRequestContext

KEEPALIVE_EXPIRY = 2.0


def _expire_idle_early(init):
    @functools.wraps(init)
    def wrapped(self, *args, limits=None, **kwargs):
        if limits is None:
            limits = httpx.Limits(
                max_connections=100, max_keepalive_connections=20, keepalive_expiry=KEEPALIVE_EXPIRY
            )
        return init(self, *args, limits=limits, **kwargs)

    return wrapped


def install() -> None:
    """Send `Connection: close` with every Playwright API request and expire idle httpx connections early."""
    httpx.Client.__init__ = _expire_idle_early(httpx.Client.__init__)
    httpx.AsyncClient.__init__ = _expire_idle_early(httpx.AsyncClient.__init__)
    inner_fetch = APIRequestContext._inner_fetch

    async def fetch_on_fresh_connection(
        self, request, url, method=None, headers=None, *args, **kwargs
    ):
        sent = headers or (request.headers if request else None) or {}
        sent = {name: value for name, value in sent.items() if name.lower() != "connection"}
        sent["Connection"] = "close"
        return await inner_fetch(self, request, url, method, sent, *args, **kwargs)

    APIRequestContext._inner_fetch = fetch_on_fresh_connection
