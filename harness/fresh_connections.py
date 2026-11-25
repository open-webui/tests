"""Every Playwright API request goes out on a fresh connection.

Playwright sends `page.request`, `context.request` and `route.fetch` from one process-wide pool
of kept connections, shared by every test. Open WebUI closes a connection after 5 s idle, and a
request sent on a kept connection just as the server closes it fails with "socket hang up"
before the server sees it. With `Connection: close` no connection stays in the pool.
"""

from __future__ import annotations

from playwright._impl._fetch import APIRequestContext


def install() -> None:
    """Send `Connection: close` with every Playwright API request."""
    inner_fetch = APIRequestContext._inner_fetch

    async def fetch_on_fresh_connection(
        self, request, url, method=None, headers=None, *args, **kwargs
    ):
        sent = headers or (request.headers if request else None) or {}
        sent = {name: value for name, value in sent.items() if name.lower() != "connection"}
        sent["Connection"] = "close"
        return await inner_fetch(self, request, url, method, sent, *args, **kwargs)

    APIRequestContext._inner_fetch = fetch_on_fresh_connection
