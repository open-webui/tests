"""Journey: the web search settings that shape a search of several queries and the pages it loads.

The admin's Concurrent Requests caps how many queries of one search go to the engine at once.
Bypass Web Loader hands back the engine's snippets in place of the pages, which are then never
fetched. Trust Proxy Environment decides whether the pages of a search are fetched through the
proxy the environment names: with it on, a page whose host only the proxy knows is read; with it
off, that page cannot be reached. The engine is the external one on a local listener, driven
through the search route the chat uses for a forced web search; the native `search_web` and
`fetch_url` tools are covered in e2e/retrieval.

Discriminates: passes on dev ebc6add67; in a backend copy whose search ignores the concurrency
setting the one-at-a-time case fails, with the loader bypass inverted both bypass cases fail, and
with the loader's `trust_env` always off the trusted case fails.
"""

from __future__ import annotations

import threading
import time

import pytest

from harness.http_proxy import serving_proxy
from harness.listener import json_answer, text_answer
from harness.web_retrieval import (
    LOCAL_WEB_FETCH,
    save_web_settings,
    serve_search_results,
    web_settings_restored,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

WEB_SEARCH = "/api/v1/retrieval/process/web/search"
PAGE_TEXT = "Kittiwakes nest on the north cliff"
BEHIND_PROXY = "cliffs.behind-proxy.invalid"


@pytest.fixture(scope="module")
def proxy():
    with serving_proxy() as serving:
        yield serving


@pytest.fixture(scope="module")
def searching(instance_with, proxy):
    return instance_with(
        {
            **LOCAL_WEB_FETCH,
            "HTTP_PROXY": f"http://127.0.0.1:{proxy.port}",
            "NO_PROXY": "127.0.0.1,localhost",
        }
    )


@pytest.fixture
def searching_admin(searching):
    with searching.client() as client, web_settings_restored(client):
        yield client


def snippets_only(**changes) -> dict:
    return {"BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL": True, **changes}


def search(client, *queries: str) -> dict:
    response = client.post(WEB_SEARCH, json={"queries": list(queries)})
    assert response.status_code == 200, response.text
    return response.json()


class InFlight:
    """An engine answer that takes a while and counts how many searches overlap."""

    def __init__(self, links: list[str]) -> None:
        self.links = links
        self.current = 0
        self.most = 0
        self.lock = threading.Lock()

    def __call__(self, request):
        with self.lock:
            self.current += 1
            self.most = max(self.most, self.current)
        time.sleep(0.4)
        with self.lock:
            self.current -= 1
        hits = [{"link": link, "title": link, "snippet": "a cliff"} for link in self.links]
        return json_answer(hits)


@pytest.mark.parametrize("limit, overlapping", [(1, 1), (3, 3)])
def test_concurrent_requests_cap_the_queries_sent_at_once(
    limit, overlapping, searching_admin, listener
):
    link = f"{listener.base_url}/cliffs"
    engine = InFlight([link])
    save_web_settings(
        searching_admin,
        **serve_search_results(listener, [link]),
        **snippets_only(BYPASS_WEB_SEARCH_WEB_LOADER=True, WEB_SEARCH_CONCURRENT_REQUESTS=limit),
    )
    listener.route("POST", "/search", engine)

    search(searching_admin, "puffins", "gannets", "kittiwakes")

    assert len(listener.requests_to("/search")) == 3
    assert engine.most == overlapping


@pytest.mark.parametrize("bypass", [True, False], ids=["snippets", "pages"])
def test_bypass_web_loader_hands_back_the_snippets_without_fetching_pages(
    bypass, searching_admin, listener
):
    link = f"{listener.base_url}/cliffs"
    listener.route("GET", "/cliffs", text_answer(f"<p>{PAGE_TEXT}</p>"))
    save_web_settings(
        searching_admin,
        **serve_search_results(listener, [link]),
        **snippets_only(BYPASS_WEB_SEARCH_WEB_LOADER=bypass),
    )

    [doc] = search(searching_admin, "kittiwakes")["docs"]

    if bypass:
        assert doc["content"] == f"snippet of {link}"
        assert listener.requests_to("/cliffs") == [], "the page was fetched though bypassed"
    else:
        assert PAGE_TEXT in doc["content"]


@pytest.mark.parametrize("trusted", [True, False], ids=["trusted", "untrusted"])
def test_trust_proxy_environment_decides_whether_pages_go_through_the_proxy(
    trusted, searching_admin, listener, proxy
):
    proxy.known[BEHIND_PROXY] = listener.port
    listener.route("GET", "/cliffs", text_answer(f"<p>{PAGE_TEXT}</p>"))
    link = f"http://{BEHIND_PROXY}/cliffs"
    save_web_settings(
        searching_admin,
        **serve_search_results(listener, [link]),
        **snippets_only(BYPASS_WEB_SEARCH_WEB_LOADER=False, WEB_SEARCH_TRUST_ENV=trusted),
    )
    asked_before = len(proxy.requests_for(BEHIND_PROXY))

    docs = search(searching_admin, "kittiwakes").get("docs") or []

    read = any(PAGE_TEXT in doc["content"] for doc in docs)
    proxied = proxy.requests_for(BEHIND_PROXY)[asked_before:]
    assert read is trusted, f"the page was {'not ' if trusted else ''}read: {docs}"
    assert bool(proxied) is trusted
