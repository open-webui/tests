"""Regression: web loader settings saved by the admin, and web search results that fail to embed.

- Admin loader settings ignored (PR #26749, commit f7e7f3210, issue #26747; commit 304cbe456 for
  the search call site): `get_web_loader` dispatched on the `WEB_LOADER_ENGINE` read from the
  environment at import, so the engine and endpoints saved in Admin Settings > Web Search were
  never used and pages were always fetched directly. Here each engine points at a local service,
  which must be the one that gets the page.
- Silent embedding failures (PR #26883, commit 6c7478c1c): a failed vector write after a web
  search was logged at debug level and answered as a healthy collection, so the chat said
  "searched N sites" and then found no sources. It is a 500 naming the embedding settings now.
- Batch parser locked to the first URL (PR #27367, commit acf586c00): the parser picked for the
  first page of a search was reused for every later page, so an HTML page after a feed lost its
  entities and a feed after an HTML page lost its CDATA text.

Twin of unit/retrieval/test_web_loader_configuration.py, which keeps the urllib3-future socket
options (no route sets them) and the Playwright session cleanup (no browser in the unit lane).

Discriminates: passes on dev bbfa876af; dispatching on the env `WEB_LOADER_ENGINE` fails the three
engine cases, dropping `loader_config` at the search call site fails the search engine case,
pacing the search at a fixed 10 pages a second fails the pacing case, logging the vector write
failure fails the embedding case and keeping the first page's parser fails both batch orders.
"""

from __future__ import annotations

import time

import pytest

from harness.listener import json_answer, text_answer
from harness.web_retrieval import (
    LOCAL_WEB_FETCH,
    RETRIEVAL_CONFIG,
    save_web_settings,
    serve_search_results,
    web_settings_restored,
)

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

EMBEDDING_CONFIG = ("/api/v1/retrieval/embedding", "/api/v1/retrieval/embedding/update")
WEB_SEARCH = "/api/v1/retrieval/process/web/search"
LOADER_KEY = "loader-key"

# engine: (settings, with {base} and {port} filled from the listener; the path it calls)
ENGINES = {
    "external": (
        {"EXTERNAL_WEB_LOADER_URL": "{base}/external", "EXTERNAL_WEB_LOADER_API_KEY": LOADER_KEY},
        "/external",
    ),
    "firecrawl": (
        {"FIRECRAWL_API_BASE_URL": "{base}", "FIRECRAWL_API_KEY": LOADER_KEY},
        "/v2/scrape",
    ),
    "playwright": ({"PLAYWRIGHT_WS_URL": "ws://127.0.0.1:{port}/playwright"}, "/playwright"),
}


@pytest.fixture(scope="module")
def fetching_instance(instance_with):
    return instance_with(LOCAL_WEB_FETCH)


@pytest.fixture
def fetching_admin(fetching_instance):
    with fetching_instance.client() as client, web_settings_restored(client):
        yield client


def engine_settings(engine, listener):
    settings, _ = ENGINES[engine]
    filled = {
        name: value.format(base=listener.base_url, port=listener.port)
        for name, value in settings.items()
    }
    return {"WEB_LOADER_ENGINE": engine, **filled}


def serve_loader_backends(listener):
    listener.route("POST", "/external", json_answer([{"page_content": "from the external loader"}]))
    listener.route("POST", "/v2/scrape", json_answer({"data": {"markdown": "from firecrawl"}}))


@pytest.mark.parametrize("engine", ENGINES)
def test_page_is_loaded_by_the_admin_selected_engine(fetching_admin, listener, engine):
    page = f"{listener.base_url}/page"
    listener.route("GET", "/page", text_answer("<p>fetched directly</p>"))
    serve_loader_backends(listener)
    save_web_settings(fetching_admin, **engine_settings(engine, listener))

    fetching_admin.post("/api/v1/retrieval/process/web?process=false", json={"url": page})

    _, called_path = ENGINES[engine]
    calls = listener.requests_to(called_path)
    assert calls, f"the {engine} loader saved by the admin was never called"
    if engine != "playwright":  # a browser endpoint takes no key and no page in the handshake
        assert calls[0].headers["Authorization"] == f"Bearer {LOADER_KEY}"
        assert page in calls[0].body.decode()


def test_search_results_are_loaded_by_the_admin_selected_engine(fetching_admin, listener):
    pages = [f"{listener.base_url}/first", f"{listener.base_url}/second"]
    serve_loader_backends(listener)
    save_web_settings(
        fetching_admin,
        **serve_search_results(listener, pages),
        **engine_settings("external", listener),
        BYPASS_WEB_SEARCH_WEB_LOADER=False,
        BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=True,
    )

    response = fetching_admin.post(WEB_SEARCH, json={"queries": ["engine"]})

    assert response.status_code == 200, response.text
    calls = listener.requests_to("/external")
    assert [call.json()["urls"] for call in calls] == [pages]
    assert calls[0].headers["Authorization"] == f"Bearer {LOADER_KEY}"
    assert listener.requests_to("/first") == [], "the page was fetched directly"


def test_search_results_are_paced_at_the_admin_saved_rate(fetching_admin, listener):
    pages = [f"{listener.base_url}/first", f"{listener.base_url}/second"]
    listener.route(
        "POST", "/browse", lambda request: json_answer({"content": "text", **request.json()})
    )
    save_web_settings(
        fetching_admin,
        **serve_search_results(listener, pages),
        WEB_LOADER_ENGINE="microsoft_web_iq",
        MICROSOFT_WEB_IQ_API_BASE_URL=listener.base_url,
        MICROSOFT_WEB_IQ_API_KEY="web-iq-key",
        WEB_LOADER_CONCURRENT_REQUESTS=2,
        BYPASS_WEB_SEARCH_WEB_LOADER=False,
        BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=True,
    )

    started = time.monotonic()
    response = fetching_admin.post(WEB_SEARCH, json={"queries": ["pace"]})
    elapsed = time.monotonic() - started

    assert response.status_code == 200, response.text
    assert elapsed >= 0.45, f"two pages at 2 per second took {elapsed:.2f}s"  # one interval


def save_embedding_url(client, url):
    current = client.get(EMBEDDING_CONFIG[0]).json()
    saved = client.post(
        EMBEDDING_CONFIG[1], json={**current, "openai_config": {"url": url, "key": "sk-embed"}}
    )
    assert saved.status_code == 200, saved.text


@pytest.fixture
def snippet_search(admin, preserve, listener):
    """Web search on the shared instance, embedding the result snippets without loading pages."""
    preserve(RETRIEVAL_CONFIG, EMBEDDING_CONFIG)
    pages = [f"{listener.base_url}/first", f"{listener.base_url}/second"]
    with admin.client() as client:
        save_web_settings(
            client,
            **serve_search_results(listener, pages),
            BYPASS_WEB_SEARCH_WEB_LOADER=True,
            BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=False,
        )
        yield client, pages


def test_web_search_reports_a_failed_embedding(snippet_search, listener):
    client, _ = snippet_search
    save_embedding_url(client, f"{listener.base_url}/no-embeddings-here")

    response = client.post(WEB_SEARCH, json={"queries": ["embed"]})

    assert response.status_code == 500, response.text
    detail = response.json()["detail"]
    assert "embed" in detail.lower() and "Documents" in detail


def test_web_search_returns_its_collection_when_embedding_works(snippet_search, upstream):
    client, pages = snippet_search
    save_embedding_url(client, upstream.base_url)

    response = client.post(WEB_SEARCH, json={"queries": ["embed"]})

    assert response.status_code == 200, response.text
    assert len(response.json()["collection_names"]) == 1
    assert response.json()["filenames"] == pages
    assert upstream.requests_to("/embeddings")


FEED = (
    "<?xml version='1.0'?><feed><entry><summary><![CDATA[feed payload]]></summary></entry></feed>"
)
PAGE = "<html><head><title>page</title></head><body><p>caf&eacute; au lait</p></body></html>"


@pytest.mark.parametrize("first", ["feed.xml", "page.html"])
def test_each_search_result_is_parsed_as_its_own_type(fetching_admin, listener, first):
    listener.route("GET", "/feed.xml", text_answer(FEED, content_type="application/xml"))
    listener.route("GET", "/page.html", text_answer(PAGE))
    second = "page.html" if first == "feed.xml" else "feed.xml"
    pages = [f"{listener.base_url}/{name}" for name in (first, second)]
    save_web_settings(
        fetching_admin,
        **serve_search_results(listener, pages),
        WEB_LOADER_ENGINE="safe_web",
        BYPASS_WEB_SEARCH_WEB_LOADER=False,
        BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=True,
    )

    response = fetching_admin.post(WEB_SEARCH, json={"queries": ["mixed"]})

    assert response.status_code == 200, response.text
    text_by_page = {
        doc["metadata"]["source"].rsplit("/", 1)[1]: doc["content"]
        for doc in response.json()["docs"]
    }
    assert "feed payload" in text_by_page["feed.xml"], "the feed was not parsed as XML"
    assert "café au lait" in text_by_page["page.html"], "the page was not parsed as HTML"
