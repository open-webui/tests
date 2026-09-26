"""Journey: the model reads a web page with `fetch_url`, within the admin's limits.

With web search switched on and asked for in the chat, the model is offered `search_web` and
`fetch_url`. `fetch_url` hands back the page's text, cut at the admin's maximum content length
with a note that it was cut, and an error for an address it may not fetch. An account without
the web search permission is offered neither tool. Pages come from a local service, so the
instance runs with local fetching allowed.

Discriminates: in a backend copy, `fetch_url` skipping the maximum length turned the truncation
test red, and the web tools offered without the permission check turned the permission test
red.
"""

from __future__ import annotations

import json

import pytest

from harness.actors import admin_of, create_user
from harness.listener import text_answer
from harness.tool_calls import offered_tools, run_tool
from harness.web_retrieval import (
    LOCAL_WEB_FETCH,
    RETRIEVAL_CONFIG,
    save_web_settings,
    serve_search_results,
)

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

WEB_TOOLS = {"search_web", "fetch_url"}
WITH_WEB_SEARCH = {"features": {"web_search": True}}
DEFAULT_PERMISSIONS = "/api/v1/users/default/permissions"
PAGE = "<html><body><h1>Heron watch</h1><p>{body}</p></body></html>"


@pytest.fixture
def fetching(instance_with, preserve, listener):
    """The local-fetch instance with web search on, answered by the listener."""
    launched = instance_with(LOCAL_WEB_FETCH)
    preserve(RETRIEVAL_CONFIG, "permissions", on=launched)
    with admin_of(launched).client() as client:
        save_web_settings(client, **serve_search_results(listener, [f"{listener.base_url}/page"]))
    return launched


def fetch(launched, actor, url: str) -> str:
    with actor.client() as client:
        return run_tool(client, launched.upstream, "fetch_url", {"url": url}, **WITH_WEB_SEARCH)


def test_a_page_is_fetched_as_text(fetching, listener):
    listener.route("GET", "/page", text_answer(PAGE.format(body="Two herons at the jetty.")))

    content = fetch(fetching, create_user(fetching), f"{listener.base_url}/page")

    assert "Heron watch" in content and "Two herons at the jetty." in content, content
    assert "<p>" not in content


def test_a_long_page_is_cut_at_the_maximum_length(fetching, listener):
    listener.route("GET", "/page", text_answer(PAGE.format(body="heron " * 400)))
    with admin_of(fetching).client() as client:
        save_web_settings(client, WEB_FETCH_MAX_CONTENT_LENGTH=100)

    content = fetch(fetching, create_user(fetching), f"{listener.base_url}/page")

    assert content.endswith("\n\n[Content truncated...]"), content[-80:]
    assert len(content) == 100 + len("\n\n[Content truncated...]")


def test_an_address_that_may_not_be_fetched_is_an_error(fetching):
    result = json.loads(fetch(fetching, create_user(fetching), "file:///etc/passwd"))

    assert "error" in result and "root:" not in result["error"], result


def test_the_web_tools_follow_the_web_search_permission(fetching):
    account = create_user(fetching)
    with account.client() as client:
        permitted = offered_tools(client, fetching.upstream, **WITH_WEB_SEARCH)
        not_asked = offered_tools(client, fetching.upstream)
    with admin_of(fetching).client() as client:
        current = client.get(DEFAULT_PERMISSIONS).json()
        barred = {**current, "features": {**current["features"], "web_search": False}}
        client.post(DEFAULT_PERMISSIONS, json=barred).raise_for_status()
    with account.client() as client:
        refused = offered_tools(client, fetching.upstream, **WITH_WEB_SEARCH)

    assert WEB_TOOLS <= permitted
    assert not WEB_TOOLS & not_asked, "the web tools were offered without web search"
    assert not WEB_TOOLS & refused, f"an account without web search was offered {refused}"
