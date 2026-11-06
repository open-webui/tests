"""Journey: a chat with Web Search on searches DuckDuckGo when the model asks.

With the `duckduckgo` engine, switching Web Search on in the chat input's Integrations menu offers
the model the `search_web` tool; its call sends the query to DuckDuckGo through ddgs and hands
the model the links, titles and snippets DuckDuckGo listed, which the chat shows as the tool's
result. DuckDuckGo is played by `harness/duckduckgo.py` on an instance of its own. Twin of
integration/deps/test_web_search_stack.py.

Discriminates: passes on dev ef67cc3fa; in a backend copy opening `DDGS` with a `proxi` keyword
(a bump dropping `proxy`), the tool answers with an error in place of the results.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.actors import create_user
from harness.duckduckgo import Result, duckduckgo_env, duckduckgo_settings, serving_duckduckgo
from harness.web_retrieval import save_web_settings, web_settings_restored
from utils.chat_ui import conversation, expect_reply, send, turn_on_integration

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

HERONS = Result(
    "https://example.org/herons",
    "Herons of the estuary",
    "Grey herons wait motionless in the shallows.",
)
EGRETS = Result("https://example.org/egrets", "Egrets in winter", "Little egrets gather at dusk.")
QUESTION = "Where do herons wait?"


@pytest.fixture(scope="module")
def duckduckgo():
    with serving_duckduckgo() as fake:
        fake.results = [HERONS, EGRETS]
        yield fake


@pytest.fixture
def searcher(instance_with, duckduckgo):
    searching = instance_with(duckduckgo_env(duckduckgo))
    if not searching.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    with searching.client() as client, web_settings_restored(client):
        save_web_settings(client, **duckduckgo_settings(result_count=2))
        yield searching


def _tool_results(searcher) -> list[str]:
    follow_up = reply.conversation(searcher.upstream.chat_requests()[-1])
    return [entry["content"] for entry in follow_up if entry["role"] == "tool"]


def test_the_model_searches_duckduckgo_and_the_chat_shows_what_it_found(
    page_for, searcher, duckduckgo
):
    searcher.upstream.queue(
        reply.tool_call("search_web", {"query": "herons"}, match=reply.answering(QUESTION)),
        reply.text("By the water.", match=reply.answering(QUESTION)),
    )
    page = page_for(create_user(searcher))
    turn_on_integration(page, "Web Search")
    send(page, QUESTION)
    expect_reply(page, "By the water.")

    [found] = _tool_results(searcher)
    assert json.loads(found) == [
        {"title": result.title, "link": result.link, "snippet": result.snippet}
        for result in (HERONS, EGRETS)
    ]
    assert duckduckgo.queries()[-1:] == ["herons"]
    conversation(page).get_by_text("View Result from search_web").click()
    expect(conversation(page).get_by_text(HERONS.snippet)).to_be_visible()
