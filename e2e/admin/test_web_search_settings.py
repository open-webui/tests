"""Journey: the Search Result Count and Domain Filter List of Admin Settings > Web Search.

With web search on an external engine, the admin saves a result count and a domain filter list
in the Web Search tab. When the model searches during a user's chat, the engine is asked for that
many results, and the hits the model is given back are only the ones from the listed domains.
SearXNG picked as the engine, with a query URL and a language, is asked the model's query in that
language, and its results are what the model reads.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose Web Search form sends the stored
settings back in place of the edited ones, every test fails.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.listener import json_answer
from harness.web_retrieval import RETRIEVAL_CONFIG, save_web_settings, serve_search_results
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

QUESTION = "What do the harbour pages say about the ferry?"
HITS = [
    {"link": "https://harbour.example/ferry", "title": "Ferry", "snippet": "Ferry at noon."},
    {"link": "https://elsewhere.example/ferry", "title": "Other", "snippet": "Ferry at one."},
]


@pytest.fixture
def engine(admin, preserve, listener):
    """Web search on, on an external engine that answers with one hit per domain."""
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        save_web_settings(client, **serve_search_results(listener, []))
    listener.route("POST", "/search", json_answer(HITS))
    return listener


def web_search_tab(page: Page) -> Locator:
    page.goto("/admin/settings/web")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("switch", name="Web Search", exact=True)).to_be_checked()
    return settings


DOMAIN_FILTER = "Enter domains separated by commas (e.g., example.com,site.org,!excludedsite.com)"


def save(page: Page, settings: Locator) -> None:
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def search_in_a_chat(page: Page, upstream) -> list[dict]:
    """The model searches once; returns what the provider was sent after the search."""
    answering = reply.answering(QUESTION)
    upstream.queue(
        reply.tool_call("search_web", {"query": "harbour ferry"}, "call_search", match=answering),
        reply.text("The ferry leaves at noon.", match=answering),
    )
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="Integrations", exact=True).last.click()
    page.get_by_role("menu").get_by_role("button", name="Web Search").click()
    page.keyboard.press("Escape")
    send(page, QUESTION)
    expect_reply(page, "The ferry leaves at noon.")
    return [body for body in upstream.chat_requests() if answering(body)]


def test_the_engine_is_asked_for_the_saved_result_count(
    page_for, admin, make_user, upstream, engine
):
    admin_page = page_for(admin)
    settings = web_search_tab(admin_page)
    settings.get_by_placeholder("Search Result Count").fill("2")
    save(admin_page, settings)

    search_in_a_chat(page_for(make_user()), upstream)

    [search] = engine.requests_to("/search")
    assert search.json()["count"] == 2


def test_the_model_gets_only_the_hits_from_the_listed_domains(
    page_for, admin, make_user, upstream, engine
):
    admin_page = page_for(admin)
    settings = web_search_tab(admin_page)
    settings.get_by_placeholder("Search Result Count").fill("2")
    settings.get_by_placeholder(DOMAIN_FILTER).fill("harbour.example")
    save(admin_page, settings)

    requests = search_in_a_chat(page_for(make_user()), upstream)

    tool_results = json.dumps(
        [entry for entry in requests[-1]["messages"] if entry.get("role") == "tool"]
    )
    assert "harbour.example/ferry" in tool_results
    assert "elsewhere.example" not in tool_results


def test_searxng_is_asked_the_query_in_the_saved_language(
    page_for, admin, make_user, upstream, engine
):
    engine.route(
        "GET",
        "/searxng",
        json_answer(
            {
                "results": [
                    {
                        "url": "https://harbour.example/ferry",
                        "title": "Ferry",
                        "content": "Die Faehre faehrt um zwoelf.",
                        "score": 1.0,
                    }
                ]
            }
        ),
    )
    admin_page = page_for(admin)
    settings = web_search_tab(admin_page)
    settings.get_by_role("combobox").filter(
        has=admin_page.get_by_role("option", name="searxng", exact=True)
    ).select_option("searxng")
    settings.get_by_placeholder("Enter Searxng Query URL").fill(f"{engine.base_url}/searxng")
    settings.get_by_placeholder("Enter Searxng search language").fill("de")
    settings.get_by_placeholder("Search Result Count").fill("1")
    save(admin_page, settings)

    requests = search_in_a_chat(page_for(make_user()), upstream)

    [search] = engine.requests_to("/searxng")
    assert "q=harbour+ferry" in search.path or "q=harbour%20ferry" in search.path, search.path
    assert "language=de" in search.path
    assert "Die Faehre" in json.dumps(requests[-1]["messages"])
