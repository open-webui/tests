"""Journey: the Search Result Count and Domain Filter List of Admin Settings > Web Search.

With web search on an external engine, the admin saves a result count and a domain filter list
in the Web Search tab. When the model searches during a user's chat, the engine is asked for that
many results, and the hits the model is given back are only the ones from the listed domains.
SearXNG picked as the engine, with a query URL and a language, is asked the model's query in that
language, and its results are what the model reads.

With web search off, the tab still saves the loader settings when no engine is picked or the
picked engine is missing its address, while with web search on it keeps asking for them (issue
#32831).

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose Web Search form sends the stored
settings back in place of the edited ones, every test fails. On dev ca8f451cc both loader tests are
red: the engine list or the engine's own fields block the save even with web search off.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.listener import json_answer
from harness.web_retrieval import RETRIEVAL_CONFIG, save_web_settings, serve_search_results
from utils.admin_settings import save
from utils.chat_ui import expect_reply, send, turn_on_integration

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
    with page.expect_response(lambda response: response.url.endswith("/retrieval/config")):
        page.goto("/admin/settings/web")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("switch", name="Web Search", exact=True)).to_be_checked()
    return settings


DOMAIN_FILTER = "Enter domains separated by commas (e.g., example.com,site.org,!excludedsite.com)"


def search_in_a_chat(page: Page, upstream) -> list[dict]:
    """The model searches once; returns what the provider was sent after the search."""
    answering = reply.answering(QUESTION)
    upstream.queue(
        reply.tool_call("search_web", {"query": "harbour ferry"}, "call_search", match=answering),
        reply.text("The ferry leaves at noon.", match=answering),
    )
    turn_on_integration(page, "Web Search")
    send(page, QUESTION)
    expect_reply(page, "The ferry leaves at noon.")
    return [body for body in upstream.chat_requests() if answering(body)]


def test_the_engine_is_asked_for_the_saved_result_count(
    page_for, admin, make_user, upstream, engine
):
    admin_page = page_for(admin)
    settings = web_search_tab(admin_page)
    settings.get_by_placeholder("Search Result Count").fill("2")
    save(settings)

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
    save(settings)

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
    save(settings)

    requests = search_in_a_chat(page_for(make_user()), upstream)

    [search] = engine.requests_to("/searxng")
    assert "q=harbour+ferry" in search.path or "q=harbour%20ferry" in search.path, search.path
    assert "language=de" in search.path
    assert "Die Faehre" in json.dumps(requests[-1]["messages"])


NO_ENGINE = {"WEB_SEARCH_ENGINE": ""}
SEARXNG_WITHOUT_URL = {"WEB_SEARCH_ENGINE": "searxng", "SEARXNG_QUERY_URL": ""}
UNSET_ENGINES = pytest.mark.parametrize(
    "engine", [NO_ENGINE, SEARXNG_WITHOUT_URL], ids=["no engine", "searxng without its url"]
)


def loader_tab(page_for, admin, web_search_on: bool, engine: dict) -> tuple[Page, Locator]:
    with admin.client() as client:
        save_web_settings(client, ENABLE_WEB_SEARCH=web_search_on, **engine)
    page = page_for(admin, path="/admin/settings/web")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_placeholder("Timeout", exact=True)).to_be_visible()
    return page, settings


def saved_loader_timeout(admin) -> str | None:
    with admin.client() as client:
        current = client.get(RETRIEVAL_CONFIG[0])
    assert current.status_code == 200, current.text
    return current.json()["web"]["WEB_LOADER_TIMEOUT"]


@pytest.mark.regression
@UNSET_ENGINES
def test_the_loader_settings_save_with_web_search_off(page_for, admin, preserve, engine):
    # open-webui/open-webui#32831: an empty engine field blocked the save and nothing was sent
    preserve(RETRIEVAL_CONFIG)
    page, settings = loader_tab(page_for, admin, web_search_on=False, engine=engine)
    settings.get_by_placeholder("Timeout", exact=True).fill("15")
    save(settings)

    assert saved_loader_timeout(admin) == "15"


@UNSET_ENGINES
def test_web_search_on_still_asks_for_the_engine(page_for, admin, preserve, engine):
    preserve(RETRIEVAL_CONFIG)
    page, settings = loader_tab(page_for, admin, web_search_on=True, engine=engine)
    timeout_before = saved_loader_timeout(admin)
    settings.get_by_placeholder("Timeout", exact=True).fill("16")
    settings.get_by_role("button", name="Save", exact=True).click()

    if engine == NO_ENGINE:
        missing = settings.get_by_role("combobox").filter(
            has=page.get_by_role("option", name="searxng", exact=True)
        )
    else:
        missing = settings.get_by_placeholder("Enter Searxng Query URL")
    expect(missing).to_be_focused()
    assert saved_loader_timeout(admin) == timeout_before
