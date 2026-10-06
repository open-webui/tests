"""Journey: each web search engine set up in Admin Settings > Web Search answers a user's chat.

The admin switches Web Search on, picks the engine, fills in its key and address fields and a
result count, and saves. A user turns Web Search on in the chat input's Integrations menu and asks;
the model searches with `search_web`, which reaches the engine's API with the query and the key,
and is handed the links, titles and snippets the engine listed, which the chat shows as the tool's
result. The model then reads the first hit with `fetch_url` and cites it: the reply lists that page
as its one source, and opening the marker shows the page's text and link. Every engine's API is
played by `harness/search_apis.py`, the fixed public hosts through its proxy, and the hits are
pages on the same local service, on an instance that may fetch loopback addresses. Twin of
integration/retrieval/test_web_search_engines.py, which has the domain filter per engine.

Not here: DuckDuckGo, SearXNG and the external engine (modules of their own), Azure AI Search (not
in the admin panel's engine list) and Sougou (its Tencent Cloud SDK is not among the backend's
requirements).

Red on dev ebc6add67: the YaCy form marks the username and password as required, so a YaCy
without them, which the docs call optional and the backend supports, cannot be saved.

Discriminates: passes on dev ebc6add67 but for YaCy without credentials; in a backend copy whose
`search_web` tool hands the model the engine's hits without their snippets every engine case
fails; in a frontend build without the two `required` marks the YaCy case passes.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.listener import text_answer
from harness.search_apis import (
    API_KEY,
    ENGINES,
    ENGINES_BY_NAME,
    Engine,
    Hit,
    SearchApis,
    carries,
    search_apis_env,
    serving_search_apis,
)
from harness.web_retrieval import LOCAL_WEB_FETCH, web_settings_restored
from utils.chat_ui import chat_input, conversation, expect_reply, last_reply, send

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

PAGE_TEXT = "The harbour ferry leaves at noon from pier two."
PAGE = f"<html><body><h1>Ferry times</h1><p>{PAGE_TEXT}</p></body></html>"
KEYLESS = {"yacy", "openserp"}


@pytest.fixture(scope="module")
def apis():
    with serving_search_apis() as serving:
        serving.listener.route("GET", "/pages/ferry", text_answer(PAGE))
        yield serving


@pytest.fixture(scope="module")
def searcher(instance_with, apis):
    launched = instance_with({**LOCAL_WEB_FETCH, **search_apis_env(apis)})
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    return launched


@pytest.fixture
def searcher_admin(searcher):
    admin = admin_of(searcher)
    with admin.client() as client, web_settings_restored(client):
        yield admin


def hits_on(apis: SearchApis) -> list[Hit]:
    return [
        Hit(f"{apis.base_url}/pages/ferry", "Ferry times", "The ferry leaves at noon."),
        Hit(f"{apis.base_url}/pages/buses", "Bus times", "Buses leave every hour."),
    ]


def fill_engine_form(page: Page, engine: Engine, values: dict[str, str]) -> None:
    """Web Search switched on and `engine` picked, with the fields named in `values` filled in."""
    with page.expect_response(lambda response: response.url.endswith("/retrieval/config")):
        page.goto("/admin/settings/web")
    settings = page.get_by_role("dialog")
    web_search = settings.get_by_role("switch", name="Web Search", exact=True)
    if web_search.get_attribute("aria-checked") != "true":
        web_search.click()
    settings.get_by_role("combobox").filter(
        has=page.get_by_role("option", name="searxng", exact=True)
    ).select_option(engine.name)
    for placeholder, setting in engine.form.items():
        if setting in values:
            settings.get_by_placeholder(placeholder, exact=True).fill(values[setting])
    settings.get_by_placeholder("Search Result Count").fill("2")
    settings.get_by_role("button", name="Save", exact=True).click()


def set_up_engine(page: Page, engine: Engine, apis: SearchApis) -> None:
    fill_engine_form(page, engine, engine.filled_settings(apis))
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def turn_web_search_on(page: Page) -> None:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="Integrations", exact=True).last.click()
    page.get_by_role("menu").get_by_role("button", name="Web Search").click()
    page.keyboard.press("Escape")


def tool_results(searcher, question: str) -> list[str]:
    follow_up = [
        body for body in searcher.upstream.chat_requests() if reply.answering(question)(body)
    ]
    return [entry["content"] for entry in follow_up[-1]["messages"] if entry["role"] == "tool"]


@pytest.mark.parametrize("engine", ENGINES, ids=lambda engine: engine.name)
def test_the_engine_set_up_by_the_admin_answers_a_chat_search(
    engine, page_for, searcher, searcher_admin, apis
):
    hits = hits_on(apis)
    engine.serve(apis, hits)
    set_up_engine(page_for(searcher_admin), engine, apis)
    question = f"When does the ferry leave, says {engine.name}?"
    answering = reply.answering(question)
    searcher.upstream.queue(
        reply.tool_call("search_web", {"query": "ferry harbour"}, "call_search", match=answering),
        reply.tool_call("fetch_url", {"url": hits[0].link}, "call_fetch", match=answering),
        reply.text("At noon [1].", match=answering),
    )
    page = page_for(create_user(searcher))

    turn_web_search_on(page)
    send(page, question)
    expect_reply(page, "At noon")

    found, fetched = tool_results(searcher, question)
    assert json.loads(found) == engine.found(hits)
    assert PAGE_TEXT in fetched
    [sent] = engine.searches(apis)[-1:]
    assert carries(sent, "ferry harbour"), f"{engine.name} was not asked the query"
    if engine.name not in KEYLESS:
        assert carries(sent, API_KEY), f"{engine.name} was not sent the admin's key"

    chat = conversation(page)
    last_reply(page).get_by_role("button", name="Toggle details").click()
    chat.get_by_text("View Result from search_web").click()
    expect(chat.get_by_text(engine.found(hits)[0]["snippet"]).first).to_be_visible()
    expect(chat.get_by_role("button", name="Toggle 1 source")).to_be_visible()
    site = hits[0].link.split("/")[2]
    last_reply(page).get_by_role("button", name=f"View source: {site}").click()
    citation = page.get_by_role("dialog")
    expect(citation.get_by_role("link", name=hits[0].link, exact=True)).to_be_visible()
    expect(citation).to_contain_text(PAGE_TEXT)


def test_yacy_without_credentials_can_be_saved(page_for, searcher, searcher_admin, apis):
    yacy = ENGINES_BY_NAME["yacy"]
    page = page_for(searcher_admin)

    fill_engine_form(page, yacy, {"YACY_QUERY_URL": f"{apis.base_url}/yacy"})

    expect(
        page.get_by_text("Settings saved successfully!").first,
        "the form refuses a YaCy without username and password, which the docs call optional",
    ).to_be_visible()
