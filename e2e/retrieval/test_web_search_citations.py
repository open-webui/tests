"""Journey: a chat with Web Search on cites the page the model read, and only that page.

A person switches Web Search on in the chat input's Integrations menu and asks a question. The
model searches with `search_web` and reads one of the hits with `fetch_url`; the reply lists that
page as its one source, its inline marker names the page's site, and opening it shows the page's
text with a link to the page. The search hits the model never opened are not offered as sources
(open-webui/open-webui#29631). The search engine and the page are a local service, so the
instance runs with local fetching allowed.

The test runs in the main chat and in every way a side chat starts and every state it can be in
(`utils/chat_surface.py`), opened on the searching instance with side chats allowed there; the
sources show on a side chat answer on dev c15f28649 and on dev e791cbdde alike.

Discriminates: passes on dev 176d31d1d; in a frontend copy, the source dialog taking marker `[n]`
as the n+1th source left the marker opening nothing; in a backend copy, `fetch_url` left out of the
tools whose results are cited turned the reply sourceless.
"""

from __future__ import annotations

import dataclasses
import re

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.listener import text_answer
from harness.web_retrieval import (
    LOCAL_WEB_FETCH,
    save_web_settings,
    serve_search_results,
    web_settings_restored,
)
from utils.chat_surface import EVERYWHERE, MAIN_CHAT, ChatOpener
from utils.chat_ui import (
    conversation,
    expect_reply,
    last_reply,
    send,
    turn_on_integration,
)

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

QUESTION = "How many herons are at the jetty?"
PAGE_TEXT = "Two grey herons stand at the jetty every morning."
PAGE = f"<html><body><h1>Heron watch</h1><p>{PAGE_TEXT}</p></body></html>"


@pytest.fixture
def searcher(instance_with, listener):
    """The local-fetch instance searching the listener, whose hits are the page and one more."""
    searching = instance_with(LOCAL_WEB_FETCH)
    if not searching.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    listener.route("GET", "/herons", text_answer(PAGE))
    hits = [f"{listener.base_url}/herons", f"{listener.base_url}/egrets"]
    with admin_of(searching).client() as client, web_settings_restored(client):
        save_web_settings(client, **serve_search_results(listener, hits))
        yield searching


@pytest.fixture
def searching_chat_for(chat_for: ChatOpener, searcher, preserve) -> ChatOpener:
    """`chat_for` on the searching instance: its provider answers and side chats are allowed."""
    admin = admin_of(searcher)
    if chat_for.place != MAIN_CHAT:
        preserve("permissions", on=searcher)
        with admin.client() as client:
            permissions = client.get("/api/v1/users/default/permissions").json()
            permissions["chat"].update(side_chat=True)
            saved = client.post("/api/v1/users/default/permissions", json=permissions)
        assert saved.status_code == 200, saved.text
    return dataclasses.replace(chat_for, upstream=searcher.upstream, admin=admin)


# several models: each answer's sources come with that answer, as with one model
@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_the_page_the_model_read_is_the_reply_source(searching_chat_for, searcher, listener):
    page_url = f"{listener.base_url}/herons"
    answering = reply.answering(QUESTION)
    searcher.upstream.queue(
        reply.tool_call("search_web", {"query": "herons jetty"}, "call_search", match=answering),
        reply.tool_call("fetch_url", {"url": page_url}, "call_fetch", match=answering),
        reply.text("Two herons [1].", match=answering),
    )
    chat = searching_chat_for(create_user(searcher))
    turn_on_integration(chat, "Web Search")
    send(chat, QUESTION)
    expect_reply(chat, "Two herons")

    assert listener.requests_to("/search"), "the model's search never reached the engine"
    shown = conversation(chat)
    expect(
        shown.get_by_role("button", name="Toggle 1 source"),
        f"the reply in the {chat.place} lists no source",
    ).to_be_visible()
    site = page_url.split("/")[2]
    marker = last_reply(chat).get_by_role("button", name=f"View source: {site}")
    marker.click()
    citation = chat.page.get_by_role("dialog")
    expect(citation.get_by_role("link", name=page_url, exact=True)).to_be_visible()
    expect(citation).to_contain_text(PAGE_TEXT)
    citation.get_by_role("button", name="Close citation modal").click()

    shown.get_by_role("button", name="Toggle 1 source").click()
    listed = shown.get_by_role("button", name=re.compile(r"^View source: \d+ http"))
    expect(listed).to_have_count(1)
    expect(listed).to_have_accessible_name(f"View source: 1 {page_url}")
