"""Journey: with Tool Search on, a tool the model runs through call_tool shows as that tool.

Dev 7b47c3b22 lets the model dispatch a deferred tool through `call_tool(tool_name, params)`. The
chat unwraps the call before approvals and execution, so the reply shows the target tool by name
with its own input and result (also after a reload), and an approval prompt names it. A call to
an unknown tool comes back as an error the reply continues past. The next turn replays the call
the model actually made, so the provider sees its own `call_tool` call in the history. A
dispatched tool shows on the reply as a direct call does: knowledge sources listed with each
`[n]` marker opening its passage (after a reload too), a workspace tool's citation, a page read
with `fetch_url`, a picture from `generate_image` kept as a stored file and an HTML embed.

With Ask for approval, a round mixing a dispatched call, a direct call and a refused dispatch asks
about each call in turn and runs only the allowed ones, and a second round asks again. A pending
prompt survives a reload and another tab of the account can allow it; switching to Full access
runs the pending calls; a denied dispatch is asked anew on Regenerate while Continue runs nothing;
a temporary chat asks in a dialog naming the dispatched tool.

Since dev 003299641 the search itself is not shown on the reply, only the tool it found, and one
`call_tool` call may carry several tool calls: each shows as its own tool with its input and
result (also on a fresh load of the chat), the reply goes on, and the provider gets one
combined result. With Ask for approval each call of a batch asks in turn, and only the allowed
one runs.

Since dev 0f3bd0a7e Admin Settings > Interface offers a search engine for Tool Search: picking
the Decisions API shows its address, key, model, relevance threshold and prompt fields (the
prompt showing the default until one is typed); they are saved, shown again on a fresh load and
used by the next search. A decisions address with credentials is refused with the server's
message and nothing is saved. Picking Embeddings names the Documents embedding model with a link
to its settings and is kept.

Discriminates: passes on dev 6276e81ce; in a backend copy without the streaming handler's unwrap
the display and approval tests fail (the reply shows call_tool), without the replay of the
model's original call the display test fails on the next turn's history, and without the unknown
tool check the failure test fails (a different error reaches the model and the chat). Skipping
citation sources for dispatched calls fails the knowledge and page tests; no unwrap at all also
fails the citation event, embed and picture tests.


tool check the failure test fails (a different error reaches the model and the chat). In backend
copies of dev 81f5b8adf, skipping the Ask pause for a round holding a dispatched call fails every
approval test but the temporary chat one, which fails when its confirmation is skipped for a
dispatched call; marking every pending call approved on one approval, or resuming with full
access after an approval, fails the mixed round test. A frontend build whose Full access switch
leaves pending calls alone fails the Full access test. In a backend copy of dev f659c0d12 that
shows the search call the display test fails; one that runs a batch as a single call fails both
batch tests; one whose settings read back BM25 fails the decisions and embeddings tests; one
without the decisions check fails the refused address test; and one that names no embedding
model fails the embeddings test.
"""

from __future__ import annotations

import base64
import json
import re
import uuid
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.channel_chat import serve_openai_images
from harness.chat import ask
from harness.image_engines import IMAGES_CONFIG, PNG_BASE64, save_image_settings
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.listener import json_answer, text_answer
from harness.python_tools import EVERYONE_READS, python_tool
from harness.upstream import MOCK_MODEL_ID
from harness.web_retrieval import (
    LOCAL_WEB_FETCH,
    RETRIEVAL_CONFIG,
    save_web_settings,
    serve_search_results,
)
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    conversation,
    expect_reply,
    last_reply,
    listed_source,
    regenerate_buttons,
    send,
    set_tool_approval_mode,
    turn_on_integration,
)
from utils.model_selector import select_model

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
REJECTED = "Error: tool call rejected by user."
SHOUTING_TOOL = '''
class Tools:
    def shout(self, text: str) -> str:
        """Shout the text."""
        return text.upper()

    def mumble(self, text: str) -> str:
        """Mumble the text."""
        return text.lower()
'''


@pytest.fixture
def tool_search_on(admin, preserve) -> None:
    """Tool Search defers every workspace tool and leaves the built-in ones loaded."""
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        client.post(
            CHAT_CONFIG[1],
            json={
                **current,
                "ENABLE_TOOL_SEARCH": True,
                "TOOL_SEARCH_DEFER_THRESHOLD": 0,
                "TOOL_SEARCH_ALWAYS_LOADED": [],
                "TOOL_SEARCH_DEFER_BUILTIN_TOOLS": False,
            },
        ).raise_for_status()


@pytest.fixture
def tool_approval_on(admin, tool_search_on) -> None:
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        client.post(
            CHAT_CONFIG[1], json={**current, "ENABLE_TOOL_PERMISSIONS": True}
        ).raise_for_status()


@pytest.fixture
def shouter(admin) -> Iterator[str]:
    """A preset of the scripted model with the shouting tool attached; yields its name."""
    name = f"Shouter {uuid.uuid4().hex[:8]}"
    model_id = name.lower().replace(" ", "-")
    with python_tool(admin, SHOUTING_TOOL) as tool_id, admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": name,
                "meta": {"toolIds": [tool_id]},
                "params": {},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert created.status_code == 200, created.text
        yield name
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def call_tool(tool_name: str, params: dict, prompt: str, call_id: str) -> reply.Reply:
    arguments = {"tool_name": tool_name, "params": json.dumps(params)}
    return reply.tool_call("call_tool", arguments, call_id=call_id, match=reply.answering(prompt))


def requests_for(upstream, prompt: str) -> list[dict]:
    return [body for body in upstream.chat_requests() if reply.answering(prompt)(body)]


def tool_results(upstream, prompt: str) -> list[str]:
    final = requests_for(upstream, prompt)[-1]
    return [str(entry["content"]) for entry in reply.conversation(final) if entry["role"] == "tool"]


def expect_shout_shown(page: Page, word: str) -> None:
    last_reply(page).get_by_text("View Result from shout").click()
    shown = re.compile(rf"Input\s*text\s*{word}\s*Output\s*{word.upper()}")
    expect(last_reply(page)).to_contain_text(shown)
    # since dev 003299641 the search itself is not shown, only the tool it found
    expect(last_reply(page)).not_to_contain_text("search_tools")
    expect(last_reply(page)).not_to_contain_text("call_tool")


def approval_button(page: Page, name: str):
    button = conversation(page).get_by_role("button", name=name, exact=True)
    expect(button).to_have_count(1, timeout=REPLY_TIMEOUT_MS)
    return button


def test_a_dispatched_call_shows_as_its_tool_and_is_replayed_as_sent(
    tool_search_on, page_for, make_user, upstream, shouter
):
    prompt = f"shout ahoy {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.tool_call(
            "search_tools", {"query": "shout"}, call_id="call_s", match=reply.answering(prompt)
        ),
        call_tool("shout", {"text": "ahoy"}, prompt, "call_c"),
        reply.text("Shouted it.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())
    select_model(page, shouter)
    send(page, prompt)
    expect_reply(page, "Shouted it.")

    offered = [tool["function"]["name"] for tool in requests_for(upstream, prompt)[0]["tools"]]
    assert {"search_tools", "call_tool"} <= set(offered) and "shout" not in offered
    assert tool_results(upstream, prompt)[-1] == "AHOY"
    expect_shout_shown(page, "ahoy")

    page.reload()
    expect_shout_shown(page, "ahoy")

    follow_up = f"and again {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Once more.", match=reply.answering(follow_up)))
    send(page, follow_up)
    expect_reply(page, "Once more.")
    [history] = requests_for(upstream, follow_up)
    calls = [
        call["function"]
        for entry in history["messages"]
        for call in entry.get("tool_calls") or []
        if call["id"] == "call_c"
    ]
    assert len(calls) == 1 and calls[0]["name"] == "call_tool", calls
    assert json.loads(calls[0]["arguments"]) == {
        "tool_name": "shout",
        "params": json.dumps({"text": "ahoy"}),
    }


def test_a_call_to_an_unknown_tool_shows_the_error_and_the_reply_goes_on(
    tool_search_on, page_for, make_user, upstream, shouter
):
    prompt = f"whisper hush {uuid.uuid4().hex[:6]}"
    upstream.queue(
        call_tool("whisper", {"text": "hush"}, prompt, "call_w"),
        reply.text("No whispering here.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())
    select_model(page, shouter)
    send(page, prompt)
    expect_reply(page, "No whispering here.")

    error = 'Error: Tool "whisper" is not available. Search for an available tool.'
    assert tool_results(upstream, prompt) == [error]
    last_reply(page).get_by_text("View Result from call_tool").click()
    expect(last_reply(page)).to_contain_text('Tool "whisper" is not available')


def test_an_approval_prompt_names_the_dispatched_tool(
    tool_approval_on, page_for, make_user, upstream, shouter
):
    page = page_for(make_user())
    select_model(page, shouter)
    set_tool_approval_mode(page, "Ask for approval")

    allowed = f"shout yes {uuid.uuid4().hex[:6]}"
    upstream.queue(
        call_tool("shout", {"text": "granted"}, allowed, "call_a"),
        reply.text("Allowed and shouted.", match=reply.answering(allowed)),
    )
    send(page, allowed)
    approval_button(page, "Allow")
    last_reply(page).get_by_text("Allow shout?").click()
    expect(last_reply(page)).to_contain_text(re.compile(r"Input\s*text\s*granted"))
    approval_button(page, "Allow").click()
    expect_reply(page, "Allowed and shouted.")
    assert tool_results(upstream, allowed) == ["GRANTED"]

    denied = f"shout no {uuid.uuid4().hex[:6]}"
    upstream.queue(
        call_tool("shout", {"text": "refused"}, denied, "call_d"),
        reply.text("Not shouted.", match=reply.answering(denied)),
    )
    send(page, denied)
    approval_button(page, "Deny")
    expect(last_reply(page)).to_contain_text("Allow shout?")
    approval_button(page, "Deny").click()
    expect_reply(page, "Not shouted.")
    expect(last_reply(page)).to_contain_text("Denied shout")
    assert tool_results(upstream, denied)[-1:] == [REJECTED]


# --- what a dispatched tool shows on the reply: sources, pictures, embeds ----------------------

NOTES = ("herons", "Grey herons nest in colonies near water and hunt fish in the shallows.")
LEDGER = ("ledger", "The heron survey budget for the spring is four thousand.")
SOURCE_TAG = re.compile(r'<source id="(\d+)" name="([^"]+)"')
FILE_CONTENT_URL = re.compile(r"/api/v1/files/[^/]+/content$")
REPORTING_TOOL = '''
from fastapi.responses import HTMLResponse


class Tools:
    async def harbour_report(self, harbour: str, __event_emitter__=None) -> str:
        """Report on a harbour, citing the harbour log."""
        await __event_emitter__(
            {
                "type": "citation",
                "data": {
                    "document": [f"The lights of {harbour} were lit at dusk."],
                    "metadata": [{"source": "Harbour log"}],
                    "source": {"name": "Harbour log"},
                },
            }
        )
        return f"report for {harbour}"

    def harbour_chart(self, harbour: str) -> HTMLResponse:
        """Show a chart of a harbour."""
        page = f'<p id="chart">Chart of {harbour}</p>'
        return HTMLResponse(content=page, headers={"Content-Disposition": "inline"})
'''


@pytest.fixture
def builtins_deferred(admin, tool_search_on) -> None:
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        client.post(
            CHAT_CONFIG[1], json={**current, "TOOL_SEARCH_DEFER_BUILTIN_TOOLS": True}
        ).raise_for_status()


@pytest.fixture
def reporter(admin) -> Iterator[str]:
    """A preset of the scripted model with the harbour reporting tool attached; yields its name."""
    name = f"Reporter {uuid.uuid4().hex[:8]}"
    model_id = name.lower().replace(" ", "-")
    with python_tool(admin, REPORTING_TOOL) as tool_id, admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": name,
                "meta": {"toolIds": [tool_id]},
                "params": {},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert created.status_code == 200, created.text
        yield name
        client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def library(admin, make_user):
    """Two notes in a knowledge base a fresh reader may read; yields reader, base and texts."""
    reader = make_user()
    tag = uuid.uuid4().hex[:8]
    grant = [{"principal_type": "user", "principal_id": reader.id, "permission": "read"}]
    names = {f"{stem}-{tag}.txt": text for stem, text in (NOTES, LEDGER)}
    with admin.client() as client, knowledge_base(client, f"Birds {tag}", grant) as knowledge_id:
        for filename, text in names.items():
            add_text_file(client, knowledge_id, filename, text)
        yield reader, knowledge_id, names


def expect_markers_open_their_sources(page: Page, source_ids: dict[str, str], texts: dict) -> None:
    expect(conversation(page).get_by_role("button", name="Toggle 2 sources")).to_be_visible()
    markers = last_reply(page).get_by_role("button", name=re.compile(r"^View source: "))
    expect(markers).to_have_count(2)
    for position in (1, 2):
        name = source_ids[str(position)]
        expect(markers.nth(position - 1)).to_have_accessible_name(f"View source: {name}")
        markers.nth(position - 1).click()
        citation = page.get_by_role("dialog")
        expect(citation).to_contain_text(texts[name])
        citation.get_by_role("button", name="Close citation modal").click()
        expect(citation).to_be_hidden()


def test_knowledge_sources_from_a_dispatched_search_are_listed_and_cited(
    builtins_deferred, page_for, library, upstream
):
    reader, knowledge_id, texts = library
    prompt = f"what do my notes say? {uuid.uuid4().hex[:6]}"
    arguments = {"query": "heron", "knowledge_ids": [knowledge_id], "count": 10}
    upstream.queue(
        call_tool("query_knowledge_files", arguments, prompt, "call_k"),
        reply.text("Herons nest [1] and the budget is set [2].", match=reply.answering(prompt)),
    )
    page = page_for(reader)
    send(page, prompt)
    expect_reply(page, "Herons nest")

    final = requests_for(upstream, prompt)[-1]
    source_ids = dict(SOURCE_TAG.findall(json.dumps(final["messages"]).replace('\\"', '"')))
    assert set(source_ids.values()) == set(texts), source_ids
    expect_markers_open_their_sources(page, source_ids, texts)

    expect(page).to_have_url(re.compile(r"/c/[0-9a-f-]+$"))
    page.reload()
    expect_reply(page, "Herons nest")
    expect_markers_open_their_sources(page, source_ids, texts)


def test_a_citation_a_dispatched_tool_emits_is_listed(
    tool_search_on, page_for, make_user, upstream, reporter
):
    prompt = f"report on Pula {uuid.uuid4().hex[:6]}"
    upstream.queue(
        call_tool("harbour_report", {"harbour": "Pula"}, prompt, "call_r"),
        reply.text("Reported.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())
    select_model(page, reporter)
    send(page, prompt)
    expect_reply(page, "Reported.")

    assert tool_results(upstream, prompt) == ["report for Pula"]
    conversation(page).get_by_role("button", name="Toggle 1 source").click()
    listed_source(page, "Harbour log").click()
    expect(page.get_by_role("dialog")).to_contain_text("The lights of Pula were lit at dusk.")


def test_an_embed_a_dispatched_tool_returns_is_shown(
    tool_search_on, page_for, make_user, upstream, reporter
):
    prompt = f"chart Pula {uuid.uuid4().hex[:6]}"
    upstream.queue(
        call_tool("harbour_chart", {"harbour": "Pula"}, prompt, "call_e"),
        reply.text("Charted.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())
    select_model(page, reporter)
    send(page, prompt)
    expect_reply(page, "Charted.")

    [told] = tool_results(upstream, prompt)
    assert "harbour_chart: Embedded UI result is active" in told
    chart = last_reply(page).frame_locator("iframe").locator("#chart")
    expect(chart).to_have_text("Chart of Pula")
    page.reload()
    expect(last_reply(page).frame_locator("iframe").locator("#chart")).to_have_text("Chart of Pula")


@pytest.fixture
def image_engine(admin, preserve, listener) -> None:
    preserve(IMAGES_CONFIG)
    with admin.client() as client:
        save_image_settings(client, **serve_openai_images(listener))


def test_a_picture_drawn_through_call_tool_shows_and_is_stored(
    builtins_deferred, image_engine, page_for, make_user, upstream, listener
):
    person = make_user()
    prompt = f"draw a lighthouse {uuid.uuid4().hex[:6]}"
    upstream.queue(
        call_tool("generate_image", {"prompt": "a lighthouse"}, prompt, "call_i"),
        reply.text("Here is your lighthouse.", match=reply.answering(prompt)),
    )
    page = page_for(person)
    turn_on_integration(page, "Image")
    send(page, prompt)
    expect_reply(page, "Here is your lighthouse.")

    picture = last_reply(page).get_by_role("img")
    expect(picture).to_be_visible()
    source = picture.get_attribute("src")
    assert FILE_CONTENT_URL.search(source), f"the picture is not a stored file: {source}"
    [generation] = listener.requests_to("/images/generations")
    assert generation.json()["prompt"] == "a lighthouse"
    [told] = tool_results(upstream, prompt)
    assert json.loads(told)["status"] == "success"

    expect(page).to_have_url(re.compile(r"/c/[0-9a-f-]+$"))
    page.reload()
    expect_reply(page, "Here is your lighthouse.")
    expect(last_reply(page).get_by_role("img")).to_have_attribute("src", source)
    with person.client() as client:
        assert client.get(source).content == base64.b64decode(PNG_BASE64)


PAGE_TEXT = "Two grey herons stand at the jetty every morning."


@pytest.fixture
def searcher(instance_with, preserve, listener):
    """The local-fetch instance searching the listener, Tool Search deferring its builtins."""
    searching = instance_with(LOCAL_WEB_FETCH)
    if not searching.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    preserve(RETRIEVAL_CONFIG, CHAT_CONFIG, on=searching)
    listener.route("GET", "/herons", text_answer(f"<html><body><p>{PAGE_TEXT}</p></body></html>"))
    with admin_of(searching).client() as client:
        save_web_settings(client, **serve_search_results(listener, [f"{listener.base_url}/herons"]))
        current = client.get(CHAT_CONFIG[0]).json()
        client.post(
            CHAT_CONFIG[1],
            json={
                **current,
                "ENABLE_TOOL_SEARCH": True,
                "TOOL_SEARCH_DEFER_THRESHOLD": 0,
                "TOOL_SEARCH_ALWAYS_LOADED": [],
                "TOOL_SEARCH_DEFER_BUILTIN_TOOLS": True,
            },
        ).raise_for_status()
    return searching


@pytest.mark.slow
def test_a_page_read_through_call_tool_is_the_reply_source(page_for, searcher, listener):
    page_url = f"{listener.base_url}/herons"
    prompt = f"how many herons? {uuid.uuid4().hex[:6]}"
    searcher.upstream.queue(
        call_tool("search_web", {"query": "herons jetty"}, prompt, "call_s"),
        call_tool("fetch_url", {"url": page_url}, prompt, "call_f"),
        reply.text("Two herons [1].", match=reply.answering(prompt)),
    )
    page = page_for(create_user(searcher))
    turn_on_integration(page, "Web Search")
    send(page, prompt)
    expect_reply(page, "Two herons")

    assert listener.requests_to("/search"), "the dispatched search never reached the engine"
    expect(conversation(page).get_by_role("button", name="Toggle 1 source")).to_be_visible()
    site = page_url.split("/")[2]
    last_reply(page).get_by_role("button", name=f"View source: {site}").click()
    citation = page.get_by_role("dialog")
    expect(citation.get_by_role("link", name=page_url, exact=True)).to_be_visible()
    expect(citation).to_contain_text(PAGE_TEXT)


def shout_directly(text: str, call_id: str) -> dict:
    arguments = json.dumps({"text": text})
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "shout", "arguments": arguments},
    }


def dispatched(tool_name: str, params: dict, call_id: str) -> dict:
    arguments = json.dumps({"tool_name": tool_name, "params": json.dumps(params)})
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "call_tool", "arguments": arguments},
    }


def test_a_mixed_round_asks_each_call_in_turn_and_a_second_round_asks_again(
    tool_approval_on, page_for, make_user, upstream, shouter
):
    page = page_for(make_user())
    select_model(page, shouter)
    set_tool_approval_mode(page, "Ask for approval")

    prompt = f"shout twice {uuid.uuid4().hex[:6]}"
    round_one = [
        dispatched("shout", {"text": "one"}, "call_1"),
        shout_directly("two", "call_2"),
        dispatched("whisper", {"text": "hush"}, "call_3"),
    ]
    upstream.queue(
        reply.Reply(tool_calls=round_one, match=reply.answering(prompt)),
        call_tool("shout", {"text": "three"}, prompt, "call_4"),
        reply.text("Shouted what I could.", match=reply.answering(prompt)),
    )
    send(page, prompt)
    for word, action in (("one", "Allow"), ("two", "Deny"), ("three", "Allow")):
        buttons = last_reply(page).get_by_role("button", name=action, exact=True)
        expect(buttons).not_to_have_count(0, timeout=REPLY_TIMEOUT_MS)
        card = last_reply(page).get_by_text("Allow shout?")
        closed_group = last_reply(page).get_by_role("button", name="Toggle details", expanded=False)
        # the card hides briefly while the next call renders; clicking an open group closes it
        expect(card.or_(closed_group).first).to_be_visible()
        if not card.is_visible() and closed_group.count():
            closed_group.last.click()
        card.click()
        expect(last_reply(page)).to_contain_text(re.compile(rf"Input\s*text\s*{word}"))
        buttons.last.click()
    expect_reply(page, "Shouted what I could.")

    unavailable = 'Error: Tool "whisper" is not available. Search for an available tool.'
    final = requests_for(upstream, prompt)[-1]
    results = {
        entry["tool_call_id"]: entry["content"]
        for entry in reply.conversation(final)
        if entry["role"] == "tool"
    }
    assert results == {
        "call_1": "ONE",
        "call_2": REJECTED,
        "call_3": unavailable,
        "call_4": "THREE",
    }
    expect(last_reply(page)).to_contain_text("Denied shout")


def test_a_pending_dispatch_survives_a_reload_and_another_tab_can_allow_it(
    tool_approval_on, page_for, make_user, upstream, shouter
):
    account = make_user()
    page = page_for(account)
    select_model(page, shouter)
    set_tool_approval_mode(page, "Ask for approval")
    prompt = f"shout later {uuid.uuid4().hex[:6]}"
    upstream.queue(
        call_tool("shout", {"text": "patience"}, prompt, "call_p"),
        reply.text("Shouted after all.", match=reply.answering(prompt)),
    )
    send(page, prompt)
    approval_button(page, "Allow")
    expect(page).to_have_url(re.compile(r"/c/"))

    page.reload()
    approval_button(page, "Allow")
    expect(last_reply(page)).to_contain_text("Allow shout?")
    other_tab = page_for(account, path=page.url)
    expect(last_reply(other_tab)).to_contain_text("Allow shout?")
    approval_button(other_tab, "Allow").click()

    expect_reply(other_tab, "Shouted after all.")
    expect_reply(page, "Shouted after all.")
    expect(conversation(page).get_by_role("button", name="Allow", exact=True)).to_have_count(0)
    assert tool_results(upstream, prompt) == ["PATIENCE"]


def test_switching_to_full_access_runs_the_pending_calls_and_ask_asks_again_later(
    tool_approval_on, page_for, make_user, upstream, shouter
):
    page = page_for(make_user())
    select_model(page, shouter)
    set_tool_approval_mode(page, "Ask for approval")
    prompt = f"shout both {uuid.uuid4().hex[:6]}"
    both = [
        dispatched("shout", {"text": "first"}, "call_1"),
        shout_directly("second", "call_2"),
    ]
    upstream.queue(
        reply.Reply(tool_calls=both, match=reply.answering(prompt)),
        reply.text("Both shouted.", match=reply.answering(prompt)),
    )
    send(page, prompt)
    approval_button(page, "Allow")
    set_tool_approval_mode(page, "Full access")
    expect_reply(page, "Both shouted.")
    assert tool_results(upstream, prompt) == ["FIRST", "SECOND"]

    set_tool_approval_mode(page, "Ask for approval")
    again = f"shout again {uuid.uuid4().hex[:6]}"
    upstream.queue(
        call_tool("shout", {"text": "asked"}, again, "call_3"),
        reply.text("Asked again.", match=reply.answering(again)),
    )
    send(page, again)
    approval_button(page, "Allow").click()
    expect_reply(page, "Asked again.")


def test_a_denied_dispatch_is_asked_again_on_regenerate_and_continue_runs_nothing(
    tool_approval_on, page_for, make_user, upstream, shouter
):
    page = page_for(make_user())
    select_model(page, shouter)
    set_tool_approval_mode(page, "Ask for approval")
    prompt = f"shout maybe {uuid.uuid4().hex[:6]}"
    upstream.queue(
        call_tool("shout", {"text": "maybe"}, prompt, "call_m"),
        reply.text("Not shouted.", match=reply.answering(prompt)),
    )
    send(page, prompt)
    approval_button(page, "Deny")
    expect(regenerate_buttons(page)).to_have_count(0)
    approval_button(page, "Deny").click()
    expect_reply(page, "Not shouted.")

    upstream.queue(reply.text(" Really not.", match=reply.answering(prompt)))
    last_reply(page).hover()
    conversation(page).get_by_role("button", name="Continue Response").click()
    expect_reply(page, "Really not.")
    continued = requests_for(upstream, prompt)[-1]
    assert tool_results(upstream, prompt) == [REJECTED]
    assert reply.conversation(continued)[-1]["role"] == "assistant"

    upstream.queue(
        call_tool("shout", {"text": "maybe"}, prompt, "call_r"),
        reply.text("Shouted this time.", match=reply.answering(prompt)),
    )
    last_reply(page).hover()
    regenerate_buttons(page).last.click()
    page.get_by_text("Try Again").click()
    approval_button(page, "Allow").click()
    expect_reply(page, "Shouted this time.")
    assert tool_results(upstream, prompt) == ["MAYBE"]


def test_a_temporary_chat_asks_before_running_the_dispatched_tool(
    tool_approval_on, page_for, make_user, upstream, shouter
):
    page = page_for(make_user())
    select_model(page, shouter)
    set_tool_approval_mode(page, "Ask for approval")
    page.get_by_role("button", name="Temporary Chat").click()

    for word, action, answer, result in (
        ("hidden", "Deny", "Kept quiet.", "Error: tool call was not approved."),
        ("loud", "Allow", "Shouted it.", "LOUD"),
    ):
        prompt = f"shout {word} {uuid.uuid4().hex[:6]}"
        upstream.queue(
            call_tool("shout", {"text": word}, prompt, "call_t"),
            reply.text(answer, match=reply.answering(prompt)),
        )
        send(page, prompt)
        dialog = page.get_by_role("dialog")
        expect(dialog).to_contain_text("Tool Permissions", timeout=REPLY_TIMEOUT_MS)
        expect(dialog).to_contain_text(re.compile(rf'shout\s*\{{\s*"text": "{word}"'))
        dialog.get_by_role("button", name=action, exact=True).click()
        expect_reply(page, answer)
        assert tool_results(upstream, prompt)[-1:] == [result]
    assert "/c/" not in page.url


# --- one call_tool call carrying several tool calls -------------------------------------------


def expect_batch_shown(page: Page) -> None:
    group = last_reply(page).get_by_text(re.compile(r"^Explored shout, mumble"))
    expect(group).to_be_visible()
    group.click()
    for name, shown in (
        ("shout", r"Input\s*text\s*ahoy\s*Output\s*AHOY"),
        ("mumble", r"Input\s*text\s*HUSH\s*Output\s*hush"),
    ):
        last_reply(page).get_by_text(f"View Result from {name}").click()
        expect(last_reply(page)).to_contain_text(re.compile(shown))
    expect(last_reply(page)).not_to_contain_text("call_tool")


def test_a_batch_of_calls_shows_each_tool_and_the_reply_goes_on(
    tool_search_on, page_for, make_user, upstream, shouter
):
    prompt = f"shout and mumble {uuid.uuid4().hex[:6]}"
    calls = [
        {"tool_name": "shout", "params": json.dumps({"text": "ahoy"})},
        {"tool_name": "mumble", "params": json.dumps({"text": "HUSH"})},
    ]
    upstream.queue(
        reply.tool_call(
            "call_tool", {"calls": calls}, call_id="call_b", match=reply.answering(prompt)
        ),
        reply.text("Shouted and mumbled.", match=reply.answering(prompt)),
    )
    account = make_user()
    page = page_for(account)
    select_model(page, shouter)
    send(page, prompt)
    expect_reply(page, "Shouted and mumbled.")
    expect_batch_shown(page)

    [combined] = tool_results(upstream, prompt)
    results = json.loads(combined)["results"]
    assert [(entry["tool_name"], entry["output"][0]["text"]) for entry in results] == [
        ("shout", "AHOY"),
        ("mumble", "hush"),
    ], combined

    expect(page).to_have_url(re.compile(r"/c/[\w-]+$"))
    chat_path = page.url[page.url.index("/c/") :]
    expect_batch_shown(page_for(account, path=chat_path))


# --- the search engine in Admin Settings > Interface -------------------------------------------

DECISIONS_PROMPT = "Rate how well this tool fits the request."
DECISIONS_FIELDS = [
    "TOOL_SEARCH_ENGINE",
    "TOOL_SEARCH_DECISIONS_API_BASE_URL",
    "TOOL_SEARCH_DECISIONS_API_KEY",
    "TOOL_SEARCH_DECISIONS_MODEL",
    "TOOL_SEARCH_DECISIONS_THRESHOLD",
    "TOOL_SEARCH_DECISIONS_PROMPT",
]


@pytest.fixture
def interface_settings(admin, tool_search_on, page_for) -> Locator:
    """Admin Settings > Interface open, Tool Search on with BM25 and no auto permissions."""
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        client.post(
            CHAT_CONFIG[1],
            json={**current, "TOOL_SEARCH_ENGINE": "bm25", "ENABLE_TOOL_PERMISSIONS_AUTO": False},
        ).raise_for_status()
    return open_interface_settings(page_for(admin, path="/admin/settings/interface"))


def open_interface_settings(page: Page) -> Locator:
    settings = page.get_by_role("dialog")
    # the settings load their models before they show, which takes a few seconds on a busy runner
    settings.get_by_label("Search engine").wait_for()
    expect(settings.get_by_role("switch", name="Tool Search")).to_be_checked()
    return settings


def save_interface(settings: Locator) -> None:
    settings.get_by_role("button", name="Save", exact=True).click()


def stored_chat_config(admin) -> dict:
    with admin.client() as client:
        return client.get(CHAT_CONFIG[0]).json()


def default_decisions_prompt(admin) -> str:
    return stored_chat_config(admin)["DEFAULT_TOOL_SEARCH_DECISIONS_PROMPT"]


def decisions_field(settings: Locator, label: str) -> Locator:
    return settings.get_by_label(label, exact=True)


def test_an_admin_picks_the_decisions_engine_and_it_is_kept_and_used(
    interface_settings, page_for, admin, make_user, upstream, listener
):
    listener.route(
        "POST",
        "/v1/decisions",
        lambda request: json_answer(
            {
                "answers": [
                    {"type": "predicate", "name": question["name"], "probability": 0.9}
                    for question in request.json()["questions"]
                ]
            }
        ),
    )
    settings = interface_settings
    engine = settings.get_by_label("Search engine")
    expect(engine).to_have_value("bm25")
    expect(decisions_field(settings, "API Base URL")).to_have_count(0)
    engine.select_option(label="Decisions API")
    # the prompt field shows the default prompt until the admin types one
    prompt = settings.get_by_placeholder(default_decisions_prompt(admin), exact=True)
    expect(prompt).to_be_visible()
    decisions_field(settings, "API Base URL").fill(f"{listener.base_url}/v1")
    decisions_field(settings, "API Key").fill("sk-typed-in-the-form")
    decisions_field(settings, "Model").fill("judge-2")
    decisions_field(settings, "Relevance threshold").fill("0.65")
    prompt.fill(DECISIONS_PROMPT)
    save_interface(settings)
    expect(settings.page.get_by_text("Settings saved successfully!")).to_be_visible()

    stored = stored_chat_config(admin)
    assert {key: stored[key] for key in DECISIONS_FIELDS} == {
        "TOOL_SEARCH_ENGINE": "decisions",
        "TOOL_SEARCH_DECISIONS_API_BASE_URL": f"{listener.base_url}/v1",
        "TOOL_SEARCH_DECISIONS_API_KEY": "sk-typed-in-the-form",
        "TOOL_SEARCH_DECISIONS_MODEL": "judge-2",
        "TOOL_SEARCH_DECISIONS_THRESHOLD": 0.65,
        "TOOL_SEARCH_DECISIONS_PROMPT": DECISIONS_PROMPT,
    }

    settings = open_interface_settings(page_for(admin, path="/admin/settings/interface"))
    expect(settings.get_by_label("Search engine")).to_have_value("decisions")
    expect(decisions_field(settings, "API Base URL")).to_have_value(f"{listener.base_url}/v1")
    expect(decisions_field(settings, "Model")).to_have_value("judge-2")
    expect(decisions_field(settings, "API Key")).to_have_value("sk-typed-in-the-form")
    expect(decisions_field(settings, "Relevance threshold")).to_have_value("0.65")
    expect(settings.get_by_placeholder(default_decisions_prompt(admin))).to_have_value(
        DECISIONS_PROMPT
    )

    upstream.queue(
        reply.tool_call("search_tools", {"query": "shout loudly"}, call_id="call_s"),
        reply.text("searched"),
    )
    with python_tool(admin, SHOUTING_TOOL) as tool_id, make_user().client() as client:
        _, message = ask(client, "find a shouting tool", tool_ids=[tool_id])
    assert message["content"] == "searched"
    [decided] = listener.requests_to("/v1/decisions")
    assert decided.headers.get("Authorization") == "Bearer sk-typed-in-the-form"
    body = decided.json()
    assert (body["model"], body["input"]) == ("judge-2", "shout loudly")
    assert body["questions"][0]["instructions"].startswith(DECISIONS_PROMPT)


def test_a_decisions_address_with_credentials_is_refused_with_a_message(interface_settings, admin):
    settings = interface_settings
    settings.get_by_label("Search engine").select_option(label="Decisions API")
    decisions_field(settings, "API Base URL").fill("https://user:secret@decide.test/v1")
    decisions_field(settings, "Model").fill("judge-2")
    save_interface(settings)

    expect(
        settings.page.get_by_text(
            "Decisions API requires an HTTP(S) base URL without credentials, query, or fragment."
        )
    ).to_be_visible()
    stored = stored_chat_config(admin)
    assert stored["TOOL_SEARCH_ENGINE"] == "bm25", "a refused setup was saved"
    assert "secret" not in json.dumps(stored)


def test_the_embeddings_engine_names_the_documents_embedding_model_and_is_kept(
    interface_settings, page_for, admin
):
    with admin.client() as client:
        model = client.get("/api/v1/retrieval/embedding").json()["RAG_EMBEDDING_MODEL"]
    assert model, "the instance has no embedding model to name"
    settings = interface_settings
    settings.get_by_label("Search engine").select_option(label="Embeddings")
    expect(settings).to_contain_text(f"Uses the Documents embedding model: {model}.")
    link = settings.get_by_role("link", name="Configure in Documents settings")
    expect(link).to_have_attribute("href", "/admin/settings/documents")
    save_interface(settings)
    expect(settings.page.get_by_text("Settings saved successfully!")).to_be_visible()
    assert stored_chat_config(admin)["TOOL_SEARCH_ENGINE"] == "embeddings"

    reopened = open_interface_settings(page_for(admin, path="/admin/settings/interface"))
    expect(reopened.get_by_label("Search engine")).to_have_value("embeddings")


def test_each_call_of_a_batch_asks_for_approval_in_turn(
    tool_approval_on, page_for, make_user, upstream, shouter
):
    page = page_for(make_user())
    select_model(page, shouter)
    set_tool_approval_mode(page, "Ask for approval")

    prompt = f"shout and mumble {uuid.uuid4().hex[:6]}"
    calls = [
        {"tool_name": "shout", "params": json.dumps({"text": "one"})},
        {"tool_name": "mumble", "params": json.dumps({"text": "TWO"})},
    ]
    upstream.queue(
        reply.tool_call(
            "call_tool", {"calls": calls}, call_id="call_b", match=reply.answering(prompt)
        ),
        reply.text("Shouted, not mumbled.", match=reply.answering(prompt)),
    )
    send(page, prompt)
    for name, word, action in (("shout", "one", "Allow"), ("mumble", "TWO", "Deny")):
        buttons = last_reply(page).get_by_role("button", name=action, exact=True)
        expect(buttons).not_to_have_count(0, timeout=REPLY_TIMEOUT_MS)
        card = last_reply(page).get_by_text(f"Allow {name}?")
        closed_group = last_reply(page).get_by_role("button", name="Toggle details", expanded=False)
        # the pending calls sit in a folded group; clicking an open group closes it
        expect(card.or_(closed_group).first).to_be_visible()
        if not card.is_visible() and closed_group.count():
            closed_group.last.click()
        card.click()
        expect(last_reply(page)).to_contain_text(re.compile(rf"Input\s*text\s*{word}"))
        buttons.last.click()
    expect_reply(page, "Shouted, not mumbled.")
    expect(last_reply(page)).to_contain_text("Denied mumble")

    [combined] = tool_results(upstream, prompt)
    results = json.loads(combined)["results"]
    assert [(entry["tool_name"], entry["status"]) for entry in results] == [
        ("shout", "completed"),
        ("mumble", "rejected"),
    ], combined
    assert results[0]["output"][0]["text"] == "ONE"
