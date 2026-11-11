"""Journey: a side chat does what a person does in a normal chat.

A side chat (`/side <question>` beside a chat, or the New Chat tab shortcut) is a chat of its own
with its own message box, saved under its question. A person expects the everyday chat actions to
work there as in the chat: picking another model, attaching a file, an image or a knowledge base
(and seeing its source), a tool and its result, web search, Stop, Copy, code blocks and reasoning.
The saved side chat shares and exports with its exchange from its sidebar menu, and without the
Side Chat permission neither `/side` nor the shortcut opens one. The basic side chat flows
(context, closing, reopening, deleting, dropping a chat beside another) are in test_side_chat.py.
A question the model asks with Ask User shows in the side chat, and answering it finishes the
reply. Since 3b1a95383 the shortcut opens a side chat on an empty new chat too; with no chat so
far the model gets only the side question.

What a side chat does not offer yet on dev c15f28649 is pinned as plain tests, red until it
lands: Regenerate, Edit, Continue Response and rating under its messages
(open-webui/open-webui#34014) and follow-up suggestions (open-webui/open-webui#34015), which pass
on the build of fix PR open-webui/open-webui#34064; an HTML block opening as an artifact
(open-webui/open-webui#34026, red on that build too, the PR leaves it out); and its tab taking
the new title when the chat is renamed in the sidebar, which no filed issue covers and which is
red on that build too.

Discriminates: passes on dev f659c0d12 apart from those gaps. In a backend copy whose chat Stop
stops nothing, that never offers the web tools, never names attached knowledge to the model, whose
timestamp tool returns no timestamp and that applies no model system prompt, the stop, web search,
knowledge, tool and model tests fail on exactly those. The file, image, Copy, code block,
reasoning, share, export, shortcut and permission tests guard frontend behaviour; their mutation
build was held back while release fix builds had the machine, so they are not yet shown red.
The Ask User test is red on the f659c0d12 build, a real bug: the server pauses the side chat's
reply for the question, but the side chat never shows it (open-webui/open-webui#33890, side
chats saved since 3b1a95383).
The empty new chat test passes on dev e8c0ad85e and fails on a copy of the dev e8c0ad85e build that
offers a New Chat tab only on a saved chat, as before 3b1a95383.

A question still waiting in a side chat shows again after a reload and can be dismissed, and
the reply then carries on (5cbd9bc5c, PR open-webui/open-webui#33901). Passes on dev e8c0ad85e
and fails on that build with 5cbd9bc5c reverted (the question never shows).

A command the model runs from a side chat runs on the terminal picked for the chat, and with the
terminal taken off the side chat's model gets no terminal tools. Fails on a copy of the dev
c15f28649 build whose side chat leaves the picked terminal out of its request.
"""

from __future__ import annotations

import base64
import json
import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.python_tools import EVERYONE_READS
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from harness.upstream import MOCK_MODEL_ID
from harness.web_retrieval import RETRIEVAL_CONFIG, save_web_settings, serve_search_results
from utils.cached_chat import calling, choose_terminal, offered_tools
from utils.chat_surface import main_chat
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    chat_input,
    close_menu,
    conversation,
    expect_reply,
    menu_entries,
    send,
    side_chat_panel,
)
from utils.clipboard import read_text
from utils.menus import late_menu_entry
from utils.sidebar_chats import chat_menu, chat_row

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PARENT_CHAT = [
    {"role": "user", "content": "Which pass links Lienz and Matrei?"},
    {"role": "assistant", "content": "The Felbertauern road links them through a tunnel."},
]
SIDE_ANSWER = "The Felbertauern tunnel is 5.3 kilometres long."
# a 1x1 PNG
PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def side_chat_allowed(admin, preserve) -> None:
    """Allow Side Chat on in the default permissions."""
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"].update(side_chat=True)
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


def tagged(text: str) -> str:
    return f"{text} ({uuid.uuid4().hex[:8]})"


def open_chat(page_for, account) -> Page:
    """A page on a stored chat of `account` with one answered question."""
    with account.client() as client:
        chat_id, _ = seed_chat(client, PARENT_CHAT)
    page = page_for(account, path=f"/c/{chat_id}")
    expect_reply(page, PARENT_CHAT[1]["content"])
    return page


def open_side_chat(
    page: Page, upstream, question: str, *replies: reply.Reply, shows: str | None = None
) -> Locator:
    """Ask `question` with `/side` and wait for `shows` (the last scripted reply's text) in its
    panel; returns the panel. With `shows=""` it returns as soon as the panel is there."""
    for scripted in replies:
        scripted.match = reply.answering(question)
        upstream.queue(scripted)
    send(page, f"/side {question}")
    panel = side_chat_panel(page, question)
    shows = replies[-1].content if shows is None else shows
    if shows:
        expect(side_reply(panel)).to_contain_text(shows, timeout=REPLY_TIMEOUT_MS)
    else:
        expect(panel).to_be_visible()
    return panel


def side_input(panel: Locator) -> Locator:
    return panel.get_by_role("textbox")


def side_send(panel: Locator, text: str) -> None:
    expect(side_input(panel)).to_be_visible()
    side_input(panel).click()
    panel.page.keyboard.type(text)
    panel.page.keyboard.press("Enter")


def side_ask(panel: Locator, upstream, question: str, *replies: reply.Reply) -> None:
    """Send `question` in the side chat and wait for its last scripted reply there."""
    for scripted in replies:
        scripted.match = reply.answering(question)
        upstream.queue(scripted)
    side_send(panel, question)
    expect(side_reply(panel)).to_contain_text(replies[-1].content, timeout=REPLY_TIMEOUT_MS)


def side_messages(panel: Locator) -> Locator:
    """The side chat's messages, a question or a reply each, with the buttons under them."""
    return panel.get_by_role("log").get_by_role("listitem")


def side_reply(panel: Locator) -> Locator:
    return side_messages(panel).last


def action(message: Locator, name: str) -> Locator:
    """The button `name` under a side chat message, once it shows on hovering the message."""
    message.hover()
    button = message.get_by_role("button", name=name, exact=True)
    expect(button, f"no {name} button under the side chat message").to_be_visible()
    return button


def side_composer_menu(panel: Locator, button: str) -> Locator:
    expect(side_input(panel)).to_be_visible()
    panel.get_by_role("button", name=button, exact=True).click()
    menu = panel.page.get_by_role("menu")
    expect(menu_entries(menu).first).to_be_visible()
    return menu


def sent_for(upstream, question: str) -> dict:
    """The one provider request that answered `question`."""
    sent = [body for body in upstream.chat_requests() if reply.answering(question)(body)]
    assert sent, f"the provider got no request for {question!r}"
    return sent[-1]


def last_user_content(body: dict) -> object:
    return [entry for entry in reply.conversation(body) if entry["role"] == "user"][-1]["content"]


# --- models, files, tools, search and knowledge -------------------------------------------------


@pytest.fixture
def second_model(admin):
    """A preset on the scripted model with a system prompt of its own; its name, then its prompt."""
    model_id = f"side-{uuid.uuid4().hex[:8]}"
    name = f"Side model {model_id[-8:]}"
    prompt = f"You answer as the {name}."
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": name,
        "meta": {},
        "params": {"system": prompt},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        saved = client.post("/api/v1/models/create", json=form)
        assert saved.status_code == 200, saved.text
    yield model_id, name, prompt
    with admin.client() as client:
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_a_side_chat_answers_on_the_model_picked_in_its_own_selector(
    page_for, make_user, upstream, side_chat_allowed, second_model
):
    model_id, name, prompt = second_model
    page = open_chat(page_for, make_user())
    question = tagged("By the way, how long is that tunnel?")
    panel = open_side_chat(page, upstream, question, reply.text(SIDE_ANSWER))

    panel.get_by_role("button", name=re.compile("Select a model|Selected model")).click()
    page.get_by_role("combobox", name="Search In Models").fill(name)
    page.get_by_role("listbox", name="Available models").get_by_role(
        "option", name=f"Select {name} model"
    ).click()
    follow_up = tagged("And who built it?")
    side_ask(panel, upstream, follow_up, reply.text("It was built in the 1960s."))

    sent = json.dumps(sent_for(upstream, follow_up)["messages"])
    assert prompt in sent, f"the side chat's next message did not go to {name}: {sent}"
    assert prompt not in json.dumps(sent_for(upstream, question)["messages"]), (
        "the side chat's first message already went to the picked model"
    )
    # the main chat keeps its own model
    expect(
        page.get_by_role("main").get_by_role("button", name=re.compile("Selected model")).first
    ).not_to_contain_text(name)


def side_upload(panel: Locator, name: str, content: bytes, mime_type: str) -> None:
    menu = side_composer_menu(panel, "More")
    with panel.page.expect_file_chooser() as chooser:
        menu.get_by_role("menuitem", name="Upload Files").click()
    chooser.value.set_files({"name": name, "mimeType": mime_type, "buffer": content})


def test_a_file_attached_in_a_side_chat_reaches_its_model(
    page_for, make_user, upstream, side_chat_allowed, file_context_on
):
    page = open_chat(page_for, make_user())
    question = tagged("By the way, how long is that tunnel?")
    panel = open_side_chat(page, upstream, question, reply.text(SIDE_ANSWER))
    secret = f"toll-{uuid.uuid4().hex[:8]}"

    with page.expect_response(lambda response: "/process/status" in response.url) as processing:
        side_upload(panel, "tolls.txt", f"The toll code is {secret}.".encode(), "text/plain")
    expect(panel.get_by_role("button", name="Remove File", exact=True)).to_be_visible()
    # the side chat refuses to send while the file is still being processed
    processing.value.finished()
    asked = tagged("What does the toll note say?")
    side_ask(panel, upstream, asked, reply.text("The note names the toll code."))

    sent = json.dumps(sent_for(upstream, asked)["messages"])
    assert secret in sent, f"the side chat's model never got the attached file: {sent}"
    expect(side_messages(panel).nth(-2).get_by_text("tolls.txt")).to_be_visible()


def test_an_image_attached_in_a_side_chat_is_sent_to_its_model(
    page_for, make_user, upstream, side_chat_allowed
):
    page = open_chat(page_for, make_user())
    question = tagged("By the way, how long is that tunnel?")
    panel = open_side_chat(page, upstream, question, reply.text(SIDE_ANSWER))

    side_upload(panel, "portal.png", PIXEL_PNG, "image/png")
    expect(panel.get_by_role("button", name="Show image preview")).to_be_visible()
    asked = tagged("What does the tunnel portal look like?")
    side_ask(panel, upstream, asked, reply.text("A plain concrete portal."))

    content = last_user_content(sent_for(upstream, asked))
    images = (
        [part for part in content if part.get("type") == "image_url"]
        if isinstance(content, list)
        else []
    )
    assert len(images) == 1, f"the side chat's model did not get the image: {content}"


def test_a_tool_runs_in_a_side_chat_and_its_result_shows_there(
    page_for, make_user, upstream, side_chat_allowed
):
    page = open_chat(page_for, make_user())
    question = tagged("By the way, what time is it now?")

    panel = open_side_chat(
        page,
        upstream,
        question,
        reply.tool_call("get_current_timestamp", {}),
        reply.text("It is just past noon."),
    )

    expect(side_reply(panel).get_by_text("View Result from get_current_timestamp")).to_be_visible()
    results = [
        str(entry["content"])
        for entry in sent_for(upstream, question)["messages"]
        if entry.get("role") == "tool"
    ]
    assert any("timestamp" in result for result in results), (
        f"the side chat's model never got the tool's result: {results}"
    )


PLAN_QUESTION = {
    "id": "plan",
    "header": "Plan",
    "question": "Which plan fits you best?",
    "options": [
        {"label": "Basic", "description": "The cheapest one"},
        {"label": "Pro", "description": "The fastest one"},
    ],
}


def test_a_question_the_model_asks_in_a_side_chat_shows_there_and_its_answer_finishes_the_reply(
    page_for, make_user, upstream, side_chat_allowed
):
    """Red on f659c0d12 (open-webui/open-webui#33890): the side chat never shows the question."""
    page = open_chat(page_for, make_user())
    question = tagged("By the way, which plan should I take?")
    panel = open_side_chat(
        page,
        upstream,
        question,
        reply.tool_call("ask_user", {"questions": [PLAN_QUESTION]}),
        reply.text("Pro it is."),
        shows="",
    )

    asked = panel.get_by_text("Which plan fits you best?")
    expect(asked, "the side chat never showed the model's question").to_be_visible(
        timeout=REPLY_TIMEOUT_MS
    )
    card = panel.locator("section", has_text="Which plan fits you best?")
    card.get_by_role("button", name=re.compile(r"^Pro\b")).click()

    expect(side_reply(panel)).to_contain_text("Pro it is.", timeout=REPLY_TIMEOUT_MS)
    results = [
        str(entry["content"])
        for entry in sent_for(upstream, question)["messages"]
        if entry.get("role") == "tool"
    ]
    assert results, "the side chat's model never got the answer"
    assert json.loads(results[-1])["answers"]["plan"]["label"] == "Pro", results


def test_a_question_in_a_side_chat_still_shows_after_a_reload_and_can_be_dismissed(
    page_for, make_user, upstream, side_chat_allowed
):
    page = open_chat(page_for, make_user())
    question = tagged("By the way, which plan suits a family?")
    panel = open_side_chat(
        page,
        upstream,
        question,
        reply.tool_call("ask_user", {"questions": [PLAN_QUESTION]}),
        reply.text("Then I will pick one for you."),
        shows="",
    )
    expect(panel.get_by_text("Which plan fits you best?")).to_be_visible(timeout=REPLY_TIMEOUT_MS)

    page.reload()
    panel = side_chat_panel(page, question)
    asked = panel.get_by_text("Which plan fits you best?")
    expect(asked, "the waiting question was gone after a reload (#33890)").to_be_visible(
        timeout=REPLY_TIMEOUT_MS
    )
    panel.locator("section", has_text="Which plan fits you best?").get_by_role(
        "button", name="Cancel", exact=True
    ).click()

    expect(side_reply(panel)).to_contain_text(
        "Then I will pick one for you.", timeout=REPLY_TIMEOUT_MS
    )
    expect(asked).to_have_count(0)
    results = [
        str(entry["content"])
        for entry in sent_for(upstream, question)["messages"]
        if entry.get("role") == "tool"
    ]
    assert results, "the side chat's model was never told the question was dismissed"


@pytest.fixture
def web_search_on(admin, preserve, listener) -> str:
    """Web search on, its engine listing one page; the page's link."""
    link = f"https://example.org/felbertauern-{uuid.uuid4().hex[:8]}"
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        save_web_settings(client, **serve_search_results(listener, [link]))
    return link


def test_web_search_turned_on_in_a_side_chat_searches_the_web(
    page_for, make_user, upstream, side_chat_allowed, web_search_on
):
    page = open_chat(page_for, make_user())
    question = tagged("By the way, how long is that tunnel?")
    panel = open_side_chat(page, upstream, question, reply.text(SIDE_ANSWER))

    switch = side_composer_menu(panel, "More").get_by_role(
        "menuitemcheckbox", name="Web Search", exact=True
    )
    switch.click()
    expect(switch).to_have_attribute("aria-checked", "true")
    close_menu(page)
    asked = tagged("Search the toll prices")
    side_ask(
        panel,
        upstream,
        asked,
        reply.tool_call("search_web", {"query": "felbertauern toll"}),
        reply.text("The toll is listed online [1]."),
    )

    offered = {tool["function"]["name"] for tool in sent_for(upstream, asked).get("tools") or []}
    assert "search_web" in offered, f"the side chat's model was offered no web search: {offered}"
    results = json.dumps(sent_for(upstream, asked)["messages"])
    assert web_search_on in results, f"the search's result never reached the model: {results}"
    search = side_reply(panel).get_by_role("button", name="View Result from search_web")
    expect(search).to_be_visible()


def test_a_knowledge_base_attached_in_a_side_chat_is_searched_and_cited(
    page_for, admin, make_user, upstream, side_chat_allowed
):
    name = f"Tunnel handbook {uuid.uuid4().hex[:8]}"
    with (
        admin.client() as client,
        knowledge_base(client, name, access_grants=[EVERYONE_READS]) as knowledge_id,
    ):
        add_text_file(client, knowledge_id, "tunnel.txt", "The tunnel opened in 1967.")
        page = open_chat(page_for, make_user())
        question = tagged("By the way, how long is that tunnel?")
        panel = open_side_chat(page, upstream, question, reply.text(SIDE_ANSWER))

        menu = side_composer_menu(panel, "More")
        menu.get_by_role("menuitem", name="Add Context").click()
        menu.get_by_role("menuitem", name="Attach Knowledge").click()
        late_menu_entry(menu, name).first.click()
        expect(panel.locator("form").get_by_text(name).first).to_be_visible()
        asked = tagged("When did it open?")
        side_ask(
            panel,
            upstream,
            asked,
            reply.tool_call("query_knowledge_files", {"query": "opened"}),
            reply.text("It opened in 1967."),
        )

        sent = sent_for(upstream, asked)
        attached = str(last_user_content(sent))
        assert 'type="collection"' in attached and name in attached, (
            f"the side chat's question did not carry the attached knowledge: {attached}"
        )
        assert "1967" in json.dumps(sent["messages"]), (
            "searching the attached knowledge from the side chat found nothing"
        )
        side_reply(panel).get_by_role("button", name="Toggle 1 source").click()
        source = re.compile(r"^View source: \d+ tunnel\.txt$")
        expect(side_reply(panel).get_by_role("button", name=source)).to_be_visible()


def test_stop_ends_a_side_chat_reply_and_keeps_what_was_written(
    page_for, make_user, upstream, side_chat_allowed
):
    page = open_chat(page_for, make_user())
    question = tagged("By the way, tell me the whole story of the tunnel")
    pieces = [f"part-{index} " for index in range(40)]

    panel = open_side_chat(
        page, upstream, question, reply.text(pieces, chunk_delay=0.3), shows="part-1"
    )
    stop = panel.get_by_role("button", name="Stop")
    stop.click()

    expect(stop).to_be_hidden()
    stopped = side_reply(panel).inner_text()
    page.wait_for_timeout(1500)  # five more piece intervals: nothing more arrives
    assert side_reply(panel).inner_text() == stopped
    assert "part-39" not in stopped
    asked = tagged("Shorter please")
    side_ask(panel, upstream, asked, reply.text("It is a long tunnel."))


def test_copy_under_a_side_chat_reply_puts_the_reply_on_the_clipboard(
    page_for, make_user, upstream, side_chat_allowed
):
    page = open_chat(page_for, make_user())
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    question = tagged("By the way, how long is that tunnel?")
    panel = open_side_chat(page, upstream, question, reply.text(SIDE_ANSWER))

    action(side_reply(panel), "Copy").click()

    expect(page.get_by_text("Copying to clipboard was successful!")).to_be_visible()
    assert read_text(page).strip() == SIDE_ANSWER


def test_a_code_block_in_a_side_chat_reply_shows_its_language_and_copies(
    page_for, make_user, upstream, side_chat_allowed
):
    page = open_chat(page_for, make_user())
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    question = tagged("By the way, how do I print the length in Python?")
    code = 'print("5.3 km")'

    panel = open_side_chat(
        page,
        upstream,
        question,
        reply.text(f"```python\n{code}\n```\n\nThat is all."),
        shows="That is all.",
    )

    box = side_reply(panel)
    expect(box.get_by_text("python", exact=True)).to_be_visible()
    expect(box.get_by_text(code, exact=True)).to_be_visible()
    box.get_by_role("button", name="Copy", exact=True).first.click()
    expect(box.get_by_role("button", name="Copied", exact=True)).to_be_visible()
    assert read_text(page) == code


def test_reasoning_in_a_side_chat_reply_is_folded_away_above_the_answer(
    page_for, make_user, upstream, side_chat_allowed
):
    page = open_chat(page_for, make_user())
    question = tagged("By the way, how long is that tunnel?")
    thought = f"weighing the portals {uuid.uuid4().hex[:8]}"

    panel = open_side_chat(page, upstream, question, reply.text(SIDE_ANSWER, reasoning=thought))

    label = side_reply(panel).get_by_text("Thought for")
    expect(label).to_be_visible()
    expect(panel.get_by_text(thought)).to_be_hidden()
    label.click()
    expect(panel.get_by_text(thought)).to_be_visible()


# --- what a side chat does not offer yet ---------------------------------------------------------


def test_an_html_block_in_a_side_chat_reply_opens_as_an_artifact(
    page_for, make_user, upstream, side_chat_allowed
):
    """Red: open-webui/open-webui#34026, not fixed by PR open-webui/open-webui#34064."""
    page = open_chat(page_for, make_user())
    question = tagged("By the way, draw me a sign for the tunnel")
    card = f"Tunnel {uuid.uuid4().hex[:6]}"

    open_side_chat(
        page,
        upstream,
        question,
        reply.text(f"```html\n<h1>{card}</h1>\n```\n\nThat is all."),
        shows="That is all.",
    )

    expect(page.locator("#artifacts-container").frame_locator("iframe").locator("h1")).to_have_text(
        card
    )


def test_regenerate_in_a_side_chat_replaces_the_reply(
    page_for, make_user, upstream, side_chat_allowed
):
    """Red on dev: open-webui/open-webui#34014, fix PR open-webui/open-webui#34064."""
    page = open_chat(page_for, make_user())
    question = tagged("By the way, how long is that tunnel?")
    panel = open_side_chat(page, upstream, question, reply.text("first try"))
    upstream.queue(reply.text("second try", match=reply.answering(question)))

    action(side_reply(panel), "Regenerate").click()
    page.get_by_text("Try Again").click()

    expect(side_reply(panel)).to_contain_text("second try", timeout=REPLY_TIMEOUT_MS)
    expect(side_messages(panel)).to_have_count(2)


def test_editing_a_side_chat_question_resends_it(page_for, make_user, upstream, side_chat_allowed):
    """Red on dev: open-webui/open-webui#34014, fix PR open-webui/open-webui#34064."""
    page = open_chat(page_for, make_user())
    question = tagged("By the way, how lnog is that tunnel?")
    panel = open_side_chat(page, upstream, question, reply.text("answer to the typo"))
    fixed = tagged("By the way, how long is that tunnel?")
    upstream.queue(reply.text(SIDE_ANSWER, match=reply.answering(fixed)))

    asked = side_messages(panel).first
    action(asked, "Edit").click()
    asked.locator("textarea").fill(fixed)
    page.locator("#confirm-edit-message-button").click()

    expect(side_reply(panel)).to_contain_text(SIDE_ANSWER, timeout=REPLY_TIMEOUT_MS)
    assert last_user_content(sent_for(upstream, fixed)) == fixed


def test_continue_in_a_side_chat_carries_the_reply_on(
    page_for, make_user, upstream, side_chat_allowed
):
    """Red on dev: open-webui/open-webui#34014, fix PR open-webui/open-webui#34064."""
    page = open_chat(page_for, make_user())
    question = tagged("By the way, how long is that tunnel?")
    panel = open_side_chat(page, upstream, question, reply.text("The tunnel is"))
    upstream.queue(reply.text(" 5.3 kilometres long.", match=reply.answering(question)))

    action(side_reply(panel), "Continue Response").click()

    expect(side_reply(panel)).to_contain_text(
        "The tunnel is 5.3 kilometres long.", timeout=REPLY_TIMEOUT_MS
    )


def test_a_side_chat_reply_can_be_rated(page_for, make_user, upstream, side_chat_allowed):
    """Red on dev: open-webui/open-webui#34014, fix PR open-webui/open-webui#34064."""
    account = make_user()
    page = open_chat(page_for, account)
    question = tagged("By the way, how long is that tunnel?")
    panel = open_side_chat(page, upstream, question, reply.text(SIDE_ANSWER))

    good = action(side_reply(panel), "Good Response")
    with page.expect_response(
        lambda response: (
            response.request.method == "POST" and response.url.endswith("/evaluations/feedback")
        )
    ):
        good.click()

    with account.client() as client:
        listed = client.get("/api/v1/evaluations/feedbacks/user")
    assert listed.status_code == 200, listed.text
    ratings = [item["data"].get("rating") for item in listed.json()["items"]]
    assert ratings == [1], f"the thumbs up in the side chat was not saved: {ratings}"


@pytest.fixture
def follow_up_generation_on(admin, preserve) -> None:
    preserve("tasks")
    with admin.client() as client:
        current = client.get("/api/v1/tasks/config").json()
        saved = client.post(
            "/api/v1/tasks/config/update", json={**current, "ENABLE_FOLLOW_UP_GENERATION": True}
        )
    assert saved.status_code == 200, saved.text


def test_a_side_chat_reply_offers_follow_ups(
    page_for, make_user, upstream, side_chat_allowed, follow_up_generation_on
):
    """Red on dev: open-webui/open-webui#34015, fix PR open-webui/open-webui#34064."""
    page = open_chat(page_for, make_user())
    question = tagged("By the way, how long is that tunnel?")
    suggestion = f"When was it opened? {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.text(json.dumps({"follow_ups": [suggestion]}), match=reply.answering('"follow_ups"'))
    )

    panel = open_side_chat(page, upstream, question, reply.text(SIDE_ANSWER))

    expect(panel.get_by_text(suggestion)).to_be_visible(timeout=REPLY_TIMEOUT_MS)


# --- the saved side chat, shortcuts and the permission -------------------------------------------


def open_sidebar(page: Page) -> Locator:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    expect(sidebar.get_by_role("button", name="Chats", exact=True)).to_be_visible()
    return sidebar


def test_a_side_chat_renamed_in_the_sidebar_shows_its_new_name_on_its_tab(
    page_for, make_user, upstream, side_chat_allowed
):
    """Red: the tab of an open side chat keeps its old title; no issue filed yet."""
    page = open_chat(page_for, make_user())
    question = tagged("By the way, how long is that tunnel?")
    open_side_chat(page, upstream, question, reply.text(SIDE_ANSWER))
    sidebar = open_sidebar(page)
    renamed = tagged("Tunnel length")

    chat_menu(sidebar, question).get_by_role("menuitem", name="Rename").click()
    page.keyboard.press("Control+A")
    page.keyboard.type(renamed)
    page.keyboard.press("Enter")
    expect(chat_row(sidebar, renamed)).to_be_visible()

    expect(page.get_by_role("tab", name=renamed, exact=True)).to_be_visible()
    expect(side_chat_panel(page, renamed)).to_contain_text(SIDE_ANSWER)


def test_a_side_chat_shared_from_the_sidebar_shows_its_exchange(
    page_for, make_user, upstream, side_chat_allowed
):
    account = make_user()
    page = open_chat(page_for, account)
    question = tagged("By the way, how long is that tunnel?")
    open_side_chat(page, upstream, question, reply.text(SIDE_ANSWER))

    chat_menu(open_sidebar(page), question).get_by_role("menuitem", name="Share").click()
    dialog = page.get_by_role("dialog").filter(has_text="Share Chat")
    dialog.get_by_role("button", name="Copy Link").click()
    link = dialog.get_by_role("link", name="You have shared this chat before")
    expect(link).to_be_visible()

    shared = page_for(account, path=link.get_attribute("href"))
    expect(conversation(shared).get_by_text(question)).to_be_visible()
    expect(conversation(shared).get_by_text(SIDE_ANSWER)).to_be_visible()


def test_a_side_chat_exported_from_the_sidebar_holds_its_exchange(
    page_for, make_user, upstream, side_chat_allowed, tmp_path
):
    page = open_chat(page_for, make_user())
    question = tagged("By the way, how long is that tunnel?")
    open_side_chat(page, upstream, question, reply.text(SIDE_ANSWER))

    menu = chat_menu(open_sidebar(page), question)
    menu.get_by_role("menuitem", name="Download").hover()
    with page.expect_download() as download_info:
        page.get_by_role("menu").get_by_role("menuitem", name="Plain text (.txt)").click()
    saved = tmp_path / download_info.value.suggested_filename
    download_info.value.save_as(saved)

    text = saved.read_text()
    assert f"### USER\n{question}\n\n### ASSISTANT\n{SIDE_ANSWER}" in text, (
        f"the exported side chat lacks its exchange: {text}"
    )


def test_the_new_chat_tab_shortcut_opens_a_side_chat_that_answers(
    page_for, make_user, upstream, side_chat_allowed
):
    page = open_chat(page_for, make_user())
    expect(chat_input(page)).to_be_visible()

    page.keyboard.press("Control+Alt+O")

    panel = page.get_by_role("tabpanel", name="New Chat 1", exact=True)
    expect(side_input(panel)).to_be_visible()
    question = tagged("By the way, how long is that tunnel?")
    upstream.queue(reply.text(SIDE_ANSWER, match=reply.answering(question)))
    side_send(panel, question)
    expect(side_reply(side_chat_panel(page, question))).to_contain_text(
        SIDE_ANSWER, timeout=REPLY_TIMEOUT_MS
    )
    sent = [entry["content"] for entry in reply.conversation(sent_for(upstream, question))]
    assert PARENT_CHAT[1]["content"] in sent, f"the side chat lacked the chat so far: {sent}"


def test_a_side_chat_opens_from_an_empty_new_chat_and_answers_alone(
    page_for, make_user, upstream, side_chat_allowed
):
    account = make_user()
    page = page_for(account)
    expect(chat_input(page)).to_be_visible()

    page.keyboard.press("Control+Alt+O")

    panel = page.get_by_role("tabpanel", name="New Chat 1", exact=True)
    expect(side_input(panel)).to_be_visible()
    question = tagged("Which pass links Lienz and Matrei?")
    upstream.queue(reply.text(SIDE_ANSWER, match=reply.answering(question)))
    side_send(panel, question)
    expect(side_reply(side_chat_panel(page, question))).to_contain_text(
        SIDE_ANSWER, timeout=REPLY_TIMEOUT_MS
    )
    sent = reply.conversation(sent_for(upstream, question))
    exchanged = [entry["content"] for entry in sent if entry["role"] in ("user", "assistant")]
    assert exchanged == [question], f"the side chat of an empty chat was sent {exchanged}"
    with account.client() as client:
        listed = client.get("/api/v1/chats/").json()
    assert [chat["title"] for chat in listed] == [question], (
        f"expected only the side chat to be saved, the account has {listed}"
    )
    # the address follows the selected tab (dev da850248d)
    main_chat(page).show()
    expect(page, "the empty chat became a chat of its own").not_to_have_url(re.compile("/c/"))


@pytest.fixture
def side_chat_denied(admin, preserve) -> None:
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"].update(side_chat=False)
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


def test_without_side_chat_neither_the_command_nor_the_shortcut_opens_one(
    page_for, make_user, upstream, side_chat_denied
):
    page = open_chat(page_for, make_user())
    question = tagged("By the way, how long is that tunnel?")
    asked_before = len(upstream.chat_requests())

    send(page, f"/side {question}")

    expect(page.get_by_text("You do not have permission to use side chat.")).to_be_visible()
    expect(page.get_by_role("tab", name=question)).to_have_count(0)
    page.keyboard.press("Control+Alt+O")
    # the shortcut opens a "New Chat 1" tab when allowed (see the shortcut test above)
    expect(page.get_by_role("tab", name=re.compile("^New Chat"))).to_have_count(0)
    expect(page.get_by_role("tab", name=question)).to_have_count(0)
    later = upstream.chat_requests()[asked_before:]
    assert not any(reply.answering(question)(body) for body in later), (
        "the refused side question was still sent to the model"
    )


def test_a_command_the_model_runs_from_a_side_chat_runs_on_the_terminal_picked_for_the_chat(
    page_for, make_user, admin, preserve, upstream, side_chat_allowed, open_terminal
):
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    connection = open_terminal.connection(config={"access_grants": [read_grant(person.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    page = open_chat(page_for, person)
    choose_terminal(page, connection["name"])
    marker = f"berth-{uuid.uuid4().hex[:8]}.txt"

    question = tagged("By the way, note the berth for me.")
    panel = open_side_chat(
        page,
        upstream,
        question,
        calling("run_command", {"command": f"echo four > {marker}", "wait": 10}),
        reply.text("Noted in the terminal."),
    )

    assert "run_command" in offered_tools(sent_for(upstream, question)), (
        "the side chat's model was not offered the picked terminal's tools"
    )
    written = open_terminal.home / marker
    assert written.exists() and written.read_text() == "four\n", (
        f"the side chat's command did not run on the picked terminal: {marker} is missing there"
    )

    # with the terminal taken off the chat, the side chat's model has no terminal either
    composer = page.get_by_role("main")
    composer.get_by_role("button", name=connection["name"], exact=True).click()
    page.get_by_role("menu").get_by_role("menuitemradio", name=connection["name"]).click()
    expect(composer.get_by_role("button", name=connection["name"], exact=True)).to_have_count(0)
    close_menu(page)
    again = tagged("And which berth was it?")
    side_ask(panel, upstream, again, reply.text("I cannot look without a terminal."))
    assert "run_command" not in offered_tools(sent_for(upstream, again)), (
        "the side chat's model was offered terminal tools with no terminal picked"
    )
