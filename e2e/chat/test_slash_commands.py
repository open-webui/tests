"""Journey: the built-in commands of the composer's / menu do what their menu rows say.

Typing / lists the chat commands above the saved prompts. Fork copies the open conversation into
a new chat that answers from it. `/model <id>` typed and sent switches the chat to that model, so
the next message goes there, and an unknown id is refused and leaves the model alone. Model in the
menu opens the model selector, Settings the settings dialog, Compare text the compare view with an
empty original once the typed `/compare` is taken out of the message, and Temporary turns a new
chat temporary.

Status, picked or typed, opens the chat's Status panel and Compact, typed, has the chat's earlier
turns summarized; neither is sent to the model.

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`), and the Fork,
Status, Compact and typed /model tests in every way a side chat starts and every state it can be
in. In a side chat /status and /fork were offered and did nothing, /compact was not offered, and
all of them typed went to the model as a message (open-webui/open-webui#34018, fix PR
open-webui/open-webui#34064). On dev e791cbdde, which holds that PR, every side chat variant passes
but Fork's: the fork is made and never opened, so the side chat stays on the original
(open-webui/open-webui#34182).

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose / menu commands and typed /model
do nothing every test fails. The Compare text test passes on the dev 3b8205505 build and fails in
a build of it whose / menu leaves out Compare text. For the side chats the dev c15f28649 build is
the mutant of #34018.
"""

from __future__ import annotations

import json
import re
import threading
import uuid

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import (
    EVERYWHERE,
    MAIN_CHAT,
    REPLY_TIMEOUT_MS,
    SIDE_CHAT,
    TEMPORARY_SIDE,
    ChatSurface,
    side_tab_name,
    stored_chat_holding,
)
from utils.chat_ui import chat_input, expect_reply, replies, send
from utils.model_selector import model_search

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PILOT_PROMPT = "You are the harbour pilot."
CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
SUMMARY = "The ferry timetable was discussed."
# a saved chat only: the main chat offers neither Fork nor a manual Compact in a temporary chat
SAVED_CHAT_PLACES = tuple(place for place in EVERYWHERE if place != TEMPORARY_SIDE)


def _command_menu(chat: ChatSurface, typed: str) -> Locator:
    expect(chat_input(chat)).to_be_visible()
    chat_input(chat).click()
    chat.page.keyboard.type(f"/{typed}")
    return chat.page.get_by_role("tooltip")


def _type_and_send(chat: ChatSurface, text: str) -> None:
    expect(chat_input(chat)).to_be_visible()
    chat_input(chat).click()
    chat.page.keyboard.type(text)
    chat.page.keyboard.press("Enter")


def _ask(chat: ChatSurface, upstream, question: str, answer: str) -> None:
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    send(chat, question)
    expect_reply(chat, answer)
    # the answer shows before the reply ends, and /compact waits for the end
    expect(chat.stop_button).to_be_hidden(timeout=REPLY_TIMEOUT_MS)


def _sent_as_a_message(upstream, text: str) -> list[dict]:
    return [body for body in upstream.chat_requests() if reply.answering(text)(body)]


def _system_text(request: dict) -> str:
    return "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )


@pytest.fixture
def pilot_model(make_user):
    """A fresh admin and a model of theirs with its own system prompt; yields (account, id)."""
    account = make_user(role="admin")
    model_id = f"pilot-{uuid.uuid4().hex[:6]}"
    form = {
        "id": model_id,
        "name": "Harbour pilot",
        "base_model_id": MOCK_MODEL_ID,
        "meta": {},
        "params": {"system": PILOT_PROMPT},
    }
    with account.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        yield account, model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.mark.chat_surfaces(*SAVED_CHAT_PLACES)
def test_fork_copies_the_conversation_into_a_new_chat(chat_for, make_user, upstream):
    """In a side chat: open-webui/open-webui#34018; on dev e791cbdde and 782f4b705 the fork is
    made but never opened (open-webui/open-webui#34182). Since c63565927 a tab beside a saved
    chat leaves the address on that chat, so a side tab shows the fork by its title."""
    account = make_user()
    chat = chat_for(account)
    page = chat.page
    question = f"When did the ferry leave? ({uuid.uuid4().hex[:6]})"
    upstream.queue(reply.text("The ferry left at nine.", match=reply.answering(question)))
    send(chat, question)
    expect_reply(chat, "The ferry left at nine.")
    expect(page).to_have_url(re.compile(r"/c/"))
    original_id = stored_chat_holding(account, question)["id"]

    with page.expect_response(
        lambda response: (
            response.request.method == "POST"
            and response.url.endswith(f"/api/v1/chats/{original_id}/fork")
        )
    ) as answer:
        _command_menu(chat, "fork").get_by_role("button", name=re.compile(r"^Fork:")).click()
    assert answer.value.status == 200, f"forking the chat got {answer.value.status}"
    fork = answer.value.json()
    assert fork["id"] != original_id, "the fork kept the original chat's id"

    if chat.is_side:
        # a tab beside a saved chat leaves the address on that chat (c63565927)
        expect(chat.tab, f"the {chat.place} did not open the fork").to_have_accessible_name(
            side_tab_name(fork["title"])
        )
    else:
        expect(page).to_have_url(re.compile(rf"/c/{fork['id']}$"))
    expect_reply(chat, "The ferry left at nine.")
    follow_up = f"When does it come back? ({uuid.uuid4().hex[:6]})"
    upstream.queue(reply.text("Back at five.", match=reply.answering(follow_up)))
    send(chat, follow_up)
    expect_reply(chat, "Back at five.")
    sent = upstream.chat_requests()[-1]["messages"]
    assert ("assistant", "The ferry left at nine.") in [
        (entry["role"], entry["content"]) for entry in sent
    ]
    answered_in = stored_chat_holding(account, follow_up)
    assert answered_in["id"] == fork["id"], (
        f"the next message went to {answered_in['title']!r}, not to the fork {fork['title']!r}"
    )


# one model picked: the command replaces the chat's pick, whether it held one model or several
@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_a_typed_model_command_switches_the_chats_model(chat_for, pilot_model, upstream):
    """In a side chat: open-webui/open-webui#34018."""
    account, model_id = pilot_model
    chat = chat_for(account)
    page = chat.page
    _type_and_send(chat, f"/model {model_id}")

    expect(page.get_by_text(f"Model switched to: {model_id}")).to_be_visible()
    expect(chat.panel.get_by_role("button", name="Selected model: Harbour pilot")).to_be_visible()
    upstream.queue(reply.text("Pilot speaking.", match=reply.answering("who is aboard")))
    send(chat, "who is aboard?")
    expect_reply(chat, "Pilot speaking.")
    assert PILOT_PROMPT in _system_text(upstream.chat_requests()[-1])


# the refusal is the typed /model's other branch, which the test above crosses with every place
def test_an_unknown_model_in_the_command_is_refused(chat_for, make_user, upstream):
    chat = chat_for(make_user())
    page = chat.page
    _type_and_send(chat, "/model no-such-model")

    expect(page.get_by_text("Model not found: no-such-model")).to_be_visible()
    expect(
        chat.panel.get_by_role("button", name=f"Selected model: {MOCK_MODEL_ID}")
    ).to_be_visible()
    expect(replies(chat)).to_have_count(0)


def test_model_in_the_menu_opens_the_model_selector(chat_for, make_user):
    chat = chat_for(make_user())
    menu = _command_menu(chat, "model")
    menu.get_by_role("button", name=re.compile(r"^Model:")).click()

    expect(model_search(chat.page)).to_be_visible()


def test_settings_in_the_menu_opens_the_settings(chat_for, make_user):
    chat = chat_for(make_user())
    menu = _command_menu(chat, "settings")
    menu.get_by_role("button", name=re.compile(r"^Settings")).click()

    expect(chat.page.get_by_role("dialog").get_by_role("tab", name="General")).to_be_visible()


def test_compare_text_in_the_menu_opens_the_compare_view(chat_for, make_user):
    chat = chat_for(make_user())
    menu = _command_menu(chat, "compare")
    menu.get_by_role("button", name="Compare text").click()

    compare = chat.page.get_by_role("dialog", name="Compare text")
    expect(compare.get_by_role("textbox", name="Original", exact=True)).to_have_text("")
    expect(compare.get_by_role("button", name="Use in prompt")).to_be_visible()
    expect(chat_input(chat)).to_have_text("")


def test_temporary_in_the_menu_turns_the_new_chat_temporary(chat_for, make_user, upstream):
    account = make_user()
    chat = chat_for(account)
    page = chat.page
    menu = _command_menu(chat, "temporary")
    menu.get_by_role("button", name=re.compile(r"^Temporary:")).click()

    upstream.queue(reply.text("Off the record.", match=reply.answering("quietly")))
    send(chat, "quietly, what time is it?")
    expect_reply(chat, "Off the record.")
    expect(page).not_to_have_url(re.compile(r"/c/"))
    with account.client() as client:
        assert client.get("/api/v1/chats/", params={"page": 1}).json() == []


@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_status_picked_or_typed_shows_the_chats_status_and_sends_nothing(
    chat_for, make_user, upstream
):
    """In a side chat: open-webui/open-webui#34018."""
    chat = chat_for(make_user())
    _ask(chat, upstream, "When did the ferry leave?", "The ferry left at nine.")
    status = chat.panel.get_by_text(re.compile(r"^Context\b"))
    close = chat.panel.get_by_role("button", name="Close", exact=True)

    _command_menu(chat, "status").get_by_role("button", name=re.compile(r"^Status:")).click()
    expect(status, f"/status picked in the {chat.place} showed no status").to_be_visible()
    expect(chat.panel.get_by_text("Chat ID", exact=True)).to_be_visible()
    close.click()
    expect(status).to_be_hidden()

    _type_and_send(chat, "/status")
    expect(status, f"/status typed in the {chat.place} showed no status").to_be_visible()
    expect(chat_input(chat)).to_have_text("")
    sent = _sent_as_a_message(upstream, "/status")
    assert not sent, f"/status typed in the {chat.place} went to the model: {sent}"


@pytest.fixture
def compaction(admin, preserve):
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        updated = client.post(CHAT_CONFIG[1], json={**current, "ENABLE_CONTEXT_COMPACTION": True})
    assert updated.status_code == 200, updated.text


@pytest.mark.chat_surfaces(*SAVED_CHAT_PLACES)
def test_compact_is_offered_and_typed_summarizes_the_earlier_turns(
    chat_for, make_user, upstream, compaction
):
    """In a side chat: open-webui/open-webui#34018."""
    chat = chat_for(make_user())
    question = f"When did the ferry leave? ({uuid.uuid4().hex[:8]})"
    _ask(chat, upstream, question, "The ferry left at nine.")

    def is_summary_of_the_question(body: dict) -> bool:
        return not body.get("stream") and question in json.dumps(body["messages"])

    upstream.queue(reply.text(SUMMARY, match=is_summary_of_the_question))
    menu = _command_menu(chat, "compact")
    expect(
        menu.get_by_role("button", name=re.compile(r"^Compact:")),
        f"/compact is not offered in the {chat.place}",
    ).to_be_enabled()
    chat_input(chat).press("Escape")
    expect(chat_input(chat)).to_have_text("/compact")
    with chat.page.expect_response(lambda response: response.url.endswith("/compact")) as done:
        chat_input(chat).press("Enter")

    assert done.value.json().get("compacted") is True, done.value.text()
    summaries = [body for body in upstream.chat_requests() if is_summary_of_the_question(body)]
    assert summaries, f"/compact typed in the {chat.place} had nothing summarized"
    sent = _sent_as_a_message(upstream, "/compact")
    assert not sent, f"/compact typed in the {chat.place} went to the model: {sent}"


@pytest.mark.chat_surfaces(MAIN_CHAT, SIDE_CHAT)
def test_compact_and_fork_in_a_menu_opened_during_the_reply_are_usable_once_it_ends(
    chat_for, make_user, upstream, compaction
):
    """The / menu opened while a reply runs kept Compact and Fork greyed out after the reply ended
    (open-webui/open-webui#35026)."""
    chat = chat_for(make_user())
    question = f"When did the ferry leave? ({uuid.uuid4().hex[:8]})"
    rest_of_reply = threading.Event()
    upstream.queue(
        reply.text(
            ["The ferry ", "left at nine."],
            match=reply.answering(question),
            hold=rest_of_reply,
            hold_after=1,
        )
    )
    send(chat, question)
    expect(chat.stop_button).to_be_visible()
    menu = _command_menu(chat, "")
    compact = menu.get_by_role("button", name=re.compile(r"^Compact:"))
    fork = menu.get_by_role("button", name=re.compile(r"^Fork:"))
    expect(compact, "Compact is usable while the reply still runs").to_be_disabled()
    expect(fork, "Fork is usable while the reply still runs").to_be_disabled()

    rest_of_reply.set()
    expect_reply(chat, "The ferry left at nine.")
    expect(chat.stop_button).to_be_hidden(timeout=REPLY_TIMEOUT_MS)
    expect(fork, "Fork stays greyed out after the reply ended").to_be_enabled()
    expect(compact, "Compact stays greyed out after the reply ended").to_be_enabled()

    def is_summary_of_the_question(body: dict) -> bool:
        return not body.get("stream") and question in json.dumps(body["messages"])

    upstream.queue(reply.text(SUMMARY, match=is_summary_of_the_question))
    with chat.page.expect_response(lambda response: response.url.endswith("/compact")) as done:
        compact.click()
    assert done.value.json().get("compacted") is True, done.value.text()


@pytest.mark.chat_surfaces(MAIN_CHAT)
def test_compacts_fill_in_a_menu_opened_during_the_reply_follows_it(
    chat_for, make_user, upstream, compaction
):
    """The / menu opened while a reply ran kept Compact's "% full" at the value it had when it
    opened, until it was opened again (open-webui/open-webui#35033)."""
    account = make_user(role="admin")
    model_id = f"window-{uuid.uuid4().hex[:6]}"
    with account.client() as client:
        form = {
            "id": model_id,
            "name": "Small window",
            "base_model_id": MOCK_MODEL_ID,
            "meta": {},
            "params": {"compact_token_threshold": 2000},
        }
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        picked = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"models": [model_id]}}
        )
        assert picked.status_code == 200, picked.text
    chat = chat_for(account)
    expect(chat.panel.get_by_role("button", name="Selected model: Small window")).to_be_visible()
    question = f"When did the ferry leave? ({uuid.uuid4().hex[:8]})"
    rest_of_reply = threading.Event()
    long_rest = "left at nine and came back at five. " * 25
    upstream.queue(
        reply.text(
            ["The ferry ", long_rest],
            match=reply.answering(question),
            hold=rest_of_reply,
            hold_after=1,
        )
    )
    send(chat, question)
    expect(chat.stop_button).to_be_visible()

    def fill_of(menu: Locator) -> Locator:
        compact = menu.get_by_role("button", name=re.compile(r"^Compact:"))
        return compact.get_by_text(re.compile(r"^\d+% full$"))

    fill = fill_of(_command_menu(chat, ""))
    expect(fill, "Compact shows no fill while the reply runs").to_be_visible()
    fill_at_open = fill.inner_text()

    rest_of_reply.set()
    expect(chat.last_reply).to_contain_text("came back at five.")
    expect(chat.stop_button).to_be_hidden(timeout=REPLY_TIMEOUT_MS)
    expect(fill, "Compact's fill in the open menu did not move with the reply").not_to_have_text(
        fill_at_open
    )
    fill_after_reply = fill.inner_text()

    chat_input(chat).press("Escape")
    chat_input(chat).press("Backspace")
    expect(chat_input(chat)).to_have_text("")
    reopened = fill_of(_command_menu(chat, ""))
    expect(
        reopened, f"the open menu showed {fill_after_reply} after the reply, a new one differs"
    ).to_have_text(fill_after_reply)
    assert int(fill_after_reply.split("%")[0]) > int(fill_at_open.split("%")[0]), (
        f"the fill went from {fill_at_open} to {fill_after_reply} over a long reply"
    )
