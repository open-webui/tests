"""Journey: the built-in commands of the composer's / menu do what their menu rows say.

Typing / lists the chat commands above the saved prompts. Fork copies the open conversation into
a new chat that answers from it. `/model <id>` typed and sent switches the chat to that model, so
the next message goes there, and an unknown id is refused and leaves the model alone. Model in the
menu opens the model selector, Settings the settings dialog and Temporary turns a new chat
temporary.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose / menu commands and typed /model
do nothing every test fails.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input, expect_reply, replies, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PILOT_PROMPT = "You are the harbour pilot."


def _command_menu(page: Page, typed: str) -> Locator:
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type(f"/{typed}")
    return page.get_by_role("tooltip")


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


def test_fork_copies_the_conversation_into_a_new_chat(page_for, make_user, upstream):
    page = page_for(make_user())
    upstream.queue(reply.text("The ferry left at nine.", match=reply.answering("ferry")))
    send(page, "When did the ferry leave?")
    expect_reply(page, "The ferry left at nine.")
    expect(page).to_have_url(re.compile(r"/c/"))
    original_url = page.url

    _command_menu(page, "fork").get_by_role("button", name=re.compile(r"^Fork:")).click()

    expect(page).not_to_have_url(original_url)
    expect(page).to_have_url(re.compile(r"/c/"))
    expect_reply(page, "The ferry left at nine.")
    upstream.queue(reply.text("Back at five.", match=reply.answering("come back")))
    send(page, "When does it come back?")
    expect_reply(page, "Back at five.")
    sent = upstream.chat_requests()[-1]["messages"]
    assert ("assistant", "The ferry left at nine.") in [
        (entry["role"], entry["content"]) for entry in sent
    ]


def test_a_typed_model_command_switches_the_chats_model(page_for, pilot_model, upstream):
    account, model_id = pilot_model
    page = page_for(account)
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type(f"/model {model_id}")
    page.keyboard.press("Enter")

    expect(page.get_by_text(f"Model switched to: {model_id}")).to_be_visible()
    expect(page.get_by_role("button", name="Selected model: Harbour pilot")).to_be_visible()
    upstream.queue(reply.text("Pilot speaking.", match=reply.answering("who is aboard")))
    send(page, "who is aboard?")
    expect_reply(page, "Pilot speaking.")
    assert PILOT_PROMPT in _system_text(upstream.chat_requests()[-1])


def test_an_unknown_model_in_the_command_is_refused(page_for, make_user, upstream):
    page = page_for(make_user())
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type("/model no-such-model")
    page.keyboard.press("Enter")

    expect(page.get_by_text("Model not found: no-such-model")).to_be_visible()
    expect(page.get_by_role("button", name=f"Selected model: {MOCK_MODEL_ID}")).to_be_visible()
    expect(replies(page)).to_have_count(0)


def test_model_in_the_menu_opens_the_model_selector(page_for, make_user):
    page = page_for(make_user())
    menu = _command_menu(page, "model")
    menu.get_by_role("button", name=re.compile(r"^Model:")).click()

    expect(page.get_by_role("textbox", name="Search In Models")).to_be_visible()


def test_settings_in_the_menu_opens_the_settings(page_for, make_user):
    page = page_for(make_user())
    menu = _command_menu(page, "settings")
    menu.get_by_role("button", name=re.compile(r"^Settings")).click()

    expect(page.get_by_role("dialog").get_by_role("tab", name="General")).to_be_visible()


def test_temporary_in_the_menu_turns_the_new_chat_temporary(page_for, make_user, upstream):
    account = make_user()
    page = page_for(account)
    menu = _command_menu(page, "temporary")
    menu.get_by_role("button", name=re.compile(r"^Temporary:")).click()

    upstream.queue(reply.text("Off the record.", match=reply.answering("quietly")))
    send(page, "quietly, what time is it?")
    expect_reply(page, "Off the record.")
    expect(page).not_to_have_url(re.compile(r"/c/"))
    with account.client() as client:
        assert client.get("/api/v1/chats/", params={"page": 1}).json() == []
