"""Journey: pinned models and notes sit in the sidebar, and a chat row previews its last messages.

A pinned model opens a new chat with that model, and Shift-hover offers to unpin it; the pin is
stored in the account's settings and survives a reload. A note pinned from the Notes page shows
under Notes in the sidebar, opens on a click and leaves the sidebar when unpinned. Hovering a chat
row shows a floating preview with the chat's title and messages, unless the account switched
Chat Hover Previews off in its interface settings.

Discriminates: passes on dev 176d31d1d; in a frontend copy each test fails when its behaviour is
cut: the pinned model link not naming its model, the unpin button not saving the settings, the
pinned note link not naming its note, the note's Unpin button not unpinning, the preview showing
no messages and the preview ignoring the hover-preview setting.
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import chat
from harness import upstream as reply
from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input, expect_reply, send
from utils.notes_page import note_row
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


PINNED_MODEL_SYSTEM_PROMPT = "You are the pinned model."


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _sidebar(page: Page) -> Locator:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    return page.get_by_role("navigation", name="Chat history")


def _pinned_models(actor: Actor) -> list[str]:
    with actor.client() as client:
        stored = client.get("/api/v1/users/user/settings")
    stored.raise_for_status()
    return stored.json()["ui"].get("pinnedModels", [])


def _save_ui_settings(actor: Actor, ui: dict) -> None:
    with actor.client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json={"ui": ui})
    assert saved.status_code == 200, saved.text


@pytest.fixture
def pinned_model(make_user) -> tuple[Actor, str]:
    """A fresh account with a workspace model on the scripted model, already pinned."""
    owner = make_user(role="admin")
    model_id = _unique("pinned-model")
    with owner.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "name": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "meta": {},
                "params": {"system": PINNED_MODEL_SYSTEM_PROMPT},
            },
        )
        assert created.status_code == 200, created.text
    _save_ui_settings(owner, {"pinnedModels": [model_id]})
    return owner, model_id


def test_a_pinned_model_opens_a_chat_with_that_model(page_for, pinned_model, upstream):
    owner, model_id = pinned_model
    page = page_for(owner)
    page.get_by_role("button", name=re.compile("^Selected model")).wait_for()
    page.get_by_role("link", name="New Chat").first.click()
    sidebar = _sidebar(page)

    sidebar.get_by_role("link", name=model_id).click()

    expect(page).to_have_url(f"{owner.base_url}/?model={model_id}")
    expect(page.get_by_role("button", name=f"Selected model: {model_id}")).to_be_visible()
    prompt = "which model is this?"
    upstream.queue(reply.text("answered by the pin", match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, "answered by the pin")
    sent = [body for body in upstream.chat_requests() if reply.answering(prompt)(body)]
    assert PINNED_MODEL_SYSTEM_PROMPT in json.dumps(sent[-1]["messages"])


def test_shift_hover_unpins_a_model_for_good(page_for, pinned_model):
    owner, model_id = pinned_model
    page = page_for(owner)
    sidebar = _sidebar(page)
    link = sidebar.get_by_role("link", name=model_id)
    expect(link).to_be_visible()

    link.hover()
    page.keyboard.down("Shift")
    tooltip_button(sidebar.locator("#pinned-models-list"), "Unpin").click()
    page.keyboard.up("Shift")

    expect(link).to_have_count(0)
    expect(sidebar.get_by_role("button", name="Models", exact=True)).to_have_count(0)
    assert _pinned_models(owner) == []
    page.reload()
    expect(chat_input(page)).to_be_visible()
    expect(link).to_have_count(0)


@pytest.fixture
def noted(make_user) -> tuple[Actor, str, str]:
    """A fresh account with one note that is not pinned."""
    author = make_user()
    title = _unique("Pinned note")
    with author.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": title, "data": {"content": {"md": "kept close"}}, "access_grants": []},
        )
        assert created.status_code == 200, created.text
    return author, created.json()["id"], title


def _is_pinned(actor: Actor, note_id: str) -> bool:
    with actor.client() as client:
        pinned = client.get("/api/v1/notes/pinned")
    pinned.raise_for_status()
    return note_id in [note["id"] for note in pinned.json()]


def _pin_from_notes_page(page: Page, title: str) -> None:
    page.goto("/notes")
    card = note_row(page, title)
    card.get_by_role("button", name="Note Menu").first.click()
    page.get_by_role("menu").get_by_role("menuitem", name="Pin to Sidebar").click()


def _notes_section(sidebar: Locator) -> Locator:
    heading = sidebar.get_by_role("button", name="Notes", exact=True)
    expect(heading).to_be_visible()
    heading.click()
    return sidebar.locator("#sidebar-pinned-notes-content")


def test_a_note_pinned_from_the_notes_page_shows_in_the_sidebar_and_opens(page_for, noted):
    author, note_id, title = noted
    page = page_for(author)
    _pin_from_notes_page(page, title)
    assert _is_pinned(author, note_id)

    section = _notes_section(_sidebar(page))
    section.get_by_role("link", name=title).click()

    expect(page).to_have_url(f"{author.base_url}/notes/{note_id}")
    expect(page.get_by_role("main").get_by_role("textbox", name="Title")).to_have_value(title)


def test_an_unpinned_note_leaves_the_sidebar(page_for, noted):
    author, note_id, title = noted
    with author.client() as client:
        assert client.post(f"/api/v1/notes/{note_id}/pin").status_code == 200
    page = page_for(author)
    sidebar = _sidebar(page)
    section = _notes_section(sidebar)
    link = section.get_by_role("link", name=title)
    expect(link).to_be_visible()

    link.hover()
    section.get_by_role("button", name="Unpin").click()

    expect(sidebar.get_by_role("button", name="Notes", exact=True)).to_have_count(0)
    assert not _is_pinned(author, note_id)
    page.reload()
    expect(chat_input(page)).to_be_visible()
    expect(page.get_by_role("link", name=title)).to_have_count(0)


FIRST_QUESTION = "what do owls eat?"
FIRST_ANSWER = "owls eat mice and voles"
SECOND_QUESTION = "and at night?"
SECOND_ANSWER = "the same, they hunt after dark"


@pytest.fixture
def talker(make_user, upstream) -> tuple[Actor, str]:
    """A fresh account with one stored chat of two exchanges."""
    account = make_user()
    upstream.queue(
        reply.text(FIRST_ANSWER, match=reply.answering(FIRST_QUESTION)),
        reply.text(SECOND_ANSWER, match=reply.answering(SECOND_QUESTION)),
    )
    with account.client() as client:
        first, _ = chat.ask(client, FIRST_QUESTION)
        chat.ask(
            client, SECOND_QUESTION, chat_id=first.chat_id, parent_id=first.assistant_message_id
        )
        renamed = client.post(f"/api/v1/chats/{first.chat_id}", json={"chat": {"title": "Owls"}})
        assert renamed.status_code == 200, renamed.text
    return account, first.chat_id


def _hover_chat_row(page: Page, title: str) -> Locator:
    sidebar = _sidebar(page)
    sidebar.get_by_role("button", name="Chats", exact=True).wait_for()
    row = sidebar.get_by_role("button", name=title)
    expect(row).to_be_visible()
    row.hover()
    return row


def _hover_preview(page: Page, chat_id: str) -> Locator:
    """The floating preview's message list, which carries the chat's id (b4db56ae9 renamed it)."""
    return page.locator(f"#chat-preview-{chat_id}")


def test_hovering_a_chat_previews_its_messages(page_for, talker):
    account, chat_id = talker
    page = page_for(account)
    _hover_chat_row(page, "Owls")

    preview = _hover_preview(page, chat_id)
    expect(preview).to_contain_text(FIRST_QUESTION)
    expect(preview).to_contain_text(FIRST_ANSWER)
    expect(preview).to_contain_text(SECOND_QUESTION)
    expect(preview).to_contain_text(SECOND_ANSWER)


def test_no_preview_shows_once_chat_hover_previews_is_switched_off(page_for, talker):
    account, chat_id = talker
    _save_ui_settings(account, {"chatHoverPreview": False})
    page = page_for(account)
    row = _hover_chat_row(page, "Owls")

    expect(row).to_be_visible()
    page.wait_for_timeout(1000)  # the preview opens 300 ms after the hover
    expect(_hover_preview(page, chat_id)).to_have_count(0)
