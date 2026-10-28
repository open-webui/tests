"""Journey: a file attached to a new chat's first message lands in that chat's own terminal.

An admin connects a Terminals orchestrator with Chat Uploads on Filesystem and the chat context
on Per chat; the orchestrator stand-in gives every chat a real Open Terminal of its own and the
shared workspace another. A person picks the terminal on the new chat page, attaches a file
from the chat input's menu and sends the first message. The chat has no terminal before it
exists, so the file waits until the message creates the chat, then uploads into that chat's
terminal: it is on that terminal's disk, the shared workspace stays without it, and the stored
message points at the uploaded file. Starting a second chat from New Chat and doing the same
puts the second file in the second chat's terminal and in neither of the others.

Discriminates: passes on dev 226dca18c; with the backend proxy no longer naming the chat to the
orchestrator (`X-Terminal-Context-Id` left out) the files land in the shared workspace and the
test fails; on a build where the new chat page uploads the attached file at once, without
waiting for the chat, the upload is refused for want of a saved chat, the attachment drops out
of the message box and the test fails (the terminal file browser journey still passes there).
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.terminal_orchestrator import SHARED
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from utils.cached_chat import ask, attach
from utils.chat_ui import chat_input
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_URL = re.compile(r"/c/([0-9a-f-]{36})$")


def _pick_per_chat_terminal(page: Page, name: str) -> None:
    """Pick the terminal on the new chat page, where it waits for the chat to exist."""
    expect(chat_input(page)).to_be_visible()
    tooltip_button(page.get_by_role("main"), "Terminal").click()
    page.get_by_role("menu").get_by_role("menuitemradio", name=name).click()
    expect(page.get_by_role("main").get_by_role("button", name=name, exact=True)).to_be_visible()


def _first_message_with_a_file(page: Page, upstream, name: str, content: str) -> str:
    """Attach `name`, send the chat's first message and return the new chat's id."""
    attach(page, name, content)
    expect(page.locator("form").get_by_text(name).first).to_be_visible()
    ask(page, upstream, f"Read {name}", reply.text(f"Read {name}."))
    expect(page).to_have_url(CHAT_URL)
    return CHAT_URL.search(page.url).group(1)


def _stored_attachment(person, chat_id: str) -> dict:
    with person.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}")
    assert stored.status_code == 200, stored.text
    messages = stored.json()["chat"]["history"]["messages"].values()
    files = [file for message in messages if message["role"] == "user" for file in message["files"]]
    assert len(files) == 1, f"the first message should carry one attachment: {files}"
    return files[0]


def test_each_new_chats_first_upload_lands_in_that_chats_own_terminal(
    page_for, make_user, admin, preserve, upstream, terminal_orchestrator
):
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    connection = terminal_orchestrator.connection()
    connection["config"]["access_grants"] = [read_grant(person.id)]
    with admin.client() as client:
        configure_terminals(client, connection)
    tag = uuid.uuid4().hex[:8]
    first_name, second_name = f"first-{tag}.txt", f"second-{tag}.txt"

    page = page_for(person)
    _pick_per_chat_terminal(page, connection["name"])
    first_chat = _first_message_with_a_file(page, upstream, first_name, "harbour log\n")

    first_home = terminal_orchestrator.home(f"chat:{first_chat}")
    assert (first_home / first_name).read_text() == "harbour log\n"
    assert not (terminal_orchestrator.home(SHARED) / first_name).exists(), (
        "the first chat's file landed in the shared workspace"
    )
    attachment = _stored_attachment(person, first_chat)
    assert (attachment["path"], attachment["session_id"]) == (
        str(first_home / first_name),
        first_chat,
    )

    page.get_by_role("link", name="New Chat").first.click()
    expect(page).not_to_have_url(CHAT_URL)
    second_chat = _first_message_with_a_file(page, upstream, second_name, "cargo list\n")

    assert second_chat != first_chat
    second_home = terminal_orchestrator.home(f"chat:{second_chat}")
    assert (second_home / second_name).read_text() == "cargo list\n"
    landed = {
        context
        for context in terminal_orchestrator.contexts()
        if (terminal_orchestrator.home(context) / second_name).exists()
    }
    assert landed == {f"chat:{second_chat}"}, (
        f"the second chat's file landed in {sorted(landed)}, not only in its own terminal"
    )
