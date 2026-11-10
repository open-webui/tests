"""Journey: a terminal scoped to the chat gives every saved chat a machine, and no other chat one.

An admin connects a Terminals orchestrator with Chat Uploads on Filesystem and shares a terminal
scoped to the chat, which gives every chat a machine of its own. A person picks the terminal on
the new chat page, attaches a file from the chat input's menu and sends the first message. The
chat has no machine before it exists, so the file waits until the message creates the chat, then
uploads into that chat's machine: it is on that machine's disk and on no other, and the stored
message points at the uploaded file. Starting a second chat from New Chat and doing the same
puts the second file in the second chat's machine and in no other.

A temporary chat is never stored, so it has no machine: its Terminal menu leaves a terminal
scoped to the chat out, and turning Temporary Chat on drops one picked before (#33866, fix
0df605850; a temporary message failed with it). A terminal with one machine for everyone stays
on offer.

Since 6f2fc7bc7 (dev after 0.13.3, titled `refac`) a terminal on an orchestrator is a template
with a scope, where 0.13.3 had a chat context on the connection; the upload test ran against a
stand-in for the Terminals 0.3 contract until then, and the temporary chat test, from
e2e/chat/test_temporary_chat.py, against a fake connection with that chat context. The real
Terminals service runs on Docker (`harness.terminals`); without its images or Docker the tests
skip.

Discriminates: passes on the dev c15f28649 build with Terminals 0.4.1 and Open Terminal 0.17.7.
With the backend no longer naming the chat to the orchestrator (`X-Terminal-Chat-Id` left out),
in a backend copy, the orchestrator refuses the chat's calls, the first message gets no reply
and the upload test fails. With the terminal list no longer telling the browser a terminal's
scope, in a backend copy, both tests fail: the file is uploaded at once, before the chat exists,
is refused and drops out of the message box, and the terminal stays picked in a temporary chat.
On 0.13.x the temporary chat test failed in a frontend copy with 0df605850 undone.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.terminal_server import read_grant
from harness.terminals import create_template, share_template
from utils.cached_chat import ask, attach, choose_terminal
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_URL = re.compile(r"/c/([0-9a-f-]{36})$")


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


def test_each_new_chats_first_upload_lands_in_that_chats_own_machine(
    page_for, make_user, admin, connect_terminals, terminals, upstream
):
    person = make_user()
    tag = uuid.uuid4().hex[:8]
    name = f"Per chat box {tag}"
    connection = connect_terminals(read_grant(person.id), config={"chat_uploads": "filesystem"})
    with admin.client() as client:
        terminal_id = create_template(client, connection, name=name, scope="chat")["terminal_id"]
        share_template(client, terminal_id, read_grant(person.id))
    first_name, second_name = f"first-{tag}.txt", f"second-{tag}.txt"

    page = page_for(person)
    choose_terminal(page, name)
    first_chat = _first_message_with_a_file(page, upstream, first_name, "harbour log\n")

    assert terminals.holding(first_name) == {f"chat:{first_chat}": "harbour log\n"}, (
        "the first chat's file is not in its own machine alone"
    )
    attachment = _stored_attachment(person, first_chat)
    assert (attachment["path"], attachment["session_id"]) == (
        f"/home/user/{first_name}",
        first_chat,
    )

    page.get_by_role("link", name="New Chat").first.click()
    expect(page).not_to_have_url(CHAT_URL)
    second_chat = _first_message_with_a_file(page, upstream, second_name, "cargo list\n")

    assert second_chat != first_chat
    assert terminals.holding(second_name) == {f"chat:{second_chat}": "cargo list\n"}, (
        "the second chat's file is not in its own machine alone"
    )
    assert terminals.holding(first_name) == {f"chat:{first_chat}": "harbour log\n"}


@pytest.mark.regression
def test_a_temporary_chat_offers_no_per_chat_terminal_and_drops_one_picked_before(
    page_for, make_user, admin, connect_terminals
):
    """#33866: a per chat terminal needs a stored chat, so a temporary message failed with it."""
    person = make_user()
    tag = uuid.uuid4().hex[:8]
    per_chat, shared = f"Per chat shell {tag}", f"Shared shell {tag}"
    connection = connect_terminals(read_grant(person.id))
    with admin.client() as client:
        for name, scope in ((per_chat, "chat"), (shared, None)):
            terminal_id = create_template(client, connection, name=name, scope=scope)["terminal_id"]
            share_template(client, terminal_id, read_grant(person.id))
    page = page_for(person)
    choose_terminal(page, per_chat)
    picked = page.get_by_role("main").get_by_role("button", name=per_chat, exact=True)

    page.get_by_role("button", name="Temporary Chat").click()
    expect(page.locator("#chat-pane").get_by_text("Temporary Chat", exact=True)).to_be_visible()

    expect(picked, "the per chat terminal stayed picked in a temporary chat").to_have_count(0)
    tooltip_button(page.get_by_role("main"), "Terminal").click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitemradio", name=shared)).to_be_visible()
    expect(menu.get_by_role("menuitemradio", name=per_chat)).to_have_count(0)
