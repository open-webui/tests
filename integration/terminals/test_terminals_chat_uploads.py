"""Journey: a terminal's Chat Uploads set to Filesystem puts a chat's files in that chat's machine.

An admin connects a Terminals orchestrator with Chat Uploads on Filesystem. Chat Uploads is a
setting of the connection, and the terminal list a signed-in person's browser reads says it for
every terminal on that connection, together with the terminal's scope; without the setting, or
with a value the backend does not know, the list says Default. A file the browser uploads for a
chat goes through the terminal proxy naming the chat. A terminal scoped to the chat gives every
chat a machine of its own: the file lands in that chat's machine and in no other, and a second
chat's upload lands in the second chat's machine. A terminal with one machine for everyone
takes every upload in that machine, with or without a chat. A terminal scoped to the chat
refuses an upload for a temporary chat, which has no machine, and starts none for it.

Since 6f2fc7bc7 (dev after 0.13.3, titled `refac`) a terminal on an orchestrator is a template
with a scope, where 0.13.3 had a chat context on the connection; these tests ran against a
stand-in for the Terminals 0.3 contract until then. The real Terminals service runs on Docker
(`harness.terminals`); without its images or Docker the tests skip.

Discriminates: passes on dev c15f28649 with Terminals 0.4.1 and Open Terminal 0.17.7; each
mutation below, made in a backend copy, turns its test red. `terminal_chat_uploads` returning
`default` for every connection fails the listing test; the proxy no longer naming the chat to
the orchestrator (`X-Terminal-Chat-Id` left out) fails the two-chats test (the orchestrator
refuses the upload); a saved chat demanded of a terminal without a scope fails the
shared-machine test (the upload without a chat is refused); the saved-chat check removed fails
the temporary-chat test (the upload is not answered 409 with the reason).
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from harness.terminals import create_template, share_template

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

FILESYSTEM = {"chat_uploads": "filesystem"}


def _terminal_for(admin, connect_terminals, person, scope: str | None) -> str:
    """A terminal of that scope shared with `person`, Chat Uploads on Filesystem; returns its id."""
    connection = connect_terminals(read_grant(person.id), config=FILESYSTEM)
    with admin.client() as client:
        terminal_id = create_template(client, connection, scope=scope)["terminal_id"]
        share_template(client, terminal_id, read_grant(person.id))
    return terminal_id


def _new_chat(client: httpx.Client) -> str:
    created = client.post("/api/v1/chats/new", json={"chat": {"title": "Uploads"}})
    assert created.status_code == 200, f"creating the chat failed: {created.text}"
    return created.json()["id"]


def _upload(client: httpx.Client, terminal_id: str, chat_id: str | None, name: str, content: bytes):
    """Upload into the chat's working directory the way the browser does for a Filesystem upload."""
    base = f"/api/v1/terminals/{terminal_id}"
    session = {"X-Session-Id": chat_id} if chat_id else {}
    cwd = client.get(f"{base}/files/cwd", headers=session)
    if cwd.status_code != 200:
        return cwd
    return client.post(
        f"{base}/files/upload",
        params={"directory": cwd.json()["cwd"]},
        headers=session,
        files={"file": (name, content, "text/plain")},
    )


def test_the_terminal_list_says_where_chat_uploads_go(
    admin, make_user, preserve, connect_terminals, terminals
):
    person = make_user()
    grants = {"access_grants": [read_grant(person.id)]}
    filesystem = connect_terminals(read_grant(person.id), config=FILESYSTEM)
    unset = terminals.connection(config=dict(grants))
    unknown = terminals.connection(config={**grants, "chat_uploads": "somewhere-else"})
    preserve(TERMINAL_SERVERS_CONFIG)
    with admin.client() as client:
        configure_terminals(client, filesystem, unset, unknown)
        # the three connections reach one orchestrator, so each offers its terminal
        template_id = create_template(client, filesystem, scope="chat")["id"]
        for connection in (filesystem, unset, unknown):
            terminal_id = f"template:{connection['id']}:{template_id}"
            share_template(client, terminal_id, read_grant(person.id))

    with person.client() as client:
        listed = client.get("/api/v1/terminals/")
    assert listed.status_code == 200, listed.text
    by_connection = {
        terminal["id"].split(":")[1]: terminal
        for terminal in listed.json()
        if terminal.get("template_id") == template_id
    }
    assert set(by_connection) == {filesystem["id"], unset["id"], unknown["id"]}, (
        f"the terminal is not offered on each of its connections: {listed.json()}"
    )

    assert by_connection[filesystem["id"]]["config"] == {"chat_uploads": "filesystem"}
    assert by_connection[filesystem["id"]]["scope"] == "chat", (
        f"the browser is not told this terminal is per chat: {by_connection[filesystem['id']]}"
    )
    assert by_connection[unset["id"]]["config"] == {"chat_uploads": "default"}
    assert by_connection[unknown["id"]]["config"] == {"chat_uploads": "default"}, (
        f"an unknown Chat Uploads value reached the browser: {by_connection[unknown['id']]}"
    )


def test_each_chats_upload_lands_in_that_chats_own_machine(
    admin, make_user, connect_terminals, terminals
):
    person = make_user()
    terminal_id = _terminal_for(admin, connect_terminals, person, "chat")
    tag = uuid.uuid4().hex[:8]
    first_name, second_name = f"first-{tag}.txt", f"second-{tag}.txt"

    with person.client() as client:
        first_chat, second_chat = _new_chat(client), _new_chat(client)
        first = _upload(client, terminal_id, first_chat, first_name, b"first chat\n")
        second = _upload(client, terminal_id, second_chat, second_name, b"second chat\n")

    assert first.status_code == 200, f"the first chat's upload failed: {first.text}"
    assert second.status_code == 200, f"the second chat's upload failed: {second.text}"
    assert terminals.holding(first_name) == {f"chat:{first_chat}": "first chat\n"}, (
        "the first chat's file is not in its own machine alone"
    )
    assert terminals.holding(second_name) == {f"chat:{second_chat}": "second chat\n"}, (
        "the second chat's file is not in its own machine alone"
    )


def test_a_terminal_with_one_machine_for_all_takes_every_upload_in_that_machine(
    admin, make_user, connect_terminals, terminals
):
    person = make_user()
    terminal_id = _terminal_for(admin, connect_terminals, person, None)
    tag = uuid.uuid4().hex[:8]
    chat_name, chatless_name = f"chat-{tag}.txt", f"chatless-{tag}.txt"

    with person.client() as client:
        in_a_chat = _upload(client, terminal_id, _new_chat(client), chat_name, b"from a chat\n")
        without_a_chat = _upload(client, terminal_id, None, chatless_name, b"from no chat\n")

    assert in_a_chat.status_code == 200, f"the chat's upload failed: {in_a_chat.text}"
    assert without_a_chat.status_code == 200, (
        f"the upload without a chat failed: {without_a_chat.status_code} {without_a_chat.text}"
    )
    assert terminals.holding(chat_name) == {"default": "from a chat\n"}
    assert terminals.holding(chatless_name) == {"default": "from no chat\n"}
    assert [machine["context_id"] for machine in terminals.instances()] == ["default"], (
        f"the terminal started more than its one machine: {terminals.instances()}"
    )


def test_a_per_chat_terminal_refuses_an_upload_for_a_temporary_chat(
    admin, make_user, connect_terminals, terminals
):
    person = make_user()
    terminal_id = _terminal_for(admin, connect_terminals, person, "chat")
    name = f"temporary-{uuid.uuid4().hex[:8]}.txt"

    with person.client() as client:
        refused = _upload(client, terminal_id, f"temporary:{uuid.uuid4()}", name, b"nowhere\n")

    assert (refused.status_code, refused.json().get("detail")) == (
        409,
        "This terminal requires a saved chat",
    ), f"a temporary chat's upload was not refused: {refused.status_code} {refused.text}"
    assert terminals.instances() == [], (
        f"a temporary chat's upload started a machine: {terminals.instances()}"
    )
