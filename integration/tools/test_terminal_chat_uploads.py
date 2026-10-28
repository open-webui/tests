"""Journey: a terminal's Chat Uploads set to Filesystem puts a chat's files in that chat's terminal.

An admin connects a Terminals orchestrator with Chat Uploads on Filesystem. The terminal list a
signed-in person's browser reads says so, and with the chat context on Per chat it says each
chat gets a terminal of its own; without the setting, or with a value the backend does not know,
the list says Default. A file the browser uploads for a chat goes through the terminal proxy
naming the chat, and the orchestrator stand-in gives every chat context a real Open Terminal of
its own: on Per chat the file lands in that chat's terminal and in no other, and a second chat's
upload lands in the second chat's terminal; with the chat context left on Default it lands in
the shared terminal. A per-chat terminal refuses an upload for a temporary chat, which has no
terminal of its own, and nothing lands anywhere.

Discriminates: passes on dev 226dca18c; each mutation below, made in a backend copy, turns
exactly one test red. `terminal_chat_uploads` returning `default` for every connection fails the
listing test; the proxy no longer adding the context header (`X-Terminal-Context-Id`) fails the
two-chats test (the files land in the shared terminal); `terminal_context_id` naming the chat
whatever the connection's chat context fails the shared-terminal test (the file lands in a chat
terminal); the proxy's saved-chat check removed fails the temporary-chat test (the upload
reaches the shared terminal).
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.terminal_orchestrator import SHARED, TerminalOrchestrator
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def _connect(admin, preserve, orchestrator: TerminalOrchestrator, person, **fields) -> dict:
    preserve(TERMINAL_SERVERS_CONFIG)
    connection = orchestrator.connection(**fields)
    connection["config"]["access_grants"] = [read_grant(person.id)]
    with admin.client() as client:
        configure_terminals(client, connection)
    return connection


def _new_chat(client: httpx.Client) -> str:
    created = client.post("/api/v1/chats/new", json={"chat": {"title": "Uploads"}})
    assert created.status_code == 200, f"creating the chat failed: {created.text}"
    return created.json()["id"]


def _upload(client: httpx.Client, connection: dict, chat_id: str, name: str, content: bytes):
    """Upload into the chat's working directory the way the browser does for a Filesystem upload."""
    base = f"/api/v1/terminals/{connection['id']}"
    session = {"X-Session-Id": chat_id}
    cwd = client.get(f"{base}/files/cwd", headers=session)
    if cwd.status_code != 200:
        return cwd
    return client.post(
        f"{base}/files/upload",
        params={"directory": cwd.json()["cwd"]},
        headers=session,
        files={"file": (name, content, "text/plain")},
    )


def _homes_holding(orchestrator: TerminalOrchestrator, name: str) -> set[str]:
    return {
        context
        for context in orchestrator.contexts()
        if (orchestrator.home(context) / name).exists()
    }


def test_the_terminal_list_says_where_chat_uploads_go(
    admin, make_user, preserve, terminal_orchestrator
):
    person = make_user()
    per_chat = terminal_orchestrator.connection(name="Per chat uploads")
    unset = terminal_orchestrator.connection(name="Unset uploads", per_chat=False)
    del unset["config"]["chat_uploads"]
    unknown = terminal_orchestrator.connection(name="Unknown uploads", per_chat=False)
    unknown["config"]["chat_uploads"] = "somewhere-else"
    preserve(TERMINAL_SERVERS_CONFIG)
    for connection in (per_chat, unset, unknown):
        connection["config"]["access_grants"] = [read_grant(person.id)]
    with admin.client() as client:
        configure_terminals(client, per_chat, unset, unknown)

    with person.client() as client:
        listed = client.get("/api/v1/terminals/")
    assert listed.status_code == 200, listed.text
    by_name = {terminal["name"]: terminal for terminal in listed.json()}

    assert by_name["Per chat uploads"]["config"] == {"chat_uploads": "filesystem"}
    assert by_name["Per chat uploads"]["contexts"] == {"chat": {"context_id": "chat_id"}}, (
        f"the browser is not told this terminal is per chat: {by_name['Per chat uploads']}"
    )
    assert by_name["Unset uploads"]["config"] == {"chat_uploads": "default"}
    assert by_name["Unknown uploads"]["config"] == {"chat_uploads": "default"}, (
        f"an unknown Chat Uploads value reached the browser: {by_name['Unknown uploads']}"
    )


def test_each_chats_upload_lands_in_that_chats_own_terminal(
    admin, make_user, preserve, terminal_orchestrator
):
    person = make_user()
    connection = _connect(admin, preserve, terminal_orchestrator, person)
    tag = uuid.uuid4().hex[:8]
    first_name, second_name = f"first-{tag}.txt", f"second-{tag}.txt"

    with person.client() as client:
        first_chat, second_chat = _new_chat(client), _new_chat(client)
        first = _upload(client, connection, first_chat, first_name, b"first chat\n")
        second = _upload(client, connection, second_chat, second_name, b"second chat\n")

    assert first.status_code == 200, f"the first chat's upload failed: {first.text}"
    assert second.status_code == 200, f"the second chat's upload failed: {second.text}"
    first_home = terminal_orchestrator.home(f"chat:{first_chat}")
    second_home = terminal_orchestrator.home(f"chat:{second_chat}")
    assert (first_home / first_name).read_text() == "first chat\n"
    assert (second_home / second_name).read_text() == "second chat\n"
    assert _homes_holding(terminal_orchestrator, first_name) == {f"chat:{first_chat}"}, (
        "the first chat's file also landed outside its own terminal"
    )
    assert _homes_holding(terminal_orchestrator, second_name) == {f"chat:{second_chat}"}, (
        "the second chat's file also landed outside its own terminal"
    )


def test_a_terminal_shared_by_all_chats_takes_the_upload_in_its_workspace(
    admin, make_user, preserve, terminal_orchestrator
):
    person = make_user()
    connection = _connect(admin, preserve, terminal_orchestrator, person, per_chat=False)
    name = f"shared-{uuid.uuid4().hex[:8]}.txt"

    with person.client() as client:
        uploaded = _upload(client, connection, _new_chat(client), name, b"for everyone\n")

    assert uploaded.status_code == 200, f"the upload failed: {uploaded.text}"
    assert (terminal_orchestrator.home(SHARED) / name).read_text() == "for everyone\n"
    assert _homes_holding(terminal_orchestrator, name) == {SHARED}


def test_a_per_chat_terminal_refuses_an_upload_for_a_temporary_chat(
    admin, make_user, preserve, terminal_orchestrator
):
    person = make_user()
    connection = _connect(admin, preserve, terminal_orchestrator, person)
    name = f"temporary-{uuid.uuid4().hex[:8]}.txt"

    with person.client() as client:
        refused = _upload(client, connection, f"temporary:{uuid.uuid4()}", name, b"nowhere\n")

    assert refused.status_code == 409, (
        f"a temporary chat's upload was not refused: {refused.status_code} {refused.text}"
    )
    assert _homes_holding(terminal_orchestrator, name) == set(), (
        "a temporary chat's file landed in a terminal"
    )
