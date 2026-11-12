"""Journey: a chat's terminal working directory, chosen for the chat or set as a folder's default.

New in 0.13.1. The chat input's working directory menu (beside the Terminal menu) picks a folder
on the selected terminal for a new chat, and the page sends it with the chat's first message as
`terminal_cwd`. A folder's settings can name a default terminal and a working directory on it,
which apply to new chats started in the folder. For a terminal the admin connected, the server
moves the new chat's terminal session to that folder before the first reply (4955624d2), so every
command the model runs in the chat runs there and the `run_command` description names it; a
folder that does not exist refuses the chat with "Working directory is unavailable". An empty
`terminal_cwd`, the menu's "No folder", keeps a folder chat in the terminal's own directory
(834c47ee0). The folder's defaults are read only for a folder the user may write to.

A real Open Terminal runs the commands; its `run_command` answer names the folder it ran in.

Discriminates: passes on dev f659c0d12. In backend copies of it: skipping the directory call in
`initialize_terminal_directory` turns the chosen-directory, later-message, missing-directory,
folder-default, override, refused-terminal and owner and writer rows red; looking the folder
up without the write check turns the reader and stranger rows red; `not requested_directory` in
place of `is None` (834c47ee0 undone) turns the "No folder" test red; applying the directory on
every message turns the later-message test red; dropping the type check turns the not-a-path test
red; applying the folder's directory to any terminal turns the other-terminal test red.
The refused-terminal test was retargeted for the terminal rework (6f2fc7bc7), where a terminal
the user has no grant on answers as one that does not exist ("not found" where it said "Access
denied"): it passes on dev c15f28649, and in a backend copy without the access checks in
`find_terminal_connection` and `get_terminal_tools` the folder lends the terminal and the chat
gets a reply.
Since dev 6f2fc7bc7 a chat the server creates stores its terminal and working directory, the
folder's where the folder lent them. In a backend copy of dev c15f28649 that stores neither, the
chosen-directory, no-directory and folder-default tests fail.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from harness import upstream as reply
from harness.access import grant
from harness.chat import ask, send_message
from harness.socket_client import connected
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from harness.upstream import conversation

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

CWD_LINE = "The current working directory is: "
UNAVAILABLE = "Working directory is unavailable"


@pytest.fixture
def member(make_user):
    return make_user()


@pytest.fixture
def terminal_id(admin, member, open_terminal, preserve) -> str:
    """The Open Terminal, connected by the admin and shared with `member`; returns its id."""
    preserve(TERMINAL_SERVERS_CONFIG)
    connection = open_terminal.connection(config={"access_grants": [read_grant(member.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    return connection["id"]


def _directory(open_terminal, name: str = "project") -> Path:
    """A fresh folder in the terminal's home, named apart from every other test's."""
    folder = open_terminal.home / f"{name}-{uuid.uuid4().hex[:8]}"
    folder.mkdir()
    return folder


def _create_folder(owner, data: dict) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/folders/", json={"name": f"Repo {uuid.uuid4().hex[:8]}", "data": data}
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _where_pwd_ran(actor, upstream, chat_id: str | None = None, **options) -> tuple[str, dict]:
    """The model runs `pwd` in a new chat (or after the last reply of `chat_id`).

    Returns the chat id and the request that carried the command's result to the model.
    """
    prompt = f"where am I? {uuid.uuid4().hex[:8]}"
    answering = reply.answering(prompt)
    upstream.queue(
        reply.tool_call("run_command", {"command": "pwd"}, match=answering),
        reply.text("looked", match=answering),
    )
    with connected(actor) as session, actor.client() as client:
        session_id = session.client.get_sid(namespace="/")
        if chat_id:
            history = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]
            options = {**options, "chat_id": chat_id, "parent_id": history["currentId"]}
        turn, message = ask(client, prompt, session_id=session_id, **options)
    assert message.get("content") == "looked", f"the chat did not finish: {message}"
    requests = [request for request in upstream.chat_requests() if answering(request)]
    assert len(requests) == 2, f"expected the command round and the answer, got {len(requests)}"
    return turn.chat_id, requests


def _ran_in(requests: list[dict]) -> str:
    """The folder Open Terminal reports the model's command ran in."""
    results = [entry["content"] for entry in conversation(requests[-1]) if entry["role"] == "tool"]
    assert results, f"run_command never ran; the model was sent {conversation(requests[-1])}"
    return json.loads(results[-1])["cwd"]


def _offered_cwd(requests: list[dict]) -> str | None:
    """The folder the `run_command` description told the model it is in."""
    for tool in requests[0].get("tools") or []:
        if tool["function"]["name"] == "run_command":
            description = tool["function"].get("description", "")
            return description.split(CWD_LINE)[-1] if CWD_LINE in description else None
    raise AssertionError(f"run_command was not offered: {requests[0].get('tools')}")


def _session_cwd(actor, terminal_id: str, chat_id: str) -> str:
    """The chat's folder as its page reads it back through the instance, e.g. after a reload."""
    with actor.client() as client:
        answer = client.get(
            f"/api/v1/terminals/{terminal_id}/files/cwd", headers={"X-Session-Id": chat_id}
        )
    assert answer.status_code == 200, answer.text
    return answer.json()["cwd"]


def _kept_with_the_chat(actor, chat_id: str) -> tuple[str | None, str | None]:
    """The terminal and the working directory stored on the chat."""
    with actor.client() as client:
        found = client.get(f"/api/v1/chats/{chat_id}")
    assert found.status_code == 200, found.text
    chat = found.json()["chat"]
    return chat.get("terminal_id"), chat.get("working_directory")


def _failed_reply(actor, **options) -> str:
    """The error a new chat's first reply ends in, as a reload shows it."""
    with actor.client() as client:
        _, message = ask(client, f"start {uuid.uuid4().hex[:8]}", **options)
    assert not message.get("content"), f"the chat got a reply: {message}"
    return str((message.get("error") or {}).get("content"))


def _refusal(actor, **options) -> str:
    """How the server refuses a new chat's first message: its status line and body."""
    with actor.client() as client:
        try:
            turn = send_message(client, f"start {uuid.uuid4().hex[:8]}", **options)
        except AssertionError as refused:
            return str(refused)
    raise AssertionError(f"the chat was accepted: {turn.chat_id}")


def test_a_new_chat_runs_the_models_commands_in_the_chosen_directory(
    member, terminal_id, open_terminal, upstream
):
    chosen = _directory(open_terminal)

    chat_id, requests = _where_pwd_ran(
        member, upstream, terminal_id=terminal_id, terminal_cwd=str(chosen)
    )

    assert _ran_in(requests) == str(chosen), "the model's command ignored the chosen directory"
    assert _offered_cwd(requests) == str(chosen), (
        "the model was not told it works in the chosen directory"
    )
    assert _session_cwd(member, terminal_id, chat_id) == str(chosen), (
        "the chat's terminal session does not keep the chosen directory"
    )
    assert _kept_with_the_chat(member, chat_id) == (terminal_id, str(chosen)), (
        "the chat does not store the terminal and directory it was started with"
    )


def test_a_new_chat_without_a_directory_runs_in_the_terminals_own(
    member, terminal_id, open_terminal, upstream
):
    _where_pwd_ran(
        member, upstream, terminal_id=terminal_id, terminal_cwd=str(_directory(open_terminal))
    )

    chat_id, requests = _where_pwd_ran(member, upstream, terminal_id=terminal_id)

    assert _ran_in(requests) == str(open_terminal.home), (
        "a chat with no directory chosen ran somewhere other than the terminal's home"
    )
    assert _kept_with_the_chat(member, chat_id) == (terminal_id, ""), (
        "a chat with no directory chosen stores one, or not its terminal"
    )


def test_a_later_message_keeps_the_chats_directory(member, terminal_id, open_terminal, upstream):
    chosen = _directory(open_terminal)
    chat_id, _ = _where_pwd_ran(member, upstream, terminal_id=terminal_id, terminal_cwd=str(chosen))

    _, requests = _where_pwd_ran(
        member,
        upstream,
        chat_id=chat_id,
        terminal_id=terminal_id,
        terminal_cwd=str(_directory(open_terminal, "elsewhere")),
    )

    assert _ran_in(requests) == str(chosen), (
        "a message in an existing chat moved the chat to another directory"
    )


def test_a_missing_directory_refuses_the_new_chat(member, terminal_id, open_terminal, upstream):
    before = len(upstream.chat_requests())

    error = _failed_reply(
        member, terminal_id=terminal_id, terminal_cwd=str(open_terminal.home / "no-such-folder")
    )

    assert UNAVAILABLE in error, f"a chat in a missing folder did not say so: {error}"
    assert len(upstream.chat_requests()) == before, "the model was asked anyway"


def test_a_directory_that_is_not_a_path_is_refused(member, terminal_id):
    error = _failed_reply(member, terminal_id=terminal_id, terminal_cwd=["/etc"])

    assert "Invalid working directory" in error, error


def test_a_folder_chat_gets_the_folders_terminal_and_directory(
    member, terminal_id, open_terminal, upstream
):
    default = _directory(open_terminal)
    folder_id = _create_folder(
        member, {"terminal_id": terminal_id, "working_directory": str(default)}
    )

    chat_id, requests = _where_pwd_ran(member, upstream, folder_id=folder_id)

    assert _ran_in(requests) == str(default), (
        "a new chat in the folder did not run in the folder's working directory"
    )
    assert _kept_with_the_chat(member, chat_id) == (terminal_id, str(default)), (
        "the folder's chat does not store the folder's terminal and directory"
    )


def test_a_directory_chosen_for_the_chat_wins_over_the_folders(
    member, terminal_id, open_terminal, upstream
):
    folder_id = _create_folder(
        member,
        {"terminal_id": terminal_id, "working_directory": str(_directory(open_terminal))},
    )
    chosen = _directory(open_terminal, "chosen")

    _, requests = _where_pwd_ran(
        member, upstream, folder_id=folder_id, terminal_id=terminal_id, terminal_cwd=str(chosen)
    )

    assert _ran_in(requests) == str(chosen)


def test_no_folder_chosen_in_a_folder_chat_runs_in_the_terminals_own(
    member, terminal_id, open_terminal, upstream
):
    folder_id = _create_folder(
        member,
        {"terminal_id": terminal_id, "working_directory": str(_directory(open_terminal))},
    )

    _, requests = _where_pwd_ran(
        member, upstream, folder_id=folder_id, terminal_id=terminal_id, terminal_cwd=""
    )

    assert _ran_in(requests) == str(open_terminal.home), (
        "picking No folder in a folder chat still ran in the folder's directory"
    )


def test_the_folders_directory_stays_with_the_folders_terminal(
    admin, member, open_terminal, upstream, preserve
):
    preserve(TERMINAL_SERVERS_CONFIG)
    access = {"access_grants": [read_grant(member.id)]}
    folders_terminal = open_terminal.connection(name="Folder box", config=access)
    other_terminal = open_terminal.connection(name="Other box", config=access)
    with admin.client() as client:
        configure_terminals(client, folders_terminal, other_terminal)
    folder_id = _create_folder(
        member,
        {
            "terminal_id": folders_terminal["id"],
            "working_directory": str(_directory(open_terminal)),
        },
    )

    _, requests = _where_pwd_ran(
        member, upstream, folder_id=folder_id, terminal_id=other_terminal["id"]
    )

    assert _ran_in(requests) == str(open_terminal.home), (
        "the folder's directory was applied to a chat on another terminal"
    )


def test_a_folder_default_terminal_the_user_may_not_use_is_refused(
    admin, member, open_terminal, upstream, preserve, make_user
):
    preserve(TERMINAL_SERVERS_CONFIG)
    private = open_terminal.connection(config={"access_grants": [read_grant(make_user().id)]})
    with admin.client() as client:
        configure_terminals(client, private)
    plain = _create_folder(member, {"terminal_id": private["id"]})
    with_directory = _create_folder(
        member,
        {"terminal_id": private["id"], "working_directory": str(_directory(open_terminal))},
    )
    before = len(upstream.chat_requests())

    plain_error = _failed_reply(member, folder_id=plain)
    directory_error = _failed_reply(member, folder_id=with_directory)

    not_found = f"Terminal unavailable: Terminal server '{private['id']}' not found"
    assert not_found in plain_error, f"the folder lent a terminal: {plain_error}"
    assert UNAVAILABLE in directory_error, directory_error
    assert len(upstream.chat_requests()) == before, "the model was asked with that terminal"


FOLDER_ROLES = {"owner": 200, "writer": 200, "reader": 403, "stranger": 403}


@pytest.mark.parametrize("role", FOLDER_ROLES)
def test_only_a_folder_the_user_may_write_to_lends_its_defaults(
    role, admin, make_user, open_terminal, upstream, preserve
):
    preserve(TERMINAL_SERVERS_CONFIG)
    owner, actor = make_user(), make_user()
    if role == "owner":
        actor = owner
    connection = open_terminal.connection(config={"access_grants": [grant("user", "*", "read")]})
    with admin.client() as client:
        configure_terminals(client, connection)
    default = _directory(open_terminal)
    folder_id = _create_folder(
        owner, {"terminal_id": connection["id"], "working_directory": str(default)}
    )
    shared = {"reader": ["read"], "writer": ["read", "write"]}.get(role, [])
    with owner.client() as client:
        granted = client.post(
            f"/api/v1/folders/{folder_id}/access/update",
            json={"access_grants": [grant("user", actor.id, level) for level in shared]},
        )
    assert granted.status_code == 200, granted.text

    if FOLDER_ROLES[role] == 403:
        refused = _refusal(actor, folder_id=folder_id)
        assert "HTTP 403" in refused and "Folder is unavailable" in refused, refused
        return
    _, requests = _where_pwd_ran(actor, upstream, folder_id=folder_id)
    assert _ran_in(requests) == str(default), f"the {role} did not get the folder's directory"
