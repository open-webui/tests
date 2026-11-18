"""Regression: a terminal call in a long saved chat cost as much as reading the whole chat.

Every call the terminal panel makes through Open WebUI (files, working directory, git status,
the shell) names the chat it belongs to, and Open WebUI checks that chat before passing the call
on: may the person write to it, is it a sub-agent of another chat, does a coding agent hold a
session in it on this terminal. From v0.13.4 that check read the chat's whole history and every
stored message with all their content, so in a chat of 130 MB one call took most of a second on
SQLite and longer on Postgres, and a panel left open used up the database connections of the
whole instance (open-webui/open-webui#34646). The fix reads only the chat's owner, sharing and
current message, and of each message only its parent and its agent session.

The narrow test times the same call in the stored 1 MB and 130 MB chats (`harness/fat_chats.py`)
and bounds the difference. The nearby tests pin what the check decides, on both refs: the owner
and a person the chat is shared with for replies get through, a person it is shared with to read
only and a chat that does not exist are refused, a sub-agent chat is let through as its parent,
and in a chat a coding agent works in, the terminal is asked in the name of the agent's owner,
whoever makes the call.

Discriminates: passes on the fix of #34646 (the 130 MB call takes under 0.05 s longer than the
1 MB one); fails on dev f8a518648 (about 0.3 s longer on SQLite and 0.8 s on Postgres). The
nearby tests pass on both.
"""

from __future__ import annotations

import statistics
import time
from typing import Iterator

import pytest

from harness.access import grant, make_group
from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    configure_terminals,
    read_grant,
    serving_terminal,
)

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

CWD = "/home/user/workspace"
TIMED_CALLS = 9
# the 130 MB call may cost this much more than the 1 MB one: about 0.04 s on SQLite with the fix
EXTRA_SECONDS_BOUND = 0.2


@pytest.fixture
def terminal() -> Iterator[FakeTerminalServer]:
    with serving_terminal() as server:
        yield server


@pytest.fixture
def people(make_user):
    return {"owner": make_user(), "writer": make_user(), "reader": make_user()}


@pytest.fixture
def terminal_id(admin, preserve, people, terminal) -> str:
    preserve(TERMINAL_SERVERS_CONFIG)
    connection = terminal.connection(
        config={"access_grants": [read_grant(person.id) for person in people.values()]}
    )
    with admin.client() as client:
        configure_terminals(client, connection)
    terminal.route("GET", "/files/cwd", json_answer({"cwd": CWD}))
    terminal.clear()
    return connection["id"]


def _cwd_call(client, terminal_id: str, chat_id: str):
    return client.get(f"/api/v1/terminals/{terminal_id}/chats/{chat_id}/files/cwd")


def _median_seconds(client, terminal_id: str, chat_id: str) -> float:
    assert _cwd_call(client, terminal_id, chat_id).status_code == 200
    timings = []
    for _ in range(TIMED_CALLS):
        started = time.perf_counter()
        answered = _cwd_call(client, terminal_id, chat_id)
        timings.append(time.perf_counter() - started)
        assert answered.status_code == 200, answered.text
    return statistics.median(timings)


def _share(owner, chat_id: str, group_id: str, share_mode: str | None) -> None:
    """Share the chat with a group as the share dialog does."""
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share", json={"share_mode": None})
        assert shared.status_code == 200, shared.text
        saved = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update",
            json={"access_grants": [grant("group", group_id, "read")], "share_mode": share_mode},
        )
    assert saved.status_code == 200, saved.text


def _asked_as(terminal: FakeTerminalServer) -> list[str | None]:
    return [request.headers.get("x-user-id") for request in terminal.requests_to("/files/cwd")]


def test_a_terminal_call_in_a_130_mb_chat_costs_about_as_much_as_in_a_1_mb_chat(
    people, terminal_id, fat_chat
):
    owner = people["owner"]
    small = fat_chat(owner, "fat-1mb")
    large = fat_chat(owner, "fat-130mb")

    with owner.client() as client:
        small_seconds = _median_seconds(client, terminal_id, small.chat_id)
        large_seconds = _median_seconds(client, terminal_id, large.chat_id)

    assert large_seconds - small_seconds < EXTRA_SECONDS_BOUND, (
        f"a terminal call took {large_seconds:.3f} s in the 130 MB chat and {small_seconds:.3f} s "
        f"in the 1 MB one; the chat's size should barely matter"
    )


def test_who_may_use_the_terminal_in_a_chat_is_decided_as_before(
    admin, people, terminal, terminal_id, fat_chat
):
    owner, writer, reader = people["owner"], people["writer"], people["reader"]
    replies = fat_chat(owner, "fat-1mb")
    read_only = fat_chat(owner, "fat-1mb")
    _share(owner, replies.chat_id, make_group(admin, [writer]), "continue")
    _share(owner, read_only.chat_id, make_group(admin, [reader]), None)

    with owner.client() as client:
        by_owner = _cwd_call(client, terminal_id, replies.chat_id)
        in_missing_chat = _cwd_call(client, terminal_id, "f0000000-0000-4000-8000-000000000000")
    with writer.client() as client:
        by_writer = _cwd_call(client, terminal_id, replies.chat_id)
    with reader.client() as client:
        by_reader = _cwd_call(client, terminal_id, read_only.chat_id)

    assert by_owner.status_code == 200, f"the owner was refused: {by_owner.text}"
    assert by_writer.status_code == 200, (
        f"a person the chat is shared with for replies was refused: {by_writer.text}"
    )
    assert (by_reader.status_code, by_reader.json().get("detail")) == (403, "Chat access denied"), (
        f"a person who may only read the chat got {by_reader.status_code} {by_reader.text}"
    )
    assert (in_missing_chat.status_code, in_missing_chat.json().get("detail")) == (
        403,
        "Chat access denied",
    ), f"a chat that does not exist got {in_missing_chat.status_code} {in_missing_chat.text}"


def test_in_a_coding_agents_chat_the_terminal_is_asked_as_the_agents_owner(
    admin, people, terminal, terminal_id, fat_chat
):
    owner, writer = people["owner"], people["writer"]
    fat = fat_chat(owner, "fat-1mb", terminal_id=terminal_id, cwd=CWD)
    plain = fat_chat(owner, "fat-1mb")
    group_id = make_group(admin, [writer])
    _share(owner, fat.chat_id, group_id, "continue")
    _share(owner, plain.chat_id, group_id, "continue")

    with writer.client() as client:
        in_agent_chat = _cwd_call(client, terminal_id, fat.chat_id)
        agent_asked_as = _asked_as(terminal)
        terminal.clear()
        in_plain_chat = _cwd_call(client, terminal_id, plain.chat_id)
        plain_asked_as = _asked_as(terminal)
    terminal.clear()
    with owner.client() as client:
        in_subagent_chat = _cwd_call(client, terminal_id, fat.subagent_chat_ids[0])
    subagent_named = [
        request.headers.get("x-session-id") for request in terminal.requests_to("/files/cwd")
    ]

    assert in_agent_chat.status_code == 200, in_agent_chat.text
    assert agent_asked_as == [owner.id], (
        f"in the agent's chat the terminal was asked as {agent_asked_as}; the agent belongs to "
        f"the owner {owner.id}"
    )
    assert in_plain_chat.status_code == 200, in_plain_chat.text
    assert plain_asked_as == [writer.id], (
        f"in a chat without an agent on this terminal it was asked as {plain_asked_as}, "
        f"not as the caller {writer.id}"
    )
    assert in_subagent_chat.status_code == 200, (
        f"a call in the agent's sub-agent chat was refused: {in_subagent_chat.text}"
    )
    assert subagent_named == [fat.subagent_chat_ids[0]], (
        f"the terminal was told the chats {subagent_named}, not the sub-agent chat"
    )
