"""Journey: a terminal's System Info reads the chat's own terminal, and only for a saved chat.

New on dev for 0.13.1 (7d4dda270 and its follow-ups, titled `refac`): the chat's Terminal menu
opens System Info, which reads `system/info` through the terminal proxy. A terminal that gives
every chat its own environment (chat context Per chat) has nothing to report before the chat is
saved, so the proxy answers 409 for a request naming no chat or a temporary chat. For a saved
chat the answer comes from that chat's own terminal; a terminal shared by all chats answers
without a chat. The orchestrator stand-in gives each chat context a real Open Terminal.

Discriminates: passes on dev f659c0d12; with the proxy's `system/info` saved-chat check removed
in a backend copy the no-chat request is answered by the shared terminal instead of a 409.
"""

from __future__ import annotations

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


def _system_info(client: httpx.Client, connection: dict, chat_id: str | None) -> httpx.Response:
    headers = {"X-Session-Id": chat_id} if chat_id else {}
    return client.get(f"/api/v1/terminals/{connection['id']}/system/info", headers=headers)


def _assert_system_details(answer: httpx.Response) -> None:
    assert answer.status_code == 200, f"System Info failed: {answer.status_code} {answer.text}"
    info = answer.json()
    for key in ("cpu", "memory", "disk", "processes"):
        assert key in info, f"System Info carries no {key}: {sorted(info)}"
    assert info["disk"]["total"] > 0, info["disk"]


def test_a_per_chat_terminal_shows_system_info_only_for_a_saved_chat(
    admin, make_user, preserve, terminal_orchestrator
):
    person = make_user()
    connection = _connect(admin, preserve, terminal_orchestrator, person)

    with person.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": {"title": "System"}})
        assert created.status_code == 200, created.text
        chat_id = created.json()["id"]
        no_chat = _system_info(client, connection, None)
        temporary = _system_info(client, connection, "temporary:unsaved-chat")
        saved = _system_info(client, connection, chat_id)

    for label, answer in (("no chat", no_chat), ("a temporary chat", temporary)):
        assert answer.status_code == 409, (
            f"System Info for {label} was not refused: {answer.status_code} {answer.text}"
        )
        assert answer.json()["error"] == "A saved chat is required for this terminal"
    _assert_system_details(saved)
    assert f"chat:{chat_id}" in terminal_orchestrator.contexts(), (
        f"the saved chat's own terminal did not answer: {terminal_orchestrator.contexts()}"
    )


def test_a_terminal_shared_by_all_chats_shows_system_info_without_a_chat(
    admin, make_user, preserve, terminal_orchestrator
):
    person = make_user()
    connection = _connect(admin, preserve, terminal_orchestrator, person, per_chat=False)

    with person.client() as client:
        answer = _system_info(client, connection, None)

    _assert_system_details(answer)
    assert terminal_orchestrator.contexts() == {SHARED}
