"""Journey: a terminal's System Info reads the chat's own machine, and only for a saved chat.

Since 0.13.1 (7d4dda270 and its follow-ups, titled `refac`) the chat's Terminal menu opens System
info, which reads `system/info` through the terminal proxy. A terminal scoped to the chat gives
every chat a machine of its own and has nothing to report before the chat is saved, so the proxy
answers 409 for a request naming no chat or a temporary chat, and starts no machine. For a saved
chat the answer comes from that chat's own machine; a terminal with one machine for everyone
answers without a chat.

Since 6f2fc7bc7 (dev after 0.13.3, titled `refac`) a terminal on an orchestrator is a template
with a scope, where 0.13.3 had a chat context on the connection; these tests ran against a
stand-in for the Terminals 0.3 contract until then. The real Terminals service runs on Docker
(`harness.terminals`); without its images or Docker the tests skip.

Discriminates: passes on dev c15f28649 with Terminals 0.4.1 and Open Terminal 0.17.7; in a
backend copy with the saved-chat check removed the no-chat request is no longer answered 409
with the reason and the per-chat test fails, and with a saved chat demanded of a terminal
without a scope the shared-machine test fails (409).
"""

from __future__ import annotations

import httpx
import pytest

from harness.terminals import execute, output_of

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def _system_info(client: httpx.Client, terminal_id: str, chat_id: str | None) -> httpx.Response:
    headers = {"X-Session-Id": chat_id} if chat_id else {}
    return client.get(f"/api/v1/terminals/{terminal_id}/system/info", headers=headers)


def _system_details(answer: httpx.Response) -> dict:
    assert answer.status_code == 200, f"System Info failed: {answer.status_code} {answer.text}"
    info = answer.json()
    for key in ("hostname", "cpu", "memory", "disk", "processes"):
        assert key in info, f"System Info carries no {key}: {sorted(info)}"
    assert info["disk"]["total"] > 0, info["disk"]
    return info


def test_a_per_chat_terminal_shows_system_info_only_for_a_saved_chat(
    make_user, shared_terminal, terminals
):
    person = make_user()
    terminal_id = shared_terminal(person, scope="chat")

    with person.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": {"title": "System"}})
        assert created.status_code == 200, created.text
        chat_id = created.json()["id"]
        no_chat = _system_info(client, terminal_id, None)
        temporary = _system_info(client, terminal_id, "temporary:unsaved-chat")
        started_by_refused = terminals.instances()
        host = output_of(execute(client, terminal_id, "hostname", chat_id)).strip()
        saved = _system_info(client, terminal_id, chat_id)

    for label, answer in (("no chat", no_chat), ("a temporary chat", temporary)):
        assert (answer.status_code, answer.json().get("detail")) == (
            409,
            "This terminal requires a saved chat",
        ), f"System Info for {label} was not refused: {answer.status_code} {answer.text}"
    assert started_by_refused == [], (
        f"a refused System Info started a machine: {started_by_refused}"
    )
    assert _system_details(saved)["hostname"] == host, (
        "the saved chat's System Info did not come from the machine its commands run in"
    )
    assert [machine["context_id"] for machine in terminals.instances()] == [f"chat:{chat_id}"], (
        f"the saved chat's own machine did not answer: {terminals.instances()}"
    )


def test_a_terminal_with_one_machine_for_all_shows_system_info_without_a_chat(
    make_user, shared_terminal, terminals
):
    person = make_user()
    terminal_id = shared_terminal(person, scope=None)

    with person.client() as client:
        host = output_of(execute(client, terminal_id, "hostname")).strip()
        answer = _system_info(client, terminal_id, None)

    assert _system_details(answer)["hostname"] == host
    assert [machine["context_id"] for machine in terminals.instances()] == ["default"]
