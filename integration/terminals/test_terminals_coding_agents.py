"""Journey: the coding agents of a terminal on a Terminals orchestrator are those of the machine.

On an orchestrator terminal each person, chat or model has a machine of its own, and a coding
agent is a program inside that machine. So the agents Open WebUI offers are the ones in the
machine the request reaches, and a turn runs there. With a machine per user, a person whose
machine has the agent is offered it and another person on the same terminal is not. With a
machine per chat, the agents are listed only for a saved chat, and the turn runs in that chat's
machine. With a machine per model, the agent's own model id names the machine; an agent id of
another terminal is refused. The agent is `harness/fake_acp_agent.py`, installed into the machine
as `opencode`.

Discriminates: passes on dev c15f28649. Mutations in a backend copy, each turning only the named
test red: `/api/models` not passing the chat on (the per-chat test); every request sent as one
and the same user (the per-user test); the check that an agent id belongs to the terminal dropped
(the per-model test).
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.chat import send_message, wait_for_reply
from harness.coding_agent import (
    DEEP,
    FAST,
    MACHINE_AGENT_ENV,
    MACHINE_AGENT_LOG,
    install_agent,
    model_id,
)
from harness.terminals import execute, output_of

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

REPLY_TIMEOUT = 60.0
UNAVAILABLE = "Agent model is unavailable in the selected terminal"


def _agents_offered(client: httpx.Client, terminal_id: str, **params) -> set[str]:
    listed = client.get("/api/models", params={"terminal_id": terminal_id, **params})
    assert listed.status_code == 200, f"the model list failed: {listed.status_code} {listed.text}"
    return {model["id"] for model in listed.json()["data"] if model["id"].startswith("agent:")}


def _new_chat(client: httpx.Client, terminal_id: str) -> str:
    created = client.post(
        "/api/v1/chats/new", json={"chat": {"title": "Agent work", "terminal_id": terminal_id}}
    )
    assert created.status_code == 200, f"creating the chat failed: {created.text}"
    return created.json()["id"]


def _agent_turn(client: httpx.Client, terminal_id: str, prompt: str, **extra) -> dict:
    turn = send_message(
        client, prompt, model=model_id(terminal_id), terminal_id=terminal_id, **extra
    )
    return wait_for_reply(client, turn, REPLY_TIMEOUT)


def _machines_whose_agent_heard(terminals, prompt: str) -> list[str]:
    return sorted(
        context
        for context, log in terminals.holding(MACHINE_AGENT_LOG).items()
        if prompt in (log or "")
    )


def test_a_person_is_offered_the_agents_of_their_own_machine_and_runs_them_there(
    make_user, shared_terminal, terminals
):
    equipped, bare = make_user(), make_user()
    terminal_id = shared_terminal(equipped, bare, env=MACHINE_AGENT_ENV)
    prompt = f"tidy the repo {uuid.uuid4().hex[:6]}"

    with equipped.client() as client:
        install_agent(client, terminal_id)
        offered = _agents_offered(client, terminal_id)
        reply = _agent_turn(client, terminal_id, prompt)
    with bare.client() as client:
        output_of(execute(client, terminal_id, "echo started"))
        offered_without = _agents_offered(client, terminal_id)
        refused = client.post(
            "/api/chat/completions",
            json={
                "model": model_id(terminal_id),
                "terminal_id": terminal_id,
                "messages": [{"role": "user", "content": "use the other machine's agent"}],
                "stream": True,
            },
        )

    assert offered == {model_id(terminal_id, FAST), model_id(terminal_id, DEEP)}, (
        f"the person was not offered the agent in their machine: {offered}"
    )
    assert reply.get("content") == f"Agent heard: {prompt} [model {FAST}, folder /home/user]", (
        f"the agent's turn did not run in the machine: {reply}"
    )
    assert _machines_whose_agent_heard(terminals, prompt) == [f"user:{equipped.id}"], (
        "the turn ran in another machine than the person's own: "
        f"{_machines_whose_agent_heard(terminals, prompt)}"
    )
    assert offered_without == set(), (
        f"a person was offered an agent that is only in someone else's machine: {offered_without}"
    )
    assert (refused.status_code, refused.json().get("detail")) == (400, UNAVAILABLE), (
        f"a person ran an agent their own machine does not have: {refused.status_code}"
    )


def test_a_per_chat_terminal_offers_its_agents_for_a_saved_chat_and_runs_them_in_its_machine(
    make_user, shared_terminal, terminals
):
    person = make_user()
    terminal_id = shared_terminal(person, scope="chat", env=MACHINE_AGENT_ENV)
    prompt = f"tidy the repo {uuid.uuid4().hex[:6]}"

    with person.client() as client:
        chat_id, other_chat = _new_chat(client, terminal_id), _new_chat(client, terminal_id)
        install_agent(client, terminal_id, chat_id=chat_id)
        output_of(execute(client, terminal_id, "echo started", chat_id=other_chat))
        without_a_chat = _agents_offered(client, terminal_id)
        for_the_chat = _agents_offered(client, terminal_id, chat_id=chat_id)
        for_the_other = _agents_offered(client, terminal_id, chat_id=other_chat)
        reply = _agent_turn(client, terminal_id, prompt, chat_id=chat_id)

    assert for_the_chat == {model_id(terminal_id, FAST), model_id(terminal_id, DEEP)}, (
        f"the chat was not offered the agent in its machine: {for_the_chat}"
    )
    assert without_a_chat == set(), f"agents were offered without a chat: {without_a_chat}"
    assert for_the_other == set(), (
        f"a chat was offered the agent of another chat's machine: {for_the_other}"
    )
    assert reply.get("content") == f"Agent heard: {prompt} [model {FAST}, folder /home/user]", (
        f"the agent's turn did not run for the chat: {reply}"
    )
    assert _machines_whose_agent_heard(terminals, prompt) == [f"chat:{chat_id}"], (
        "the turn ran in another machine than the chat's: "
        f"{_machines_whose_agent_heard(terminals, prompt)}"
    )


def test_on_a_per_model_terminal_an_agent_gets_the_machine_of_its_own_model_id(
    make_user, shared_terminal, terminals
):
    person = make_user()
    terminal_id = shared_terminal(person, scope="model", env=MACHINE_AGENT_ENV)
    own = model_id(terminal_id)
    of_another_terminal = model_id(f"{terminal_id.rsplit(':', 1)[0]}:{uuid.uuid4()}")
    prompt = f"tidy the repo {uuid.uuid4().hex[:6]}"

    with person.client() as client:
        foreign = execute(client, terminal_id, "echo hello", model_id=of_another_terminal)
        started_for_it = terminals.instances()
        install_agent(client, terminal_id, model_id=own)
        reply = _agent_turn(client, terminal_id, prompt)

    assert (foreign.status_code, foreign.json().get("detail")) == (
        403,
        "Agent model does not belong to this terminal",
    ), f"an agent of another terminal was let into this one: {foreign.status_code} {foreign.text}"
    assert started_for_it == [], f"a machine was started for it: {started_for_it}"
    assert reply.get("content") == f"Agent heard: {prompt} [model {FAST}, folder /home/user]", (
        f"the agent's turn did not run: {reply}"
    )
    assert _machines_whose_agent_heard(terminals, prompt) == [f"model:{own}"], (
        "the turn ran in another machine than the agent model's: "
        f"{_machines_whose_agent_heard(terminals, prompt)}"
    )
