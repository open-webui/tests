"""Journey: a coding agent installed in a terminal is offered as models and runs a chat turn there.

New on dev for 0.13.1 (6ef26be29, a2845c9e6 and their follow-ups, titled `refac`). An admin's
terminal connection that a person can use lists that terminal's installed coding agents as
models, `agent:<terminal>:<agent>:<model>`, in `/api/models` when the page asks for that
terminal, and only then. A chat with one runs the agent in the terminal: the terminal discovers
and starts it, Open WebUI speaks the Agent Client Protocol with it over the terminal's agent
socket, and the agent's thinking, tool calls and answer are stored as the reply. A permission
request asks the person (unless the chat has full access), a question goes through the question
card, and the next message resumes the agent's own session. A new agent chat takes its first
message as its title (18f7633af). As in any chat, an attached file reaches the agent (copied
into its terminal, with the path in the prompt), the chat's system prompt reaches it as
instructions, and Stop cancels the agent's turn while the chat goes on. The agent here is
`harness/fake_acp_agent.py`, run by a real Open Terminal as its OpenCode agent.

Discriminates: passes on dev f659c0d12. Mutations in a backend copy, each turning only the named
tests red: discovery returning no models (the listing, chat and stranger tests); the access
check in `resolve_terminal_connection` skipped (the stranger listing test); the ACP client
allowing every permission request (the reject and someone-else tests); every agent request
asking even with full access (the full access test); `resume` dropped from the session options
(the continue test); the branch check removed (the branching test); the first-message title for
agent chats removed (the title test); the attached-files note dropped from the prompt (the
attachment test); empty instructions (the system prompt test); no cancel sent to the agent on
Stop (the stop test).
"""

from __future__ import annotations

import json
import time
import uuid

import httpx
import pytest

from harness.chat import ChatTurn, send_message, wait_for_reply
from harness.coding_agent import (
    DEEP,
    FAST,
    AgentTerminal,
    connect_agent_terminal,
    model_id,
)
from harness.fake_acp_agent import REFUSAL
from harness.socket_client import connected
from harness.terminal_server import TERMINAL_SERVERS_CONFIG
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
REPLY_TIMEOUT = 60.0


@pytest.fixture
def connect(admin, preserve, agent_terminal: AgentTerminal):
    """`connect(*people, **fields)` saves a connection to the agent terminal for those people."""
    preserve(TERMINAL_SERVERS_CONFIG)
    return lambda *people, **fields: connect_agent_terminal(
        admin, agent_terminal, *people, **fields
    )


@pytest.fixture
def tool_permissions(admin, preserve):
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        saved = client.post(CHAT_CONFIG[1], json={**current, "ENABLE_TOOL_PERMISSIONS": True})
    assert saved.status_code == 200, f"switching tool permissions on failed: {saved.text}"


def agent_models(client: httpx.Client, terminal_id: str | None) -> list[dict]:
    params = {"terminal_id": terminal_id} if terminal_id else {}
    listed = client.get("/api/models", params=params)
    assert listed.status_code == 200, f"the model list failed: {listed.text}"
    return [model for model in listed.json()["data"] if model["id"].startswith("agent:")]


def ask_agent(
    client: httpx.Client, terminal_id: str, text: str, model: str = FAST, **extra
) -> ChatTurn:
    return send_message(
        client, text, model=model_id(terminal_id, model), terminal_id=terminal_id, **extra
    )


def stored(client: httpx.Client, turn: ChatTurn) -> dict:
    chat = client.get(f"/api/v1/chats/{turn.chat_id}")
    assert chat.status_code == 200, chat.text
    return chat.json()["chat"]["history"]["messages"][turn.assistant_message_id]


def pending_call(client: httpx.Client, turn: ChatTurn) -> dict:
    """The agent request waiting for the person, once it is stored."""
    deadline = time.monotonic() + REPLY_TIMEOUT
    message: dict = {}
    while time.monotonic() < deadline:
        message = stored(client, turn)
        pending = [item for item in message.get("output") or [] if item.get("status") == "pending"]
        if pending:
            return pending[0]
        assert not message.get("done"), f"the reply ended without asking: {message}"
        time.sleep(0.1)
    raise AssertionError(f"the agent never asked; last stored state: {message}")


def resolve(client: httpx.Client, turn: ChatTurn, call_id: str, **answer) -> httpx.Response:
    return client.post(
        f"/api/v1/chats/{turn.chat_id}/messages/{turn.assistant_message_id}/resolve",
        json={"call_id": call_id, **answer},
    )


def output_text(message: dict, kind: str) -> str:
    return "".join(
        part.get("text", "")
        for item in message.get("output") or []
        if item.get("type") == kind
        for part in item.get("content") or []
    )


def test_picking_a_terminal_lists_its_installed_agents_models(make_user, connect, agent_terminal):
    person = make_user()
    terminal = connect(person)

    with person.client() as client:
        without_terminal = agent_models(client, None)
        listed = agent_models(client, terminal["id"])

    assert without_terminal == [], (
        f"agent models showed without a terminal picked: {without_terminal}"
    )
    by_id = {model["id"]: model for model in listed}
    assert set(by_id) == {model_id(terminal["id"], FAST), model_id(terminal["id"], DEEP)}, (
        f"the terminal's agent models were not listed as expected: {sorted(by_id)}"
    )
    fast = by_id[model_id(terminal["id"], FAST)]
    assert fast["name"] == f"opencode/{FAST}", fast
    assert fast["owned_by"] == "agent", fast
    assert fast["agent"] == {"terminal_id": terminal["id"], "provider": "opencode", "model": FAST}
    # the terminal knows the other agents too, but none of them is installed there
    assert {model["agent"]["provider"] for model in listed} == {"opencode"}


def test_a_person_without_access_to_the_terminal_gets_no_agent_models_and_cannot_chat_with_one(
    make_user, connect, agent_terminal
):
    owner, stranger = make_user(), make_user()
    terminal = connect(owner)

    with owner.client() as client:
        assert agent_models(client, terminal["id"]), "the person with access saw no agent models"
    prompts_before = len(agent_terminal.received("session/prompt"))
    with stranger.client() as client:
        listed = agent_models(client, terminal["id"])
        refused = client.post(
            "/api/chat/completions",
            json={
                "model": model_id(terminal["id"]),
                "terminal_id": terminal["id"],
                "messages": [{"role": "user", "content": "use their agent"}],
                "stream": True,
            },
        )

    assert listed == [], f"someone without access to the terminal saw its agents: {listed}"
    assert refused.status_code == 400, f"the stranger's agent chat was not refused: {refused.text}"
    assert "unavailable" in refused.json()["detail"], refused.text
    assert len(agent_terminal.received("session/prompt")) == prompts_before, (
        "the agent ran a turn for someone without access to the terminal"
    )


def test_a_switched_off_terminal_offers_no_agents(make_user, connect):
    person = make_user()
    terminal = connect(person, enabled=False)

    with person.client() as client:
        listed = agent_models(client, terminal["id"])
        refused = client.post(
            "/api/chat/completions",
            json={
                "model": model_id(terminal["id"]),
                "terminal_id": terminal["id"],
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert listed == [], f"a switched-off terminal still offered agents: {listed}"
    assert refused.status_code == 400, refused.text


def test_a_chat_with_an_agent_model_stores_its_thinking_answer_and_usage(
    make_user, connect, agent_terminal
):
    person = make_user()
    terminal = connect(person)
    prompt = f"say hello {uuid.uuid4().hex[:6]}"

    with person.client() as client:
        turn = ask_agent(client, terminal["id"], prompt, model=DEEP)
        reply = wait_for_reply(client, turn, REPLY_TIMEOUT)

    folder = str(agent_terminal.terminal.home)
    assert "error" not in reply, f"the agent turn failed: {reply.get('error')}"
    assert reply["content"] == f"Agent heard: {prompt} [model {DEEP}, folder {folder}]", reply
    assert output_text(reply, "message") == reply["content"]
    assert output_text(reply, "reasoning") == "Thinking it over.", reply.get("output")
    assert reply["usage"] == {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
    assert agent_terminal.received("session/set_model")[-1]["modelId"] == DEEP


def test_the_reply_streams_to_the_persons_open_tabs(make_user, connect):
    person = make_user()
    terminal = connect(person)

    with person.client() as client, connected(person) as tab:
        turn = ask_agent(client, terminal["id"], "stream please", session_id=tab.sid)
        reply = wait_for_reply(client, turn, REPLY_TIMEOUT)
        final = tab.wait_for(turn.chat_id, "chat:completion", done=True)

    assert final["data"]["content"] == reply["content"], final
    live = [
        event
        for event in tab.events_of(turn.chat_id)
        if event.get("type") == "chat:completion" and not event["data"].get("done")
    ]
    assert live, "the tab got the agent's reply only once it was finished"


def test_the_next_message_continues_the_agents_own_session(make_user, connect, agent_terminal):
    person = make_user()
    terminal = connect(person)

    with person.client() as client:
        first = ask_agent(client, terminal["id"], "remember the word lantern")
        first_reply = wait_for_reply(client, first, REPLY_TIMEOUT)
        session = agent_terminal.received("session/new")[-1]
        started = agent_terminal.received("session/prompt")[-1]["sessionId"]
        second = ask_agent(
            client,
            terminal["id"],
            "what was the word?",
            chat_id=first.chat_id,
            parent_id=first.assistant_message_id,
            history=[
                {"role": "user", "content": "remember the word lantern"},
                {"role": "assistant", "content": first_reply["content"]},
            ],
        )
        second_reply = wait_for_reply(client, second, REPLY_TIMEOUT)

    assert session["cwd"] == str(agent_terminal.terminal.home)
    loads = agent_terminal.received("session/load")
    assert loads and loads[-1]["sessionId"] == started, (
        f"the second turn did not resume the agent's session {started}: {loads}"
    )
    assert second_reply["content"].startswith("Agent heard: what was the word? ["), (
        f"a resumed session was sent the whole conversation again: {second_reply['content']}"
    )


def test_branching_an_agent_conversation_is_refused(make_user, connect):
    person = make_user()
    terminal = connect(person)

    with person.client() as client:
        first = ask_agent(client, terminal["id"], "start here")
        wait_for_reply(client, first, REPLY_TIMEOUT)
        follow = {"chat_id": first.chat_id, "parent_id": first.assistant_message_id}
        wait_for_reply(client, ask_agent(client, terminal["id"], "go on", **follow), REPLY_TIMEOUT)
        branch = ask_agent(client, terminal["id"], "a different go on", **follow)
        branched = wait_for_reply(client, branch, REPLY_TIMEOUT)

    assert (branched.get("error") or {}).get("content") == (
        "Agent history cannot be branched in place. Start a new chat instead."
    ), f"a branch of an agent conversation was not refused: {branched}"


def test_an_agent_tool_call_is_stored_as_a_tool_call_with_its_result(make_user, connect):
    person = make_user()
    terminal = connect(person)

    with person.client() as client:
        reply = wait_for_reply(
            client, ask_agent(client, terminal["id"], "[tool] read it"), REPLY_TIMEOUT
        )

    calls = [item for item in reply["output"] if item["type"] == "function_call"]
    results = [item for item in reply["output"] if item["type"] == "function_call_output"]
    assert [call["name"] for call in calls] == ["Read notes.txt"], reply
    assert json.loads(calls[0]["arguments"]) == {"path": "notes.txt"}
    assert calls[0]["status"] == "completed"
    assert [result["call_id"] for result in results] == [calls[0]["call_id"]]
    assert "first line of notes" in results[0]["output"][0]["text"], results
    assert reply["content"] == "The notes start with: first line of notes"


@pytest.mark.parametrize(("action", "outcome"), [("approve", "allow-it"), ("reject", "deny-it")])
def test_an_agent_permission_request_waits_for_the_person_and_follows_their_answer(
    make_user, connect, tool_permissions, action, outcome
):
    person = make_user()
    terminal = connect(person)

    with person.client() as client:
        turn = ask_agent(
            client, terminal["id"], "[permission] clean up", params={"tool_approval_mode": "ask"}
        )
        asked = pending_call(client, turn)
        answered = resolve(client, turn, asked["call_id"], action=action)
        reply = wait_for_reply(client, turn, REPLY_TIMEOUT)

    assert asked["name"] == "agent_tool", asked
    assert json.loads(asked["arguments"])["rawInput"] == {"command": "rm -rf build"}, asked
    assert answered.status_code == 200, answered.text
    assert reply["content"] == f"Permission outcome: {outcome}", reply


def test_with_full_access_an_agent_permission_request_is_allowed_without_asking(
    make_user, connect, tool_permissions
):
    person = make_user()
    terminal = connect(person)

    with person.client() as client:
        turn = ask_agent(
            client, terminal["id"], "[permission] clean up", params={"tool_approval_mode": "full"}
        )
        reply = wait_for_reply(client, turn, REPLY_TIMEOUT)

    assert reply["content"] == "Permission outcome: allow-it", reply
    approvals = [item for item in reply["output"] if item.get("name") == "agent_tool"]
    assert [item["status"] for item in approvals] == ["completed"], approvals


def test_someone_else_cannot_answer_an_agents_permission_request(
    make_user, connect, tool_permissions
):
    person, stranger = make_user(), make_user()
    terminal = connect(person, stranger)

    with person.client() as client:
        turn = ask_agent(
            client, terminal["id"], "[permission] clean up", params={"tool_approval_mode": "ask"}
        )
        asked = pending_call(client, turn)
        with stranger.client() as other:
            meddled = resolve(other, turn, asked["call_id"], action="approve")
        still = [i for i in stored(client, turn)["output"] if i.get("call_id") == asked["call_id"]]
        resolve(client, turn, asked["call_id"], action="reject")
        reply = wait_for_reply(client, turn, REPLY_TIMEOUT)

    assert meddled.status_code in (401, 403, 404), (
        f"someone else answered the agent's request: {meddled.status_code} {meddled.text}"
    )
    assert still[0]["status"] == "pending", still
    assert reply["content"] == "Permission outcome: deny-it", reply


def test_an_agent_question_is_answered_through_the_question_card(make_user, connect):
    person = make_user()
    terminal = connect(person)

    with person.client() as client:
        turn = ask_agent(client, terminal["id"], "[ask] style the button")
        asked = pending_call(client, turn)
        wrong = resolve(client, turn, asked["call_id"], action="approve")
        answered = resolve(
            client, turn, asked["call_id"], action="answer", answers={"colour": "amber"}
        )
        reply = wait_for_reply(client, turn, REPLY_TIMEOUT)

    assert asked["name"] == "ask_user", asked
    questions = json.loads(asked["arguments"])["questions"]
    assert [q["id"] for q in questions] == ["colour"], questions
    assert "Which colour should the button be?" in questions[0]["question"], questions
    assert [o["label"] for o in questions[0]["options"]] == ["teal", "amber"], questions
    assert wrong.status_code == 400, f"a question was approved instead of answered: {wrong.text}"
    assert answered.status_code == 200, answered.text
    assert reply["content"] == "Colour answer: amber", reply


def test_an_agent_that_refuses_the_prompt_shows_its_error(make_user, connect):
    person = make_user()
    terminal = connect(person)

    with person.client() as client:
        reply = wait_for_reply(
            client, ask_agent(client, terminal["id"], "[refuse] no"), REPLY_TIMEOUT
        )

    assert (reply.get("error") or {}).get("content") == REFUSAL, reply


def test_an_agent_that_crashes_mid_turn_ends_the_reply_with_an_error(make_user, connect):
    person = make_user()
    terminal = connect(person)

    with person.client() as client:
        reply = wait_for_reply(
            client, ask_agent(client, terminal["id"], "[crash] go"), REPLY_TIMEOUT
        )

    assert reply.get("error"), f"a crashed agent left no error on the reply: {reply}"
    assert "3" in reply["error"]["content"], reply["error"]
    assert all(
        item.get("status") not in ("pending", "in_progress") for item in reply.get("output") or []
    )


def test_an_agent_that_is_not_installed_in_the_terminal_is_refused(make_user, connect):
    person = make_user()
    terminal = connect(person)

    with person.client() as client:
        refused = client.post(
            "/api/chat/completions",
            json={
                "model": model_id(terminal["id"], "some-model", provider="gemini"),
                "terminal_id": terminal["id"],
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
        works = wait_for_reply(client, ask_agent(client, terminal["id"], "hello"), REPLY_TIMEOUT)

    assert refused.status_code == 400, f"a missing agent was not refused: {refused.text}"
    assert refused.json()["detail"] == "Agent model is unavailable in the selected terminal"
    assert "error" not in works, works


def test_an_agent_cannot_share_a_turn_with_another_model(make_user, connect):
    person = make_user()
    terminal = connect(person)
    agent = model_id(terminal["id"])

    with person.client() as client:
        refused = client.post(
            "/api/chat/completions",
            json={
                "model": agent,
                "terminal_id": terminal["id"],
                "messages": [{"role": "user", "content": "both of you"}],
                "stream": True,
                "message_ids": [
                    {"model_id": agent, "message_id": str(uuid.uuid4())},
                    {"model_id": MOCK_MODEL_ID, "message_id": str(uuid.uuid4())},
                ],
            },
        )

    assert refused.status_code == 400, f"an agent shared a turn with another model: {refused.text}"
    assert refused.json()["detail"] == "Use one agent at a time in a conversation"


def test_a_new_agent_chat_is_titled_after_its_first_message(make_user, connect):
    person = make_user()
    terminal = connect(person)
    prompt = f"Refactor the parser module {uuid.uuid4().hex[:6]} and keep the tests green please"

    with person.client() as client:
        turn = ask_agent(
            client,
            terminal["id"],
            prompt,
            background_tasks={
                "title_generation": True,
                "tags_generation": False,
                "follow_up_generation": False,
            },
        )
        wait_for_reply(client, turn, REPLY_TIMEOUT)
        deadline = time.monotonic() + 30
        title = ""
        while time.monotonic() < deadline:
            title = client.get(f"/api/v1/chats/{turn.chat_id}").json()["title"]
            if title != "New Chat":
                break
            time.sleep(0.1)

    assert title == prompt[:50], f"the agent chat was not titled after its first message: {title!r}"


def test_a_file_attached_to_the_message_reaches_the_agent_in_its_terminal(
    make_user, connect, agent_terminal
):
    person = make_user()
    terminal = connect(person)

    with person.client() as client:
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "false"},
            files={"file": ("brief.txt", b"ship it on friday\n", "text/plain")},
        )
        assert uploaded.status_code == 200, f"uploading the file failed: {uploaded.text}"
        file_id = uploaded.json()["id"]
        attached = [{"type": "file", "id": file_id, "name": "brief.txt"}]
        turn = ask_agent(
            client, terminal["id"], "read the brief", message_files=attached, files=attached
        )
        reply = wait_for_reply(client, turn, REPLY_TIMEOUT)

    assert "error" not in reply, f"the agent turn failed: {reply.get('error')}"
    copy = agent_terminal.terminal.home / f"{file_id}.txt"
    assert copy.read_text() == "ship it on friday\n", "the file did not land in the terminal"
    prompt = agent_terminal.received("session/prompt")[-1]["prompt"][-1]["text"]
    assert f"Attached files in this environment:\n{copy}" in prompt, (
        f"the agent was not told where the attached file is: {prompt!r}"
    )


def test_the_chats_system_prompt_reaches_the_agent_as_instructions(
    make_user, connect, agent_terminal
):
    person = make_user()
    terminal = connect(person)
    system = f"Always answer in haiku {uuid.uuid4().hex[:6]}"

    with person.client() as client:
        turn = ask_agent(
            client,
            terminal["id"],
            "describe the repo",
            history=[{"role": "system", "content": system}],
        )
        wait_for_reply(client, turn, REPLY_TIMEOUT)

    prompt = agent_terminal.received("session/prompt")[-1]["prompt"]
    instructions = [block for block in prompt if block.get("annotations")]
    assert [block["text"] for block in instructions] == [system], (
        f"the agent did not get the system prompt as instructions: {prompt}"
    )
    assert prompt[-1]["text"] == "describe the repo"


def test_stopping_an_agent_reply_cancels_the_turn_and_the_chat_goes_on(
    make_user, connect, agent_terminal
):
    person = make_user()
    terminal = connect(person)

    with person.client() as client, connected(person) as tab:
        turn = ask_agent(client, terminal["id"], "[slow] big refactor", session_id=tab.sid)
        tab.wait_for(turn.chat_id, "chat:completion", content="Working on it... still going")
        stopped = client.post(f"/api/tasks/chat/{turn.chat_id}/stop")
        assert stopped.status_code == 200, f"stopping the reply failed: {stopped.text}"
        reply = wait_for_reply(client, turn, REPLY_TIMEOUT)
        follow = ask_agent(
            client,
            terminal["id"],
            "smaller step then",
            chat_id=turn.chat_id,
            parent_id=turn.assistant_message_id,
        )
        follow_reply = wait_for_reply(client, follow, REPLY_TIMEOUT)

    assert reply["content"] == "Working on it... still going", reply
    assert agent_terminal.received("session/cancel"), "the agent was never told to stop"
    assert "error" not in follow_reply, (
        f"the chat could not go on after the stop: {follow_reply.get('error')}"
    )
    assert follow_reply["content"].startswith("Agent heard: "), follow_reply
