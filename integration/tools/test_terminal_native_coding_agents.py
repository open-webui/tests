"""Journey: Claude Code, Codex and Pi installed in a terminal run a chat turn in their own protocol.

Open WebUI speaks ACP with most coding agents (`test_terminal_coding_agents.py`) and each of
these three in its own way: Claude Code's stream-json with control requests, the Codex app
server's threads and turns, Pi's commands, events and dialogs. For a person the three behave as
any agent: their models are offered once the terminal is picked, a turn stores thinking, answer
and token use, a tool the agent ran is stored with its result, a request to be allowed something
waits for the person and follows their answer, a question goes through the question card, a
failed turn shows its reason, Stop interrupts the agent and the chat goes on, and the next
message resumes the agent's own conversation. The chat's system prompt and the pictures in the
message, inline or uploaded with the paperclip, reach each agent the way it takes them. Codex
models also bring their reasoning efforts and service tiers as model controls: the default
applies, a pick reaches the turn, and a pick the model does not offer is refused. The agents are
`harness/fake_native_agent.py`, run by a real Open Terminal as its Claude Code, Codex and Pi.

Discriminates: passes on dev c15f28649. Mutations in a backend copy, each turning only the named
tests red here: model controls no longer built from the catalog (the listing and effort tests);
the thinking no longer stored (the turn tests); a finished tool's result dropped (the tool
tests); every request allowed without asking (the permission tests); the answers not handed back
(the question tests); the failed turn not raised (the failed turn tests); no interrupt sent on
Stop (the stop tests); `resume` dropped from the session options (the resume tests); the
instructions dropped (the system prompt tests); pictures dropped (the same tests); the effort and
tier not passed to the turn (the effort test). An uploaded picture is dropped on dev 06491f013
(#33992, fixed by PR #33993), so the picture tests stay red there until the fix merges.

New for 0.13.4. Switching a chat to another model of the same agent goes on in the agent's own
conversation and folder (1782e9bc5), while switching to another agent starts that agent's own
conversation with the chat so far. The reasoning effort of the chat's advanced parameters reaches
Claude Code and Pi (121bc0ec5), none is set when none is picked, and one the model does not offer
ends the reply with the agent's reason before the prompt. Agent model ids keep only letters,
digits, dashes and underscores, so a Pi model's `/` no longer breaks its pin (#34168, PR #34171).
Discriminates: these pass on dev 485103cde. Mutations in a backend copy: the model added to the
keys a resumed session must match (the switching tests); the effort never read from the
parameters (the effort tests); the provider dropped from those keys (the other agent test); the
model name only URL-quoted in the id (the listing test).
"""

from __future__ import annotations

import base64
import json
import re
import time
import uuid
from dataclasses import dataclass, replace

import httpx
import pytest

from harness.chat import ChatTurn, send_message, wait_for_reply
from harness.coding_agent import AgentTerminal, connect_agent_terminal, model_id
from harness.fake_native_agent import (
    COMMAND,
    NOTES,
    QUESTION,
    REFUSAL,
    THOUGHT,
    effort_refusal,
)
from harness.socket_client import connected
from harness.terminal_server import TERMINAL_SERVERS_CONFIG

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
REPLY_TIMEOUT = 60.0
PIXEL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQ"
    "DwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PIXEL_DATA = PIXEL.split(",")[1]
UPLOADED_DATA = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgaPgPAAIDAYAkYfWXAAAAAElFTkSuQmCC"
)
WHOLE_USAGE = {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}


@dataclass(frozen=True)
class Native:
    """One of the three agents, with what its stand-in does that the other two do differently."""

    provider: str
    model: str
    usage: dict
    tool: tuple[str, dict]
    asks_as: str
    allowed: str
    denied: str
    question_id: str
    other_model: str


CLAUDE = Native(
    provider="claude-code",
    model="claude-fake-haiku",
    usage={"input_tokens": 11, "output_tokens": 7},
    tool=("Read", {"path": "notes.txt"}),
    asks_as="Bash",
    allowed="allow",
    denied="deny",
    question_id=QUESTION,
    other_model="claude-fake-sonnet",
)
CODEX = Native(
    provider="codex",
    model="codex-fake",
    usage=WHOLE_USAGE,
    tool=(
        "commandExecution",
        {"id": "cmd_1", "type": "commandExecution", "command": "cat notes.txt"},
    ),
    asks_as="agent_tool",
    allowed="accept",
    denied="decline",
    question_id="colour",
    other_model="codex-fake-mini",
)
PI = Native(
    provider="pi",
    model="acme/pi-fake",
    usage=WHOLE_USAGE,
    tool=("read", {"path": "notes.txt"}),
    asks_as="agent_tool",
    allowed="confirmed",
    denied="declined",
    question_id="value",
    other_model="acme/pi-fake-1.5",
)
NATIVES = [CLAUDE, CODEX, PI]
each_agent = pytest.mark.parametrize("native", NATIVES, ids=lambda native: native.provider)


@pytest.fixture
def connect(admin, preserve, native_agent_terminal: AgentTerminal):
    """`connect(*people)` saves a connection to the terminal for those people, returns its id."""
    preserve(TERMINAL_SERVERS_CONFIG)
    return lambda *people: connect_agent_terminal(admin, native_agent_terminal, *people)["id"]


@pytest.fixture
def tool_permissions(admin, preserve):
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        saved = client.post(CHAT_CONFIG[1], json={**current, "ENABLE_TOOL_PERMISSIONS": True})
    assert saved.status_code == 200, f"switching tool permissions on failed: {saved.text}"


def _send(client: httpx.Client, terminal_id: str, native: Native, text: str, **extra) -> ChatTurn:
    return send_message(
        client,
        text,
        model=model_id(terminal_id, native.model, native.provider),
        terminal_id=terminal_id,
        **extra,
    )


def _ask(client: httpx.Client, terminal_id: str, native: Native, text: str, **extra) -> dict:
    return wait_for_reply(client, _send(client, terminal_id, native, text, **extra), REPLY_TIMEOUT)


def _output_text(message: dict, kind: str) -> str:
    return "".join(
        part.get("text", "")
        for item in message.get("output") or []
        if item.get("type") == kind
        for part in item.get("content") or []
    )


def _pending_call(client: httpx.Client, turn: ChatTurn) -> dict:
    """The agent's request waiting for the person, once it is stored."""
    deadline = time.monotonic() + REPLY_TIMEOUT
    message: dict = {}
    while time.monotonic() < deadline:
        chat = client.get(f"/api/v1/chats/{turn.chat_id}")
        assert chat.status_code == 200, chat.text
        message = chat.json()["chat"]["history"]["messages"][turn.assistant_message_id]
        pending = [item for item in message.get("output") or [] if item.get("status") == "pending"]
        if pending:
            return pending[0]
        assert not message.get("done"), f"the reply ended without asking: {message}"
        time.sleep(0.1)
    raise AssertionError(f"the agent never asked; last stored state: {message}")


def _resolve(client: httpx.Client, turn: ChatTurn, call_id: str, **answer) -> httpx.Response:
    return client.post(
        f"/api/v1/chats/{turn.chat_id}/messages/{turn.assistant_message_id}/resolve",
        json={"call_id": call_id, **answer},
    )


def _option_after(started: list[str], name: str) -> str | None:
    return started[started.index(name) + 1] if name in started else None


def test_claude_code_codex_and_pi_are_offered_with_their_models_and_codex_with_its_controls(
    make_user, connect
):
    person = make_user()
    terminal_id = connect(person)

    with person.client() as client:
        listed = client.get("/api/models", params={"terminal_id": terminal_id})
    assert listed.status_code == 200, f"the model list failed: {listed.text}"
    offered = {model["id"]: model for model in listed.json()["data"] if model.get("agent")}

    def effort(value: str) -> dict:
        return {"label": value, "params": {"reasoning_effort": value}}

    assert {model["name"] for model in offered.values()} == {
        "claude-code/claude-fake-sonnet",
        "claude-code/claude-fake-haiku",
        "codex/codex-fake",
        "codex/codex-fake-mini",
        "pi/acme/pi-fake",
        "pi/acme/pi-fake-1.5",
    }, f"the three agents' models were not offered as expected: {sorted(offered)}"
    # a pinned model opens by its id, so the id keeps nothing a link would encode (#34168)
    unsafe = [id for id in offered if not re.fullmatch(r"[A-Za-z0-9_-]+", id.rsplit(":", 1)[1])]
    assert not unsafe, f"agent model ids carry characters a link encodes: {unsafe}"
    controls = offered[model_id(terminal_id, "codex-fake", "codex")]["info"]["params"]
    assert controls["model_controls"] == {
        "reasoning_effort": {
            "label": "Reasoning effort",
            "options": {value: effort(value) for value in ("low", "medium", "high")},
            "default": "medium",
        },
        "service_tier": {
            "label": "Service tier",
            "options": {
                "standard": {"label": "Standard", "params": {"service_tier": "standard"}},
                "fast": {"label": "Fast", "params": {"service_tier": "fast"}},
            },
        },
    }, f"the Codex model's efforts and tiers did not become its controls: {controls}"
    plain = offered[model_id(terminal_id, "codex-fake-mini", "codex")]["info"]["params"]
    assert plain["model_controls"] == {}, f"a model without efforts got controls: {plain}"


@each_agent
def test_a_turn_stores_the_agents_thinking_answer_and_token_use(
    native, make_user, connect, native_agent_terminal
):
    person = make_user()
    terminal_id = connect(person)
    prompt = f"say hello {uuid.uuid4().hex[:6]}"

    with person.client() as client:
        reply = _ask(client, terminal_id, native, prompt)

    folder = str(native_agent_terminal.terminal.home)
    assert "error" not in reply, f"the turn failed: {reply.get('error')}"
    assert reply["content"] == (
        f"{native.provider} heard: {prompt} [model {native.model}, folder {folder}]"
    ), f"the answer is not what the agent said for its model and folder: {reply['content']!r}"
    assert _output_text(reply, "reasoning") == THOUGHT, reply.get("output")
    assert reply["usage"] == native.usage, f"the token use was not stored: {reply.get('usage')}"


@each_agent
def test_a_tool_the_agent_ran_is_stored_with_its_result(native, make_user, connect):
    person = make_user()
    terminal_id = connect(person)

    with person.client() as client:
        reply = _ask(client, terminal_id, native, "[tool] read it")

    calls = [item for item in reply["output"] if item["type"] == "function_call"]
    results = [item for item in reply["output"] if item["type"] == "function_call_output"]
    name, arguments = native.tool
    assert [(call["name"], call["status"]) for call in calls] == [(name, "completed")], reply
    assert json.loads(calls[0]["arguments"]).items() >= arguments.items(), calls
    assert [result["call_id"] for result in results] == [calls[0]["call_id"]], results
    assert NOTES in results[0]["output"][0]["text"], f"the tool's result was not stored: {results}"
    assert reply["content"] == f"The notes start with: {NOTES}", reply


@each_agent
@pytest.mark.parametrize("action", ["approve", "reject"])
def test_a_request_to_be_allowed_a_command_waits_for_the_person_and_follows_their_answer(
    native, action, make_user, connect, tool_permissions
):
    person = make_user()
    terminal_id = connect(person)

    with person.client() as client:
        turn = _send(
            client,
            terminal_id,
            native,
            "[permission] clean up",
            params={"tool_approval_mode": "ask"},
        )
        asked = _pending_call(client, turn)
        answered = _resolve(client, turn, asked["call_id"], action=action)
        reply = wait_for_reply(client, turn, REPLY_TIMEOUT)

    outcome = native.allowed if action == "approve" else native.denied
    assert asked["name"] == native.asks_as, asked
    assert COMMAND in asked["arguments"], f"the person was not shown the command: {asked}"
    assert answered.status_code == 200, answered.text
    assert reply["content"] == f"Permission outcome: {outcome}", (
        f"the agent did not get the person's answer: {reply}"
    )


@each_agent
def test_an_agents_question_is_answered_through_the_question_card(native, make_user, connect):
    person = make_user()
    terminal_id = connect(person)

    with person.client() as client:
        turn = _send(client, terminal_id, native, "[ask] style the button")
        asked = _pending_call(client, turn)
        answered = _resolve(
            client, turn, asked["call_id"], action="answer", answers={native.question_id: "amber"}
        )
        reply = wait_for_reply(client, turn, REPLY_TIMEOUT)

    assert asked["name"] == "ask_user", asked
    questions = json.loads(asked["arguments"])["questions"]
    assert [question["id"] for question in questions] == [native.question_id], questions
    assert QUESTION in questions[0]["question"], questions
    assert [option["label"] for option in questions[0]["options"]] == ["teal", "amber"], questions
    assert answered.status_code == 200, answered.text
    assert reply["content"] == "Colour answer: amber", (
        f"the answer did not reach the agent: {reply}"
    )


@each_agent
def test_a_failed_turn_shows_the_agents_reason(native, make_user, connect):
    person = make_user()
    terminal_id = connect(person)

    with person.client() as client:
        reply = _ask(client, terminal_id, native, "[refuse] no")

    assert (reply.get("error") or {}).get("content") == REFUSAL, (
        f"the failed turn did not show the agent's reason: {reply}"
    )


def _interrupts(terminal: AgentTerminal, native: Native) -> list[dict]:
    """The messages by which each agent is told to stop its turn."""
    received = terminal.messages(native.provider)
    if native is CLAUDE:
        return [m for m in received if (m.get("request") or {}).get("subtype") == "interrupt"]
    if native is CODEX:
        return [m for m in received if m.get("method") == "turn/interrupt"]
    return [m for m in received if m.get("type") == "abort"]


@each_agent
def test_stop_interrupts_the_agent_and_the_chat_goes_on(
    native, make_user, connect, native_agent_terminal
):
    person = make_user()
    terminal_id = connect(person)
    interrupts_before = len(_interrupts(native_agent_terminal, native))

    with person.client() as client, connected(person) as tab:
        turn = _send(client, terminal_id, native, "[slow] big refactor", session_id=tab.sid)
        tab.wait_for(turn.chat_id, "chat:completion", content="Working on it... still going")
        stopped = client.post(f"/api/tasks/chat/{turn.chat_id}/stop")
        assert stopped.status_code == 200, f"stopping the reply failed: {stopped.text}"
        reply = wait_for_reply(client, turn, REPLY_TIMEOUT)
        follow = _ask(
            client,
            terminal_id,
            native,
            "smaller step then",
            chat_id=turn.chat_id,
            parent_id=turn.assistant_message_id,
        )

    assert reply["content"] == "Working on it... still going", reply
    assert len(_interrupts(native_agent_terminal, native)) == interrupts_before + 1, (
        "the agent was not told to stop its turn"
    )
    assert follow.get("content", "").startswith(f"{native.provider} heard: smaller step then ["), (
        f"the chat could not go on after the stop: {follow}"
    )


@each_agent
def test_the_next_message_resumes_the_agents_own_conversation(
    native, make_user, connect, native_agent_terminal
):
    person = make_user()
    terminal_id = connect(person)

    with person.client() as client:
        first = _send(client, terminal_id, native, "remember the word lantern")
        first_reply = wait_for_reply(client, first, REPLY_TIMEOUT)
        second = _ask(
            client,
            terminal_id,
            native,
            "what was the word?",
            chat_id=first.chat_id,
            parent_id=first.assistant_message_id,
            history=[
                {"role": "user", "content": "remember the word lantern"},
                {"role": "assistant", "content": first_reply["content"]},
            ],
        )

    assert "error" not in second, f"the second turn failed: {second.get('error')}"
    assert second["content"].startswith(f"{native.provider} heard: what was the word? ["), (
        f"a resumed conversation was sent the whole chat again: {second['content']!r}"
    )
    ran_in = native_agent_terminal.conversations(native.provider)[-2:]
    assert len(ran_in) == 2 and ran_in[0] == ran_in[1], (
        f"the second turn did not resume the conversation of the first: {ran_in}"
    )


@each_agent
def test_the_system_prompt_and_a_picture_reach_the_agent(
    native, make_user, connect, native_agent_terminal
):
    person = make_user()
    terminal_id = connect(person)
    system = f"Always answer in haiku {uuid.uuid4().hex[:6]}"
    question = "what is on this?"

    with person.client() as client:
        uploaded = client.post(
            "/api/v1/files/",
            files={"file": ("photo.png", base64.b64decode(UPLOADED_DATA), "image/png")},
        )
        assert uploaded.status_code == 200, uploaded.text
        file_id = uploaded.json()["id"]
        # The paperclip sends the uploaded picture by its file id, a temporary chat inline.
        seen = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": PIXEL}},
            {"type": "image_url", "image_url": {"url": file_id}},
        ]
        reply = _ask(
            client,
            terminal_id,
            native,
            question,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": seen}],
        )

    assert "error" not in reply, f"the turn failed: {reply.get('error')}"
    received = native_agent_terminal.messages(native.provider)
    started = native_agent_terminal.started(native.provider)[-1]
    both = [PIXEL_DATA, UPLOADED_DATA]
    if native is CLAUDE:
        instructions = _option_after(started, "--append-system-prompt")
        blocks = [m for m in received if m.get("type") == "user"][-1]["message"]["content"]
        expected = [{"type": "base64", "media_type": "image/png", "data": data} for data in both]
        pictures = [block["source"] for block in blocks if block["type"] == "image"]
    elif native is CODEX:
        thread = [m for m in received if m.get("method") == "thread/start"][-1]["params"]
        turn = [m for m in received if m.get("method") == "turn/start"][-1]["params"]
        instructions = thread.get("developerInstructions")
        expected = [f"data:image/png;base64,{data}" for data in both]
        pictures = [part["url"] for part in turn["input"] if part["type"] == "image"]
    else:
        instructions = _option_after(started, "--append-system-prompt")
        expected = [{"type": "image", "mimeType": "image/png", "data": data} for data in both]
        pictures = [m for m in received if m.get("type") == "prompt"][-1]["images"]
    assert instructions == system, (
        f"{native.provider} did not get the system prompt as instructions: {instructions!r}"
    )
    assert pictures == expected, f"{native.provider} did not get both pictures (#33992): {pictures}"


def test_a_codex_turn_runs_with_the_models_default_effort_or_the_persons_pick(
    make_user, connect, native_agent_terminal
):
    person = make_user()
    terminal_id = connect(person)
    codex = model_id(terminal_id, CODEX.model, CODEX.provider)

    def last_turn() -> dict:
        turns = [
            m for m in native_agent_terminal.messages("codex") if m.get("method") == "turn/start"
        ]
        return {key: turns[-1]["params"].get(key) for key in ("effort", "serviceTier")}

    with person.client() as client:
        _ask(client, terminal_id, CODEX, "with the defaults")
        by_default = last_turn()
        picked = {"reasoning_effort": "high", "service_tier": "fast"}
        _ask(
            client, terminal_id, CODEX, "with my picks", params={"model_controls": {codex: picked}}
        )
        by_pick = last_turn()
        refused = client.post(
            "/api/chat/completions",
            json={
                "model": codex,
                "terminal_id": terminal_id,
                "messages": [{"role": "user", "content": "with an effort it does not have"}],
                "stream": True,
                "params": {"model_controls": {codex: {"reasoning_effort": "extreme"}}},
            },
        )

    assert by_default == {"effort": "medium", "serviceTier": None}, (
        f"the turn did not run with the model's default effort: {by_default}"
    )
    assert by_pick == {"effort": "high", "serviceTier": "fast"}, (
        f"the person's picks did not reach the turn: {by_pick}"
    )
    assert refused.status_code == 400, (
        f"an effort the model does not offer was taken: {refused.text}"
    )
    assert "reasoning_effort" in refused.json()["detail"], refused.text


@each_agent
def test_switching_to_another_model_of_the_same_agent_resumes_its_session_and_folder(
    native, make_user, connect, native_agent_terminal
):
    person = make_user()
    terminal_id = connect(person)
    home = native_agent_terminal.terminal.home
    started_in = home / f"switch-{uuid.uuid4().hex[:6]}"
    started_in.mkdir()

    with person.client() as client:
        first = _send(
            client, terminal_id, native, "remember the word lantern", terminal_cwd=str(started_in)
        )
        first_reply = wait_for_reply(client, first, REPLY_TIMEOUT)
        assert "error" not in first_reply, f"the first turn failed: {first_reply.get('error')}"
        switched = replace(native, model=native.other_model)
        # the page names the terminal's current folder again with the next message
        second = _ask(
            client,
            terminal_id,
            switched,
            "what was the word?",
            chat_id=first.chat_id,
            parent_id=first.assistant_message_id,
            terminal_cwd=str(home),
            history=[
                {"role": "user", "content": "remember the word lantern"},
                {"role": "assistant", "content": first_reply["content"]},
            ],
        )

    assert second.get("content") == (
        f"{native.provider} heard: what was the word? "
        f"[model {native.other_model}, folder {started_in}]"
    ), f"the switched model did not go on in the agent's session and folder: {second}"
    ran_in = native_agent_terminal.conversations(native.provider)[-2:]
    assert len(ran_in) == 2 and ran_in[0] == ran_in[1], (
        f"switching the model started a new conversation of the agent: {ran_in}"
    )


def test_switching_to_another_agent_starts_that_agents_own_conversation(
    make_user, connect, native_agent_terminal
):
    person = make_user()
    terminal_id = connect(person)

    with person.client() as client:
        first = _send(client, terminal_id, CLAUDE, "remember the word lantern")
        first_reply = wait_for_reply(client, first, REPLY_TIMEOUT)
        second = _ask(
            client,
            terminal_id,
            CODEX,
            "what was the word?",
            chat_id=first.chat_id,
            parent_id=first.assistant_message_id,
            history=[
                {"role": "user", "content": "remember the word lantern"},
                {"role": "assistant", "content": first_reply["content"]},
            ],
        )

    claude_session = native_agent_terminal.conversations("claude-code")[-1]
    codex_thread = native_agent_terminal.conversations("codex")[-1]
    assert "error" not in second, f"the turn with the other agent failed: {second.get('error')}"
    assert codex_thread != claude_session, (
        f"Codex was told to resume Claude Code's conversation {claude_session}"
    )
    # a new conversation is given the chat so far
    assert "remember the word lantern" in second["content"], (
        f"the other agent did not get the conversation so far: {second['content']!r}"
    )


def _effort_messages(terminal: AgentTerminal, native: Native) -> list[str]:
    """The reasoning efforts the agent was told to use, oldest first."""
    received = terminal.messages(native.provider)
    if native is CLAUDE:
        requests = [m.get("request") or {} for m in received if m.get("type") == "control_request"]
        return [
            r["settings"]["effortLevel"]
            for r in requests
            if r.get("subtype") == "apply_flag_settings"
        ]
    return [m["level"] for m in received if m.get("type") == "set_thinking_level"]


@pytest.mark.parametrize("native", [CLAUDE, PI], ids=lambda native: native.provider)
def test_the_reasoning_effort_reaches_the_agent_and_one_its_model_lacks_stops_the_reply(
    native, make_user, connect, native_agent_terminal
):
    person = make_user()
    terminal_id = connect(person)
    told_before = len(_effort_messages(native_agent_terminal, native))

    with person.client() as client:
        plain = _ask(client, terminal_id, native, "no effort set")
        told_plain = _effort_messages(native_agent_terminal, native)[told_before:]
        picked = _ask(
            client, terminal_id, native, "think hard", params={"reasoning_effort": "high"}
        )
        told_picked = _effort_messages(native_agent_terminal, native)[told_before:]
        lacking = _ask(
            client, terminal_id, native, "think harder", params={"reasoning_effort": "max"}
        )

    assert "error" not in plain and told_plain == [], (
        f"an effort was set without one being picked: {told_plain}, {plain.get('error')}"
    )
    assert "error" not in picked, f"the turn with an effort failed: {picked.get('error')}"
    assert told_picked == ["high"], f"the picked effort did not reach the agent: {told_picked}"
    assert picked["content"].startswith(f"{native.provider} heard: think hard ["), picked
    assert (lacking.get("error") or {}).get("content") == effort_refusal("max", native.model), (
        f"an effort the model lacks did not stop the reply with the agent's reason: {lacking}"
    )
    assert not (lacking.get("content") or "").strip(), (
        f"the agent answered with an effort its model lacks: {lacking['content']!r}"
    )
