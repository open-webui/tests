"""Journey: what a terminal's coding agents look like to a person when the terminal misbehaves.

The terminal here is the fake (`harness.agent_carrier`): it lists the agents the test gives it
and carries a scripted agent's turn over the session socket, so a test can decide what a real
Open Terminal only does by accident. Listing: only agents the terminal reports as available are
offered, each model once; a catalog of another protocol version offers none; the list is asked
for the folder the page names, else for the terminal's own; and a terminal that cannot list its
agents leaves the other models listed. Running: an agent Open WebUI has no client for and a
session the terminal refuses to start end the reply with the reason; an agent works in the folder
its chat started in, also after the person browsed elsewhere in that chat, and a terminal with no
folder at all is named as the problem; an agent's plan is stored as a step of the reply, a typed
form it sends is answered in the form's types or declined, and a Cursor agent is signed in before
its session starts; an attachment the terminal does not take fails the turn before the agent is
asked, and a file of another terminal is refused. The socket: after a hang-up Open WebUI
comes back, says how much it has read and sends again what was never acknowledged, so the reply
is whole and the agent gets each message once; output lost on the way ends the reply with an
error; a socket that stays away for a minute ends it with "Agent environment disconnected". In a
chat shared for replies a collaborator's message goes on in the owner's agent session on the
owner's terminal, which the collaborator has no access to alone.

Regression (open-webui#34062): while an agent streamed its reply, the whole chat was saved again
about once a second, so a reply wrote more the longer it ran and the longer the chat already was.
Each command the agent was allowed without asking saved it once more. Two tests count how often
the stored chat changes while a scripted agent streams for about one and a half and for about
four and a half seconds, and while it runs three and twelve commands it need not ask for: the two
counts are equal when saving follows neither the length of the reply nor what the agent does in
it. A page opening the chat mid-reply is still given what has streamed. How the streamed text is
kept mid-reply is left open: the fix keeps it in the reply's stream that page loads read, not in
the reply's stored message. Both red on dev c15f28649 (the longer reply and the one with more
commands saved the chat more often); fixed on dev (aa19e7f86, 8588bd2a7), green there.

Discriminates: others pass on c15f28649. Mutations in a backend copy, each turning only the named
test red: `available` no longer checked in discovery (the available test); the protocol version
check dropped (the version test); `cwd` not passed on by `/api/models` (the folder listing test);
the discovery error not caught in `/api/models` (the failing terminal test); the provider check
dropped in the runner (the no-client test); the start error's detail replaced (the refused start
test); the previous session's folder ignored (the folder test); the attachment error check
dropped (the attachment test); the other-terminal check dropped (the other terminal test);
pictures dropped from the prompt (the picture test, and the picture tests of
test_terminal_native_coding_agents.py); plan updates ignored (the plan test); typed answers
handed back as text (the typed form test); no sign-in sent for Cursor (the Cursor test); the
cursor always sent as 0 (the hang-up test); the unacknowledged input not sent again (the
unacknowledged test, by its time limit); the sequence check dropped (the lost output test); the
60 s limit raised to 600 s (the stays-away test, by its time limit); the session's terminal owner
replaced by the sender (the collaborator test).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Iterator

import httpx
import pytest

from harness.access import grant
from harness.agent_carrier import (
    FOLDER,
    MODEL,
    SESSION,
    AgentRun,
    ScriptedAgents,
    agent,
    catalog,
    heard,
    serving_agents,
)
from harness.backends import read_rows
from harness.chat import ChatTurn, send_message, wait_for_reply
from harness.coding_agent import model_id
from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

REPLY_TIMEOUT = 60.0
UNAVAILABLE = "Agent model is unavailable in the selected terminal"
DISCONNECTED = "Agent environment disconnected. Reconnect before continuing this chat."
PIXEL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQ"
    "DwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def agents(admin, preserve) -> Iterator[ScriptedAgents]:
    """The fake terminal's scripted agents; `agents.terminal_id` once `connect` has saved it."""
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as terminal:
        yield serving_agents(terminal, "opencode", "aider", "cursor")


@pytest.fixture
def connect(admin, agents):
    """`connect(*people)` saves the fake terminal for those people and returns its id."""

    def save(*people) -> str:
        connection = agents.terminal.connection()
        connection["config"]["access_grants"] = [read_grant(person.id) for person in people]
        with admin.client() as client:
            configure_terminals(client, connection)
        return connection["id"]

    return save


def _listed(client: httpx.Client, terminal_id: str, **params) -> tuple[set[str], set[str]]:
    """The agent models and the other models the model list shows for the terminal."""
    listed = client.get("/api/models", params={"terminal_id": terminal_id, **params})
    assert listed.status_code == 200, f"the model list failed: {listed.status_code} {listed.text}"
    ids = {model["id"] for model in listed.json()["data"]}
    agent_ids = {listed_id for listed_id in ids if listed_id.startswith("agent:")}
    return agent_ids, ids - agent_ids


def _ask(
    client: httpx.Client, terminal_id: str, text: str, provider: str = "opencode", **extra
) -> tuple[ChatTurn, dict]:
    turn = send_message(
        client,
        text,
        model=model_id(terminal_id, MODEL, provider),
        terminal_id=terminal_id,
        **extra,
    )
    return turn, wait_for_reply(client, turn, REPLY_TIMEOUT)


def _error(reply: dict) -> str | None:
    return (reply.get("error") or {}).get("content")


def _with_error(client: httpx.Client, turn: ChatTurn, reply: dict) -> dict:
    """The done reply once its error is stored: a turn that fails after the agent started is
    marked done a few milliseconds before the error is written."""
    deadline = time.monotonic() + 10
    while not _error(reply) and time.monotonic() < deadline:
        time.sleep(0.1)
        stored = client.get(f"/api/v1/chats/{turn.chat_id}")
        stored.raise_for_status()
        reply = stored.json()["chat"]["history"]["messages"][turn.assistant_message_id]
    return reply


def _pending(client: httpx.Client, turn: ChatTurn) -> dict:
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


def _prompt(run: AgentRun) -> list[dict]:
    prompts = run.requests("session/prompt")
    assert prompts, f"the agent was never asked on this session: {run.received}"
    return prompts[-1]["prompt"]


def test_only_agents_the_terminal_reports_as_available_are_offered_each_model_once(
    make_user, agents, connect
):
    person = make_user()
    terminal_id = connect(person)
    twice = [{"id": "alpha", "name": "Alpha"}, {"id": "alpha", "name": "Alpha"}, {"id": "beta"}]
    missing = agent("gemini", {"id": "hidden"}, available=False)
    agents.catalog = catalog(agent("opencode", *twice), missing)

    with person.client() as client:
        listed = client.get("/api/models", params={"terminal_id": terminal_id}).json()["data"]
        agents.catalog = catalog(agent("opencode", *twice), {**missing, "available": True})
        once_installed, _ = _listed(client, terminal_id)

    offered = sorted(model["id"] for model in listed if model["id"].startswith("agent:"))
    assert offered == [
        model_id(terminal_id, "alpha"),
        model_id(terminal_id, "beta"),
    ], f"the available agent's models were not offered once each: {offered}"
    assert model_id(terminal_id, "hidden", "gemini") in once_installed, (
        f"the same agent was not offered once the terminal reported it available: {once_installed}"
    )


def test_a_catalog_of_another_protocol_version_offers_no_agents(make_user, agents, connect):
    person = make_user()
    terminal_id = connect(person)

    with person.client() as client:
        agents.catalog = catalog(agent(), protocol_version=2)
        newer, others = _listed(client, terminal_id)
        agents.catalog = catalog(agent(), protocol_version=1)
        current, _ = _listed(client, terminal_id)

    assert newer == set(), f"agents of an unknown protocol version were offered: {newer}"
    assert MOCK_MODEL_ID in others, f"the other models went with them: {others}"
    assert current == {model_id(terminal_id, MODEL)}, (
        f"the same catalog in the known version was not offered: {current}"
    )


def test_the_agents_are_listed_for_the_folder_the_page_names_else_the_terminals_own(
    make_user, agents, connect
):
    person = make_user()
    terminal_id = connect(person)
    agents.catalog = lambda folder: catalog(
        agent("opencode", {"id": f"model-of-{(folder or 'nowhere').rsplit('/', 1)[-1]}"})
    )

    with person.client() as client:
        named, _ = _listed(client, terminal_id, cwd="/srv/atlas")
        own, _ = _listed(client, terminal_id)

    assert named == {model_id(terminal_id, "model-of-atlas")}, (
        f"the agents were not listed for the folder the page named: {named}"
    )
    assert own == {model_id(terminal_id, f"model-of-{FOLDER.rsplit('/', 1)[-1]}")}, (
        f"without a folder the agents were not listed for the terminal's own: {own}"
    )


def test_a_terminal_that_cannot_list_its_agents_leaves_the_other_models_listed(
    admin, make_user, agents, connect
):
    person = make_user()
    terminal_id = connect(person)
    failures = {
        "an error": json_answer({"detail": "agents are broken"}, status=500),
        "a refusal": json_answer({"detail": "Invalid API key"}, status=401),
        "a page instead of JSON": (200, {"Content-Type": "text/html"}, b"<h1>proxy login</h1>"),
        "a catalog without ids": json_answer(
            {"protocol_version": 1, "agents": [{"available": True, "models": [{}]}]}
        ),
    }

    with person.client() as client:
        healthy, _ = _listed(client, terminal_id)
        seen = {}
        for name, answer in failures.items():
            agents.catalog = lambda folder, answer=answer: answer
            seen[name] = _listed(client, terminal_id)

    assert healthy == {model_id(terminal_id, MODEL)}, f"the healthy terminal offered {healthy}"
    for name, (offered, others) in seen.items():
        assert offered == set(), f"a terminal answering with {name} still offered agents: {offered}"
        assert MOCK_MODEL_ID in others, f"with {name} the other models were not listed: {others}"


def test_an_agent_open_webui_has_no_client_for_ends_the_reply_with_the_reason(
    make_user, agents, connect
):
    person = make_user()
    terminal_id = connect(person)
    agents.catalog = catalog(agent("aider"), agent("opencode"))

    with person.client() as client:
        _, unknown = _ask(client, terminal_id, "refactor with aider", provider="aider")
        started_for_it = len(agents.starts)
        _, known = _ask(client, terminal_id, "refactor with opencode")

    assert _error(unknown) == "Unsupported terminal agent", (
        f"the reply of an agent without a client did not say why it failed: {unknown}"
    )
    assert started_for_it == 0, "an agent session was started for an agent without a client"
    assert known.get("content") == "Scripted agent heard: refactor with opencode", known


def test_a_session_the_terminal_refuses_to_start_ends_the_reply_with_its_reason(
    make_user, agents, connect
):
    person = make_user()
    terminal_id = connect(person)
    busy = "Another agent turn is running in this conversation"

    with person.client() as client:
        agents.refuse_start = (409, {"detail": busy})
        turn, refused = _ask(client, terminal_id, "start while busy")
        refused = _with_error(client, turn, refused)
        agents.refuse_start = None
        _, after = _ask(
            client,
            terminal_id,
            "start again",
            chat_id=turn.chat_id,
            parent_id=turn.assistant_message_id,
        )

    assert _error(refused) == busy, f"the terminal's reason did not reach the reply: {refused}"
    assert after.get("content") == "Scripted agent heard: start again", (
        f"the chat did not go on once the terminal started a session: {after}"
    )


def test_a_start_the_terminal_rejects_as_invalid_ends_the_reply_without_the_rejected_request(
    make_user, agents, connect
):
    """Regression (open-webui#34172): the reply showed the terminal's validation answer, the whole
    system prompt included. Red on dev 782f4b705."""
    person = make_user()
    terminal_id = connect(person)
    system_prompt = "Answer in the voice of the release notes. " * 1700
    agents.refuse_start = (
        422,
        {
            "detail": [
                {
                    "type": "string_too_long",
                    "loc": ["body", "instructions"],
                    "msg": "String should have at most 65536 characters",
                    "input": system_prompt,
                    "ctx": {"max_length": 65536},
                }
            ]
        },
    )

    with person.client() as client:
        turn, refused = _ask(
            client,
            terminal_id,
            "start with a long system prompt",
            history=[{"role": "system", "content": system_prompt}],
        )
        refused = _with_error(client, turn, refused)

    started = agents.terminal.requests_to("/agents/opencode/sessions")
    assert started and json.loads(started[-1].body)["instructions"] == system_prompt, (
        "the system prompt did not reach the terminal as the agent's instructions"
    )
    error = _error(refused)
    assert error, f"the refused start left no error on the reply: {refused}"
    assert system_prompt[:60] not in error and "string_too_long" not in error, (
        f"the reply shows the terminal's validation answer: {error[:300]}"
    )
    assert len(error) < 200, f"the reply's error is not a short message: {error[:300]}"


def test_an_agent_works_in_the_folder_its_chat_started_in(make_user, agents, connect):
    person = make_user()
    terminal_id = connect(person)

    with person.client() as client:
        first, _ = _ask(client, terminal_id, "start in atlas", terminal_cwd="/srv/atlas")
        chosen = agents.starts[-1].options["cwd"]
        browsed = client.post(
            f"/api/v1/terminals/{terminal_id}/files/cwd",
            json={"path": "/srv/elsewhere"},
            headers={"X-Session-Id": first.chat_id},
        )
        assert browsed.status_code == 200, f"moving the chat's terminal failed: {browsed.text}"
        _ask(
            client,
            terminal_id,
            "go on",
            chat_id=first.chat_id,
            parent_id=first.assistant_message_id,
        )
        continued = agents.starts[-1].options["cwd"]
        agents.folder = "/srv/elsewhere"
        _ask(client, terminal_id, "start where the terminal is")
        default = agents.starts[-1].options["cwd"]
        agents.folder = None
        _, nowhere = _ask(client, terminal_id, "start nowhere")

    assert chosen == "/srv/atlas", f"the agent did not start in the chat's chosen folder: {chosen}"
    assert continued == "/srv/atlas", (
        f"the next turn followed the file browser to another folder: {continued}"
    )
    assert default == "/srv/elsewhere", (
        f"a chat without a chosen folder did not start in the terminal's own: {default}"
    )
    assert _error(nowhere) == "Agent working directory is unavailable", (
        f"a terminal without a folder was not named as the problem: {nowhere}"
    )
    assert len(agents.starts) == 3, "a session was started without a folder to work in"


def _upload(client: httpx.Client, name: str) -> list[dict]:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "false"},
        files={"file": (name, b"ship it on friday\n", "text/plain")},
    )
    assert uploaded.status_code == 200, f"uploading the file failed: {uploaded.text}"
    return [{"type": "file", "id": uploaded.json()["id"], "name": name}]


def test_an_attachment_the_terminal_does_not_take_fails_the_turn_before_the_agent_is_asked(
    make_user, agents, connect
):
    person = make_user()
    terminal_id = connect(person)
    landed = f"{FOLDER}/brief-{uuid.uuid4().hex[:6]}.txt"

    with person.client() as client:
        attached = _upload(client, "brief.txt")
        agents.terminal.route("POST", "/files/upload", json_answer({"detail": "disk full"}, 507))
        _, refused = _ask(
            client, terminal_id, "read the brief", message_files=attached, files=attached
        )
        started_for_it = len(agents.starts)
        agents.terminal.route("POST", "/files/upload", json_answer({"path": landed}))
        _, taken = _ask(
            client, terminal_id, "read the brief", message_files=attached, files=attached
        )

    assert _error(refused) == "An attachment could not be uploaded to the agent environment", (
        f"the failed upload was not named on the reply: {refused}"
    )
    assert started_for_it == 0, "the agent was started although its attachment never arrived"
    assert not _error(taken), f"the turn failed with the upload taken: {taken}"
    assert _prompt(agents.starts[-1])[-1]["text"].endswith(
        f"Attached files in this environment:\n{landed}"
    ), f"the agent was not told where the file is: {_prompt(agents.starts[-1])}"


def test_a_file_of_another_terminal_is_refused_and_one_of_the_agents_own_is_named_to_it(
    make_user, agents, connect
):
    person = make_user()
    terminal_id = connect(person)

    def notes(of_terminal: str) -> list[dict]:
        return [
            {
                "type": "filesystem",
                "terminal_id": of_terminal,
                "path": "/srv/notes.txt",
                "name": "notes.txt",
            }
        ]

    with person.client() as client:
        _, foreign = _ask(client, terminal_id, "read the notes", files=notes("another-terminal"))
        started_for_it = len(agents.starts)
        _, own = _ask(client, terminal_id, "read the notes", files=notes(terminal_id))

    assert _error(foreign) == "Attachment belongs to another terminal", (
        f"a file of another terminal was not refused: {foreign}"
    )
    assert started_for_it == 0, "the agent was started with a file of another terminal"
    assert not _error(own), f"a file of the agent's own terminal failed the turn: {own}"
    assert _prompt(agents.starts[-1])[-1]["text"].endswith(
        "Attached files in this environment:\n/srv/notes.txt"
    ), f"the agent was not told about the file: {_prompt(agents.starts[-1])}"


def test_a_picture_in_the_message_reaches_an_agent_that_takes_pictures(make_user, agents, connect):
    person = make_user()
    terminal_id = connect(person)
    seen = [
        {"type": "text", "text": "what is on this?"},
        {"type": "image_url", "image_url": {"url": PIXEL}},
    ]

    with person.client() as client:
        _, reply = _ask(
            client, terminal_id, "what is on this?", messages=[{"role": "user", "content": seen}]
        )

    assert not _error(reply), f"the turn with a picture failed: {reply}"
    pictures = [block for block in _prompt(agents.starts[-1]) if block.get("type") == "image"]
    assert pictures == [{"type": "image", "mimeType": "image/png", "data": PIXEL.split(",")[1]}], (
        f"the agent did not get the picture: {_prompt(agents.starts[-1])}"
    )


def test_an_agents_plan_is_stored_as_a_plan_step_of_the_reply(make_user, agents, connect):
    person = make_user()
    terminal_id = connect(person)
    entries = [
        {"content": "Read the parser", "priority": "high", "status": "completed"},
        {"content": "Write the tests", "priority": "medium", "status": "pending"},
    ]

    def plan_first(run: AgentRun, params: dict) -> None:
        update = {"sessionUpdate": "plan", "entries": entries}
        run.emit({"method": "session/update", "params": {"sessionId": SESSION, "update": update}})
        run.say("Planned.")

    agents.turn = plan_first
    with person.client() as client:
        _, reply = _ask(client, terminal_id, "plan the refactor")

    steps = [item for item in reply.get("output") or [] if item.get("name") == "Plan"]
    assert [(step["type"], step["status"]) for step in steps] == [("function_call", "completed")], (
        f"the agent's plan was not stored as one finished step: {reply.get('output')}"
    )
    assert json.loads(steps[0]["arguments"]) == {"entries": entries}, steps
    assert reply["content"] == "Planned.", reply


def test_a_typed_form_from_an_agent_is_answered_in_its_types_or_declined(
    make_user, agents, connect
):
    person = make_user()
    terminal_id = connect(person)
    form = {
        "mode": "form",
        "message": "Before the release",
        "requestedSchema": {
            "type": "object",
            "properties": {
                "tag": {"type": "boolean", "title": "Tag it"},
                "copies": {"type": "integer", "title": "Copies"},
                "note": {"type": "string", "title": "Note"},
            },
            "required": ["tag", "copies"],
        },
    }
    answers = []

    def ask_the_form(run: AgentRun, params: dict) -> None:
        answers.append(run.ask("elicitation/create", {"sessionId": SESSION, **form})["result"])
        run.say("Noted.")

    def answer_with(client: httpx.Client, **given) -> dict:
        turn = send_message(
            client, "release it", model=model_id(terminal_id, MODEL), terminal_id=terminal_id
        )
        asked = _pending(client, turn)
        answered = client.post(
            f"/api/v1/chats/{turn.chat_id}/messages/{turn.assistant_message_id}/resolve",
            json={"call_id": asked["call_id"], "action": "answer", "answers": given},
        )
        assert answered.status_code == 200, f"answering the form failed: {answered.text}"
        wait_for_reply(client, turn, REPLY_TIMEOUT)
        return asked

    agents.turn = ask_the_form
    with person.client() as client:
        asked = answer_with(client, tag="Yes", copies="3", note="(Skip)")
        answer_with(client, tag="No", copies="three")

    questions = {q["id"]: q for q in json.loads(asked["arguments"])["questions"]}
    assert list(questions) == ["tag", "copies", "note"], f"the form's fields: {questions}"
    assert [o["label"] for o in questions["tag"]["options"]] == ["Yes", "No"], questions["tag"]
    assert [o["label"] for o in questions["note"]["options"]] == ["(Skip)"], questions["note"]
    assert answers[0] == {"action": "accept", "content": {"tag": True, "copies": 3}}, (
        f"the agent did not get the answers in the form's types: {answers[0]}"
    )
    assert answers[1] == {"action": "decline"}, (
        f"a number that is none was not declined: {answers[1]}"
    )


def test_a_cursor_agent_is_signed_in_before_its_session_starts(make_user, agents, connect):
    person = make_user()
    terminal_id = connect(person)
    agents.catalog = catalog(agent("cursor"), agent("opencode"))

    with person.client() as client:
        _, by_cursor = _ask(client, terminal_id, "hello cursor", provider="cursor")
        cursor_run = agents.starts[-1]
        _, by_opencode = _ask(client, terminal_id, "hello opencode")
        opencode_run = agents.starts[-1]

    assert by_cursor.get("content") == "Scripted agent heard: hello cursor", by_cursor
    assert by_opencode.get("content") == "Scripted agent heard: hello opencode", by_opencode
    asked = [message.get("method") for message in cursor_run.received]
    assert cursor_run.requests("authenticate") == [{"methodId": "cursor_login"}], (
        f"the Cursor agent was not signed in with its login: {asked}"
    )
    assert asked.index("authenticate") < asked.index("session/new"), asked
    assert opencode_run.requests("authenticate") == [], "another agent was signed in as Cursor"


def test_a_reply_goes_on_after_the_agent_socket_hung_up(make_user, agents, connect):
    person = make_user()
    terminal_id = connect(person)
    read_before_the_hang_up = []

    def hang_up_halfway(run: AgentRun, params: dict) -> None:
        run.say("First half. ")
        run.wait_until_read()
        read_before_the_hang_up.append(run.acknowledged)
        run.drop()
        run.say("Second half.")

    agents.turn = hang_up_halfway
    with person.client() as client:
        _, reply = _ask(client, terminal_id, "a long task")

    run = agents.starts[-1]
    assert not _error(reply), f"the reply failed after the socket hung up: {reply}"
    assert reply["content"] == "First half. Second half.", (
        f"the reply is not whole after the reconnect: {reply['content']!r}"
    )
    assert [sign_in.get("cursor") for sign_in in run.connections] == [
        0,
        read_before_the_hang_up[0],
    ], f"Open WebUI did not come back saying how much it had read: {run.connections}"


def test_a_message_the_terminal_never_acknowledged_is_sent_again_and_handled_once(
    make_user, agents, connect
):
    person = make_user()
    terminal_id = connect(person)
    agents.swallow = lambda message: message.get("method") == "session/prompt"

    with person.client() as client:
        _, reply = _ask(client, terminal_id, "did you get this?")

    run = agents.starts[-1]
    prompts = [frame for frame in run.inputs if '"session/prompt"' in frame["data"]]
    assert reply.get("content") == "Scripted agent heard: did you get this?", (
        f"the reply did not finish after the unacknowledged message: {reply}"
    )
    assert len(run.connections) == 2, f"Open WebUI did not reconnect: {run.connections}"
    assert len(prompts) == 2 and prompts[0] == prompts[1], (
        f"the unacknowledged message was not sent again as it was: {prompts}"
    )
    assert len(run.requests("session/prompt")) == 1, "the agent was asked twice"


def test_agent_output_lost_on_the_way_ends_the_reply_with_an_error(make_user, agents, connect):
    person = make_user()
    terminal_id = connect(person)

    def lose_the_middle(run: AgentRun, params: dict) -> None:
        run.say("One. ")
        run.skip_a_frame()
        run.say("Three.")

    with person.client() as client:
        _, whole = _ask(client, terminal_id, "count")
        agents.turn = lose_the_middle
        lost, broken = _ask(client, terminal_id, "count")
        broken = _with_error(client, lost, broken)

    assert whole.get("content") == "Scripted agent heard: count", whole
    assert _error(broken) == "Agent protocol output was lost", (
        f"a reply with a piece missing was not marked as failed: {broken}"
    )
    assert "Three." not in broken.get("content", ""), (
        f"the reply went on past the missing piece: {broken['content']!r}"
    )


def _stored_chat(instance, chat_id: str):
    return read_rows(instance, "SELECT chat FROM chat WHERE id = :id", {"id": chat_id})[0]["chat"]


def _saves_while(
    client: httpx.Client, instance, agents: ScriptedAgents, terminal_id: str, work
) -> tuple[int, str, dict]:
    """One agent turn doing `work(run)`: how often the stored chat changed meanwhile, what a page
    opening the chat right after is given of the reply, and the finished reply."""
    working, worked, release = threading.Event(), threading.Event(), threading.Event()

    def turn(run: AgentRun, params: dict) -> None:
        working.set()
        work(run)
        run.wait_until_read()
        worked.set()
        release.wait(REPLY_TIMEOUT)

    agents.turn = turn
    sent = send_message(
        client, "go on", model=model_id(terminal_id, MODEL), terminal_id=terminal_id
    )
    try:
        assert working.wait(REPLY_TIMEOUT), "the agent was never asked"
        stored = _stored_chat(instance, sent.chat_id)
        saves = 0
        deadline = time.monotonic() + REPLY_TIMEOUT
        while not worked.is_set():
            assert time.monotonic() < deadline, "the agent never finished its work"
            latest = _stored_chat(instance, sent.chat_id)
            saves += latest != stored
            stored = latest
        opened = client.get(f"/api/v1/chats/{sent.chat_id}")
        assert opened.status_code == 200, opened.text
        shown = opened.json()["chat"]["history"]["messages"][sent.assistant_message_id]
    finally:
        release.set()
    return saves, shown.get("content") or "", wait_for_reply(client, sent, REPLY_TIMEOUT)


def _streaming(pieces: int):
    def work(run: AgentRun) -> None:
        for number in range(pieces):
            run.say(f"piece {number} ")
            # the agent's own pace: the bug saved the chat once for every second the reply ran
            time.sleep(0.1)

    return work


def _running_commands(commands: int):
    def work(run: AgentRun) -> None:
        for number in range(commands):
            request = {
                "sessionId": SESSION,
                "toolCall": {"toolCallId": f"shell-{number}", "title": "List the folder"},
                "options": [{"optionId": "allow-it", "name": "Allow", "kind": "allow_once"}],
            }
            allowed = run.ask("session/request_permission", request)["result"]["outcome"]
            assert allowed.get("optionId") == "allow-it", f"the command was not allowed: {allowed}"
            # a command takes its time; the sampling between two of them sees each save
            time.sleep(0.1)

    return work


def test_an_agent_reply_saves_the_chat_no_more_often_the_longer_it_streams(
    make_user, agents, connect, instance
):
    """open-webui#34062: the chat was saved again for every second an agent's reply streamed."""
    person = make_user()
    terminal_id = connect(person)

    with person.client() as client:
        short_saves, short_shown, short = _saves_while(
            client, instance, agents, terminal_id, _streaming(15)
        )
        long_saves, long_shown, long = _saves_while(
            client, instance, agents, terminal_id, _streaming(45)
        )

    assert long_saves == short_saves, (
        f"a reply three times as long saved the whole chat {long_saves} times while it streamed, "
        f"the short one {short_saves} times"
    )
    for shown in (short_shown, long_shown):
        assert shown.startswith("piece 0 "), (
            f"a page opening the chat mid-reply was not given what had streamed: {shown!r}"
        )
    for reply, pieces in ((short, 15), (long, 45)):
        whole = "".join(f"piece {number} " for number in range(pieces))
        assert reply.get("content") == whole, f"the finished reply is not whole: {reply}"
        assert not _error(reply), f"the reply failed: {reply}"


def test_an_agent_allowed_its_commands_without_asking_saves_the_chat_no_more_often_the_more_it_runs(
    make_user, agents, connect, instance
):
    """open-webui#34062: each command an agent was allowed unasked saved the whole chat again."""
    person = make_user()
    terminal_id = connect(person)

    with person.client() as client:
        few_saves, _, few = _saves_while(
            client, instance, agents, terminal_id, _running_commands(3)
        )
        many_saves, _, many = _saves_while(
            client, instance, agents, terminal_id, _running_commands(12)
        )

    assert many_saves == few_saves, (
        f"twelve commands allowed without asking saved the whole chat {many_saves} times, "
        f"three of them {few_saves} times"
    )
    for reply, commands in ((few, 3), (many, 12)):
        calls = [item for item in reply.get("output") or [] if item.get("type") == "function_call"]
        assert len(calls) == commands, f"the reply does not hold its {commands} commands: {reply}"
        assert not _error(reply), f"the reply failed: {reply}"


@pytest.mark.slow
def test_an_agent_socket_that_stays_away_for_a_minute_ends_the_reply(make_user, agents, connect):
    person = make_user()
    terminal_id = connect(person)

    def leave_for_good(run: AgentRun, params: dict) -> None:
        run.say("Working on it. ")
        run.wait_until_read()
        agents.terminal.refuse_sessions(503, f"/agents/sessions/{run.id}/connect")
        run.drop()

    agents.turn = leave_for_good
    with person.client() as client:
        turn = send_message(
            client, "a long task", model=model_id(terminal_id, MODEL), terminal_id=terminal_id
        )
        started = time.monotonic()
        reply = _with_error(client, turn, wait_for_reply(client, turn, timeout=100))
        waited = time.monotonic() - started
        agents.turn = heard
        _, after = _ask(
            client, terminal_id, "again", chat_id=turn.chat_id, parent_id=turn.assistant_message_id
        )

    assert _error(reply) == DISCONNECTED, f"the reply did not end as disconnected: {reply}"
    assert reply["content"] == "Working on it. ", f"what had arrived was not kept: {reply}"
    assert waited >= 50, f"the reply was given up after only {waited:.0f}s without the socket"
    assert after.get("content") == "Scripted agent heard: again", (
        f"the chat did not go on once the terminal was back: {after}"
    )


def _share_for_replies(owner, chat_id: str, *members) -> None:
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share", json={"share_mode": None})
        assert shared.status_code == 200, f"sharing the chat failed: {shared.text}"
        saved = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update",
            json={
                "access_grants": [grant("user", member.id, "read") for member in members],
                "share_mode": "continue",
            },
        )
    assert saved.status_code == 200, f"saving the chat's access failed: {saved.text}"


def test_a_collaborators_message_goes_on_in_the_owners_agent_session_on_the_owners_terminal(
    make_user, agents, connect
):
    owner, guest = make_user(), make_user()
    terminal_id = connect(owner)

    with owner.client() as client:
        first, _ = _ask(client, terminal_id, "remember the word lantern")
    _share_for_replies(owner, first.chat_id, guest)
    with guest.client() as client:
        alone = client.post(
            "/api/chat/completions",
            json={
                "model": model_id(terminal_id, MODEL),
                "terminal_id": terminal_id,
                "messages": [{"role": "user", "content": "my own chat"}],
                "stream": True,
            },
        )
        _, reply = _ask(
            client,
            terminal_id,
            "what was the word?",
            chat_id=first.chat_id,
            parent_id=first.assistant_message_id,
        )

    assert (alone.status_code, alone.json().get("detail")) == (400, UNAVAILABLE), (
        f"the collaborator can use the terminal alone: {alone.status_code} {alone.text}"
    )
    assert reply.get("content") == "Scripted agent heard: what was the word?", (
        f"the collaborator's message did not reach the owner's agent: {reply}"
    )
    theirs = agents.starts[-1]
    assert theirs.options["resume"] == SESSION, (
        f"the collaborator's message did not go on in the owner's session: {theirs.options}"
    )
    assert theirs.headers.get("x-user-id") == owner.id, (
        "the collaborator's message did not run as the terminal's owner: "
        f"{theirs.headers.get('x-user-id')} (owner {owner.id}, collaborator {guest.id})"
    )
