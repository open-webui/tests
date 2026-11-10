"""Journey: an automation on a terminal the admin connected directly.

New on dev after 0.13.3 (titled `refac`, 404d27a36 to 473a0fb24). An automation names a terminal
and a working directory. Saving it, new or edited, is refused for a terminal its owner cannot
use, and the same check runs again before every run, so a terminal withdrawn since then fails
the run before the model is asked. A run offers the model the terminal's tools, in a chat and in
a channel, and starts in the saved working directory. An automation without a terminal of its
own takes the default terminal of its model. A terminal that authenticates people by their
Open WebUI session gets a token of the owner although nobody is signed in. The tools that read
and type in a person's open shell are never offered to a run. A coding agent installed in the
terminal can be the automation's model: only with that terminal picked, only writing to a chat,
and only while the terminal still has it; the editor reads the agents to offer from
`GET /api/v1/automations/models`, which needs the automations permission and, for a saved
automation, its owner.

`test_terminals_background_runs.py` has the same journey for a terminal of a Terminals
orchestrator; the one case of it here is a terminal with a machine per model, which an
automation may only use with a model its owner may use and whose run works in that model's
machine (the real Terminals stack of `harness.terminals`, skipped without its images). The
plain terminal here is the fake of `harness.terminal_server`, and a real Open Terminal with the
stand-in coding agent (`harness.coding_agent`) where the working directory or an agent
matters; those tests skip without `open-terminal`.

Discriminates: passes on dev c15f28649 with Open Terminal 0.17.7 and Terminals 0.4.1; each
mutation below, made in a backend copy, turns its tests red. The working directory not stored
fails the kept-terminal test, the three refused-save tests (the refused edit is compared with
what was kept) and the three working-directory tests. The terminal check dropped from Create
fails the refused-save and agent tests, from Update the refused-save tests, from a run the
withdrawn-terminal test. The terminal left out of a chat run fails every chat run here (none is
offered `run_command`), out of a channel run the channel test; the working directory dropped
from a chat run fails the chat and the agent run, from a channel run the channel test. The
model's default terminal ignored fails the default-terminal test. A run without its token fails
the session test (the run ends in an error). The shell tools let through for a run fail the
shell test. Each of the three agent refusals removed fails its own case. The permission check or
the owner check dropped from the agent list fails the editor test, and every terminal counting
as shared fails it and the not-shared case. The model check dropped for a terminal with a
machine per model fails the per-model test.
"""

from __future__ import annotations

import json
import uuid
from typing import Iterator

import httpx
import pytest

from harness import upstream as reply
from harness.access import make_group
from harness.backends import wait_until
from harness.channel_quotes import enable_channels
from harness.chat import ask
from harness.coding_agent import DEEP, FAST, connect_agent_terminal, model_id
from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from harness.terminals import create_template, share_template
from harness.upstream import MOCK_MODEL_ID, conversation

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SCHEDULE = "DTSTART:20990101T090000\nRRULE:FREQ=DAILY"
RUN_WAIT = 60.0
UNAVAILABLE = "The configured terminal is unavailable for this automation."
SHELL_TOOLS = {"read_user_terminal", "send_user_terminal_input"}
USER_SHELL_PATH = "/api/terminals/user/output"


def _operation(operation_id: str) -> dict:
    return {
        "operationId": operation_id,
        "description": operation_id,
        "responses": {"200": {"description": "ok"}},
    }


SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "terminal", "version": "1"},
    "paths": {
        "/execute": {"post": _operation("run_command")},
        USER_SHELL_PATH: {"get": _operation("read_user_terminal")},
        "/api/terminals/user/input": {"post": _operation("send_user_terminal_input")},
    },
}


def _serve(server: FakeTerminalServer, printed: str = "done") -> None:
    server.route("GET", "/openapi.json", json_answer(SPEC))
    server.route("POST", "/execute", json_answer({"output": [{"data": printed}]}))


@pytest.fixture
def terminal(preserve) -> Iterator[FakeTerminalServer]:
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as server:
        _serve(server)
        yield server


@pytest.fixture
def scheduler(admin, preserve, make_user):
    """A person who may use automations and channels; their automations go afterwards."""
    preserve("admin_config", "permissions")
    enable_channels(admin)
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["features"].update(automations=True, channels=True)
        granted = client.post("/api/v1/users/default/permissions", json=permissions)
    assert granted.status_code == 200, granted.text
    account = make_user()
    yield account
    with account.client() as client:
        for automation in client.get("/api/v1/automations/list").json().get("items", []):
            client.delete(f"/api/v1/automations/{automation['id']}/delete")


@pytest.fixture
def connect_agents(admin, preserve, agent_terminal):
    """`connect_agents(*people)`: the admin connects the Open Terminal that has the agent."""
    preserve(TERMINAL_SERVERS_CONFIG)
    return lambda *people: connect_agent_terminal(admin, agent_terminal, *people)["id"]


def _connect(admin, *servers: FakeTerminalServer, shared_with=(), **fields) -> list[str]:
    """The admin saves a connection to each fake terminal, shared with those people."""
    grants = [read_grant(person.id) for person in shared_with]
    connections = [
        server.connection(**{"config": {"access_grants": grants}, **fields}) for server in servers
    ]
    with admin.client() as client:
        configure_terminals(client, *connections)
    for server in servers:
        server.clear()
    return [connection["id"] for connection in connections]


def _form(prompt: str, terminal: dict | None = None, **data) -> dict:
    """An automation as its editor sends it; `terminal` is `{server_id, cwd}`."""
    data = {"prompt": prompt, "model_id": MOCK_MODEL_ID, "rrule": SCHEDULE, **data}
    if terminal:
        data["terminal"] = terminal
    return {"name": f"Watch {uuid.uuid4().hex[:8]}", "is_active": False, "data": data}


def _saved(owner, form: dict) -> dict:
    with owner.client() as client:
        created = client.post("/api/v1/automations/create", json=form)
    assert created.status_code == 200, f"saving the automation failed: {created.text}"
    return created.json()


def _stored_terminal(owner, automation_id: str) -> dict | None:
    with owner.client() as client:
        read = client.get(f"/api/v1/automations/{automation_id}")
    assert read.status_code == 200, read.text
    return read.json()["data"].get("terminal")


def _finished_run(owner, automation_id: str) -> dict:
    """Run now, and the run once it has ended."""
    runs: list[dict] = []

    def ended() -> bool:
        nonlocal runs
        runs = client.get(f"/api/v1/automations/{automation_id}/runs").json()
        return len(runs) > before and runs[0]["status"] != "running"

    with owner.client() as client:
        before = len(client.get(f"/api/v1/automations/{automation_id}/runs").json())
        started = client.post(f"/api/v1/automations/{automation_id}/run")
        assert started.status_code == 200, f"Run now failed: {started.text}"
        assert wait_until(ended, RUN_WAIT), f"the run never ended: {runs}"
    return runs[0]


def _run(command: str, **options) -> reply.Reply:
    return reply.tool_call("run_command", {"command": command, "wait": 10}, **options)


def _requests_for(upstream, prompt: str) -> list[dict]:
    return [request for request in upstream.chat_requests() if reply.mentioning(prompt)(request)]


def _offered(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools") or []}


def _told(request: dict) -> str:
    """What the model was told its latest tool call gave."""
    results = [entry["content"] for entry in conversation(request) if entry["role"] == "tool"]
    assert results, f"no tool ran; the model was sent {conversation(request)}"
    return results[-1]


def _printed(request: dict) -> str:
    """What a real terminal printed for the model's latest command."""
    result = json.loads(_told(request))
    assert "output" in result, f"the command did not run: {result}"
    return "".join(entry["data"] for entry in result["output"])


def _channel_written_by(admin, account) -> str:
    grants = [
        {"principal_type": "user", "principal_id": account.id, "permission": permission}
        for permission in ("read", "write")
    ]
    with admin.client() as client:
        created = client.post(
            "/api/v1/channels/create",
            json={"name": f"room-{uuid.uuid4().hex[:8]}", "access_grants": grants},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _detail(response) -> tuple[int, str | None]:
    return response.status_code, response.json().get("detail")


def test_an_automation_keeps_the_terminal_and_working_directory_it_was_saved_with(
    admin, scheduler, terminal
):
    [terminal_id] = _connect(admin, terminal, shared_with=[scheduler])
    first = {"server_id": terminal_id, "cwd": "/srv/nets"}
    moved = {"server_id": terminal_id, "cwd": "/srv/buoys"}
    prompt = f"count the nets {uuid.uuid4().hex[:8]}"

    automation = _saved(scheduler, _form(prompt, first))
    after_create = _stored_terminal(scheduler, automation["id"])
    with scheduler.client() as client:
        edited = client.post(
            f"/api/v1/automations/{automation['id']}/update", json=_form(prompt, moved)
        )
        after_edit = _stored_terminal(scheduler, automation["id"])
        cleared = client.post(f"/api/v1/automations/{automation['id']}/update", json=_form(prompt))
        after_clearing = _stored_terminal(scheduler, automation["id"])

    assert automation["data"]["terminal"] == first, (
        f"the answer to Create names another terminal: {automation['data']}"
    )
    assert after_create == first, f"the saved automation holds {after_create}"
    assert edited.status_code == 200, f"the edit was refused: {edited.text}"
    assert after_edit == moved, f"after the edit the automation holds {after_edit}"
    assert cleared.status_code == 200, f"taking the terminal off was refused: {cleared.text}"
    assert after_clearing is None, f"the terminal taken off is still there: {after_clearing}"


@pytest.mark.parametrize("reason", ["not shared", "switched off", "unknown"])
def test_an_automation_cannot_be_saved_on_a_terminal_its_owner_cannot_use(
    reason, admin, scheduler, preserve
):
    preserve(TERMINAL_SERVERS_CONFIG)
    prompt = f"check the moorings {uuid.uuid4().hex[:8]}"
    with serving_terminal() as usable, serving_terminal() as other:
        usable_connection = usable.connection(config={"access_grants": [read_grant(scheduler.id)]})
        other_connection = other.connection(
            enabled=reason != "switched off",
            config={"access_grants": [] if reason == "not shared" else [read_grant(scheduler.id)]},
        )
        with admin.client() as client:
            configure_terminals(client, usable_connection, other_connection)
        unusable = (
            f"terminal-{uuid.uuid4().hex[:8]}" if reason == "unknown" else other_connection["id"]
        )
        kept = {"server_id": usable_connection["id"], "cwd": "/srv/nets"}

        automation = _saved(scheduler, _form(prompt, kept))
        with scheduler.client() as client:
            created = client.post(
                "/api/v1/automations/create", json=_form(prompt, {"server_id": unusable})
            )
            edited = client.post(
                f"/api/v1/automations/{automation['id']}/update",
                json=_form(prompt, {"server_id": unusable}),
            )
            listed = client.get("/api/v1/automations/list").json()["items"]
        after_edit = _stored_terminal(scheduler, automation["id"])

    assert _detail(created) == (400, UNAVAILABLE), (
        f"an automation was created on a terminal that is {reason}: "
        f"{created.status_code} {created.text}"
    )
    assert _detail(edited) == (400, UNAVAILABLE), (
        f"an automation was moved to a terminal that is {reason}: "
        f"{edited.status_code} {edited.text}"
    )
    assert after_edit == kept, f"the refused edit changed the automation's terminal: {after_edit}"
    assert [item["id"] for item in listed] == [automation["id"]], (
        f"the refused automation is on the list: {listed}"
    )


def test_a_run_is_refused_once_its_terminal_is_no_longer_shared_with_the_owner(
    admin, scheduler, terminal, upstream
):
    [terminal_id] = _connect(admin, terminal, shared_with=[scheduler])
    prompt = f"check the moorings {uuid.uuid4().hex[:8]}"
    upstream.queue(reply.text("Checked.", match=reply.answering(prompt)))
    automation = _saved(scheduler, _form(prompt, {"server_id": terminal_id}))

    shared = _finished_run(scheduler, automation["id"])
    asked_while_shared = len(_requests_for(upstream, prompt))
    _connect(admin, terminal)
    withdrawn = _finished_run(scheduler, automation["id"])

    assert shared["status"] == "success", f"the run on the shared terminal failed: {shared}"
    assert asked_while_shared == 1, f"the first run asked the model {asked_while_shared} times"
    assert withdrawn["status"] == "error" and UNAVAILABLE in (withdrawn.get("error") or ""), (
        f"the run on the withdrawn terminal ended as {withdrawn}"
    )
    assert len(_requests_for(upstream, prompt)) == 1, (
        "the run on the withdrawn terminal still asked the model"
    )


@pytest.mark.parametrize("destination", ["chat", "channel"])
def test_a_run_gets_the_terminal_and_starts_in_the_saved_working_directory(
    destination, admin, scheduler, connect_agents, agent_terminal, upstream
):
    terminal_id = connect_agents(scheduler)
    tag = uuid.uuid4().hex[:8]
    prompt, chosen = f"count the nets {tag}", agent_terminal.terminal.home / f"nets-{tag}"
    chosen.mkdir()
    target = {"type": "chat"}
    if destination == "channel":
        target = {"type": "channel", "channel_id": _channel_written_by(admin, scheduler)}
    upstream.queue(
        _run("pwd", match=reply.mentioning(prompt)),
        reply.text("Counted.", match=reply.mentioning(prompt)),
    )

    automation = _saved(
        scheduler,
        _form(prompt, {"server_id": terminal_id, "cwd": str(chosen)}, target=target),
    )
    run = _finished_run(scheduler, automation["id"])

    assert run["status"] == "success", f"the run failed: {run}"
    # a channel run is recorded as soon as its reply has started
    answered = wait_until(lambda: len(_requests_for(upstream, prompt)) == 2, RUN_WAIT)
    asked = _requests_for(upstream, prompt)
    assert answered, f"the run never came back from its command: {asked}"
    assert "run_command" in _offered(asked[0]), (
        f"the run was not offered its terminal: {sorted(_offered(asked[0]))}"
    )
    assert _printed(asked[-1]).strip() == str(chosen), (
        "the run's command did not run in the automation's working directory"
    )


def test_an_automation_without_a_terminal_runs_on_the_default_terminal_of_its_model(
    admin, scheduler, preserve, upstream
):
    preserve(TERMINAL_SERVERS_CONFIG)
    tag = uuid.uuid4().hex[:8]
    preset = f"harbour-{tag}"
    by_default, with_its_own = f"check the moorings {tag}", f"check the buoys {tag}"
    for prompt in (by_default, with_its_own):
        upstream.queue(
            _run("date", match=reply.answering(prompt)),
            reply.text("Checked.", match=reply.answering(prompt)),
        )
    with serving_terminal() as of_the_model, serving_terminal() as of_the_automation:
        _serve(of_the_model, "from the model's terminal")
        _serve(of_the_automation, "from the automation's terminal")
        model_terminal, own_terminal = _connect(
            admin, of_the_model, of_the_automation, shared_with=[scheduler]
        )
        with admin.client() as client:
            created = client.post(
                "/api/v1/models/create",
                json={
                    "id": preset,
                    "base_model_id": MOCK_MODEL_ID,
                    "name": f"Harbour {tag}",
                    "meta": {"terminalId": model_terminal},
                    "params": {},
                    "access_grants": [read_grant(scheduler.id)],
                },
            )
        assert created.status_code == 200, created.text
        try:
            with scheduler.client() as client:
                # the page reads the model list, which is where a new model becomes known
                client.get("/api/models").raise_for_status()
            plain = _saved(scheduler, _form(by_default, model_id=preset))
            own = _saved(
                scheduler, _form(with_its_own, {"server_id": own_terminal}, model_id=preset)
            )
            runs = [_finished_run(scheduler, automation["id"]) for automation in (plain, own)]
            calls = {
                "model": len(of_the_model.requests_to("/execute")),
                "automation": len(of_the_automation.requests_to("/execute")),
            }
        finally:
            with admin.client() as client:
                client.post("/api/v1/models/model/delete", json={"id": preset})

    assert [run["status"] for run in runs] == ["success", "success"], f"a run failed: {runs}"
    assert "from the model's terminal" in _told(_requests_for(upstream, by_default)[-1]), (
        "the automation without a terminal did not run its command on its model's terminal"
    )
    assert "from the automation's terminal" in _told(_requests_for(upstream, with_its_own)[-1]), (
        "the automation with a terminal of its own did not run its command there"
    )
    assert calls == {"model": 1, "automation": 1}, (
        f"each terminal should have run one command: {calls}"
    )


def test_a_run_signs_in_to_a_session_terminal_as_the_owner(admin, scheduler, terminal, upstream):
    [terminal_id] = _connect(admin, terminal, shared_with=[scheduler], auth_type="session")
    prompt = f"check the moorings {uuid.uuid4().hex[:8]}"
    upstream.queue(
        _run("date", match=reply.answering(prompt)),
        reply.text("Checked.", match=reply.answering(prompt)),
    )

    automation = _saved(scheduler, _form(prompt, {"server_id": terminal_id}))
    run = _finished_run(scheduler, automation["id"])

    assert run["status"] == "success", f"the run failed: {run}"
    [call] = terminal.requests_to("/execute")
    scheme, _, token = call.headers.get("authorization", "").partition(" ")
    assert scheme == "Bearer" and token, (
        f"the run's command reached the terminal without a session token: {call.headers}"
    )
    assert token != scheduler.token, "the run handed the terminal the owner's own sign-in token"
    listed = httpx.get(
        f"{scheduler.base_url}/api/v1/automations/list",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert listed.status_code == 200, (
        f"the token the terminal got is not accepted: {listed.status_code} {listed.text}"
    )
    assert [item["id"] for item in listed.json()["items"]] == [automation["id"]], (
        f"the token the terminal got does not act as the automation's owner: {listed.text}"
    )


def test_the_tools_for_a_persons_open_shell_are_not_offered_to_a_run(
    admin, scheduler, terminal, upstream
):
    [terminal_id] = _connect(
        admin,
        terminal,
        shared_with=[scheduler],
        config={"access_grants": [read_grant(scheduler.id)], "user_shell_tools": "always"},
    )
    tag = uuid.uuid4().hex[:8]
    in_a_chat, in_a_run = f"look at my shell {tag}", f"look at the shell {tag}"
    for prompt in (in_a_chat, in_a_run):
        upstream.queue(reply.text("Looked.", match=reply.answering(prompt)))

    with scheduler.client() as client:
        ask(client, in_a_chat, terminal_id=terminal_id)
    automation = _saved(scheduler, _form(in_a_run, {"server_id": terminal_id}))
    run = _finished_run(scheduler, automation["id"])

    assert run["status"] == "success", f"the run failed: {run}"
    [chat_request], [run_request] = (
        _requests_for(upstream, in_a_chat),
        _requests_for(upstream, in_a_run),
    )
    assert SHELL_TOOLS <= _offered(chat_request), (
        f"the person's own chat was not offered the shell tools: {sorted(_offered(chat_request))}"
    )
    assert "run_command" in _offered(run_request), (
        f"the run was not offered its terminal: {sorted(_offered(run_request))}"
    )
    assert not SHELL_TOOLS & _offered(run_request), (
        f"a run with nobody at the shell was offered {sorted(SHELL_TOOLS & _offered(run_request))}"
    )


def test_an_automation_on_a_terminal_with_a_machine_per_model_needs_a_model_its_owner_may_use(
    admin, scheduler, connect_terminals, terminals, upstream
):
    connection = connect_terminals(read_grant(scheduler.id))
    tag = uuid.uuid4().hex[:8]
    private, name = f"harbour-{tag}", f"moorings-{tag}.txt"
    prompt = f"check the moorings {tag}"
    with admin.client() as client:
        per_model = create_template(client, connection, scope="model")["terminal_id"]
        per_user = create_template(client, connection, scope="user")["terminal_id"]
        for terminal_id in (per_model, per_user):
            share_template(client, terminal_id, read_grant(scheduler.id))
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": private,
                "base_model_id": MOCK_MODEL_ID,
                "name": f"Harbour {tag}",
                "meta": {},
                "params": {},
            },
        )
    assert created.status_code == 200, created.text
    upstream.queue(
        _run(f"echo checked > ~/{name}", match=reply.answering(prompt)),
        reply.text("Checked.", match=reply.answering(prompt)),
    )

    try:
        with scheduler.client() as client:
            refused = client.post(
                "/api/v1/automations/create",
                json=_form(prompt, {"server_id": per_model}, model_id=private),
            )
            on_a_per_user_terminal = client.post(
                "/api/v1/automations/create",
                json=_form(prompt, {"server_id": per_user}, model_id=private),
            )
        automation = _saved(scheduler, _form(prompt, {"server_id": per_model}))
        run = _finished_run(scheduler, automation["id"])
        held = terminals.holding(name)
    finally:
        with admin.client() as client:
            client.post("/api/v1/models/model/delete", json={"id": private})

    assert _detail(refused) == (403, "Model access denied"), (
        "an automation got the machine of a model its owner may not use: "
        f"{refused.status_code} {refused.text}"
    )
    assert on_a_per_user_terminal.status_code == 200, (
        "the same automation was refused on a terminal with a machine per person: "
        f"{on_a_per_user_terminal.text}"
    )
    assert run["status"] == "success", f"the run with a model the owner may use failed: {run}"
    assert held == {f"model:{MOCK_MODEL_ID}": "checked\n"}, (
        f"the run's command ran outside the machine of its model: {held}"
    )


@pytest.mark.parametrize(
    "change, refusal",
    [
        ("without a terminal", "Select a terminal for the agent."),
        ("posting to a channel", "Agent automations require a chat destination."),
        (
            "naming an agent the terminal lacks",
            "Agent model is unavailable in the configured terminal.",
        ),
    ],
)
def test_an_agent_automation_needs_its_terminal_a_chat_and_an_agent_the_terminal_has(
    change, refusal, admin, scheduler, connect_agents
):
    terminal_id = connect_agents(scheduler)
    prompt = f"tidy the repository {uuid.uuid4().hex[:8]}"
    valid = _form(prompt, {"server_id": terminal_id}, model_id=model_id(terminal_id, FAST))
    changed = json.loads(json.dumps(valid))
    if change == "without a terminal":
        del changed["data"]["terminal"]
    if change == "posting to a channel":
        changed["data"]["target"] = {
            "type": "channel",
            "channel_id": _channel_written_by(admin, scheduler),
        }
    if change == "naming an agent the terminal lacks":
        changed["data"]["model_id"] = model_id(terminal_id, "fake-gone")

    with scheduler.client() as client:
        refused = client.post("/api/v1/automations/create", json=changed)
        accepted = client.post("/api/v1/automations/create", json=valid)

    assert _detail(refused) == (400, refusal), (
        f"an agent automation {change} was answered {refused.status_code} {refused.text}"
    )
    assert accepted.status_code == 200, (
        f"the agent automation on its terminal and a chat was refused: {accepted.text}"
    )


def test_an_agent_automation_runs_the_agent_in_its_working_directory(
    scheduler, connect_agents, agent_terminal
):
    terminal_id = connect_agents(scheduler)
    tag = uuid.uuid4().hex[:8]
    prompt, chosen = f"tidy the repository {tag}", agent_terminal.terminal.home / f"repo-{tag}"
    chosen.mkdir()

    automation = _saved(
        scheduler,
        _form(
            prompt,
            {"server_id": terminal_id, "cwd": str(chosen)},
            model_id=model_id(terminal_id, DEEP),
        ),
    )
    run = _finished_run(scheduler, automation["id"])

    assert run["status"] == "success", f"the run failed: {run}"
    with scheduler.client() as client:
        chat = client.get(f"/api/v1/chats/{run['chat_id']}").json()["chat"]
    answer = chat["history"]["messages"][chat["history"]["currentId"]]
    said = "".join(
        part.get("text", "")
        for item in answer.get("output") or []
        if item.get("type") == "message"
        for part in item.get("content") or []
    )
    assert prompt in said and DEEP in said, f"the agent did not answer the run's prompt: {said!r}"
    assert str(chosen) in said, f"the agent did not work in the saved directory: {said!r}"


def test_the_editor_reads_a_terminals_agents_only_for_who_may_use_automations_and_the_automation(
    admin, scheduler, make_user, connect_agents
):
    without_permission, another, without_the_terminal = make_user(), make_user(), make_user()
    terminal_id = connect_agents(scheduler, another, without_permission)
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["features"]["automations"] = False
        client.post("/api/v1/users/default/permissions", json=permissions).raise_for_status()
    make_group(
        admin, [scheduler, another, without_the_terminal], {"features": {"automations": True}}
    )
    automation = _saved(
        scheduler,
        _form("tidy the repository", {"server_id": terminal_id}, model_id=model_id(terminal_id)),
    )
    expected = sorted(model_id(terminal_id, model) for model in (FAST, DEEP))

    def agents(account, **params) -> tuple[int, list[str] | dict]:
        with account.client() as client:
            answer = client.get(
                "/api/v1/automations/models", params={"terminal_id": terminal_id, **params}
            )
        if answer.status_code != 200:
            return answer.status_code, answer.json()
        return 200, sorted(model["id"] for model in answer.json())

    assert agents(scheduler) == (200, expected), (
        f"the editor of a new automation was offered {agents(scheduler)}"
    )
    assert agents(scheduler, automation_id=automation["id"]) == (200, expected), (
        "the owner editing the automation was offered "
        f"{agents(scheduler, automation_id=automation['id'])}"
    )
    assert agents(another) == (200, expected), (
        f"another person who may use automations and the terminal was offered {agents(another)}"
    )
    assert agents(another, automation_id=automation["id"])[0] == 404, (
        "another person read the agents for an automation that is not theirs: "
        f"{agents(another, automation_id=automation['id'])}"
    )
    assert agents(without_the_terminal) == (200, []), (
        "a person the terminal is not shared with was offered its agents: "
        f"{agents(without_the_terminal)}"
    )
    assert agents(without_permission)[0] == 403, (
        f"a person without the automations permission read the agents: {agents(without_permission)}"
    )
