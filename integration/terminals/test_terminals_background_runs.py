"""Journey: what runs for a person in the background keeps the terminal of a Terminals orchestrator.

New on dev after 0.13.3 (titled `refac`, 404d27a36 to 473a0fb24), where a terminal on an
orchestrator is a template with a scope. A timer the model sets in a chat fires in that chat, so
on a terminal with a machine per chat its commands run in the chat's own machine. A sub-agent
the model hands a task to is offered the chat's terminal and works in the person's machine. An
automation names a terminal: saving it is refused for a terminal that was not shared with its
owner, and for a per-chat terminal when it posts to a channel, which has no chat to give a
machine to. A run works in the machine the terminal gives its owner or the chat the run writes,
in the working directory saved with it.

The real Terminals service (open-webui/terminals) runs on Docker and starts a real Open Terminal
container per machine (`harness.terminals`); without its images or Docker the tests skip.

Discriminates: passes on dev c15f28649 with Terminals 0.4.1 and Open Terminal 0.17.7; each
mutation below, made in a backend copy, turns its test red. The terminal left out of what a
fired timer sends fails the timer test, left out of a sub-agent's request the sub-agent test,
and left out of an automation's run the two run tests and the working-directory test (none is
offered `run_command`); the working directory dropped from the run fails the working-directory
test alone; every terminal counting as shared fails the not-shared test; the channel check
removed fails the channel test.

Regression, open: on a terminal with a machine per chat a sub-agent's commands run in a machine
of the sub-agent's own hidden chat. It does not find the files of the chat that handed it the
task, what it writes never reaches that chat's machine, and its machine is left running. Finding
`open-webui-subagent-gets-own-machine-on-per-chat-terminal` of the terminal suite; the issue and
the fix PR are not filed yet. The two per-chat sub-agent tests are red on dev c15f28649 for it
(the sub-agent's `cat` prints "No such file or directory") and pass with the fix branch
`fix-subagent-per-chat-terminal`.
"""

from __future__ import annotations

import json
import uuid

import pytest

from harness import upstream as reply
from harness.backends import wait_until
from harness.channel_quotes import enable_channels
from harness.chat import ask
from harness.terminal_server import read_grant
from harness.terminals import create_template, execute, output_of, share_template
from harness.upstream import MOCK_MODEL_ID, conversation

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
SCHEDULE = "DTSTART:20990101T090000\nRRULE:FREQ=DAILY"
HOME = "/home/user"
RUN_WAIT = 60.0


@pytest.fixture
def subagents_on(admin, preserve):
    """Sub-agents, in the background as well, and timers, which the first switch enables too."""
    preserve(SUBAGENTS)
    switches = {"ENABLE_SUBAGENTS": True, "SUBAGENTS_BACKGROUND_ENABLED": True}
    with admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        client.post(SUBAGENTS[1], json={**current, **switches}).raise_for_status()


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


def _run(command: str, call_id: str = "call_1", **options) -> reply.Reply:
    return reply.tool_call("run_command", {"command": command, "wait": 10}, call_id, **options)


def _requests_for(upstream, prompt: str) -> list[dict]:
    return [request for request in upstream.chat_requests() if reply.answering(prompt)(request)]


def _handed(task: str):
    """A `match` for a sub-agent's own requests; its result names the task to the chat again."""

    def matches(request: dict) -> bool:
        return reply.answering(task)(request) and not reply.answering(f"Original task: {task}")(
            request
        )

    return matches


def _offered(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools") or []}


def _printed_by_the_last_command(request: dict) -> str:
    """What the model was told its latest command printed."""
    results = [entry["content"] for entry in conversation(request) if entry["role"] == "tool"]
    assert results, f"no command ran; the model was sent {conversation(request)}"
    result = json.loads(results[-1])
    assert "output" in result, f"the command did not run: {result}"
    return "".join(entry["data"] for entry in result["output"])


def _automation(owner, terminal_id: str, prompt: str, **data):
    """Save an automation of `owner` on the terminal, the way its editor does."""
    form = {
        "name": f"Watch {uuid.uuid4().hex[:8]}",
        "is_active": False,
        "data": {
            "prompt": prompt,
            "model_id": MOCK_MODEL_ID,
            "rrule": SCHEDULE,
            "terminal": {"server_id": terminal_id},
            **data,
        },
    }
    with owner.client() as client:
        return client.post("/api/v1/automations/create", json=form)


def _finished_run(owner, automation_id: str) -> dict:
    """Run now, and the run once it has ended."""
    runs: list[dict] = []

    def ended() -> bool:
        nonlocal runs
        runs = client.get(f"/api/v1/automations/{automation_id}/runs").json()
        return bool(runs) and runs[0]["status"] != "running"

    with owner.client() as client:
        started = client.post(f"/api/v1/automations/{automation_id}/run")
        assert started.status_code == 200, f"Run now failed: {started.text}"
        assert wait_until(ended, RUN_WAIT), f"the run never ended: {runs}"
    return runs[0]


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


def test_a_timer_set_in_a_chat_fires_in_that_chats_own_machine(
    subagents_on, make_user, shared_terminal, terminals, upstream
):
    person = make_user()
    terminal_id = shared_terminal(person, scope="chat")
    tag = uuid.uuid4().hex[:8]
    prompt, ring = f"wake me at high tide {tag}", f"High tide {tag}."
    name = f"rang-{tag}.txt"
    upstream.queue(
        reply.tool_call("timer", {"prompt": ring, "at": "2s"}, match=reply.answering(prompt)),
        reply.text("Timer set.", match=reply.answering(prompt)),
        _run(f"echo rang > ~/{name} && cat ~/{name}", match=reply.answering(ring)),
        reply.text("Rang.", match=reply.answering(ring)),
    )

    with person.client() as client:
        turn, message = ask(client, prompt, terminal_id=terminal_id)
    # due after two seconds and polled every second
    fired = wait_until(lambda: len(_requests_for(upstream, ring)) == 2, RUN_WAIT)

    assert message.get("content", "").endswith("Timer set."), f"no timer was set: {message}"
    assert fired, f"the timer never came back from its command: {_requests_for(upstream, ring)}"
    rung = _requests_for(upstream, ring)
    assert "run_command" in _offered(rung[0]), (
        f"the timer's turn was not offered the chat's terminal: {sorted(_offered(rung[0]))}"
    )
    assert _printed_by_the_last_command(rung[-1]).strip() == "rang", (
        "the timer's command did not print what it wrote"
    )
    assert terminals.holding(name) == {f"chat:{turn.chat_id}": "rang\n"}, (
        f"the timer's command ran outside the chat's machine: {terminals.holding(name)}"
    )
    assert [machine["context_id"] for machine in terminals.instances()] == [
        f"chat:{turn.chat_id}"
    ], f"the timer started a machine of its own: {terminals.instances()}"


def test_a_sub_agent_works_in_the_persons_machine(
    subagents_on, make_user, shared_terminal, terminals, upstream
):
    person = make_user()
    terminal_id = shared_terminal(person)
    tag = uuid.uuid4().hex[:8]
    prompt, task = f"hand this over {tag}", f"log the catch {tag}"
    name = f"catch-{tag}.txt"
    upstream.queue(
        reply.tool_call("delegate_task", {"task": task}, match=reply.answering(prompt)),
        _run(f"echo herring > ~/{name} && cat ~/{name}", "sub_call", match=reply.answering(task)),
        reply.text("Logged.", match=reply.answering(task)),
        reply.text("Handed back.", match=reply.answering(prompt)),
    )

    with person.client() as client:
        _, message = ask(client, prompt, terminal_id=terminal_id)

    assert message.get("content") == "Handed back.", f"the chat did not finish: {message}"
    delegated = _requests_for(upstream, task)
    assert delegated, "the sub-agent never reached the model"
    assert "run_command" in _offered(delegated[0]), (
        f"the sub-agent was not offered the chat's terminal: {sorted(_offered(delegated[0]))}"
    )
    assert _printed_by_the_last_command(delegated[-1]).strip() == "herring", (
        "the sub-agent's command did not print what it wrote"
    )
    assert terminals.holding(name) == {f"user:{person.id}": "herring\n"}, (
        f"the sub-agent's command ran outside the person's machine: {terminals.holding(name)}"
    )


@pytest.mark.parametrize("background", [False, True], ids=["foreground", "background"])
def test_a_sub_agent_on_a_per_chat_terminal_works_in_the_machine_of_the_chat_that_delegated(
    background, subagents_on, make_user, shared_terminal, terminals, upstream
):
    person = make_user()
    terminal_id = shared_terminal(person, scope="chat")
    tag = uuid.uuid4().hex[:8]
    prompt, task = f"split the watch {tag}", f"read the log and add to it {tag}"
    result = f"Original task: {task}"
    from_the_chat, from_the_sub_agent = f"log-{tag}.txt", f"entry-{tag}.txt"
    upstream.queue(
        _run(f"echo high water > ~/{from_the_chat}", match=reply.answering(prompt)),
        reply.tool_call(
            "delegate_task",
            {"task": task, "background": background},
            "delegation",
            match=reply.answering(prompt),
        ),
        reply.text("Handed over.", match=reply.answering(prompt)),
        _run(
            f"cat ~/{from_the_chat}; echo herring > ~/{from_the_sub_agent}",
            "sub_call",
            match=_handed(task),
        ),
        reply.text("Logged.", match=_handed(task)),
        # a background sub-agent's result comes back as a turn of the chat
        reply.text("Noted.", match=reply.answering(result)),
    )

    with person.client() as client:
        turn, message = ask(client, prompt, terminal_id=terminal_id)
        returned = not background or wait_until(lambda: _requests_for(upstream, result), RUN_WAIT)
        delegated = [request for request in upstream.chat_requests() if _handed(task)(request)]
        read_in_the_chat = output_of(
            execute(client, terminal_id, f"cat ~/{from_the_sub_agent}", chat_id=turn.chat_id)
        )

    assert message.get("content") == "Handed over.", f"the chat did not finish: {message}"
    assert returned, f"the sub-agent's result never came back to the chat: {delegated}"
    assert len(delegated) == 2, f"the sub-agent did not run its command: {delegated}"
    assert _printed_by_the_last_command(delegated[-1]).strip() == "high water", (
        "the sub-agent did not find the file its chat wrote"
    )
    assert read_in_the_chat.strip() == "herring", (
        f"the chat does not find the file its sub-agent wrote: {read_in_the_chat!r}"
    )
    assert terminals.holding(from_the_sub_agent) == {f"chat:{turn.chat_id}": "herring\n"}, (
        "the sub-agent's file is outside the machine of the chat that delegated: "
        f"{terminals.holding(from_the_sub_agent)}"
    )
    assert [machine["context_id"] for machine in terminals.instances()] == [
        f"chat:{turn.chat_id}"
    ], f"the sub-agent got a machine of its own: {terminals.instances()}"


@pytest.mark.parametrize("scope", ["user", "chat"])
def test_an_automation_runs_its_commands_in_the_machine_its_terminal_gives_it(
    scope, scheduler, shared_terminal, terminals, upstream
):
    terminal_id = shared_terminal(scheduler, scope=scope)
    tag = uuid.uuid4().hex[:8]
    prompt, name = f"check the moorings {tag}", f"moorings-{tag}.txt"
    answering = reply.answering(prompt)
    upstream.queue(
        _run(f"echo checked > ~/{name} && pwd", match=answering),
        reply.text("Checked.", match=answering),
    )

    saved = _automation(scheduler, terminal_id, prompt)
    assert saved.status_code == 200, f"saving the automation failed: {saved.text}"
    run = _finished_run(scheduler, saved.json()["id"])

    assert run["status"] == "success", f"the run failed: {run}"
    context = {"user": f"user:{scheduler.id}", "chat": f"chat:{run['chat_id']}"}[scope]
    asked = _requests_for(upstream, prompt)
    assert asked and "run_command" in _offered(asked[0]), (
        f"the run was not offered its terminal: {asked and sorted(_offered(asked[0]))}"
    )
    assert _printed_by_the_last_command(asked[-1]).strip() == HOME, (
        "the run's command did not run in the machine's home"
    )
    assert terminals.holding(name) == {context: "checked\n"}, (
        f"the run's command ran outside its machine ({context}): {terminals.holding(name)}"
    )


def test_an_automation_run_starts_in_the_working_directory_saved_with_it(
    scheduler, shared_terminal, upstream
):
    terminal_id = shared_terminal(scheduler)
    tag = uuid.uuid4().hex[:8]
    prompt, chosen = f"count the nets {tag}", f"{HOME}/nets-{tag}"
    answering = reply.answering(prompt)
    upstream.queue(_run("pwd", match=answering), reply.text("Counted.", match=answering))
    with scheduler.client() as client:
        output_of(execute(client, terminal_id, f"mkdir {chosen}"))

    saved = _automation(
        scheduler, terminal_id, prompt, terminal={"server_id": terminal_id, "cwd": chosen}
    )
    assert saved.status_code == 200, f"saving the automation failed: {saved.text}"
    run = _finished_run(scheduler, saved.json()["id"])

    assert run["status"] == "success", f"the run failed: {run}"
    assert _printed_by_the_last_command(_requests_for(upstream, prompt)[-1]).strip() == chosen, (
        "the run's command did not run in the automation's working directory"
    )


def test_an_automation_cannot_be_saved_on_a_terminal_not_shared_with_its_owner(
    admin, scheduler, connect_terminals
):
    connection = connect_terminals(read_grant(scheduler.id))
    with admin.client() as client:
        terminal_id = create_template(client, connection)["terminal_id"]
    prompt = f"check the moorings {uuid.uuid4().hex[:8]}"

    unshared = _automation(scheduler, terminal_id, prompt)
    with admin.client() as client:
        share_template(client, terminal_id, read_grant(scheduler.id))
    shared = _automation(scheduler, terminal_id, prompt)

    assert (unshared.status_code, unshared.json().get("detail")) == (
        403,
        "Terminal access denied",
    ), f"an automation was saved on someone else's terminal: {unshared.status_code} {unshared.text}"
    assert shared.status_code == 200, f"the shared terminal was refused too: {shared.text}"


def test_a_per_chat_terminal_refuses_an_automation_that_posts_to_a_channel(
    admin, scheduler, connect_terminals
):
    connection = connect_terminals(read_grant(scheduler.id))
    with admin.client() as client:
        per_chat = create_template(client, connection, scope="chat")["terminal_id"]
        per_user = create_template(client, connection, scope="user")["terminal_id"]
        for terminal_id in (per_chat, per_user):
            share_template(client, terminal_id, read_grant(scheduler.id))
    target = {"type": "channel", "channel_id": _channel_written_by(admin, scheduler)}
    prompt = f"post the forecast {uuid.uuid4().hex[:8]}"

    refused = _automation(scheduler, per_chat, prompt, target=target)
    to_a_chat = _automation(scheduler, per_chat, prompt)
    per_user_to_the_channel = _automation(scheduler, per_user, prompt, target=target)

    assert (refused.status_code, refused.json().get("detail")) == (
        400,
        "This terminal requires a chat destination.",
    ), f"a per-chat terminal was saved for a channel: {refused.status_code} {refused.text}"
    assert to_a_chat.status_code == 200, (
        f"the per-chat terminal was refused for a chat as well: {to_a_chat.text}"
    )
    assert per_user_to_the_channel.status_code == 200, (
        f"a per-user terminal was refused for the channel: {per_user_to_the_channel.text}"
    )
