"""Journey: what a sub-agent is sent and how the server runs it.

The model's delegation call starts a chat of the sub-agent's own: its task (and the context the
model gave) is the user message, the default system prompt leads, and it is offered the parent's
tools except the ones that would let it delegate again or change memory. That chat stays out of
the person's chat list and search while it works and after it reported back. A call naming a file
the chat does not hold, or an empty task, comes back to the model as an error before anything
runs. Sub-agents of one reply run side by side. The admin's concurrent limit holds the sub-agents
of an automation run, whose reply waits for them, one at a time.

Since dev 8522af2d6 a delegation from a chat runs on after the reply ends and its report comes back
to the lead as a message of its own (integration/tools/test_background_subagents.py), so the
tests wait for that report before they read the sub-agent's requests. Only an automation run (and
a channel) still waits for its sub-agents, and only there the concurrent limit applies; the chat's
own limit is the background one.

`test_the_subagent_gets_the_task_with_its_context_and_no_way_to_delegate_again` and
`test_the_subagents_of_one_reply_run_side_by_side` are red on dev 62f70a844: since de73bb830 a chat
request whose reply message is already stored in the chat, the way automations, sub-agents and
timers prepare their reply, is refused with 409 and the reply is never written
(open-webui/open-webui#32066, fixed by 641c82df1).

Discriminates: passes on dev 176d31d1d. In backend copies each test turns red with its edit: the
context left off the task, the default prompt dropped, the delegation tool left in the sub-agent's
kit, the internal marker dropped, the file check removed, the empty-task check removed, the limit
forced to one or to none. The chat list test and the automation limit test were retargeted on
dev 9d19883f1, where the chat limit test had turned red because a chat's sub-agents no longer
wait for a place.

`test_a_subagent_run_has_no_client_address_for_the_ip_placeholders` is a regression test: a
sub-agent's request filled the client IP header placeholders (#33833) with 127.0.0.1, the address
of the request sub-agents build for themselves (issue open-webui/open-webui#33882). Red on dev
1fc5b8f6d (sends 127.0.0.1), green with fix 2ea13e650 (branch fix/user-ip-empty-for-automations).
"""

from __future__ import annotations

import json
import threading
import time
import uuid

import pytest

from harness import backends
from harness import upstream as reply
from harness.actors import create_user
from harness.chat import ask
from harness.second_provider import OPENAI_CONFIG
from harness.subagents import delegated, wait_for_report
from harness.tool_calls import without_hidden_calls
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
DEFAULT_PROMPT_START = "You are a sub-agent working on a specific task"
CONCURRENT_ENV = {"ENABLE_SUBAGENTS": "true", "SUBAGENTS_MAX_CONCURRENT": "1"}
SLOW_ANSWER_SECONDS = 3.0
SCHEDULE = "DTSTART:20990101T090000\nRRULE:FREQ=DAILY"


@pytest.fixture
def subagents_on(admin, preserve):
    preserve(SUBAGENTS)

    def switch_on(**settings) -> None:
        with admin.client() as client:
            current = client.get(SUBAGENTS[0]).json()
            body = {**current, "ENABLE_SUBAGENTS": True, **settings}
            client.post(SUBAGENTS[1], json=body).raise_for_status()

    return switch_on


def unique(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:8]}"


def offered(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools") or []}


def tool_result(request: dict) -> str:
    last = reply.conversation(request)[-1]
    assert last["role"] == "tool", last
    return last["content"]


def two_delegations(tasks: list[str], prompt: str) -> reply.Reply:
    calls = [
        {
            "id": f"call_{n}",
            "type": "function",
            "function": {"name": "delegate_task", "arguments": f'{{"task": "{task}"}}'},
        }
        for n, task in enumerate(tasks)
    ]
    return reply.Reply(tool_calls=calls, match=reply.answering(prompt))


def test_the_subagent_gets_the_task_with_its_context_and_no_way_to_delegate_again(
    subagents_on, make_user, upstream
):
    subagents_on()
    prompt, task = unique("hand this over"), unique("check the ledger")
    upstream.queue(
        reply.tool_call(
            "delegate_task",
            {"task": task, "context": "the ledger is in March"},
            match=reply.answering(prompt),
        ),
        reply.text("Ledger checked.", match=reply.answering(task)),
        reply.text("Handed over.", match=reply.answering(prompt)),
    )
    with make_user().client() as client:
        ask(client, prompt, features={"memory": True})
    wait_for_report(upstream, task)

    [sub] = [r for r in upstream.chat_requests() if delegated(task)(r)]
    parent = upstream.chat_requests()[0]
    system, user = sub["messages"][0], reply.conversation(sub)[-1]
    assert system["role"] == "system"
    assert system["content"].startswith(DEFAULT_PROMPT_START)
    assert user["content"] == f"{task}\n\n## Context\nthe ledger is in March"
    assert {"delegate_task", "timer"} <= offered(parent)
    assert not {"delegate_task", "timer"} & offered(sub), (
        "a sub-agent was offered the tools to delegate again"
    )
    assert {"update_memory", "add_memory"} <= offered(parent)
    assert not {"update_memory", "add_memory", "delete_memory"} & offered(sub)


def dispatched_chat_id(stored: dict) -> str:
    """The id of the sub-agent's chat, from the handle the delegation call returned."""
    [handle] = [
        json.loads(item["output"][0]["text"])
        for item in without_hidden_calls(stored["output"])
        if item["type"] == "function_call_output"
    ]
    return handle["subagent_chat_id"]


def test_the_subagents_own_chat_is_not_in_the_persons_chat_list(subagents_on, make_user, upstream):
    subagents_on()
    prompt, task = unique("hand this over"), unique("look it up")
    answered = threading.Event()
    upstream.queue(
        reply.tool_call("delegate_task", {"task": task}, match=reply.answering(prompt)),
        reply.text("Found it.", hold=answered, match=delegated(task)),
        reply.text("Looked up.", match=reply.answering(prompt)),
    )
    with make_user().client() as client:
        try:
            turn, stored = ask(client, prompt)
            sub_id = dispatched_chat_id(stored)
            working = [chat["id"] for chat in client.get("/api/v1/chats/").json()]
        finally:
            answered.set()
        wait_for_report(upstream, task)
        listed = [chat["id"] for chat in client.get("/api/v1/chats/").json()]
        # the lead chat holds the task in its delegation call, the sub-agent's chat is never found
        searched = [
            chat["id"] for chat in client.get("/api/v1/chats/search", params={"text": task}).json()
        ]
        opened = client.get(f"/api/v1/chats/{sub_id}")

    assert opened.status_code == 200, f"the sub-agent's own chat was not stored: {opened.text}"
    assert working == [turn.chat_id], f"the list showed {working} while the sub-agent worked"
    assert listed == [turn.chat_id], f"the list showed {listed} once the sub-agent reported"
    assert searched == [turn.chat_id], f"searching for the task found {searched}"


def _sent_ip(entry) -> str:
    """What the provider received for the X-Real-IP header, empty when it was not sent."""
    return {name.lower(): value for name, value in entry.headers.items()}.get("x-real-ip", "")


def test_a_subagent_run_has_no_client_address_for_the_ip_placeholders(
    subagents_on, admin, preserve, make_user, upstream
):
    subagents_on()
    preserve(OPENAI_CONFIG)
    with admin.client() as client:
        config = client.get(OPENAI_CONFIG[0]).json()
        connection = config["OPENAI_API_CONFIGS"].get("0", {})
        config["OPENAI_API_CONFIGS"] = {
            "0": {**connection, "headers": {"X-Real-IP": "{{USER_IP}}"}}
        }
        client.post(OPENAI_CONFIG[1], json=config).raise_for_status()
    prompt, task = unique("hand this over"), unique("count the boxes")
    upstream.queue(
        reply.tool_call("delegate_task", {"task": task}, match=reply.answering(prompt)),
        reply.text("Twelve boxes.", match=reply.answering(task)),
        reply.text("Counted.", match=reply.answering(prompt)),
    )
    with make_user().client() as client:
        ask(client, prompt)
    wait_for_report(upstream, task)

    requests = [entry for entry in upstream.requests_to("/chat/completions") if entry.body]
    sub_agent = [_sent_ip(entry) for entry in requests if delegated(task)(entry.body)]
    chatted = [_sent_ip(entry) for entry in requests if reply.answering(prompt)(entry.body)]
    assert chatted and all(chatted), f"the person's chat sent no client address: {chatted}"
    # the sub-agent's request is the server's own, so no address may stand in for a client (#33882)
    assert sub_agent == [""], f"a sub-agent run sent {{{{USER_IP}}}} as {sub_agent}"


def test_a_file_the_chat_does_not_hold_is_refused_before_the_subagent_starts(
    subagents_on, make_user, upstream
):
    subagents_on()
    prompt, task = unique("hand this over"), unique("read the attachment")
    upstream.queue(
        reply.tool_call(
            "delegate_task",
            {"task": task, "file_ids": ["no-such-file"]},
            match=reply.answering(prompt),
        ),
        reply.text("Could not start it.", match=reply.answering(prompt)),
    )
    with make_user().client() as client:
        ask(client, prompt)

    assert tool_result(upstream.chat_requests()[-1]) == (
        "Error: file_ids not attached or unavailable: no-such-file"
    )
    assert [r for r in upstream.chat_requests() if reply.answering(task)(r)] == []


def test_an_empty_task_is_refused_before_the_subagent_starts(subagents_on, make_user, upstream):
    subagents_on()
    prompt = unique("hand this over")
    upstream.queue(
        reply.tool_call("delegate_task", {"task": "   "}, match=reply.answering(prompt)),
        reply.text("Nothing to do.", match=reply.answering(prompt)),
    )
    with make_user().client() as client:
        ask(client, prompt)

    assert tool_result(upstream.chat_requests()[-1]) == "Error: task must not be empty."
    assert len(upstream.chat_requests()) == 2


def _started(upstream, tasks: list[str]) -> list[dict]:
    return [r for r in upstream.chat_requests() if any(delegated(t)(r) for t in tasks)]


def _wait_for_the_first_start(upstream, tasks: list[str]) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not _started(upstream, tasks):
        time.sleep(0.05)
    assert _started(upstream, tasks), "no sub-agent ever started"


def _slow_answers(tasks: list[str], **options) -> list[reply.Reply]:
    return [reply.text("Done.", match=delegated(task), **options) for task in tasks]


def test_the_subagents_of_one_reply_run_side_by_side(subagents_on, make_user, upstream):
    subagents_on()
    prompt = unique("split the work")
    tasks = [unique("first errand"), unique("second errand")]
    answered = threading.Event()
    upstream.queue(
        two_delegations(tasks, prompt),
        *_slow_answers(tasks, hold=answered),
        reply.text("Both are on it.", match=reply.answering(prompt)),
    )
    with make_user().client() as client:
        try:
            ask(client, prompt)
            # each waits on its answer, so a second start shows they run at once
            assert backends.wait_until(lambda: len(_started(upstream, tasks)) == 2, timeout=15.0), (
                f"{len(_started(upstream, tasks))} of two sub-agents started side by side"
            )
        finally:
            answered.set()
        for task in tasks:
            wait_for_report(upstream, task)


def _create_automation(client, prompt: str) -> str:
    created = client.post(
        "/api/v1/automations/create",
        json={
            "name": unique("nightly split"),
            "is_active": True,
            "data": {"prompt": prompt, "model_id": MOCK_MODEL_ID, "rrule": SCHEDULE},
        },
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _automation_reply(client, automation_id: str) -> dict:
    """The automation run's final reply, once the run is recorded."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        runs = client.get(f"/api/v1/automations/{automation_id}/runs").json()
        if runs:
            assert runs[0]["status"] == "success", runs[0]
            stored = client.get(f"/api/v1/chats/{runs[0]['chat_id']}").json()
            history = stored["chat"]["history"]
            return history["messages"][history["currentId"]]
        time.sleep(0.2)
    raise AssertionError("the automation run was never recorded")


@pytest.mark.slow
def test_a_concurrent_limit_of_one_runs_an_automations_subagents_one_after_the_other(
    instance_with,
):
    launched = instance_with(CONCURRENT_ENV)
    upstream = launched.upstream
    prompt = unique("split the nightly work")
    tasks = [unique("first errand"), unique("second errand")]
    upstream.queue(
        two_delegations(tasks, prompt),
        *_slow_answers(tasks, delay=SLOW_ANSWER_SECONDS),
        reply.text("Both are back.", match=reply.answering(prompt)),
    )
    owner = create_user(launched, role="admin")
    with owner.client() as client:
        automation_id = _create_automation(client, prompt)
        try:
            started = client.post(f"/api/v1/automations/{automation_id}/run")
            assert started.status_code == 200, started.text
            _wait_for_the_first_start(upstream, tasks)
            time.sleep(1.0)  # shows the second does not start while the first one works
            at_once = len(_started(upstream, tasks))
            final = _automation_reply(client, automation_id)
        finally:
            client.delete(f"/api/v1/automations/{automation_id}/delete")

    assert at_once == 1, f"{at_once} sub-agents ran at once under a limit of one"
    assert len(_started(upstream, tasks)) == 2
    assert final["content"] == "Both are back.", final
    results = [
        entry["content"]
        for entry in reply.conversation(upstream.chat_requests()[-1])
        if entry["role"] == "tool"
    ]
    assert len(results) == 2 and all(result.endswith("\n\nDone.") for result in results), (
        f"the run's reply did not get both sub-agents' answers: {results}"
    )
