"""Journey: what a foreground sub-agent is sent and how the server runs it.

The model's delegation call starts a chat of the sub-agent's own: its task (and the context the
model gave) is the user message, the default system prompt leads, and it is offered the parent's
tools except the ones that would let it delegate again or change memory. That chat stays out of
the person's chat list. A call naming a file the chat does not hold, or
an empty task, comes back to the model as an error before anything runs. Sub-agents of one reply
run side by side up to the admin's concurrent limit.

`test_a_concurrent_limit_of_one_runs_the_subagents_one_after_the_other`,
`test_the_subagent_gets_the_task_with_its_context_and_no_way_to_delegate_again` and
`test_the_subagents_of_one_reply_run_side_by_side` are red on dev 62f70a844: since de73bb830 a chat
request whose reply message is already stored in the chat, the way automations, sub-agents and
timers prepare their reply, is refused with 409 and the reply is never written
(open-webui/open-webui#32066).

Discriminates: passes on dev 176d31d1d. In backend copies each test turns red with its edit: the
context left off the task, the default prompt dropped, the delegation tool left in the sub-agent's
kit, the internal marker dropped, the file check removed, the empty-task check removed, the limit
forced to one or to none.
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness import upstream as reply
from harness.actors import create_user
from harness.chat import ask, send_message, wait_for_reply

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
DEFAULT_PROMPT_START = "You are a sub-agent working on a specific task"
CONCURRENT_ENV = {"ENABLE_SUBAGENTS": "true", "SUBAGENTS_MAX_CONCURRENT": "1"}
SLOW_ANSWER_SECONDS = 3.0


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
    last = request["messages"][-1]
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

    [sub] = [r for r in upstream.chat_requests() if reply.answering(task)(r)]
    parent = upstream.chat_requests()[0]
    system, user = sub["messages"][0], sub["messages"][-1]
    assert system["role"] == "system"
    assert system["content"].startswith(DEFAULT_PROMPT_START)
    assert user["content"] == f"{task}\n\n## Context\nthe ledger is in March"
    assert {"delegate_task", "timer"} <= offered(parent)
    assert not {"delegate_task", "timer"} & offered(sub), (
        "a sub-agent was offered the tools to delegate again"
    )
    assert {"update_memory", "add_memory"} <= offered(parent)
    assert not {"update_memory", "add_memory", "delete_memory"} & offered(sub)


def test_the_subagents_own_chat_is_not_in_the_persons_chat_list(subagents_on, make_user, upstream):
    subagents_on()
    prompt, task = unique("hand this over"), unique("look it up")
    upstream.queue(
        reply.tool_call("delegate_task", {"task": task}, match=reply.answering(prompt)),
        reply.text("Found it.", match=reply.answering(task)),
        reply.text("Looked up.", match=reply.answering(prompt)),
    )
    with make_user().client() as client:
        turn, _ = ask(client, prompt)
        listed = client.get("/api/v1/chats/").json()
        searched = client.get("/api/v1/chats/search", params={"text": task}).json()

    assert [chat["id"] for chat in listed] == [turn.chat_id]
    assert searched == []


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


def _run_two_subagents(launched, tasks: list[str], seconds_before_looking: float) -> int:
    """Start two slow sub-agents in one reply and count how many had started by then."""
    upstream = launched.upstream
    prompt = unique("split the work")
    upstream.queue(
        two_delegations(tasks, prompt),
        *[
            reply.text("Done.", delay=SLOW_ANSWER_SECONDS, match=reply.answering(task))
            for task in tasks
        ],
        reply.text("Both are back.", match=reply.answering(prompt)),
    )
    account = create_user(launched)
    with account.client() as client:
        turn = send_message(client, prompt)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not _started(upstream, tasks):
            time.sleep(0.05)
        time.sleep(seconds_before_looking)
        started = len(_started(upstream, tasks))
        wait_for_reply(client, turn)
    return started


def _started(upstream, tasks: list[str]) -> list[dict]:
    return [r for r in upstream.chat_requests() if any(reply.answering(t)(r) for t in tasks)]


def test_the_subagents_of_one_reply_run_side_by_side(subagents_on, instance, upstream):
    subagents_on()
    tasks = [unique("first errand"), unique("second errand")]

    started = _run_two_subagents(instance, tasks, seconds_before_looking=1.0)

    assert started == 2


@pytest.mark.slow
def test_a_concurrent_limit_of_one_runs_the_subagents_one_after_the_other(instance_with):
    launched = instance_with(CONCURRENT_ENV)
    tasks = [unique("first errand"), unique("second errand")]

    started = _run_two_subagents(launched, tasks, seconds_before_looking=1.0)

    assert started == 1
    assert len(_started(launched.upstream, tasks)) == 2
