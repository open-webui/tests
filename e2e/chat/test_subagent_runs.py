"""Journey: the model hands work to a sub-agent in a chat and its result comes back to the chat.

The scripted model calls the delegation tool; its reply ends with one row for the call while the
sub-agent runs its own loop against the same provider (tool calls of its own, then an answer).
The sub-agent's report then shows as a "Sub-agent <ending> · <task>" row that opens to its
result, and the model carries on from it. The call's row opens to the task and the context it
was given, a long task is cut in the row names, the sub-agent's own tool calls stay out of the
chat, and the limits an admin sets (iterations, output length) show in the result. A call keeps
to one row while the sub-agent works and after its result is in, also after thinking (one group
naming the task once until it is opened).

Each request is tied to its own prompt: the sub-agent's latest user message is its task, the
parent's is what the person typed, and the report is matched by `harness.subagents.report_of`.

Since dev 8522af2d6 every sub-agent of a chat works on after the reply that started it, and its
result comes back as a report row of its own (e85500e77), where it used to be the call's tool
result. The tests were retargeted on dev 9d19883f1. Tests that now cover the same as newer ones
were removed: a failing sub-agent, several in one reply, a reload after the run and a failure next
to a working sub-agent are e2e/chat/test_subagent_threads.py (finished, failed and stopped rows,
the reload, the Threads list) and e2e/chat/test_background_subagents.py (a reload while it works,
the failing report, two finishing together); the row reading Executing while the sub-agent works
is now the Threads list's working entry; the answer within the output limit is
integration/tools/test_subagents_admin_config.py, at the limit itself. The two limit-of-one tests
went with the concurrent limit, which no longer holds a chat's sub-agents (it still holds an
automation's, integration/tools/test_subagent_runs.py); the chat's limit is the background one.

Discriminates: written on dev 9d19883f1. In backend copies the limit tests turn red when the
iteration limit is dropped or the truncation removed. The row tests pin the labels the build draws.

Every test runs in the main chat and in a side chat.
"""

from __future__ import annotations

import re
import threading
import uuid

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from harness.subagents import delegated, report_of
from utils.chat_ui import REPLY_TIMEOUT_MS, Chat, conversation, expect_reply, replies, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
CLOCK = "get_current_timestamp"


@pytest.fixture
def subagents_on(admin, preserve):
    """`subagents_on(SUBAGENTS_MAX_ITERATIONS=2)` switches sub-agents on with those limits."""
    preserve(SUBAGENTS)

    def switch_on(**limits) -> None:
        with admin.client() as client:
            current = client.get(SUBAGENTS[0]).json()
            settings = {**current, "ENABLE_SUBAGENTS": True, **limits}
            client.post(SUBAGENTS[1], json=settings).raise_for_status()

    return switch_on


def unique(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:8]}"


def delegation(task: str, prompt: str, call_id: str = "call_1", **arguments) -> reply.Reply:
    return reply.tool_call(
        "delegate_task", {"task": task, **arguments}, call_id, match=reply.answering(prompt)
    )


def call_row(chat: Chat, task: str) -> Locator:
    """The delegation call's own row, named for the task."""
    name = re.compile(rf"^View Result from {re.escape(task)}")
    return conversation(chat).get_by_role("button", name=name)


def result_row(chat: Chat, task: str, ending: str = "finished") -> Locator:
    name = re.compile(rf"^Sub-agent {ending} · {re.escape(task)}")
    return conversation(chat).get_by_role("button", name=name)


def task_rows(chat: Chat, task: str) -> Locator:
    """Every row of the chat that names the task, whatever it reads."""
    return conversation(chat).get_by_role("button").filter(has_text=task)


def unfinished_rows(chat: Chat) -> Locator:
    return conversation(chat).get_by_role("button", name=re.compile("^(Executing|Preparing) "))


def sub_requests(upstream, task: str) -> list[dict]:
    return [r for r in upstream.chat_requests() if delegated(task)(r)]


def test_a_subagent_runs_its_own_loop_and_the_chat_carries_on_from_its_report(
    subagents_on, chat_for, make_user, upstream
):
    subagents_on()
    prompt, task = unique("please look into this"), unique("read the clock twice")
    answer = unique("It is late")
    upstream.queue(
        delegation(task, prompt),
        reply.tool_call(CLOCK, {}, "clock_1", match=delegated(task)),
        reply.tool_call(CLOCK, {}, "clock_2", match=delegated(task)),
        reply.text(answer, match=delegated(task)),
        reply.text("A helper is on it.", match=reply.answering(prompt)),
        reply.text("The helper says it is late.", match=report_of(task)),
    )
    chat = chat_for(make_user())

    send(chat, prompt)

    expect(result_row(chat, task)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    expect_reply(chat, "The helper says it is late.")
    expect(conversation(chat)).to_contain_text("A helper is on it.")
    expect(
        conversation(chat).get_by_role("button", name="Checked the time"),
        "the sub-agent's own tool calls show in the chat",
    ).to_have_count(0)
    assert len(sub_requests(upstream, task)) == 3
    [report] = [r for r in upstream.chat_requests() if report_of(task)(r)]
    assert answer in reply.conversation(report)[-1]["content"]


def test_the_call_opens_to_the_task_and_the_result_row_to_the_answer(
    subagents_on, chat_for, make_user, upstream
):
    subagents_on()
    prompt, task = unique("delegate this"), unique("name the capital")
    answer = unique("Vienna")
    upstream.queue(
        delegation(task, prompt, context="the country is Austria"),
        reply.text(answer, match=delegated(task)),
        reply.text("Handed over.", match=reply.answering(prompt)),
        reply.text("Done delegating.", match=report_of(task)),
    )
    chat = chat_for(make_user())
    send(chat, prompt)
    expect_reply(chat, "Done delegating.")

    expect(conversation(chat)).not_to_contain_text(answer)
    call_row(chat, task).click()
    expect(conversation(chat)).to_contain_text("the country is Austria")
    result_row(chat, task).click()
    expect(conversation(chat).get_by_text(answer, exact=True)).to_be_visible()


def test_a_long_task_is_cut_in_the_row_names(subagents_on, chat_for, make_user, upstream):
    subagents_on()
    prompt = unique("delegate a long one")
    task = (f"{uuid.uuid4().hex[:8]} " + "and then some more words " * 6).strip()
    upstream.queue(
        delegation(task, prompt),
        reply.text("Long one done.", match=delegated(task)),
        reply.text("Sent it off.", match=reply.answering(prompt)),
        reply.text("The long task is over.", match=report_of(task)),
    )
    chat = chat_for(make_user())

    send(chat, prompt)

    expect_reply(chat, "The long task is over.")
    shortened = task[:60].strip()
    expect(call_row(chat, shortened)).to_be_visible()
    expect(result_row(chat, shortened)).to_be_visible()
    expect(task_rows(chat, task)).to_have_count(0)


def test_a_subagent_stops_at_the_iteration_limit_and_says_so(
    subagents_on, chat_for, make_user, upstream
):
    subagents_on(SUBAGENTS_MAX_ITERATIONS=2)
    prompt, task = unique("delegate a long errand"), unique("keep checking the clock")
    upstream.queue(
        delegation(task, prompt),
        *[reply.tool_call(CLOCK, {}, f"clock_{n}", match=delegated(task)) for n in range(3)],
        reply.text("Sent it off.", match=reply.answering(prompt)),
        reply.text("It gave up.", match=report_of(task)),
    )
    chat = chat_for(make_user())

    send(chat, prompt)

    expect_reply(chat, "It gave up.")
    result_row(chat, task, "failed").click()
    # since d87837372 the request after the last allowed round turns tool calls off for an answer,
    # and a call the model still makes there fails with the limit; since #34176 it keeps the tools
    expect(conversation(chat)).to_contain_text("Tool-call iteration limit reached.")
    requests = sub_requests(upstream, task)
    assert len(requests) == 3
    assert requests[-1].get("tool_choice") == "none", requests[-1]
    assert requests[-1].get("tools") == requests[0].get("tools"), requests[-1]


def test_a_subagent_answer_longer_than_the_output_limit_is_cut(
    subagents_on, chat_for, make_user, upstream
):
    subagents_on(SUBAGENTS_MAX_OUTPUT=20)
    prompt, task = unique("delegate a report"), unique("write the report")
    kept, dropped = "K" * 20, "the tail nobody should see"
    upstream.queue(
        delegation(task, prompt),
        reply.text(kept + dropped, match=delegated(task)),
        reply.text("Sent it off.", match=reply.answering(prompt)),
        reply.text("Report received.", match=report_of(task)),
    )
    chat = chat_for(make_user())

    send(chat, prompt)

    expect_reply(chat, "Report received.")
    result_row(chat, task).click()
    expect(conversation(chat)).to_contain_text(kept)
    expect(conversation(chat)).to_contain_text("[output truncated]")
    expect(conversation(chat)).not_to_contain_text(dropped)


def test_one_delegation_shows_one_row_while_it_works_and_one_result(
    subagents_on, chat_for, make_user, upstream
):
    subagents_on()
    prompt, task = unique("hand this over"), unique("look it up")
    answered = threading.Event()
    upstream.queue(
        delegation(task, prompt),
        reply.text("Found it.", hold=answered, match=delegated(task)),
        reply.text("A helper looks.", match=reply.answering(prompt)),
        reply.text("The helper found it.", match=report_of(task)),
    )
    chat = chat_for(make_user())

    try:
        send(chat, prompt)

        expect_reply(chat, "A helper looks.")
        expect(call_row(chat, task)).to_be_visible()
        expect(task_rows(chat, task)).to_have_count(1)
        expect(unfinished_rows(chat)).to_have_count(0)
    finally:
        answered.set()

    expect_reply(chat, "The helper found it.")
    expect(result_row(chat, task)).to_be_visible()
    expect(task_rows(chat, task)).to_have_count(2)
    expect(unfinished_rows(chat)).to_have_count(0)
    assert len(sub_requests(upstream, task)) == 1


def test_a_delegation_after_thinking_folds_into_one_group_that_names_the_task_once(
    subagents_on, chat_for, make_user, upstream
):
    subagents_on()
    prompt, task = unique("think then hand over"), unique("check the shelf")
    answered = threading.Event()
    upstream.queue(
        reply.Reply(
            reasoning="A helper should check this.",
            tool_calls=delegation(task, prompt).tool_calls,
            match=reply.answering(prompt),
        ),
        reply.text("Shelf checked.", hold=answered, match=delegated(task)),
        reply.text("A helper checks.", match=reply.answering(prompt)),
        reply.text("The shelf is fine.", match=report_of(task)),
    )
    chat = chat_for(make_user())

    try:
        send(chat, prompt)

        expect_reply(chat, "A helper checks.")
        delegating_reply = replies(chat).filter(has_text="A helper checks.")
        group = delegating_reply.get_by_role("button", name="Toggle details")
        expect(group).to_contain_text(f"Explored {task}")
        expect(task_rows(chat, task)).to_have_count(1)
    finally:
        answered.set()

    expect_reply(chat, "The shelf is fine.")
    expect(task_rows(chat, task)).to_have_count(2)
    group.click()
    # opened, the group lists its steps under the header: the call's own row joins it
    expect(call_row(chat, task)).to_be_visible()
    expect(task_rows(chat, task)).to_have_count(3)
    expect(unfinished_rows(chat)).to_have_count(0)
