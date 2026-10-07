"""Journey: a background sub-agent keeps working after the reply ends and reports back in the chat.

The model delegates with `background` set: its reply ends with a row named for the task while the
sub-agent works on and the page shows no running reply. When the sub-agent finishes, its report
is stored as a "Background sub-agent finished" row that opens to the result, and the model
continues from it without the person typing. A reload while it works keeps the chat and the
report still arrives. Two accounts working at once each get only their own report, the admin's
background limit refuses a delegation over the cap and reports the other, a failing sub-agent
is reported with its error, and sub-agents that finish while the reply is written come back as
one report of two tasks. Stopping the reply leaves the background sub-agent running; stopping
the chat during a foreground sub-agent ends the reply and leaves the chat usable.

Before PR #31576 (open-webui/open-webui#31566) the open page did not follow a report on its own:
the server told it to reload the chat while the chat's stored current message still pointed at the
earlier reply, so the page kept showing that one and neither the report nor the follow-up appeared
until the person reloaded. Every test that reads the report waits for it in the open page.

Discriminates: passes on dev a5bc78300; on dev 176d31d1d the tests that wait for the report fail on
that bug. In backend copies the tests turn red when the dispatch waits for the sub-agent, the cap
is ignored, a failure is reported as completed, the reports of two chats are mixed up, two finished
sub-agents are reported one by one, the stop of the reply reaches the background sub-agent, or the
stop of the chat leaves the foreground one running. In a frontend copy with the dispatch row named
as a foreground one, the report row's result left out and its count of tasks dropped, the row tests
turn red while an unrelated journey stays green.
"""

from __future__ import annotations

import json
import re
import time
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    conversation,
    expect_reply,
    last_reply,
    replies,
    send,
    stop_button,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
SLOW_SECONDS = 8.0
STOP_SECONDS = 20.0


@pytest.fixture
def background_on(admin, preserve):
    """`background_on(SUBAGENTS_MAX_ASYNC=1)` turns background sub-agents on with a limit."""
    preserve(SUBAGENTS)

    def switch_on(**limits) -> None:
        with admin.client() as client:
            current = client.get(SUBAGENTS[0]).json()
            settings = {
                **current,
                "ENABLE_SUBAGENTS": True,
                "SUBAGENTS_BACKGROUND_ENABLED": True,
                **limits,
            }
            client.post(SUBAGENTS[1], json=settings).raise_for_status()

    return switch_on


def unique(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:8]}"


def dispatch(task: str, prompt: str, call_id: str = "call_1") -> reply.Reply:
    return reply.tool_call(
        "delegate_task", {"task": task, "background": True}, call_id, match=reply.answering(prompt)
    )


def report_of(task: str):
    """A `match` for the request that carries the finished sub-agent's report back to the model."""

    def matches(body: dict) -> bool:
        users = [entry for entry in body.get("messages", []) if entry.get("role") == "user"]
        content = str(users[-1].get("content")) if users else ""
        return "[ASYNC SUBAGENT COMPLETE" in content and task in content

    return matches


def dispatched_row(page: Page, task: str) -> Locator:
    name = re.escape(f'View Result from Background sub-agent: "{task}"')
    return conversation(page).get_by_role("button", name=re.compile(name))


def report_row(page: Page) -> Locator:
    return conversation(page).get_by_role(
        "button", name=re.compile("Background sub-agent finished")
    )


def expect_report_row(page: Page) -> None:
    expect(
        report_row(page),
        "the report never showed in the open chat (it shows only after a reload)",
    ).to_be_visible(timeout=REPLY_TIMEOUT_MS)


def expect_in_chat(page: Page, text: str) -> None:
    """The text shows in the chat, whichever reply is last: a report can add one after it."""
    expect(conversation(page)).to_contain_text(text, timeout=REPLY_TIMEOUT_MS)


def running_task_ids(admin) -> set[str]:
    with admin.client() as client:
        return set(client.get("/api/tasks").json()["tasks"])


def wait_until(condition, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not condition():
        assert time.monotonic() < deadline, "the condition never held"
        time.sleep(0.1)


def two_dispatches(tasks: list[str], prompt: str) -> reply.Reply:
    calls = [
        {
            "id": f"call_{number}",
            "type": "function",
            "function": {
                "name": "delegate_task",
                "arguments": json.dumps({"task": task, "background": True}),
            },
        }
        for number, task in enumerate(tasks)
    ]
    return reply.Reply(tool_calls=calls, match=reply.answering(prompt))


def test_the_reply_ends_while_the_subagent_keeps_working_and_its_report_then_continues_the_chat(
    background_on, page_for, make_user, upstream, admin
):
    background_on()
    prompt, task = unique("hand this over"), unique("check the ledger")
    answer = unique("the ledger balances")
    upstream.queue(
        dispatch(task, prompt),
        reply.text(answer, delay=SLOW_SECONDS, match=reply.answering(task)),
        reply.text("It is being checked.", match=reply.answering(prompt)),
        reply.text("The ledger is fine.", match=report_of(task)),
    )
    page = page_for(make_user())
    idle_tasks = running_task_ids(admin)

    send(page, prompt)

    expect_in_chat(page, "It is being checked.")
    expect(dispatched_row(page, task)).to_be_visible()
    expect(stop_button(page)).to_have_count(0)
    expect(report_row(page)).to_have_count(0)
    assert running_task_ids(admin) - idle_tasks, "the sub-agent had finished before the reply ended"
    expect_report_row(page)
    expect_reply(page, "The ledger is fine.")
    report_row(page).click()
    expect(conversation(page)).to_contain_text(answer)
    assert len([r for r in upstream.chat_requests() if report_of(task)(r)]) == 1


def test_the_report_and_the_follow_up_show_in_the_open_chat_without_a_reload(
    background_on, page_for, make_user, upstream
):
    background_on()
    prompt, task = unique("hand this over"), unique("check the books")
    upstream.queue(
        dispatch(task, prompt),
        reply.text("all good", delay=2.0, match=reply.answering(task)),
        reply.text("It is being checked.", match=reply.answering(prompt)),
        reply.text("The books are fine.", match=report_of(task)),
    )
    page = page_for(make_user())
    send(page, prompt)
    expect_reply(page, "It is being checked.")

    expect_report_row(page)
    expect_reply(page, "The books are fine.")


def test_a_reload_while_the_subagent_works_keeps_the_chat_and_the_report_still_comes(
    background_on, page_for, make_user, upstream
):
    background_on()
    prompt, task = unique("hand this over"), unique("sum the receipts")
    answer = unique("forty receipts")
    upstream.queue(
        dispatch(task, prompt),
        reply.text(answer, delay=SLOW_SECONDS, match=reply.answering(task)),
        reply.text("Counting them.", match=reply.answering(prompt)),
        reply.text("Forty in all.", match=report_of(task)),
    )
    page = page_for(make_user())
    send(page, prompt)
    expect_reply(page, "Counting them.")

    page.reload()

    expect_reply(page, "Counting them.")
    expect(dispatched_row(page, task)).to_be_visible()
    expect(report_row(page)).to_have_count(0)
    expect(stop_button(page)).to_have_count(0)
    expect_report_row(page)
    expect_reply(page, "Forty in all.")
    report_row(page).click()
    expect(conversation(page)).to_contain_text(answer)


def test_two_accounts_at_once_each_get_only_their_own_report(
    background_on, page_for, make_user, upstream
):
    background_on()
    prompts = [unique("hand over the north"), unique("hand over the south")]
    tasks = [unique("count north doors"), unique("count south doors")]
    answers = [unique("north has three"), unique("south has nine")]
    follow_ups = ["North report read.", "South report read."]
    upstream.queue(
        *[dispatch(task, prompt) for task, prompt in zip(tasks, prompts)],
        *[
            reply.text(answer, delay=4.0, match=reply.answering(task))
            for task, answer in zip(tasks, answers)
        ],
        *[reply.text("Working.", match=reply.answering(prompt)) for prompt in prompts],
        *[reply.text(text, match=report_of(task)) for task, text in zip(tasks, follow_ups)],
    )
    pages = [page_for(make_user()), page_for(make_user())]

    for page, prompt in zip(pages, prompts):
        send(page, prompt)
    for page in pages:
        expect_in_chat(page, "Working.")
    for page, follow_up, answer in zip(pages, follow_ups, answers):
        expect_report_row(page)
        expect_reply(page, follow_up)
        report_row(page).click()
        expect(conversation(page)).to_contain_text(answer)

    for page, other_task, other_answer in zip(pages, reversed(tasks), reversed(answers)):
        expect(conversation(page)).not_to_contain_text(other_task)
        expect(conversation(page)).not_to_contain_text(other_answer)
    for task, answer in zip(tasks, answers):
        [report] = [r for r in upstream.chat_requests() if report_of(task)(r)]
        other_answer = answers[1 - answers.index(answer)]
        assert answer in str(report["messages"][-1]["content"])
        assert other_answer not in str(report["messages"][-1]["content"])


def test_a_background_limit_of_one_refuses_one_of_two_subagents_and_reports_the_other(
    background_on, page_for, make_user, upstream
):
    background_on(SUBAGENTS_MAX_ASYNC=1)
    prompt = unique("split this up")
    tasks = [unique("first errand"), unique("second errand")]
    upstream.queue(
        two_dispatches(tasks, prompt),
        *[reply.text("done", delay=3.0, match=reply.answering(task)) for task in tasks],
        reply.text("Two were asked for.", match=reply.answering(prompt)),
        # the two calls of one reply race for the single slot
        reply.text("One came back.", match=lambda body: any(report_of(t)(body) for t in tasks)),
    )
    page = page_for(make_user())

    send(page, prompt)

    expect_in_chat(page, "Two were asked for.")
    replies(page).filter(has_text="Two were asked for.").get_by_text("Explored", exact=True).click()
    for task in tasks:
        dispatched_row(page, task).click()
    expect(
        conversation(page).get_by_text("Async subagent capacity reached (1 running)")
    ).to_have_count(1)
    expect(conversation(page).get_by_text('"status": "dispatched"')).to_have_count(1)
    expect_report_row(page)
    expect_reply(page, "One came back.")
    started = [
        r
        for r in upstream.chat_requests()
        if any(reply.answering(t)(r) and not report_of(t)(r) for t in tasks)
    ]
    assert len(started) == 1


def test_a_failing_background_subagent_is_reported_and_the_chat_carries_on(
    background_on, page_for, make_user, upstream
):
    background_on()
    prompt, task = unique("hand this over"), unique("call the broken model")
    upstream.queue(
        dispatch(task, prompt),
        reply.error(500, "the helper model is down", match=reply.answering(task)),
        reply.text("Sent it off.", match=reply.answering(prompt)),
        reply.text("The helper failed.", match=report_of(task)),
    )
    page = page_for(make_user())

    send(page, prompt)

    expect_in_chat(page, "Sent it off.")
    expect_report_row(page)
    expect_reply(page, "The helper failed.")
    report_row(page).click()
    expect(conversation(page)).to_contain_text("The subagent did not complete successfully.")
    expect(conversation(page)).to_contain_text("the helper model is down")


def test_stopping_the_reply_leaves_the_background_subagent_running_and_its_report_comes(
    background_on, page_for, make_user, upstream
):
    background_on()
    prompt, task = unique("hand this over"), unique("watch the queue")
    answer = unique("the queue is empty")
    upstream.queue(
        dispatch(task, prompt),
        reply.text(answer, delay=3.0, match=reply.answering(task)),
        reply.text("Sent it off, and more.", delay=SLOW_SECONDS, match=reply.answering(prompt)),
        reply.text("The queue is empty.", match=report_of(task)),
    )
    page = page_for(make_user())
    send(page, prompt)
    expect(dispatched_row(page, task)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    expect(stop_button(page)).to_be_visible()

    stop_button(page).click()

    expect(stop_button(page)).to_have_count(0)
    expect_report_row(page)
    expect_reply(page, "The queue is empty.")
    report_row(page).click()
    expect(conversation(page)).to_contain_text(answer)
    expect(conversation(page)).not_to_contain_text("Sent it off, and more.")


def test_stopping_the_chat_during_a_foreground_subagent_stops_it_and_the_chat_still_works(
    background_on, page_for, make_user, upstream, admin
):
    background_on()
    prompt, task, second = unique("stop this"), unique("slow foreground job"), unique("again")
    upstream.queue(
        reply.tool_call("delegate_task", {"task": task}, match=reply.answering(prompt)),
        reply.text("late answer", delay=STOP_SECONDS, match=reply.answering(task)),
        reply.text("Second reply.", match=reply.answering(second)),
    )
    page = page_for(make_user())
    idle_tasks = running_task_ids(admin)
    send(page, prompt)
    working = re.escape(f'Executing Sub-agent: "{task}"...')
    expect(last_reply(page).get_by_role("button", name=re.compile(working))).to_be_visible(
        timeout=REPLY_TIMEOUT_MS
    )

    started_tasks = running_task_ids(admin) - idle_tasks
    assert len(started_tasks) >= 2, "the reply and its sub-agent should both be running tasks"
    # stop while the sub-agent waits on its answer, not before it has asked for it
    wait_until(
        lambda: any(task in str(r["messages"]) for r in upstream.chat_requests()),
        timeout_seconds=10.0,
    )

    stop_button(page).click()

    expect(stop_button(page)).to_have_count(0)
    expect(last_reply(page).get_by_role("button", name=re.compile(working))).to_have_count(0)
    wait_until(lambda: not started_tasks & running_task_ids(admin), timeout_seconds=5.0)
    send(page, second)
    expect_reply(page, "Second reply.")
    assert all("late answer" not in str(r["messages"]) for r in upstream.chat_requests())
    assert len(upstream.chat_requests()) == 3


def test_subagents_finishing_while_the_reply_is_written_come_back_as_one_report(
    background_on, page_for, make_user, upstream
):
    background_on()
    prompt = unique("split this up")
    tasks = [unique("count the apples"), unique("count the pears")]
    upstream.queue(
        two_dispatches(tasks, prompt),
        *[reply.text(f"answer for {task}", match=reply.answering(task)) for task in tasks],
        reply.text("Sent both off.", delay=4.0, match=reply.answering(prompt)),
        reply.text("Both reports read.", match=report_of(tasks[0])),
    )
    page = page_for(make_user())

    send(page, prompt)

    expect_in_chat(page, "Sent both off.")
    expect_report_row(page)
    expect(report_row(page)).to_have_count(1)
    expect(report_row(page)).to_contain_text("2 tasks")
    expect_reply(page, "Both reports read.")
    [report] = [r for r in upstream.chat_requests() if report_of(tasks[0])(r)]
    content = str(report["messages"][-1]["content"])
    assert all(task in content and f"answer for {task}" in content for task in tasks)
