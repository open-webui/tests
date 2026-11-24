"""Journey: a sub-agent keeps working after the reply ends and reports back in the chat.

The model delegates: its reply ends with a row named for the task while the sub-agent works on
and the page shows no running reply. When the sub-agent finishes, its report is stored as a
"Sub-agent finished · <task>" row that opens to the result, and the model continues from it
without the person typing. A reload while it works keeps the chat and the report still arrives.
Two accounts working at once each get only their own report, the admin's background limit
refuses a delegation over the cap and reports the other, a failing sub-agent is reported with
its error, and sub-agents that finish while the reply is written come back together, a row and
a result each. Stopping the reply leaves the sub-agent running.

Since dev 8522af2d6 every delegation of a chat works this way: the switch for background
sub-agents and the delegation's `background` argument are gone, the report rows read how the
sub-agent ended (e85500e77), and no sub-agent of a chat holds its reply any more, so the test of
stopping the chat during a foreground sub-agent went with it. Stopping a sub-agent itself is
e2e/chat/test_subagent_edges.py.

Before PR #31576 (open-webui/open-webui#31566) the open page did not follow a report on its own:
the server told it to reload the chat while the chat's stored current message still pointed at the
earlier reply, so the page kept showing that one and neither the report nor the follow-up appeared
until the person reloaded. Every test that reads the report waits for it in the open page.

`test_a_background_limit_of_one_refuses_one_of_two_subagents_and_reports_the_other`,
`test_a_failing_background_subagent_is_reported_and_the_chat_carries_on`,
`test_a_reload_while_the_subagent_works_keeps_the_chat_and_the_report_still_comes`,
`test_stopping_the_reply_leaves_the_background_subagent_running_and_its_report_comes`,
`test_subagents_finishing_while_the_reply_is_written_come_back_together`,
`test_the_reply_ends_while_the_subagent_keeps_working_and_its_report_then_continues_the_chat`,
`test_the_report_and_the_follow_up_show_in_the_open_chat_without_a_reload` and
`test_two_accounts_at_once_each_get_only_their_own_report` are red on dev 62f70a844: since de73bb830
a chat request whose reply message is already stored in the chat, the way automations, sub-agents
and timers prepare their reply, is refused with 409 and the reply is never written
(open-webui/open-webui#32066, fixed by 641c82df1).

`test_a_reload_while_the_subagent_works_keeps_the_chat_and_the_report_still_comes` is red on dev
131ddc32e: since d5038fec4 a chat that reloads as the report arrives keeps the copy of the report
the socket sent, which lacks what marks it as a report, so the page shows the raw report as a
plain message until the next reload (open-webui/open-webui#32801).


Discriminates: passes on dev a5bc78300; on dev 176d31d1d the tests that wait for the report fail on
that bug. In backend copies the tests turn red when the dispatch waits for the sub-agent, the cap
is ignored, a failure is reported as completed, the reports of two chats are mixed up, two finished
sub-agents are reported one by one, or the stop of the reply reaches the sub-agent. In a frontend
copy with the report row's result left out, the row tests turn red while an unrelated journey stays
green. Retargeted on dev 9d19883f1 to the rows of e85500e77.

Every test runs in the main chat and in a side chat. On dev c15f28649 seven side chat variants
are red and green on head e791cbdde (fix PR open-webui/open-webui#34064): a side chat did not
follow a report the server sent while the chat was open, so the report and the follow-up showed
only after a reload. No filed issue (open-webui/open-webui#34014 to #34036) covers it. The reload
test is green on dev: a reload makes the side chat the page's own.
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from harness.subagents import report_of
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    Chat,
    conversation,
    expect_reply,
    replies,
    send,
    stop_button,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
SLOW_SECONDS = 8.0


@pytest.fixture
def background_on(admin, preserve):
    """`background_on(SUBAGENTS_MAX_ASYNC=1)` turns sub-agents on with a background limit."""
    preserve(SUBAGENTS)

    def switch_on(**limits) -> None:
        with admin.client() as client:
            current = client.get(SUBAGENTS[0]).json()
            settings = {**current, "ENABLE_SUBAGENTS": True, **limits}
            client.post(SUBAGENTS[1], json=settings).raise_for_status()

    return switch_on


def unique(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:8]}"


def dispatch(task: str, prompt: str, call_id: str = "call_1") -> reply.Reply:
    return reply.tool_call("delegate_task", {"task": task}, call_id, match=reply.answering(prompt))


def dispatched_row(chat: Chat, task: str) -> Locator:
    """The delegation call in the reply, alone or as the group of steps it folds into."""
    name = re.compile(rf"^(View Result from|Explored) {re.escape(task)}")
    return conversation(chat).get_by_role("button", name=name)


def report_row(chat: Chat, ending: str = "finished") -> Locator:
    """The rows of the sub-agents' reports that read they `ending` ("finished", "failed")."""
    return conversation(chat).get_by_role("button", name=re.compile(f"^Sub-agent {ending} · "))


def expect_report_row(chat: Chat, ending: str = "finished") -> None:
    expect(
        report_row(chat, ending),
        "the report never showed in the open chat (it shows only after a reload)",
    ).to_be_visible(timeout=REPLY_TIMEOUT_MS)


def expect_in_chat(chat: Chat, text: str) -> None:
    """The text shows in the chat, whichever reply is last: a report can add one after it."""
    expect(conversation(chat)).to_contain_text(text, timeout=REPLY_TIMEOUT_MS)


def running_task_ids(admin) -> set[str]:
    with admin.client() as client:
        return set(client.get("/api/tasks").json()["tasks"])


def two_dispatches(tasks: list[str], prompt: str) -> reply.Reply:
    calls = [
        {
            "id": f"call_{number}",
            "type": "function",
            "function": {
                "name": "delegate_task",
                "arguments": json.dumps({"task": task}),
            },
        }
        for number, task in enumerate(tasks)
    ]
    return reply.Reply(tool_calls=calls, match=reply.answering(prompt))


def test_the_reply_ends_while_the_subagent_keeps_working_and_its_report_then_continues_the_chat(
    background_on, chat_for, make_user, upstream, admin
):
    """In a side chat on dev: the report shows only after a reload (no issue filed)."""
    background_on()
    prompt, task = unique("hand this over"), unique("check the ledger")
    answer = unique("the ledger balances")
    upstream.queue(
        dispatch(task, prompt),
        reply.text(answer, delay=SLOW_SECONDS, match=reply.answering(task)),
        reply.text("It is being checked.", match=reply.answering(prompt)),
        reply.text("The ledger is fine.", match=report_of(task)),
    )
    chat = chat_for(make_user())
    idle_tasks = running_task_ids(admin)

    send(chat, prompt)

    expect_in_chat(chat, "It is being checked.")
    expect(dispatched_row(chat, task)).to_be_visible()
    expect(stop_button(chat)).to_have_count(0)
    expect(report_row(chat)).to_have_count(0)
    assert running_task_ids(admin) - idle_tasks, "the sub-agent had finished before the reply ended"
    expect_report_row(chat)
    expect_reply(chat, "The ledger is fine.")
    report_row(chat).click()
    expect(conversation(chat)).to_contain_text(answer)
    assert len([r for r in upstream.chat_requests() if report_of(task)(r)]) == 1


def test_the_report_and_the_follow_up_show_in_the_open_chat_without_a_reload(
    background_on, chat_for, make_user, upstream
):
    """In a side chat on dev: the report shows only after a reload (no issue filed)."""
    background_on()
    prompt, task = unique("hand this over"), unique("check the books")
    upstream.queue(
        dispatch(task, prompt),
        reply.text("all good", delay=2.0, match=reply.answering(task)),
        reply.text("It is being checked.", match=reply.answering(prompt)),
        reply.text("The books are fine.", match=report_of(task)),
    )
    chat = chat_for(make_user())
    send(chat, prompt)
    expect_reply(chat, "It is being checked.")

    expect_report_row(chat)
    expect_reply(chat, "The books are fine.")


def test_a_reload_while_the_subagent_works_keeps_the_chat_and_the_report_still_comes(
    background_on, chat_for, make_user, upstream
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
    chat = chat_for(make_user())
    send(chat, prompt)
    expect_reply(chat, "Counting them.")

    chat = chat.reload()

    expect_reply(chat, "Counting them.")
    expect(dispatched_row(chat, task)).to_be_visible()
    expect(report_row(chat)).to_have_count(0)
    expect(stop_button(chat)).to_have_count(0)
    expect_report_row(chat)
    expect_reply(chat, "Forty in all.")
    report_row(chat).click()
    expect(conversation(chat)).to_contain_text(answer)


def test_two_accounts_at_once_each_get_only_their_own_report(
    background_on, chat_for, make_user, upstream
):
    """In a side chat on dev: the report shows only after a reload (no issue filed)."""
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
    chats = [chat_for(make_user()), chat_for(make_user())]

    for chat, prompt in zip(chats, prompts):
        send(chat, prompt)
    for chat in chats:
        expect_in_chat(chat, "Working.")
    for chat, follow_up, answer in zip(chats, follow_ups, answers):
        expect_report_row(chat)
        expect_reply(chat, follow_up)
        report_row(chat).click()
        expect(conversation(chat)).to_contain_text(answer)

    for chat, other_task, other_answer in zip(chats, reversed(tasks), reversed(answers)):
        expect(conversation(chat)).not_to_contain_text(other_task)
        expect(conversation(chat)).not_to_contain_text(other_answer)
    for task, answer in zip(tasks, answers):
        [report] = [r for r in upstream.chat_requests() if report_of(task)(r)]
        other_answer = answers[1 - answers.index(answer)]
        reported = str(reply.conversation(report)[-1]["content"])
        assert answer in reported
        assert other_answer not in reported


def test_a_background_limit_of_one_refuses_one_of_two_subagents_and_reports_the_other(
    background_on, chat_for, make_user, upstream
):
    """In a side chat on dev: the report shows only after a reload (no issue filed)."""
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
    chat = chat_for(make_user())

    send(chat, prompt)

    expect_in_chat(chat, "Two were asked for.")
    replies(chat).filter(has_text="Two were asked for.").get_by_text("Explored", exact=True).click()
    for task in tasks:
        dispatched_row(chat, task).click()
    expect(
        conversation(chat).get_by_text("Async subagent capacity reached (1 running)")
    ).to_have_count(1)
    expect(conversation(chat).get_by_text('"status": "dispatched"')).to_have_count(1)
    expect_report_row(chat)
    expect_reply(chat, "One came back.")
    started = [
        r
        for r in upstream.chat_requests()
        if any(reply.answering(t)(r) and not report_of(t)(r) for t in tasks)
    ]
    assert len(started) == 1


def test_a_failing_background_subagent_is_reported_and_the_chat_carries_on(
    background_on, chat_for, make_user, upstream
):
    """In a side chat on dev: the report shows only after a reload (no issue filed)."""
    background_on()
    prompt, task = unique("hand this over"), unique("call the broken model")
    upstream.queue(
        dispatch(task, prompt),
        reply.error(500, "the helper model is down", match=reply.answering(task)),
        reply.text("Sent it off.", match=reply.answering(prompt)),
        reply.text("The helper failed.", match=report_of(task)),
    )
    chat = chat_for(make_user())

    send(chat, prompt)

    expect_in_chat(chat, "Sent it off.")
    expect_report_row(chat, "failed")
    expect_reply(chat, "The helper failed.")
    report_row(chat, "failed").click()
    expect(conversation(chat)).to_contain_text("The subagent did not complete successfully.")
    expect(conversation(chat)).to_contain_text("the helper model is down")


def test_stopping_the_reply_leaves_the_background_subagent_running_and_its_report_comes(
    background_on, chat_for, make_user, upstream
):
    """In a side chat on dev: the report shows only after a reload (no issue filed)."""
    background_on()
    prompt, task = unique("hand this over"), unique("watch the queue")
    answer = unique("the queue is empty")
    upstream.queue(
        dispatch(task, prompt),
        reply.text(answer, delay=3.0, match=reply.answering(task)),
        reply.text("Sent it off, and more.", delay=SLOW_SECONDS, match=reply.answering(prompt)),
        reply.text("The queue is empty.", match=report_of(task)),
    )
    chat = chat_for(make_user())
    send(chat, prompt)
    expect(dispatched_row(chat, task)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    expect(stop_button(chat)).to_be_visible()

    stop_button(chat).click()

    expect(stop_button(chat)).to_have_count(0)
    expect_report_row(chat)
    expect_reply(chat, "The queue is empty.")
    report_row(chat).click()
    expect(conversation(chat)).to_contain_text(answer)
    expect(conversation(chat)).not_to_contain_text("Sent it off, and more.")


def test_subagents_finishing_while_the_reply_is_written_come_back_together(
    background_on, chat_for, make_user, upstream
):
    """In a side chat on dev: the report shows only after a reload (no issue filed)."""
    background_on()
    prompt = unique("split this up")
    tasks = [unique("count the apples"), unique("count the pears")]
    upstream.queue(
        two_dispatches(tasks, prompt),
        *[reply.text(f"answer for {task}", match=reply.answering(task)) for task in tasks],
        reply.text("Sent both off.", delay=4.0, match=reply.answering(prompt)),
        reply.text("Both reports read.", match=report_of(tasks[0])),
    )
    chat = chat_for(make_user())

    send(chat, prompt)

    expect_in_chat(chat, "Sent both off.")
    expect(
        report_row(chat),
        "the reports never showed in the open chat (they show only after a reload)",
    ).to_have_count(2, timeout=REPLY_TIMEOUT_MS)
    expect_reply(chat, "Both reports read.")
    for task in tasks:
        row = report_row(chat).filter(has_text=task)
        expect(row, f"no result row for {task}").to_have_count(1)
        row.click()
        expect(conversation(chat)).to_contain_text(f"answer for {task}")
    [report] = [r for r in upstream.chat_requests() if report_of(tasks[0])(r)]
    content = str(reply.conversation(report)[-1]["content"])
    assert all(task in content and f"answer for {task}" in content for task in tasks)
