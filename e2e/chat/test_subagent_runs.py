"""Journey: the model hands work to a sub-agent in a chat and the answer comes back to the reply.

The scripted model calls the delegation tool; the sub-agent then runs its own loop against the
same provider (tool calls of its own, then an answer) and the parent carries on with what it
returned. The chat shows one row per sub-agent, named after its task, that opens to the task and
the answer; the sub-agent's own tool calls stay out of the parent's reply. The limits an admin
sets (iterations, output length) hold, a failing sub-agent shows its error, several sub-agents
in one reply each get their row and result, and a reload keeps all of it. A call keeps to one
row from the moment it is made until its result is in, also after thinking (one group naming the
task once until it is opened) and across a reload while the sub-agent still works; a failing
sub-agent next to a slow one ends with both results; under a limit of one, two sub-agents run
one after the other and a stopped one frees its place.

Each request is tied to its own prompt: the sub-agent's latest user message is its task, the
parent's is what the person typed.

`test_a_failing_subagent_shows_its_error_and_the_parent_still_replies`,
`test_a_reload_after_the_run_keeps_the_rows_and_the_reply`,
`test_a_subagent_answer_longer_than_the_output_limit_is_cut`,
`test_a_subagent_runs_its_own_loop_and_the_parent_carries_on_with_its_answer`,
`test_a_subagent_stops_at_the_iteration_limit_and_says_so`,
`test_an_answer_within_the_output_limit_is_handed_over_whole`,
`test_several_subagents_in_one_reply_each_get_a_row_and_a_result` and
`test_the_result_row_opens_to_the_task_and_the_answer` are red on dev 62f70a844: since de73bb830 a
chat request whose reply message is already stored in the chat, the way automations, sub-agents and
timers prepare their reply, is refused with 409 and the reply is never written
(open-webui/open-webui#32066, fixed by 641c82df1).

`test_the_row_reads_executing_while_the_subagent_is_still_working` is red on dev 93fc3fcb7
for the same reason: the sub-agent is refused at once, so its row never reads Executing.

Discriminates: passes on dev 176d31d1d. In backend copies the limit tests turn red when the
iteration limit is dropped, the truncation is removed (or cut at `>=`), the failed status is
swallowed, only the first call of a turn runs or the answer is replaced; on a build with the
task cut at 600 characters, the executing label renamed and the tool output not drawn, the
matching row tests turn red while the two that read only the provider's requests stay green.
In a backend copy where a sub-agent's result never comes back the one-row, after-thinking,
reload-while-working and failing-next-to-slow tests turn red; where a delegation is listed twice
the one-row and after-thinking tests; where the limit is ignored the one-after-the-other test;
where a finished or stopped sub-agent keeps its place the stop test.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import admin_of
from harness.backends import wait_until
from utils.chat_ui import REPLY_TIMEOUT_MS, expect_reply, last_reply, send, stop_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
CLOCK = "get_current_timestamp"
ONE_AT_A_TIME = {"ENABLE_SUBAGENTS": "true", "SUBAGENTS_MAX_CONCURRENT": "1"}


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


def sub_row(page: Page, task: str) -> Locator:
    name = re.escape(f'View Result from Sub-agent: "{task}"')
    return last_reply(page).get_by_role("button", name=re.compile(name))


def sub_requests(upstream, task: str) -> list[dict]:
    return [r for r in upstream.chat_requests() if reply.answering(task)(r)]


def test_a_subagent_runs_its_own_loop_and_the_parent_carries_on_with_its_answer(
    subagents_on, page_for, make_user, upstream
):
    subagents_on()
    prompt, task = unique("please look into this"), unique("read the clock twice")
    answer = unique("It is late")
    upstream.queue(
        delegation(task, prompt),
        reply.tool_call(CLOCK, {}, "clock_1", match=reply.answering(task)),
        reply.tool_call(CLOCK, {}, "clock_2", match=reply.answering(task)),
        reply.text(answer, match=reply.answering(task)),
        reply.text("The helper says it is late.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())

    send(page, prompt)

    expect_reply(page, "The helper says it is late.")
    expect(sub_row(page, task)).to_be_visible()
    expect(
        last_reply(page).get_by_role("button", name=re.compile("View Result from"))
    ).to_have_count(1)
    assert len(sub_requests(upstream, task)) == 3
    parent_followup = upstream.chat_requests()[-1]["messages"][-1]
    assert (parent_followup["role"], parent_followup["content"]) == ("tool", answer)


def test_the_result_row_opens_to_the_task_and_the_answer(
    subagents_on, page_for, make_user, upstream
):
    subagents_on()
    prompt, task = unique("delegate this"), unique("name the capital")
    answer = unique("Vienna")
    upstream.queue(
        delegation(task, prompt, context="the country is Austria"),
        reply.text(answer, match=reply.answering(task)),
        reply.text("Done delegating.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())
    send(page, prompt)
    expect_reply(page, "Done delegating.")

    expect(last_reply(page)).not_to_contain_text(answer)
    sub_row(page, task).click()

    # exact matches: the task also sits in the row's own name
    expect(last_reply(page).get_by_text(task, exact=True)).to_be_visible()
    expect(last_reply(page).get_by_text("the country is Austria", exact=True)).to_be_visible()
    expect(last_reply(page).get_by_text(answer, exact=True)).to_be_visible()


def test_the_row_reads_executing_while_the_subagent_is_still_working(
    subagents_on, page_for, make_user, upstream
):
    subagents_on()
    prompt, task = unique("delegate slowly"), unique("take your time")
    upstream.queue(
        delegation(task, prompt),
        reply.text("Finally.", delay=4.0, match=reply.answering(task)),
        reply.text("It came back.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())

    send(page, prompt)

    working = re.escape(f'Executing Sub-agent: "{task}"...')
    expect(last_reply(page).get_by_role("button", name=re.compile(working))).to_be_visible(
        timeout=REPLY_TIMEOUT_MS
    )
    expect_reply(page, "It came back.")
    expect(sub_row(page, task)).to_be_visible()


def test_a_long_task_is_cut_in_the_row_name(subagents_on, page_for, make_user, upstream):
    subagents_on()
    prompt = unique("delegate a long one")
    task = f"{uuid.uuid4().hex[:8]} " + "and then some more words " * 6
    upstream.queue(
        delegation(task, prompt),
        reply.text("Long one done.", match=reply.answering(task)),
        reply.text("The long task is over.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())

    send(page, prompt)

    expect_reply(page, "The long task is over.")
    shortened = re.escape(f'Sub-agent: "{task[:60]}..."')
    expect(last_reply(page).get_by_role("button", name=re.compile(shortened))).to_be_visible()


def test_a_subagent_stops_at_the_iteration_limit_and_says_so(
    subagents_on, page_for, make_user, upstream
):
    subagents_on(SUBAGENTS_MAX_ITERATIONS=2)
    prompt, task = unique("delegate a long errand"), unique("keep checking the clock")
    upstream.queue(
        delegation(task, prompt),
        *[reply.tool_call(CLOCK, {}, f"clock_{n}", match=reply.answering(task)) for n in range(3)],
        reply.text("It gave up.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())

    send(page, prompt)

    expect_reply(page, "It gave up.")
    sub_row(page, task).click()
    # since d87837372 the request after the last allowed round withdraws the tools for an answer,
    # and a call the model still makes there fails with the limit
    expect(last_reply(page)).to_contain_text("Tool-call iteration limit reached.")
    requests = sub_requests(upstream, task)
    assert len(requests) == 3
    assert requests[-1].get("tool_choice") == "none" and not requests[-1].get("tools"), requests[-1]


def test_a_subagent_answer_longer_than_the_output_limit_is_cut(
    subagents_on, page_for, make_user, upstream
):
    subagents_on(SUBAGENTS_MAX_OUTPUT=20)
    prompt, task = unique("delegate a report"), unique("write the report")
    kept, dropped = "K" * 20, "the tail nobody should see"
    upstream.queue(
        delegation(task, prompt),
        reply.text(kept + dropped, match=reply.answering(task)),
        reply.text("Report received.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())

    send(page, prompt)

    expect_reply(page, "Report received.")
    sub_row(page, task).click()
    expect(last_reply(page)).to_contain_text(f"{kept}\n\n[output truncated]")
    expect(last_reply(page)).not_to_contain_text(dropped)
    handed_over = upstream.chat_requests()[-1]["messages"][-1]["content"]
    assert handed_over == f"{kept}\n\n[output truncated]"


def test_an_answer_within_the_output_limit_is_handed_over_whole(
    subagents_on, page_for, make_user, upstream
):
    subagents_on(SUBAGENTS_MAX_OUTPUT=20)
    prompt, task = unique("delegate a short one"), unique("write the note")
    upstream.queue(
        delegation(task, prompt),
        reply.text("exactly twenty chars", match=reply.answering(task)),
        reply.text("Note received.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())

    send(page, prompt)

    expect_reply(page, "Note received.")
    assert upstream.chat_requests()[-1]["messages"][-1]["content"] == "exactly twenty chars"


def test_a_failing_subagent_shows_its_error_and_the_parent_still_replies(
    subagents_on, page_for, make_user, upstream
):
    subagents_on()
    prompt, task = unique("delegate a doomed job"), unique("call the broken model")
    upstream.queue(
        delegation(task, prompt),
        reply.error(500, "the helper model is down", match=reply.answering(task)),
        reply.text("The helper failed.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())

    send(page, prompt)

    expect_reply(page, "The helper failed.")
    sub_row(page, task).click()
    expect(last_reply(page)).to_contain_text("Error:")
    expect(last_reply(page)).to_contain_text("the helper model is down")
    failed = upstream.chat_requests()[-1]["messages"][-1]
    assert failed["role"] == "tool"
    assert "the helper model is down" in failed["content"]


def test_several_subagents_in_one_reply_each_get_a_row_and_a_result(
    subagents_on, page_for, make_user, upstream
):
    subagents_on()
    prompt = unique("split this up")
    tasks = [unique("count the apples"), unique("count the pears")]
    answers = [unique("seven apples"), unique("nine pears")]
    calls = [
        {
            "id": f"call_{n}",
            "type": "function",
            "function": {"name": "delegate_task", "arguments": f'{{"task": "{task}"}}'},
        }
        for n, task in enumerate(tasks)
    ]
    upstream.queue(
        reply.Reply(tool_calls=calls, match=reply.answering(prompt)),
        *[reply.text(a, match=reply.answering(t)) for t, a in zip(tasks, answers)],
        reply.text("Both helpers reported.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())

    send(page, prompt)

    expect_reply(page, "Both helpers reported.")
    # two calls in one turn fold into a group of steps
    last_reply(page).get_by_text("Explored", exact=True).click()
    for task, answer in zip(tasks, answers):
        expect(sub_row(page, task)).to_be_visible()
        sub_row(page, task).click()
        expect(last_reply(page)).to_contain_text(answer)
    handed_over = {
        entry["tool_call_id"]: entry["content"]
        for entry in reply.conversation(upstream.chat_requests()[-1])
        if entry["role"] == "tool"
    }
    assert handed_over == {"call_0": answers[0], "call_1": answers[1]}


def test_a_reload_after_the_run_keeps_the_rows_and_the_reply(
    subagents_on, page_for, make_user, upstream
):
    subagents_on()
    prompt, task = unique("delegate then reload"), unique("find the answer")
    answer = unique("forty two")
    upstream.queue(
        delegation(task, prompt),
        reply.text(answer, match=reply.answering(task)),
        reply.text("Reload me.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())
    send(page, prompt)
    expect_reply(page, "Reload me.")

    page.reload()

    expect_reply(page, "Reload me.")
    expect(sub_row(page, task)).to_be_visible()
    sub_row(page, task).click()
    expect(last_reply(page)).to_contain_text(answer)


def task_rows(page: Page, task: str) -> Locator:
    """Every row of the last reply that names the task, whatever state it reads."""
    return last_reply(page).get_by_role("button").filter(has_text=task)


def working_row(page: Page, task: str) -> Locator:
    name = re.escape(f'Executing Sub-agent: "{task}"...')
    return last_reply(page).get_by_role("button", name=re.compile(name))


def unfinished_rows(page: Page) -> Locator:
    return last_reply(page).get_by_role("button", name=re.compile("^(Executing|Preparing) "))


def two_delegations(tasks: list[str], prompt: str) -> reply.Reply:
    calls = [
        {
            "id": f"call_{n}",
            "type": "function",
            "function": {"name": "delegate_task", "arguments": json.dumps({"task": task})},
        }
        for n, task in enumerate(tasks)
    ]
    return reply.Reply(tool_calls=calls, match=reply.answering(prompt))


def test_one_delegation_shows_one_row_from_the_call_to_its_result(
    subagents_on, page_for, make_user, upstream
):
    subagents_on()
    prompt, task = unique("hand this over"), unique("look it up")
    answered = threading.Event()
    upstream.queue(
        delegation(task, prompt),
        reply.text("Found it.", hold=answered, match=reply.answering(task)),
        reply.text("The helper found it.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())

    try:
        send(page, prompt)

        expect(working_row(page, task)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
        expect(task_rows(page, task)).to_have_count(1)
    finally:
        answered.set()

    expect_reply(page, "The helper found it.")
    expect(sub_row(page, task)).to_be_visible()
    expect(task_rows(page, task)).to_have_count(1)
    expect(unfinished_rows(page)).to_have_count(0)
    assert len(sub_requests(upstream, task)) == 1


def test_a_delegation_after_thinking_folds_into_one_group_that_names_the_task_once(
    subagents_on, page_for, make_user, upstream
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
        reply.text("Shelf checked.", hold=answered, match=reply.answering(task)),
        reply.text("The shelf is fine.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())

    try:
        send(page, prompt)

        group = last_reply(page).get_by_role("button", name="Toggle details")
        expect(group).to_contain_text(f'Exploring Sub-agent: "{task}"', timeout=REPLY_TIMEOUT_MS)
        expect(task_rows(page, task)).to_have_count(1)
    finally:
        answered.set()

    expect_reply(page, "The shelf is fine.")
    expect(group).to_contain_text(f'Explored Sub-agent: "{task}"')
    expect(task_rows(page, task)).to_have_count(1)
    group.click()
    # opened, the group lists its steps under the header: the call's own row joins it
    expect(sub_row(page, task)).to_be_visible()
    expect(task_rows(page, task)).to_have_count(2)
    expect(unfinished_rows(page)).to_have_count(0)


def test_a_reload_while_the_subagent_works_shows_it_working_and_then_its_result(
    subagents_on, page_for, make_user, upstream
):
    subagents_on()
    prompt, task = unique("hand over then reload"), unique("dig through the archive")
    answer = unique("box nine")
    answered = threading.Event()
    upstream.queue(
        delegation(task, prompt),
        reply.text(answer, hold=answered, match=reply.answering(task)),
        reply.text("The archive had it.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())

    try:
        send(page, prompt)
        expect(working_row(page, task)).to_be_visible(timeout=REPLY_TIMEOUT_MS)

        page.reload()

        # reloaded mid-run the row reads Preparing, not Executing: either way it is still working
        expect(unfinished_rows(page).filter(has_text=task)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
        expect(task_rows(page, task)).to_have_count(1)
    finally:
        answered.set()

    expect_reply(page, "The archive had it.")
    expect(task_rows(page, task)).to_have_count(1)
    expect(unfinished_rows(page)).to_have_count(0)
    sub_row(page, task).click()
    expect(last_reply(page)).to_contain_text(answer)
    assert len(sub_requests(upstream, task)) == 1


def test_a_subagent_failing_next_to_one_still_working_ends_with_both_results(
    subagents_on, page_for, make_user, upstream
):
    subagents_on()
    prompt = unique("split this in two")
    doomed, slow = unique("ask the broken model"), unique("count the stairs")
    answer = unique("ninety steps")
    answered = threading.Event()
    upstream.queue(
        two_delegations([doomed, slow], prompt),
        reply.error(500, "the helper model is down", match=reply.answering(doomed)),
        reply.text(answer, hold=answered, match=reply.answering(slow)),
        reply.text("One failed and one counted.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())

    try:
        send(page, prompt)
        group = last_reply(page).get_by_role("button", name="Toggle details")
        expect(group).to_contain_text("Exploring", timeout=REPLY_TIMEOUT_MS)
        group.click()

        # the results of one turn arrive together, so the failed one waits for the slow one
        expect(working_row(page, slow)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
        assert wait_until(lambda: len(sub_requests(upstream, doomed)) == 1, timeout=10.0)
    finally:
        answered.set()

    expect_reply(page, "One failed and one counted.")
    expect(sub_row(page, doomed)).to_be_visible()
    expect(sub_row(page, slow)).to_be_visible()
    expect(unfinished_rows(page)).to_have_count(0)
    handed_over = {
        entry["tool_call_id"]: entry["content"]
        for entry in reply.conversation(upstream.chat_requests()[-1])
        if entry["role"] == "tool"
    }
    assert set(handed_over) == {"call_0", "call_1"}, handed_over
    assert "the helper model is down" in handed_over["call_0"]
    assert handed_over["call_1"] == answer


@pytest.mark.slow
def test_a_limit_of_one_runs_two_subagents_one_after_the_other_and_both_come_back(
    instance_with, page_for
):
    # an instance of its own: the limit is read once, when the first sub-agent of a server runs
    booted = instance_with(ONE_AT_A_TIME)
    upstream = booted.upstream
    prompt = unique("split this in a queue")
    tasks = [unique("weigh the flour"), unique("weigh the sugar")]
    answers = [unique("one kilo"), unique("two kilos")]
    answered = threading.Event()

    def asked() -> list[dict]:
        return [r for task in tasks for r in sub_requests(upstream, task)]

    upstream.queue(
        two_delegations(tasks, prompt),
        *[reply.text(a, hold=answered, match=reply.answering(t)) for t, a in zip(tasks, answers)],
        reply.text("Both are weighed.", match=reply.answering(prompt)),
    )
    page = page_for(admin_of(booted))

    try:
        send(page, prompt)
        assert wait_until(lambda: len(asked()) >= 1, timeout=30.0), "no sub-agent ever started"
        time.sleep(2.0)  # shows the second sub-agent does not start while the first one works
        assert len(asked()) == 1, f"{len(asked())} sub-agents ran at once under a limit of one"
    finally:
        answered.set()

    expect_reply(page, "Both are weighed.")
    expect(unfinished_rows(page)).to_have_count(0)
    handed_over = {
        entry["tool_call_id"]: entry["content"]
        for entry in reply.conversation(upstream.chat_requests()[-1])
        if entry["role"] == "tool"
    }
    assert handed_over == {"call_0": answers[0], "call_1": answers[1]}


@pytest.mark.slow
def test_stopping_a_subagent_under_a_limit_of_one_frees_the_place_for_the_next(
    instance_with, page_for
):
    booted = instance_with(ONE_AT_A_TIME)
    upstream = booted.upstream
    prompt, task = unique("hand over then stop"), unique("wait at the door")
    second, next_task = unique("hand over once more"), unique("ring the bell")
    never_answered = threading.Event()
    upstream.queue(
        delegation(task, prompt),
        reply.text("too late", hold=never_answered, match=reply.answering(task)),
        delegation(next_task, second),
        reply.text("It rang.", match=reply.answering(next_task)),
        reply.text("The bell rang.", match=reply.answering(second)),
    )
    page = page_for(admin_of(booted))

    try:
        send(page, prompt)
        expect(working_row(page, task)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
        assert wait_until(lambda: len(sub_requests(upstream, task)) == 1, timeout=10.0)

        stop_button(page).click()
        expect(stop_button(page)).to_have_count(0)
        send(page, second)

        expect_reply(page, "The bell rang.")
        expect(sub_row(page, next_task)).to_be_visible()
        expect(unfinished_rows(page)).to_have_count(0)
    finally:
        never_answered.set()
