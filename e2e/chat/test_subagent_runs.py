"""Journey: the model hands work to a sub-agent in a chat and the answer comes back to the reply.

The scripted model calls the delegation tool; the sub-agent then runs its own loop against the
same provider (tool calls of its own, then an answer) and the parent carries on with what it
returned. The chat shows one row per sub-agent, named after its task, that opens to the task and
the answer; the sub-agent's own tool calls stay out of the parent's reply. The limits an admin
sets (iterations, output length) hold, a failing sub-agent shows its error, several sub-agents
in one reply each get their row and result, and a reload keeps all of it.

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
(open-webui/open-webui#32066).

Discriminates: passes on dev 176d31d1d. In backend copies the limit tests turn red when the
iteration limit is dropped, the truncation is removed (or cut at `>=`), the failed status is
swallowed, only the first call of a turn runs or the answer is replaced; on a build with the
task cut at 600 characters, the executing label renamed and the tool output not drawn, the
matching row tests turn red while the two that read only the provider's requests stay green.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import REPLY_TIMEOUT_MS, expect_reply, last_reply, send

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
    expect(last_reply(page)).to_contain_text("Tool-call limit reached (2 iterations).")
    # one request to start and one after each of the two allowed rounds
    assert len(sub_requests(upstream, task)) == 3
    failed = upstream.chat_requests()[-1]["messages"][-1]
    assert failed["role"] == "tool"
    assert failed["content"].startswith("Error:")
    assert "Tool-call limit reached (2 iterations)." in failed["content"]


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
        for entry in upstream.chat_requests()[-1]["messages"]
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
