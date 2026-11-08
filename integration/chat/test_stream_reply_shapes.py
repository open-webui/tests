"""Journey: reply shapes the streaming handler turns into what a user sees and keeps.

Many local models write their reasoning into the text itself between tags (`<think>` and its
kin, or a pair the chat settings name), often split across chunks, with attributes, or never
closed. The handler lifts that into a reasoning item and keeps the answer, unless the chat
turned tag detection off. A reply of only whitespace is stored empty, an admin can stop a reply
mid-stream and keep what arrived, and a model that keeps calling tools is cut off at the
configured number of rounds with a message saying so. Since d87837372 (0.13.1) the call the
limit cuts off gets an error result, so the stored calls and results stay paired. Since then the
tool loop also stops on its own: a call that failed twice in a row with the same arguments is not
run a third time, and a reply the token limit ended does not run its unfinished tool calls; in
both cases the model is asked once more without tools for a final answer, and a reply that still
calls a tool then ends with an error. A failing call with changed arguments and a succeeding call
repeated unchanged keep running.

Discriminates: in a backend copy, dropping the attribute parsing fails the attributes case,
ignoring the chat's own tag pair fails the custom-tags case, looking for the end tag only past
the text already scanned fails the split case, leaving stopped items `in_progress` fails the
stop test and removing the iteration limit message fails the limit test. On dev e8c0ad85e, the
repetition check answering None fails the failed-twice test, the check counting succeeded calls
too fails the repeat-success test, the token limit not marking the calls incomplete fails the
token limit and still-calls tests, and no final answer request without tools fails the
failed-twice, token limit and still-calls tests.
"""

from __future__ import annotations

import uuid

import pytest

from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import ask, wait_for_reply
from harness.inflight import start_slow_reply, wait_until_streaming
from harness.python_tools import python_tool
from harness.socket_client import connected
from harness.tool_calls import without_hidden_calls

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def _items(message: dict, kind: str) -> list[dict]:
    return [item for item in without_hidden_calls(message["output"]) if item["type"] == kind]


def _reasoning_text(message: dict) -> list[str]:
    return [item["content"][0]["text"] for item in _items(message, "reasoning")]


def _ask_streamed(make_user, upstream, pieces: list[str], **options) -> dict:
    upstream.queue(reply.text(pieces))
    with make_user().client() as client:
        _, message = ask(client, "think it over", **options)
    return message


def test_tags_split_across_chunks_become_a_reasoning_item(make_user, upstream):
    pieces = ["Sure. <thi", "nk>pondering", " more</thi", "nk>The answer", " is 4."]

    message = _ask_streamed(make_user, upstream, pieces)

    [reasoning] = _items(message, "reasoning")
    assert reasoning["content"][0]["text"] == "pondering more"
    assert reasoning["status"] == "completed"
    assert (reasoning["start_tag"], reasoning["end_tag"]) == ("<think>", "</think>")
    assert [item["type"] for item in without_hidden_calls(message["output"])] == [
        "message",
        "reasoning",
        "message",
    ]
    assert without_hidden_calls(message["output"])[-1]["content"][0]["text"] == "The answer is 4."
    assert "pondering" not in message["content"]
    assert "<think>" not in message["content"]


def test_a_tag_keeps_its_attributes(make_user, upstream):
    message = _ask_streamed(make_user, upstream, ['<think effort="high">deep</think>ok'])

    [reasoning] = _items(message, "reasoning")
    assert reasoning["attributes"] == {"effort": "high"}
    assert _reasoning_text(message) == ["deep"]
    assert message["content"] == "ok"


def test_the_chats_own_tag_pair_is_detected(make_user, upstream):
    message = _ask_streamed(
        make_user,
        upstream,
        ["<plan>step one</plan>Done."],
        params={"reasoning_tags": ["<plan>", "</plan>"]},
    )

    assert _reasoning_text(message) == ["step one"]
    assert message["content"] == "Done."


def test_detection_switched_off_keeps_the_tags_as_text(make_user, upstream):
    message = _ask_streamed(
        make_user, upstream, ["<think>visible</think>kept"], params={"reasoning_tags": False}
    )

    assert _items(message, "reasoning") == []
    assert message["content"] == "<think>visible</think>kept"


def test_an_unclosed_tag_is_closed_when_the_stream_ends(make_user, upstream):
    message = _ask_streamed(make_user, upstream, ["<think>never ", "closes"])

    [reasoning] = _items(message, "reasoning")
    assert reasoning["status"] == "completed"
    assert reasoning["content"][0]["text"] == "never closes"
    assert not message.get("error")


def test_a_reply_of_only_whitespace_is_stored_empty(make_user, upstream):
    message = _ask_streamed(make_user, upstream, ["   ", "\n"])

    assert message["content"] == ""
    assert [item["content"][0]["text"] for item in _items(message, "message")] == [""]


def test_an_admin_stops_a_reply_and_keeps_what_arrived(admin, upstream):
    with admin.client() as client, connected(admin) as tab:
        turn = start_slow_reply(client, upstream, chunk_delay=0.2)
        [task_id] = client.get(f"/api/tasks/chat/{turn.chat_id}").json()["task_ids"]
        wait_until_streaming(tab, turn.chat_id)
        stopped = client.post(f"/api/tasks/stop/{task_id}")
        assert stopped.status_code == 200, stopped.text
        assert stopped.json()["status"] is True
        # with Redis the stop answers before the stopped reply is saved
        message = wait_for_reply(client, turn)

    assert message["content"].startswith("part-0 ")
    assert "part-19" not in message["content"]
    assert [item["status"] for item in without_hidden_calls(message["output"])] == ["incomplete"]


def test_only_an_admin_may_stop_a_task_by_id(make_user, upstream):
    with make_user().client() as client:
        turn = start_slow_reply(client, upstream, chunk_delay=0.05)
        [task_id] = client.get(f"/api/tasks/chat/{turn.chat_id}").json()["task_ids"]
        refused = client.post(f"/api/tasks/stop/{task_id}")

    assert refused.status_code in (401, 403), refused.text


@pytest.mark.slow
def test_a_model_that_keeps_calling_tools_is_cut_off(instance_with):
    limited = instance_with({"CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS": "2"})
    limited.upstream.queue(
        *(reply.tool_call("get_current_timestamp", {}, call_id=f"call_{n}") for n in range(3)),
    )
    with admin_of(limited).client() as client:
        _, message = ask(client, "what time is it, again and again?")

    # the third call is asked for but never run
    assert len(limited.upstream.chat_requests()) == 3
    assert len(_items(message, "function_call")) == 3
    results = [item["output"][0]["text"] for item in _items(message, "function_call_output")]
    assert len(results) == 3, results
    assert all("current_timestamp" in result for result in results[:2]), results
    assert results[2] == "Error: Tool-call iteration limit reached.", (
        f"the third call was not cut off: {results[2]}"
    )
    assert "Tool-call iteration limit reached." in str(message.get("error"))


# --- the tool loop stops (d87837372) ------------------------------------------------------------

BERTH_TOOL_SOURCE = '''
class Tools:
    def open_berth(self, berth: str) -> str:
        """Open a berth in the harbour.

        :param berth: The berth to open.
        """
        raise RuntimeError("the berth gate is jammed")
'''

FINAL_ANSWER_PROMPT = "Tool use has stopped for this response."
REPEATED_FAILURE = "Error: the same tool call failed twice. Finish using the available results."
INCOMPLETE_CALL = "Error: the model response was incomplete; its tool calls were not executed."


@pytest.fixture(scope="module")
def berth_tool(admin):
    with python_tool(admin, BERTH_TOOL_SOURCE, name="Berths") as tool_id:
        yield tool_id


def _tool_loop(make_user, upstream, *replies, tool_ids=()) -> tuple[dict, list[str], list[dict]]:
    """The stored reply, its tool results and the provider requests for a scripted tool loop."""
    prompt = f"open the berths ({uuid.uuid4().hex[:8]})"
    answering = reply.answering(prompt)
    for scripted in replies:
        scripted.match = answering
    upstream.queue(*replies)
    with make_user().client() as client:
        _, message = ask(client, prompt, tool_ids=list(tool_ids))
    results = [item["output"][0]["text"] for item in _items(message, "function_call_output")]
    return message, results, [request for request in upstream.chat_requests() if answering(request)]


def _asks_for_a_final_answer(request: dict) -> bool:
    instructions = [
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    ]
    return (
        not request.get("tools")
        and request.get("tool_choice") == "none"
        and any(FINAL_ANSWER_PROMPT in text for text in instructions)
    )


def test_a_call_that_failed_twice_unchanged_is_not_run_again_and_the_model_answers(
    make_user, upstream, berth_tool
):
    message, results, requests = _tool_loop(
        make_user,
        upstream,
        *(reply.tool_call("open_berth", {"berth": "north"}, f"call_{n}") for n in range(3)),
        reply.text("The north berth gate is jammed."),
        tool_ids=[berth_tool],
    )

    assert len(results) == 3, results
    assert all("jammed" in result for result in results[:2]), results
    assert results[2] == REPEATED_FAILURE, f"the third unchanged call was run: {results[2]}"
    assert [_asks_for_a_final_answer(request) for request in requests] == [
        False,
        False,
        False,
        True,
    ], "only the request after the refused repeat should ask for a final answer without tools"
    assert message["content"] == "The north berth gate is jammed."
    assert not message.get("error"), message.get("error")


def test_a_failing_call_with_changed_arguments_keeps_running(make_user, upstream, berth_tool):
    _, results, requests = _tool_loop(
        make_user,
        upstream,
        *(
            reply.tool_call("open_berth", {"berth": berth}, f"call_{berth}")
            for berth in ("north", "south", "east")
        ),
        reply.text("Every gate is jammed."),
        tool_ids=[berth_tool],
    )

    assert len(results) == 3 and all("jammed" in result for result in results), results
    assert not any(_asks_for_a_final_answer(request) for request in requests)


def test_a_call_that_succeeds_can_repeat_unchanged(make_user, upstream):
    _, results, requests = _tool_loop(
        make_user,
        upstream,
        *(reply.tool_call("get_current_timestamp", {}, f"call_{n}") for n in range(3)),
        reply.text("Still the same hour."),
    )

    assert len(results) == 3 and all("current_timestamp" in result for result in results), results
    assert not any(_asks_for_a_final_answer(request) for request in requests)


def test_a_reply_ended_by_the_token_limit_does_not_run_its_tool_calls(make_user, upstream):
    message, results, requests = _tool_loop(
        make_user,
        upstream,
        reply.tool_call("get_current_timestamp", {}, finish_reason="length"),
        reply.text("I ran out of room before the call."),
    )

    assert len(results) == 1 and results[0].startswith(INCOMPLETE_CALL), (
        f"the unfinished call was run: {results}"
    )
    assert [_asks_for_a_final_answer(request) for request in requests] == [False, True]
    assert message["content"] == "I ran out of room before the call."


def test_a_model_that_still_calls_a_tool_after_tool_use_stopped_ends_with_an_error(
    make_user, upstream
):
    message, results, requests = _tool_loop(
        make_user,
        upstream,
        reply.tool_call("get_current_timestamp", {}, "call_1", finish_reason="length"),
        reply.tool_call("get_current_timestamp", {}, "call_2"),
    )

    assert len(requests) == 2 and _asks_for_a_final_answer(requests[1])
    assert all(not result.startswith("{") for result in results), f"a call ran: {results}"
    assert message.get("error"), f"the reply ended without an error: {message}"
