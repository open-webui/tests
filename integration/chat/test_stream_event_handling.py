"""Responses-API stream handling and the tool follow-up, seen through the stored reply.

Six 0.11.1 repairs on the path that turns a provider stream into the stored `output` items:

* #27800 / #27810 (`74a790282`): `response.completed` replaced the accumulated reply with its
  `output` whenever the key was present, so a provider ending on `output: []` erased the reply.
  The same commit put the `response.output_item.done` branch above the generic `.done` branch
  that had shadowed it, so a finished item is applied instead of dropped.
* #28312 (`fc8a9b8ed`): a delta arriving before its output item raised `UnboundLocalError`, and a
  two-part event name like `response.delta` returned `None` to a caller unpacking a tuple.
* #28872 (`883c7434f`): a stream ending inside a reasoning item that never recorded
  `started_at` crashed on `ended_at - started_at`.
* #28016 (`f3f76095d`): every tool result was stamped `completed`; an error result is `failed`.
* #28633 (`a610d77137`): a >= 400 answer to the request made after a tool ran ended the reply
  silently; the provider's message is now stored as the reply's error.

The Responses streams come from a listener registered as a Responses-API connection; the tool
cases run on the scripted provider through the web client's chat path.

Twin of unit/chat/test_stream_event_handling.py.

Discriminates: passes on dev bbfa876af. Each narrow test fails with its fix reverted: a
present-but-empty `output` replacing the stream, `output_item.done` falling to the generic `.done`
branch, the delta branch returning an unbound `new_output` (both stray events cut the stream with
an error frame), `started_at` read unguarded, every tool result stamped completed, and the
>= 400 follow-up branch removed (both follow-up error tests).
"""

from __future__ import annotations

import json

import pytest

from harness import second_provider
from harness import upstream as reply
from harness.chat import ask
from harness.filters import global_filter
from harness.listener import json_answer, text_answer
from harness.python_tools import python_tool

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

RESPONSES_MODEL = "responses-model"


@pytest.fixture
def responses_connection(admin, listener, preserve):
    """A Responses-API connection; call it with the events its next stream carries."""
    preserve(second_provider.OPENAI_CONFIG)
    with admin.client() as client:
        second_provider.attach(client, listener, RESPONSES_MODEL, api_type="responses")

    def answer_with(events: list[dict]) -> None:
        stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        listener.route("POST", "/v1/responses", text_answer(stream, "text/event-stream"))

    return answer_with


def _message_added() -> list[dict]:
    item = {"type": "message", "id": "msg_1", "role": "assistant", "content": []}
    part = {"type": "output_text", "text": ""}
    added = {"type": "response.output_item.added", "output_index": 0, "item": item}
    part_added = {"type": "response.content_part.added", "output_index": 0, "part": part}
    return [added, {**part_added, "content_index": 0}]


def _text_delta(text: str, output_index: int = 0) -> dict:
    return {
        "type": "response.output_text.delta",
        "output_index": output_index,
        "content_index": 0,
        "delta": text,
    }


def _completed(**response) -> dict:
    return {"type": "response.completed", "response": {"id": "resp_1", **response}}


def _ask_admin(admin, content: str) -> dict:
    with admin.client() as client:
        _, message = ask(client, content, model=RESPONSES_MODEL)
    return message


def test_an_empty_final_output_keeps_the_streamed_reply(admin, responses_connection):
    responses_connection(
        [*_message_added(), _text_delta("the answer "), _text_delta("is 42"), _completed(output=[])]
    )

    message = _ask_admin(admin, "what is the answer?")

    assert message["content"] == "the answer is 42", (
        "`output: []` on response.completed erased the streamed reply (#27800)"
    )


def test_a_finished_output_item_is_applied(admin, responses_connection):
    finished = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "the complete answer"}],
    }
    responses_connection(
        [
            *_message_added(),
            {"type": "response.output_item.done", "output_index": 0, "item": finished},
            _completed(),
        ]
    )

    message = _ask_admin(admin, "answer in one go")

    assert message["content"] == "the complete answer", (
        "response.output_item.done was shadowed by the generic .done branch and dropped (#27810)"
    )


PASSTHROUGH_FILTER = """
class Filter:
    def outlet(self, body: dict) -> dict:
        return body
"""


@pytest.fixture
def outlet_filter(admin):
    """A global filter, which makes the server re-read every streamed event for its outlet."""
    with admin.client() as client, global_filter(client, PASSTHROUGH_FILTER):
        yield


def _stream_over_the_api(admin, content: str) -> list[dict]:
    """What an API client streaming the completion itself receives, event by event."""
    body = {"model": RESPONSES_MODEL, "messages": [{"role": "user", "content": content}]}
    with admin.client() as client:
        with client.stream("POST", "/api/chat/completions", json={**body, "stream": True}) as sent:
            lines = sent.read().decode().splitlines()
    payloads = [line.removeprefix("data:").strip() for line in lines if line.startswith("data:")]
    return [json.loads(payload) for payload in payloads if payload != "[DONE]"]


@pytest.mark.parametrize(
    "stray_event",
    [_text_delta("lost", output_index=0), {"type": "response.delta", "delta": "x"}],
    ids=["delta-before-its-item", "two-part-event-name"],
)
def test_an_unexpected_delta_does_not_cut_the_stream(
    admin, responses_connection, outlet_filter, stray_event
):
    responses_connection([stray_event, *_message_added(), _text_delta("still here"), _completed()])

    events = _stream_over_the_api(admin, "keep going")

    errors = [event["error"] for event in events if "error" in event]
    assert not errors, f"the stray delta cut the stream short (#28312): {errors}"
    assert "still here" in [event.get("delta") for event in events]


def test_a_stream_ending_in_reasoning_finishes_cleanly(admin, responses_connection):
    reasoning = {"type": "reasoning", "id": "rs_1", "status": "in_progress", "summary": []}
    responses_connection(
        [
            {"type": "response.output_item.added", "output_index": 0, "item": reasoning},
            {
                "type": "response.reasoning_summary_text.delta",
                "output_index": 0,
                "summary_index": 0,
                "delta": "thinking it over",
            },
        ]
    )

    message = _ask_admin(admin, "think only")

    assert not message.get("error"), (
        f"closing a reasoning item without started_at failed the turn (#28872): {message['error']}"
    )
    closed = message["output"][-1]
    assert closed["type"] == "reasoning"
    assert closed["status"] == "completed"
    assert closed["summary"][0]["text"] == "thinking it over"
    assert "duration" not in closed


def test_a_plain_responses_reply_is_stored(admin, responses_connection):
    responses_connection([*_message_added(), _text_delta("plain reply"), _completed()])

    assert _ask_admin(admin, "hello")["content"] == "plain reply"


def _tool_output(message: dict) -> dict:
    return next(item for item in message["output"] if item["type"] == "function_call_output")


def test_a_failed_tool_call_is_stamped_failed(make_user, upstream):
    upstream.queue(reply.tool_call("get_current_timestamp", "{not json"), reply.text("that failed"))
    with make_user().client() as client:
        _, message = ask(client, "what time is it?")

    tool_output = _tool_output(message)
    assert tool_output["output"][0]["text"].startswith("Error:")
    assert tool_output["status"] == "failed", (
        "an error tool result was stored as completed, so it showed as a success (#28016)"
    )


def test_a_successful_tool_call_is_stamped_completed(make_user, upstream):
    upstream.queue(reply.tool_call("get_current_timestamp", {}), reply.text("It is now."))
    with make_user().client() as client:
        _, message = ask(client, "what time is it?")

    assert _tool_output(message)["status"] == "completed"
    assert message["content"].endswith("It is now.")


def test_a_provider_error_after_a_tool_ran_is_stored(make_user, upstream):
    upstream.queue(
        reply.tool_call("get_current_timestamp", {}), reply.error(502, "upstream exploded")
    )
    with make_user().client() as client:
        _, message = ask(client, "what time is it?")

    assert (message.get("error") or {}).get("content") == "upstream exploded", (
        f"a 502 after the tool ran ended the reply silently (#28633): {message}"
    )


ECHO_TOOL = '''
class Tools:
    def echo_result(self, text: str) -> str:
        """
        Return the given text as the tool result.
        :param text: The result to return.
        """
        return text
'''


@pytest.fixture
def echo_tool(admin):
    """A workspace tool whose result is whatever the model passes it."""
    with python_tool(admin, ECHO_TOOL, name="Echo") as tool_id:
        yield tool_id


@pytest.mark.parametrize(
    ("tool_result", "status"),
    [
        ("Traceback (most recent call last):", "failed"),
        ('{"error": "boom"}', "failed"),
        ('{"status": "Failed"}', "failed"),
        ('{"success": false, "message": "could not fetch"}', "failed"),
        ("The weather in Paris is 18C.", "completed"),
        ("no error occurred", "completed"),
        ('{"error": ""}', "completed"),
        ('{"success": false}', "completed"),
    ],
)
def test_every_error_shaped_tool_result_is_stamped_failed(
    admin, upstream, echo_tool, tool_result, status
):
    upstream.queue(reply.tool_call("echo_result", {"text": tool_result}), reply.text("noted"))
    with admin.client() as client:
        _, message = ask(client, "run the tool", tool_ids=[echo_tool])

    assert _tool_output(message)["status"] == status


PROVIDER_MODEL = "listener-model"


@pytest.mark.parametrize(
    ("error_body", "shown"),
    [
        ({"error": {"message": "upstream exploded"}}, "upstream exploded"),
        ({"error": "rate limited"}, "rate limited"),
        ({"detail": "model not found"}, "model not found"),
        ({"message": "bad request"}, "bad request"),
        ({"error": {"detail": "nested detail"}}, "nested detail"),
    ],
)
def test_the_providers_own_error_message_is_stored(admin, listener, preserve, error_body, shown):
    preserve(second_provider.OPENAI_CONFIG)
    with admin.client() as client:
        second_provider.attach(client, listener, PROVIDER_MODEL)
    listener.route("POST", "/v1/chat/completions", json_answer(error_body, status=502))

    with admin.client() as client:
        _, message = ask(client, "hello?", model=PROVIDER_MODEL)

    assert (message.get("error") or {}).get("content") == shown


def test_a_plain_text_error_after_a_tool_ran_names_the_status(admin, listener, preserve):
    preserve(second_provider.OPENAI_CONFIG)
    with admin.client() as client:
        second_provider.attach(client, listener, PROVIDER_MODEL)
    tool_call = second_provider.sse(
        second_provider.tool_call_delta("get_current_timestamp", {}), finish_reason="tool_calls"
    )
    answers = iter([tool_call, text_answer("gateway down", "text/plain", status=502)])
    listener.route("POST", "/v1/chat/completions", lambda _request: next(answers))

    with admin.client() as client:
        _, message = ask(client, "what time is it?", model=PROVIDER_MODEL)

    assert (message.get("error") or {}).get("content") == "Provider returned HTTP 502", message
