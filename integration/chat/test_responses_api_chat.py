"""Journey: chatting through a connection set to the OpenAI Responses API.

A chat with such a model is sent as a Responses request (the system prompt as `instructions`,
messages and earlier tool calls as `input` items, `max_output_tokens`, flat function tools) and
its event stream is assembled into the stored reply: reasoning summaries and bodies, text,
function calls whose arguments arrive in pieces, refusals, url citations, the final `output`
of `response.completed` and its usage. An API client can also call the connection without
streaming, or through the `/openai/responses` proxy. The tests at
test_stream_event_handling.py pin single repairs on the same path; these walk whole replies.

Discriminates: in a backend copy, dropping `instructions` fails the payload test, leaving a
tool in the Chat Completions shape fails the tool test, ignoring `response.reasoning_text.delta`
fails the reasoning test, skipping url citations fails the citation test, and a non-streamed
result returned unconverted fails the API client test.
"""

from __future__ import annotations

import pytest

from harness import responses_provider as responses_api
from harness.chat import ask
from harness.chat_history import seed_chat
from harness.listener import json_answer, text_answer
from harness.second_provider import OPENAI_CONFIG

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

MODEL = responses_api.RESPONSES_MODEL
SYSTEM_PROMPT = "Answer like a pirate."
FIRST_USAGE = {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
SECOND_USAGE = {"input_tokens": 20, "output_tokens": 2, "total_tokens": 22}


@pytest.fixture
def responses(admin, preserve, listener):
    preserve(OPENAI_CONFIG)
    with admin.client() as client:
        provider = responses_api.connect_responses(client, listener)
        yield provider, client


def _items(message: dict, kind: str) -> list[dict]:
    return [item for item in message["output"] if item["type"] == kind]


def test_a_request_is_sent_in_the_responses_shape(responses):
    provider, client = responses
    provider.answer(
        responses_api.events_stream(*responses_api.message("Arr."), responses_api.completed())
    )

    ask(
        client,
        "hello",
        model=MODEL,
        history=[{"role": "system", "content": SYSTEM_PROMPT}],
        params={"max_tokens": 50, "temperature": 0.5, "stop": ["END"]},
    )

    [sent] = provider.sent()
    assert sent["instructions"] == SYSTEM_PROMPT
    assert sent["input"] == [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]
    assert (sent["max_output_tokens"], sent["temperature"], sent["stream"]) == (50, 0.5, True)
    assert not {"messages", "max_tokens", "stop", "stream_options"} & sent.keys()
    timestamp_tool = next(tool for tool in sent["tools"] if tool["name"] == "get_current_timestamp")
    assert timestamp_tool["type"] == "function"
    assert timestamp_tool["strict"] is False
    assert "function" not in timestamp_tool


def test_reasoning_a_tool_call_and_the_answer_are_stored(responses):
    provider, client = responses
    provider.answer(
        responses_api.events_stream(
            *responses_api.reasoning("plan it", text="the clock decides"),
            *responses_api.function_call("get_current_timestamp", {}, index=1),
            responses_api.completed(FIRST_USAGE),
        ),
        responses_api.events_stream(
            *responses_api.message("It is ", "late."), responses_api.completed(SECOND_USAGE)
        ),
    )

    _, message = ask(client, "what time is it?", model=MODEL)

    assert message["content"] == "It is late."
    [thought] = _items(message, "reasoning")
    assert thought["status"] == "completed"
    assert thought["summary"][0]["text"] == "plan it"
    assert thought["content"][0]["text"] == "the clock decides"
    [call] = _items(message, "function_call")
    assert (call["name"], call["arguments"], call["call_id"]) == (
        "get_current_timestamp",
        "{}",
        "call_1",
    )
    [result] = _items(message, "function_call_output")
    assert "current_timestamp" in result["output"][0]["text"]
    usage = message["usage"]
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (30, 6, 36)

    follow_up = provider.sent()[-1]["input"]
    assert follow_up[-2] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "get_current_timestamp",
        "arguments": "{}",
    }
    assert follow_up[-1]["type"] == "function_call_output"
    assert follow_up[-1]["call_id"] == "call_1"
    assert "current_timestamp" in follow_up[-1]["output"]


def test_an_earlier_turn_is_replayed_as_input_items(responses):
    provider, client = responses
    image = {"type": "image", "id": "i1", "url": "data:image/png;base64,AAAA"}
    answered = {
        "type": "message",
        "id": "msg_old",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "a parrot"}],
    }
    history = [
        {"role": "user", "content": "what is this?", "files": [image]},
        {"role": "assistant", "content": "a parrot", "output": [answered]},
    ]
    chat_id, last_id = seed_chat(client, history, model=MODEL)
    provider.answer(
        responses_api.events_stream(*responses_api.message("still"), responses_api.completed())
    )

    ask(client, "and now?", model=MODEL, chat_id=chat_id, parent_id=last_id)

    first, replayed, latest = provider.sent()[-1]["input"]
    assert first["role"] == "user"
    assert first["content"] == [
        {"type": "input_text", "text": "what is this?"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "auto"},
    ]
    assert replayed["type"] == "message"
    assert replayed["content"] == [{"type": "output_text", "text": "a parrot"}]
    assert "status" not in replayed
    assert latest["content"] == [{"type": "input_text", "text": "and now?"}]


def test_a_url_citation_becomes_a_source_of_the_reply(responses):
    provider, client = responses
    citation = {
        "type": "url_citation",
        "url": "https://example.org/treasure",
        "title": "Treasure map",
        "start_index": 0,
        "end_index": 5,
    }
    finished = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "cited", "annotations": [citation]}],
    }
    provider.answer(
        responses_api.events_stream(
            *responses_api.message("cited"),
            {"type": "response.output_item.done", "output_index": 0, "item": finished},
            responses_api.completed(),
        )
    )

    _, message = ask(client, "where is it?", model=MODEL)

    assert message["content"] == "cited"
    [source] = message["sources"]
    assert source["source"] == {"name": "Treasure map", "url": "https://example.org/treasure"}


def test_the_final_output_of_a_completed_response_wins(responses):
    provider, client = responses
    final = {
        "type": "message",
        "id": "msg_9",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "the final word"}],
    }
    completed = responses_api.completed(FIRST_USAGE)
    completed["response"]["output"] = [final]
    provider.answer(
        responses_api.events_stream(
            {"type": "response.in_progress", "response": {"id": "resp_1"}},
            *responses_api.message("a draft"),
            completed,
        )
    )

    _, message = ask(client, "decide", model=MODEL)

    assert message["content"] == "the final word"
    assert message["usage"]["total_tokens"] == 14


def test_a_refusal_is_kept_in_the_stored_output(responses):
    provider, client = responses
    item = {"type": "message", "id": "msg_r", "role": "assistant", "content": []}
    empty = {"type": "refusal", "refusal": ""}
    refusal = {**empty, "refusal": "I can't help with that."}
    position = {"output_index": 0, "content_index": 0}
    provider.answer(
        responses_api.events_stream(
            {"type": "response.output_item.added", "output_index": 0, "item": item},
            {"type": "response.content_part.added", **position, "part": empty},
            {"type": "response.refusal.delta", **position, "delta": "I can't "},
            {"type": "response.refusal.delta", **position, "delta": "help with that."},
            {"type": "response.refusal.done", **position, "refusal": refusal["refusal"]},
            {"type": "response.content_part.done", **position, "part": refusal},
            responses_api.completed(),
        )
    )

    _, message = ask(client, "something forbidden", model=MODEL)

    [stored] = _items(message, "message")
    assert stored["content"] == [refusal]
    assert not message.get("error")


def test_an_api_client_gets_a_non_streamed_result_in_chat_completions_shape(responses):
    provider, client = responses
    result = {
        "id": "resp_7",
        "model": MODEL,
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": "checking"}]},
            {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": {"id": 7}},
        ],
        "usage": FIRST_USAGE,
    }
    provider.answer(json_answer(result))
    lookup = {"name": "lookup", "parameters": {"type": "object"}}

    answered = client.post(
        "/api/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": "find 7"},
            ],
            "stream": False,
            "max_completion_tokens": 30,
            "tools": [{"type": "function", "function": lookup}],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        },
    )

    assert answered.status_code == 200, answered.text
    body = answered.json()
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "checking"
    assert choice["message"]["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"id": 7}'}}
    ]
    assert body["usage"] == FIRST_USAGE
    [sent] = provider.sent()
    assert sent["instructions"] == SYSTEM_PROMPT
    assert sent["max_output_tokens"] == 30
    assert sent["tool_choice"] == {"type": "function", "name": "lookup"}
    assert sent["tools"] == [{"type": "function", **lookup, "strict": False}]


def test_the_responses_proxy_forwards_a_request_and_its_answer(responses):
    provider, client = responses
    result = {"id": "resp_8", "object": "response", "output": []}
    provider.answer(
        json_answer(result),
        responses_api.events_stream(*responses_api.message("streamed"), responses_api.completed()),
        json_answer({"error": {"message": "bad input"}}, status=400),
        text_answer("gateway down", "text/plain", status=502),
    )
    request = {"model": MODEL, "input": "hi", "instructions": SYSTEM_PROMPT}

    plain = client.post("/openai/responses", json=request)
    streamed = client.post("/openai/responses", json={**request, "stream": True})
    refused = client.post("/openai/responses", json=request)
    down = client.post("/openai/responses", json=request)

    assert plain.status_code == 200, plain.text
    assert plain.json() == result
    assert streamed.headers["content-type"].startswith("text/event-stream")
    assert '"delta": "streamed"' in streamed.text
    assert (refused.status_code, refused.json()) == (400, {"error": {"message": "bad input"}})
    assert (down.status_code, down.text) == (502, "gateway down")
    assert provider.sent()[0] == request
