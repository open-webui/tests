"""Journey: chatting with a model served by an Ollama connection, through the normal chat endpoint.

A chat with an Ollama model goes out as an Ollama `/api/chat` request: the messages, images,
reasoning, tools and model settings are translated into Ollama's shape, and Ollama's NDJSON
reply (thinking, text, tool calls, its own token counters) is translated back into the reply the
user sees. Each test reads both sides: what the Ollama stand-in was sent and what came back.

Discriminates: in a backend copy, dropping the images of a converted message fails the history
test, emitting a streamed chunk without its thinking fails the streamed-reply test, sending a
tool call's arguments back as a string fails the tool test, leaving `max_tokens` unrenamed in
the options fails the preset and chat-settings tests, and a non-streamed reply without Ollama's
counters fails the API client test.
"""

from __future__ import annotations

import uuid

import pytest

from harness.chat import ask
from harness.chat_history import seed_chat
from harness.listener import json_answer
from harness.ollama_provider import (
    OLLAMA_CONFIG,
    chat_line,
    chat_stream,
    connect_ollama,
    serve_ollama,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

OLLAMA_MODEL = "llama3:latest"
COUNTERS = {"prompt_eval_count": 12, "eval_count": 5, "eval_duration": 1_000_000_000}
SYSTEM_PROMPT = "Be brief."


@pytest.fixture
def ollama(admin, preserve, listener):
    """The Ollama stand-in as the only Ollama connection, and a client of an admin account."""
    preserve(OLLAMA_CONFIG)
    server = serve_ollama(listener, OLLAMA_MODEL)
    with admin.client() as client:
        connect_ollama(client, listener)
        client.get("/api/models").raise_for_status()
        yield server, client


def _output_text(message: dict, kind: str) -> list[str]:
    return [
        part["text"]
        for item in message["output"]
        if item["type"] == kind
        for part in item.get("content") or []
    ]


def _without_tools(sent: dict) -> dict:
    return {key: value for key, value in sent.items() if key != "tools"}


def test_a_streamed_reply_keeps_its_thinking_text_and_counters(ollama):
    server, client = ollama
    server.queue_chat(
        chat_stream(
            OLLAMA_MODEL,
            {"thinking": "weighing it"},
            {"content": "Hel"},
            {"content": "lo"},
            **COUNTERS,
        )
    )

    _, message = ask(client, "hi", model=OLLAMA_MODEL)

    assert message["content"] == "Hello"
    assert _output_text(message, "reasoning") == ["weighing it"]
    usage = message["usage"]
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (12, 5, 17)
    assert usage["response_token/s"] == 5.0
    [sent] = server.chat_requests()
    assert _without_tools(sent) == {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    assert "get_current_timestamp" in {tool["function"]["name"] for tool in sent["tools"]}


def test_the_history_reaches_ollama_with_images_and_thinking(ollama):
    server, client = ollama
    image = {"type": "image", "id": "i1", "url": "data:image/png;base64,AAAA"}
    reasoning = {
        "type": "reasoning",
        "id": "r1",
        "status": "completed",
        "content": [{"type": "output_text", "text": "looked closely"}],
    }
    answer = {
        "type": "message",
        "id": "m1",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "a cat"}],
    }
    history = [
        {"role": "user", "content": "what is this?", "files": [image]},
        {"role": "assistant", "content": "a cat", "output": [reasoning, answer]},
    ]
    chat_id, last_id = seed_chat(client, history, model=OLLAMA_MODEL)
    server.queue_chat(chat_stream(OLLAMA_MODEL, {"content": "still a cat"}, **COUNTERS))

    ask(
        client,
        "and now?",
        model=OLLAMA_MODEL,
        chat_id=chat_id,
        parent_id=last_id,
        history=[{"role": "system", "content": SYSTEM_PROMPT}],
    )

    assert server.chat_requests()[-1]["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "what is this?", "images": ["AAAA"]},
        {"role": "assistant", "content": "a cat", "thinking": "looked closely"},
        {"role": "user", "content": "and now?"},
    ]


def test_a_tool_call_runs_and_its_result_goes_back_in_ollamas_shape(ollama):
    server, client = ollama
    call = {"function": {"name": "get_current_timestamp", "arguments": {}}}
    server.queue_chat(
        chat_stream(OLLAMA_MODEL, {"tool_calls": [call]}),
        chat_stream(OLLAMA_MODEL, {"content": "It is late."}, **COUNTERS),
    )

    _, message = ask(client, "what time is it?", model=OLLAMA_MODEL)

    assert message["content"] == "It is late."
    [stored_call] = [item for item in message["output"] if item["type"] == "function_call"]
    assert stored_call["name"] == "get_current_timestamp"
    first, follow_up = server.chat_requests()
    assert first["stream"] is True
    assistant, tool = follow_up["messages"][-2:]
    assert assistant["content"] == ""
    assert [entry["function"] for entry in assistant["tool_calls"]] == [
        {"name": "get_current_timestamp", "arguments": {}}
    ]
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == assistant["tool_calls"][0]["id"] == stored_call["call_id"]
    assert "current_timestamp" in tool["content"]


@pytest.fixture
def ollama_preset(ollama, admin):
    """A workspace model over the Ollama model, saved with Ollama settings and a system prompt."""
    server, client = ollama
    preset_id = f"ollama-preset-{uuid.uuid4().hex[:8]}"
    params = {
        "system": SYSTEM_PROMPT,
        "temperature": 0.3,
        "num_ctx": 4096,
        "max_tokens": 64,
        "stop": ["###"],
        "keep_alive": "5m",
        "think": True,
        "format": '{"type": "object"}',
        "custom_params": {"top_k": "40"},
    }
    form = {
        "id": preset_id,
        "name": preset_id,
        "base_model_id": OLLAMA_MODEL,
        "meta": {},
        "params": params,
    }
    created = client.post("/api/v1/models/create", json=form)
    assert created.status_code == 200, created.text
    client.get("/api/models").raise_for_status()
    yield server, client, preset_id
    client.post("/api/v1/models/model/delete", json={"id": preset_id})


def test_a_presets_settings_become_ollama_options(ollama_preset):
    server, client, preset_id = ollama_preset
    server.queue_chat(chat_stream(OLLAMA_MODEL, {"content": "{}"}, **COUNTERS))

    ask(client, "hi", model=preset_id)

    sent = server.chat_requests()[-1]
    assert sent["model"] == OLLAMA_MODEL
    assert sent["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert (sent["keep_alive"], sent["think"], sent["format"]) == ("5m", True, {"type": "object"})
    options = sent["options"]
    assert options["temperature"] == 0.3
    assert options["num_ctx"] == 4096
    assert options["num_predict"] == 64
    assert options["top_k"] == 40
    assert options["stop"] == ["###"]
    assert "max_tokens" not in options


def test_chat_settings_move_ollamas_root_fields_out_of_the_options(ollama):
    server, client = ollama
    server.queue_chat(chat_stream(OLLAMA_MODEL, {"content": "ok"}, **COUNTERS))
    params = {"temperature": 0.1, "max_tokens": 20, "keep_alive": "1h", "format": "json"}

    ask(client, "hi", model=OLLAMA_MODEL, params=params)

    sent = server.chat_requests()[-1]
    assert (sent["keep_alive"], sent["format"]) == ("1h", "json")
    assert sent["options"] == {"temperature": 0.1, "num_predict": 20}


def test_an_api_client_gets_a_non_streamed_reply_in_openai_shape(ollama):
    server, client = ollama
    tool_call = {"function": {"name": "lookup", "arguments": {"city": "Graz"}}}
    finished = chat_line(
        OLLAMA_MODEL,
        {"content": "", "thinking": "need the weather", "tool_calls": [tool_call]},
        done_reason="stop",
        **COUNTERS,
    )
    server.queue_chat(json_answer(finished))
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": schema}}]

    answered = client.post(
        "/api/chat/completions",
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "weather?"}],
            "stream": False,
            "stop": ["END"],
            "tools": tools,
            "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
        },
    )

    assert answered.status_code == 200, answered.text
    body = answered.json()
    message = body["choices"][0]["message"]
    assert message["reasoning_content"] == "need the weather"
    [call] = message["tool_calls"]
    assert call["function"] == {"name": "lookup", "arguments": '{"city": "Graz"}'}
    assert (body["usage"]["prompt_tokens"], body["usage"]["completion_tokens"]) == (12, 5)
    sent = server.chat_requests()[-1]
    assert sent["stream"] is False
    assert sent["options"]["stop"] == ["END"]
    assert sent["format"] == schema
    assert sent["tools"] == tools


@pytest.mark.parametrize(
    "status,body,shown",
    [
        pytest.param(404, {"error": "model 'x' not found"}, "model 'x' not found", id="json"),
        pytest.param(500, "upstream exploded", None, id="not-json"),
    ],
)
def test_an_ollama_error_is_stored_on_the_reply(ollama, status, body, shown):
    server, client = ollama
    answer = (
        json_answer(body, status=status)
        if isinstance(body, dict)
        else (status, {"Content-Type": "text/plain"}, body.encode())
    )
    server.queue_chat(answer)

    _, message = ask(client, "hi", model=OLLAMA_MODEL)

    error = message.get("error")
    assert error, f"the failed Ollama call left no error on the reply: {message}"
    assert (shown or "") in str(error)
    assert message["content"] == ""
