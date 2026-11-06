"""Journey: provider wire formats and compatible APIs under `ENABLE_ORJSON` on and off.

The switch moves the JSON Open WebUI writes to Ollama and OpenAI connections, the conversion of
Ollama's NDJSON into OpenAI event streams (tool call arguments arrive as objects and are dumped
to strings), the Responses API events, the Ollama passthrough routes, the embeddings routes and
the re-serialization of every chunk once a stream filter is installed. Real providers write raw
UTF-8 on one line; the scripted one escapes it, so the connections here answer with raw bytes.
Two instances share one database and one scripted provider, one per value, and each test runs on
both with the same expected values: the stored reply, the tool call arguments, what the provider
was sent and every frame a client splits with `str.splitlines()`.

Discriminates: passes on dev 176d31d1d. In backend copies of dev 176d31d1d, the orjson codec
writing mojibake, orjson request parsing that mangles non-ASCII and orjson response rendering that
replaces non-ASCII between them turn the orjson case of every test red, and the stdlib codec
writing mojibake turns the stdlib cases red.
"""

from __future__ import annotations

import json

import pytest

from harness import responses_provider as responses_api
from harness.actors import admin_of
from harness.chat import ask
from harness.json_codecs import ALL_MIXED, CODECS, MIXED_TEXT, codec_pair
from harness.listener import listening, text_answer
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, serve_ollama
from harness.plugins import installed_function
from harness.raw_provider import RAW_MODEL_ID, chunk, connect
from harness.second_provider import OPENAI_CONFIG
from harness.tool_calls import without_hidden_calls
from harness.upstream import conversation

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

OLLAMA_MODEL = "llama3:latest"
PIECES = list(MIXED_TEXT.values())
FULL_TEXT = "".join(PIECES)
# the stored reply is trimmed at its ends
STORED_TEXT = FULL_TEXT.strip()
ARGUMENTS = {
    "city": MIXED_TEXT["german"],
    "note": MIXED_TEXT["chinese"],
    "tags": [MIXED_TEXT["emoji"]],
}
LOOKUP = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
PASS_THROUGH_STREAM_FILTER = """
class Filter:
    def stream(self, event):
        return event
"""


@pytest.fixture(scope="module")
def pair(instance_with):
    return codec_pair(instance_with, {"ENABLE_OPENAI_API_PASSTHROUGH": "true"})


def _raw(payload: dict) -> str:
    """One JSON line as a real provider writes it: raw UTF-8, line separators escaped as Go does."""
    text = json.dumps(payload, ensure_ascii=False)
    return text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _raw_ndjson(*lines: dict):
    body = "".join(f"{_raw(line)}\n" for line in lines)
    return 200, {"Content-Type": "application/x-ndjson"}, body.encode()


def _ollama_line(message: dict, **final) -> dict:
    line = {"model": OLLAMA_MODEL, "message": {"role": "assistant", **message}, "done": bool(final)}
    return {**line, **final}


def _ollama_reply(*messages: dict):
    finished = _ollama_line({"content": ""}, done_reason="stop", eval_count=5, prompt_eval_count=3)
    return _raw_ndjson(*(_ollama_line(message) for message in messages), finished)


@pytest.fixture
def ollama(pair, request, preserve):
    """The Ollama stand-in as the only Ollama connection of the instance under test."""
    instance = pair[request.param]
    preserve(OLLAMA_CONFIG, on=instance)
    with listening() as listener:
        server = serve_ollama(listener, OLLAMA_MODEL)
        with admin_of(instance).client() as client:
            connect_ollama(client, listener)
            client.get("/api/models", params={"refresh": "true"}).raise_for_status()
            yield server, client


def _parsed_frames(text: str) -> list[dict]:
    """The event stream split the way `str.splitlines()` splits; every frame must parse."""
    frames = []
    for line in text.splitlines():
        if not line.strip() or line.strip() == "data: [DONE]":
            continue
        assert line.startswith("data:"), f"a frame was cut in the middle: {line!r}"
        frames.append(json.loads(line.removeprefix("data:")))
    return frames


def _content(frames: list[dict]) -> str:
    return "".join(
        choice["delta"].get("content") or "" for frame in frames for choice in frame["choices"]
    )


def _streamed_calls(frames: list[dict]) -> list[dict]:
    calls: dict[int, dict] = {}
    for frame in frames:
        for choice in frame["choices"]:
            for call in choice["delta"].get("tool_calls") or []:
                slot = calls.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                function = call.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
    return list(calls.values())


def _stored_calls(message: dict) -> list[dict]:
    return [
        item for item in without_hidden_calls(message["output"]) if item["type"] == "function_call"
    ]


def _user_texts(sent: dict) -> list[str]:
    return [entry["content"] for entry in sent["messages"] if entry["role"] == "user"]


# --- Ollama through the chat endpoint ---------------------------------------------------------


@pytest.mark.parametrize("ollama", CODECS, indirect=True)
def test_an_ollama_reply_with_mixed_text_is_stored_and_the_prompt_reaches_ollama_intact(ollama):
    server, client = ollama
    server.queue_chat(_ollama_reply(*({"content": piece} for piece in PIECES)))

    _, message = ask(client, ALL_MIXED, model=OLLAMA_MODEL)

    assert message["content"] == STORED_TEXT
    [sent] = server.chat_requests()
    assert _user_texts(sent) == [ALL_MIXED]


@pytest.mark.parametrize("ollama", CODECS, indirect=True)
def test_an_ollama_tool_call_with_non_ascii_arguments_is_stored_and_replayed(ollama):
    server, client = ollama
    call = {"function": {"name": "lookup", "arguments": ARGUMENTS}}
    server.queue_chat(
        _ollama_reply({"tool_calls": [call]}), _ollama_reply({"content": MIXED_TEXT["japanese"]})
    )

    _, message = ask(client, "find it", model=OLLAMA_MODEL)

    [stored] = _stored_calls(message)
    assert json.loads(stored["arguments"]) == ARGUMENTS
    assert message["content"] == MIXED_TEXT["japanese"]
    _, follow_up = server.chat_requests()
    replayed = [entry for entry in conversation(follow_up) if entry.get("tool_calls")]
    assert [entry["function"] for entry in replayed[0]["tool_calls"]] == [
        {"name": "lookup", "arguments": ARGUMENTS}
    ]


@pytest.mark.parametrize("ollama", CODECS, indirect=True)
def test_a_streamed_api_reply_from_ollama_splits_into_parseable_frames(ollama):
    server, client = ollama
    server.queue_chat(_ollama_reply(*({"content": piece} for piece in PIECES)))

    streamed = client.post(
        "/api/chat/completions",
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": ALL_MIXED}],
            "stream": True,
        },
    )

    assert streamed.status_code == 200, streamed.text
    assert _content(_parsed_frames(streamed.text)) == FULL_TEXT
    assert _user_texts(server.chat_requests()[-1]) == [ALL_MIXED]


@pytest.mark.parametrize("ollama", CODECS, indirect=True)
def test_a_streamed_api_tool_call_from_ollama_carries_its_arguments_as_a_string(ollama):
    server, client = ollama
    call = {"function": {"name": "lookup", "arguments": ARGUMENTS}}
    server.queue_chat(_ollama_reply({"tool_calls": [call]}))

    streamed = client.post(
        "/api/chat/completions",
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "find it"}],
            "stream": True,
            "tools": [LOOKUP],
        },
    )

    assert streamed.status_code == 200, streamed.text
    [seen] = _streamed_calls(_parsed_frames(streamed.text))
    assert seen["name"] == "lookup"
    assert isinstance(seen["arguments"], str)
    assert json.loads(seen["arguments"]) == ARGUMENTS


@pytest.mark.parametrize("ollama", CODECS, indirect=True)
def test_a_non_streamed_api_reply_from_ollama_keeps_text_and_arguments(ollama):
    server, client = ollama
    call = {"function": {"name": "lookup", "arguments": ARGUMENTS}}
    reply = _ollama_line(
        {"content": ALL_MIXED, "tool_calls": [call]},
        done_reason="stop",
        prompt_eval_count=3,
        eval_count=5,
        eval_duration=1_000_000_000,
    )
    server.queue_chat((200, {"Content-Type": "application/json"}, _raw(reply).encode()))

    answered = client.post(
        "/api/chat/completions",
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": ALL_MIXED}],
            "stream": False,
            "tools": [LOOKUP],
        },
    )

    assert answered.status_code == 200, answered.text
    message = answered.json()["choices"][0]["message"]
    assert message["content"] == ALL_MIXED
    [seen] = message["tool_calls"]
    assert json.loads(seen["function"]["arguments"]) == ARGUMENTS
    assert _user_texts(server.chat_requests()[-1]) == [ALL_MIXED]


# --- Ollama routes a client calls directly ----------------------------------------------------


def _ndjson_lines(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@pytest.mark.parametrize("ollama", CODECS, indirect=True)
def test_the_ollama_chat_route_passes_ndjson_lines_that_parse(ollama):
    server, client = ollama
    server.queue_chat(_ollama_reply(*({"content": piece} for piece in PIECES)))

    streamed = client.post(
        "/ollama/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": ALL_MIXED}],
            "stream": True,
        },
    )

    assert streamed.status_code == 200, streamed.text
    lines = _ndjson_lines(streamed.text)
    assert "".join(line["message"]["content"] for line in lines) == FULL_TEXT
    assert _user_texts(server.chat_requests()[-1]) == [ALL_MIXED]


@pytest.mark.parametrize("ollama", CODECS, indirect=True)
def test_the_ollama_generate_route_passes_ndjson_lines_that_parse(ollama):
    server, client = ollama
    pieces = [{"model": OLLAMA_MODEL, "response": piece, "done": False} for piece in PIECES]
    finished = {"model": OLLAMA_MODEL, "response": "", "done": True, "done_reason": "stop"}
    server.listener.route("POST", "/api/generate", _raw_ndjson(*pieces, finished))

    streamed = client.post(
        "/ollama/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": ALL_MIXED, "stream": True},
    )

    assert streamed.status_code == 200, streamed.text
    assert "".join(line["response"] for line in _ndjson_lines(streamed.text)) == FULL_TEXT
    assert server.sent("/api/generate")[-1]["prompt"] == ALL_MIXED


@pytest.mark.parametrize("ollama", CODECS, indirect=True)
def test_the_ollama_generate_route_answers_a_non_streamed_call_with_the_same_text(ollama):
    server, client = ollama
    reply = {"model": OLLAMA_MODEL, "response": ALL_MIXED, "done": True, "done_reason": "stop"}
    server.listener.route(
        "POST", "/api/generate", (200, {"Content-Type": "application/json"}, _raw(reply).encode())
    )

    answered = client.post(
        "/ollama/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": ALL_MIXED, "stream": False},
    )

    assert answered.status_code == 200, answered.text
    assert answered.json()["response"] == ALL_MIXED
    assert server.sent("/api/generate")[-1]["prompt"] == ALL_MIXED


@pytest.mark.parametrize("ollama", CODECS, indirect=True)
def test_the_ollama_show_route_returns_a_modelfile_with_mixed_text(ollama):
    server, client = ollama
    info = {"modelfile": f"FROM {OLLAMA_MODEL}\n# {ALL_MIXED}\n", "details": {"family": "llama"}}
    server.listener.route(
        "POST", "/api/show", (200, {"Content-Type": "application/json"}, _raw(info).encode())
    )

    shown = client.post("/ollama/api/show", json={"model": OLLAMA_MODEL})

    assert shown.status_code == 200, shown.text
    assert shown.json()["modelfile"] == info["modelfile"]


@pytest.mark.parametrize("ollama", CODECS, indirect=True)
def test_the_ollama_embed_route_sends_the_mixed_input_unchanged(ollama):
    server, client = ollama
    inputs = list(MIXED_TEXT.values())

    embedded = client.post("/ollama/api/embed", json={"model": OLLAMA_MODEL, "input": inputs})

    assert embedded.status_code == 200, embedded.text
    assert len(embedded.json()["embeddings"]) == len(inputs)
    assert server.sent("/api/embed")[-1]["input"] == inputs


# --- Responses API ----------------------------------------------------------------------------


def _raw_events(*events: dict):
    body = "".join(f"data: {_raw(event)}\n\n" for event in events)
    return text_answer(body, "text/event-stream")


@pytest.fixture
def responses(pair, request, preserve):
    instance = pair[request.param]
    preserve(OPENAI_CONFIG, on=instance)
    with listening() as listener, admin_of(instance).client() as client:
        provider = responses_api.connect_responses(client, listener)
        yield provider, client


@pytest.mark.parametrize("responses", CODECS, indirect=True)
def test_a_responses_reply_with_mixed_text_and_a_function_call_is_stored_equal(responses):
    provider, client = responses
    provider.answer(
        _raw_events(
            *responses_api.message(*PIECES),
            *responses_api.function_call("lookup", ARGUMENTS, index=1),
            responses_api.completed(),
        ),
        _raw_events(*responses_api.message(MIXED_TEXT["japanese"]), responses_api.completed()),
    )

    _, message = ask(client, ALL_MIXED, model=responses_api.RESPONSES_MODEL)

    [stored] = _stored_calls(message)
    assert (stored["name"], json.loads(stored["arguments"])) == ("lookup", ARGUMENTS)
    [first_message] = [
        item for item in message["output"] if item["type"] == "message" and item["content"]
    ][:1]
    assert first_message["content"][0]["text"] == FULL_TEXT
    # the reply keeps the text from before the tool call and after it
    assert message["content"].startswith(FULL_TEXT)
    assert message["content"].endswith(MIXED_TEXT["japanese"])
    first, follow_up = provider.sent()
    assert first["input"][0]["content"] == [{"type": "input_text", "text": ALL_MIXED}]
    replayed = [item for item in follow_up["input"] if item.get("name") == "lookup"]
    assert json.loads(replayed[0]["arguments"]) == ARGUMENTS


@pytest.mark.parametrize("responses", CODECS, indirect=True)
def test_a_streamed_responses_reply_reaches_an_api_client_as_parseable_frames(responses):
    provider, client = responses
    provider.answer(_raw_events(*responses_api.message(*PIECES), responses_api.completed()))

    streamed = client.post(
        "/api/chat/completions",
        json={
            "model": responses_api.RESPONSES_MODEL,
            "messages": [{"role": "user", "content": ALL_MIXED}],
            "stream": True,
        },
    )

    assert streamed.status_code == 200, streamed.text
    frames = _parsed_frames(streamed.text)
    # API clients get Chat Completions chunks since dev e789513ea (#33389)
    assert {frame.get("object") for frame in frames} == {"chat.completion.chunk"}, frames
    assert _content(frames) == FULL_TEXT
    assert provider.sent()[-1]["input"][0]["content"] == [{"type": "input_text", "text": ALL_MIXED}]


# --- a raw OpenAI-shaped provider -------------------------------------------------------------


@pytest.fixture
def raw_provider(pair, request, preserve):
    instance = pair[request.param]
    preserve(OPENAI_CONFIG, on=instance)
    admin = admin_of(instance)
    with listening() as listener:
        provider = connect(admin, listener)
        yield provider, admin


def _raw_sse(pieces: list[str]) -> bytes:
    deltas = [{"role": "assistant", "content": ""}, *({"content": piece} for piece in pieces)]
    frames = [f"data: {_raw(chunk(delta))}\n\n" for delta in deltas]
    frames.append(f"data: {_raw(chunk({}, 'stop'))}\n\ndata: [DONE]\n\n")
    return "".join(frames).encode()


def _raw_request() -> dict:
    return {
        "model": RAW_MODEL_ID,
        "messages": [{"role": "user", "content": ALL_MIXED}],
        "stream": True,
    }


@pytest.mark.parametrize("raw_provider", CODECS, indirect=True)
def test_a_raw_utf8_provider_stream_through_a_filter_splits_into_parseable_frames(raw_provider):
    provider, admin = raw_provider
    provider.stream(_raw_sse(PIECES))

    with installed_function(admin, PASS_THROUGH_STREAM_FILTER, is_global=True):
        with admin.client() as client:
            streamed = client.post("/api/chat/completions", json=_raw_request())

    assert streamed.status_code == 200, streamed.text
    assert _content(_parsed_frames(streamed.text)) == FULL_TEXT


@pytest.mark.parametrize("raw_provider", CODECS, indirect=True)
def test_a_raw_utf8_provider_reply_is_stored_the_same_through_the_web_client(raw_provider):
    provider, admin = raw_provider
    provider.stream(_raw_sse(PIECES))

    with admin.client() as client:
        _, message = ask(client, ALL_MIXED, model=RAW_MODEL_ID)

    assert message["content"] == STORED_TEXT
    [sent] = [
        json.loads(request.body)
        for request in provider.listener.requests_to("/v1/chat/completions")
    ]
    assert _user_texts(sent) == [ALL_MIXED]


@pytest.mark.parametrize("raw_provider", CODECS, indirect=True)
def test_a_raw_utf8_provider_json_reply_is_relayed_with_the_same_text(raw_provider):
    provider, admin = raw_provider
    message = {"role": "assistant", "content": ALL_MIXED}
    provider.listener.route(
        "POST",
        "/v1/chat/completions",
        (
            200,
            {"Content-Type": "application/json"},
            _raw(
                {"object": "chat.completion", "choices": [{"index": 0, "message": message}]}
            ).encode(),
        ),
    )

    with admin.client() as client:
        answered = client.post("/api/chat/completions", json={**_raw_request(), "stream": False})

    assert answered.status_code == 200, answered.text
    assert answered.json()["choices"][0]["message"]["content"] == ALL_MIXED
