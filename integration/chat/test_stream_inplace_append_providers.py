"""Journey: a reply streamed from any other provider or tool source is stored the same either way.

The streaming handler builds a reply by appending each piece to the text it already holds, and
`ENABLE_CHAT_RESPONSE_STREAM_INPLACE_APPEND` only changes how that append is done. Every test here
runs on an instance with the toggle off and one with it on, and asserts the same literal outcome
for both: the stored reply, the `response:completion` deltas a browser tab receives and what the
provider is sent on the next turn of the same chat. Tool call arguments are compared as JSON: the
server re-encodes them with its own codec, compact under orjson (the default since #31616). The
pieces arrive from an Ollama connection (text, thinking and a tool call in lines), a Responses API
connection (its own events are reduced elsewhere, so only the deltas coalesced into fewer socket
events reach the append), a direct connection served by the user's own tab, and tool calls whose
arguments are split across deltas for a workspace Python tool, an OpenAPI tool server, an MCP
server and a knowledge tool whose result carries a source. An Anthropic connection has no path of
its own here: the messages endpoint only converts the handler's stream on the way out.

The direct connection test is red on dev b859124f9 on purpose (open-webui/open-webui#31953): since
24e30d1cb the socket router checks the tab's session token again for every event the tab sends, so
the reply's pieces can overtake each other while those checks run and the stored reply comes back
scrambled or empty. It passes on dev 015dbc861 and on b859124f9 with that check taken back out of
the router.

Discriminates: with the in-place branch of the append breaking only itself, every append-in-place
case failed and every append-copies case passed; with the copying branch breaking only itself the
reverse held.
"""

from __future__ import annotations

import json
import secrets

import pytest

from harness import responses_provider as responses_api
from harness.actors import admin_of
from harness.chat import send_message, wait_for_reply
from harness.direct_connection import answering, chunk_line, direct_model
from harness.listener import json_answer
from harness.mcp_server import TOOL_SERVERS, mcp_connection, serving_mcp
from harness.ollama_provider import OLLAMA_CONFIG, chat_stream, connect_ollama, serve_ollama
from harness.python_tools import python_tool
from harness.second_provider import OPENAI_CONFIG, attach, sse
from harness.socket_client import connected

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

INPLACE_APPEND = "ENABLE_CHAT_RESPONSE_STREAM_INPLACE_APPEND"
OLLAMA_MODEL = "llama3:latest"
SECOND_MODEL = "second-model"
EVERYONE_READS = [{"principal_type": "user", "principal_id": "*", "permission": "read"}]


@pytest.fixture(params=["false", "true"], ids=["append-copies", "append-in-place"])
def streaming(request, instance_with):
    """A scratch instance with the toggle off, then on; every test below runs on both."""
    return instance_with({INPLACE_APPEND: request.param})


def converse(socket, client, model, prompts, **options):
    """Send each prompt in one chat; returns the stored replies and the first reply's deltas."""
    chat, parent, replies, deltas = None, None, [], []
    for prompt in prompts:
        turn = send_message(client, prompt, model=model, chat_id=chat, parent_id=parent, **options)
        chat, parent = turn.chat_id, turn.assistant_message_id
        replies.append(wait_for_reply(client, turn))
        socket.wait_for(chat, "chat:completion", done=True)
        if len(replies) == 1:
            deltas = [
                (event["data"]["type"], event["data"]["delta"])
                for event in socket.events_of(chat)
                if event.get("type") == "response:completion" and "delta" in event["data"]
            ]
    return replies, deltas


def shape(message):
    """The stored output without ids and clocks: (type, text or call) per item."""
    items = []
    for item in message["output"]:
        if item["type"] == "function_call":
            # re-encoded by the server's JSON codec, whose spacing is its own (#31616)
            items.append(("function_call", item["name"], json.loads(item["arguments"])))
        elif item["type"] == "function_call_output":
            items.append(("function_call_output", item["output"][0]["text"]))
        else:
            parts = item.get("content") or item.get("summary") or []
            items.append((item["type"], "".join(part["text"] for part in parts)))
    return items


# --- Ollama: lines of thinking, text and a tool call ------------------------------------


@pytest.fixture
def ollama(streaming, preserve, listener):
    preserve(OLLAMA_CONFIG, on=streaming)
    server = serve_ollama(listener, OLLAMA_MODEL)
    admin = admin_of(streaming)
    with admin.client() as client, connected(admin) as socket:
        connect_ollama(client, listener)
        client.get("/api/models").raise_for_status()
        yield server, client, socket


def test_ollama_thinking_and_text_lines_are_joined(ollama):
    server, client, socket = ollama
    server.queue_chat(
        chat_stream(
            OLLAMA_MODEL,
            {"thinking": "weigh"},
            {"thinking": "ing "},
            {"thinking": "it"},
            {"content": "Hel"},
            {"content": "lo, "},
            {"content": "wor"},
            {"content": "ld"},
        ),
        chat_stream(OLLAMA_MODEL, {"content": "fine"}),
    )

    replies, deltas = converse(socket, client, OLLAMA_MODEL, ["hi", "again"])

    assert replies[0]["content"] == "Hello, world"
    assert shape(replies[0]) == [("reasoning", "weighing it"), ("message", "Hello, world")]
    assert deltas == [
        ("response.reasoning_text.delta", "weigh"),
        ("response.reasoning_text.delta", "ing "),
        ("response.reasoning_text.delta", "it"),
        ("response.output_text.delta", "Hel"),
        ("response.output_text.delta", "lo, "),
        ("response.output_text.delta", "wor"),
        ("response.output_text.delta", "ld"),
    ]
    assert server.chat_requests()[-1]["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello, world", "thinking": "weighing it"},
        {"role": "user", "content": "again"},
    ]


def test_ollama_tool_call_then_a_streamed_answer(ollama):
    server, client, socket = ollama
    call = {"function": {"name": "get_current_timestamp", "arguments": {}}}
    server.queue_chat(
        chat_stream(
            OLLAMA_MODEL, {"thinking": "check "}, {"thinking": "clock"}, {"tool_calls": [call]}
        ),
        chat_stream(
            OLLAMA_MODEL,
            {"thinking": "read "},
            {"thinking": "it"},
            {"content": "It is "},
            {"content": "late."},
        ),
        chat_stream(OLLAMA_MODEL, {"content": "fine"}),
    )

    replies, deltas = converse(socket, client, OLLAMA_MODEL, ["what time is it?", "again"])

    [_, call_item, result_item, _, _] = replies[0]["output"]
    stamp = json.loads(result_item["output"][0]["text"])["current_timestamp"]
    assert shape(replies[0])[:2] == [
        ("reasoning", "check clock"),
        ("function_call", "get_current_timestamp", {}),
    ]
    assert shape(replies[0])[3:] == [("reasoning", "read it"), ("message", "It is late.")]
    assert replies[0]["content"] == "It is late."
    assert deltas == [
        ("response.reasoning_text.delta", "check "),
        ("response.reasoning_text.delta", "clock"),
        ("response.function_call_arguments.delta", "{}"),
        ("response.reasoning_text.delta", "read "),
        ("response.reasoning_text.delta", "it"),
        ("response.output_text.delta", "It is "),
        ("response.output_text.delta", "late."),
    ]
    first, second, third = [request["messages"] for request in server.chat_requests()]
    assert first == [{"role": "user", "content": "what time is it?"}]
    assert second[1:] == [
        {
            "role": "assistant",
            "content": "",
            "thinking": "check clock",
            "tool_calls": [
                {
                    "index": 0,
                    "id": call_item["call_id"],
                    "function": {"name": "get_current_timestamp", "arguments": {}},
                }
            ],
        },
        {
            "role": "tool",
            "content": result_item["output"][0]["text"],
            "tool_call_id": call_item["call_id"],
        },
    ]
    assert stamp > 0
    assert third[-2:] == [
        {"role": "assistant", "content": "It is late.", "thinking": "read it"},
        {"role": "user", "content": "again"},
    ]


# --- Responses API: text, reasoning and function call arguments --------------------------

RESPONSES_MODEL = responses_api.RESPONSES_MODEL

PLAN_ROUTE = '''
class Tools:
    def plan_route(self, origin: str, stops: int) -> str:
        """Plan a route.
        :param origin: Where the route starts
        :param stops: How many stops it makes
        """
        return f"route from {origin} with {stops} stops"
'''


def reasoning_in_pieces(summary: list[str], text: list[str], index: int = 0) -> list[dict]:
    item = {"type": "reasoning", "id": f"rs_{index}", "status": "in_progress", "summary": []}
    where = {"output_index": index, "summary_index": 0}
    part = {"type": "summary_text", "text": ""}
    return [
        {"type": "response.output_item.added", "output_index": index, "item": item},
        {"type": "response.reasoning_summary_part.added", **where, "part": part},
        *({"type": "response.reasoning_summary_text.delta", **where, "delta": d} for d in summary),
        {
            "type": "response.reasoning_summary_part.done",
            **where,
            "part": {**part, "text": "".join(summary)},
        },
        *(
            {"type": "response.reasoning_text.delta", "output_index": index, "delta": d}
            for d in text
        ),
        {
            "type": "response.reasoning_summary_text.done",
            "output_index": index,
            "text": "".join(summary),
        },
    ]


@pytest.fixture
def responses(streaming, preserve, listener):
    preserve(OPENAI_CONFIG, on=streaming)
    admin = admin_of(streaming)
    with admin.client() as client, connected(admin) as socket:
        provider = responses_api.connect_responses(client, listener)
        yield provider, client, socket


def test_responses_deltas_are_coalesced_into_fewer_socket_events(responses):
    provider, client, socket = responses
    provider.answer(
        responses_api.events_stream(
            *reasoning_in_pieces(["a", "b", "c", "d"], ["x", "y", "z"]),
            *responses_api.message("1", "2", "3", "4", "5", "6", "7", index=1),
            responses_api.completed(),
        ),
        responses_api.events_stream(*responses_api.message("ok"), responses_api.completed()),
    )

    replies, deltas = converse(
        socket, client, RESPONSES_MODEL, ["count", "again"], params={"stream_delta_chunk_size": 3}
    )

    [thought, answer] = replies[0]["output"]
    assert [part["text"] for part in thought["summary"]] == ["abcd"]
    assert [part["text"] for part in thought["content"]] == ["xyz"]
    assert answer["content"][0]["text"] == "1234567"
    assert deltas == [
        ("response.reasoning_summary_text.delta", "abc"),
        ("response.reasoning_summary_text.delta", "d"),
        ("response.reasoning_text.delta", "xyz"),
        ("response.output_text.delta", "123"),
        ("response.output_text.delta", "456"),
        ("response.output_text.delta", "7"),
    ]
    assert provider.sent()[-1]["input"][-2] == {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "1234567"}],
    }


def test_responses_function_call_arguments_are_coalesced_for_a_python_tool(responses, streaming):
    provider, client, socket = responses
    arguments = {"origin": "Vienna", "stops": 3}
    provider.answer(
        responses_api.events_stream(
            *reasoning_in_pieces(["pick ", "a tool"], []),
            *responses_api.function_call("plan_route", arguments, call_id="call_9", index=1),
            responses_api.completed(),
        ),
        responses_api.events_stream(
            *responses_api.message("Three ", "stops."), responses_api.completed()
        ),
        responses_api.events_stream(*responses_api.message("ok"), responses_api.completed()),
    )

    with python_tool(admin_of(streaming), PLAN_ROUTE) as tool_id:
        replies, deltas = converse(
            socket,
            client,
            RESPONSES_MODEL,
            ["plan it", "again"],
            tool_ids=[tool_id],
            params={"stream_delta_chunk_size": 2},
        )

    encoded = json.dumps(arguments)
    assert replies[0]["content"] == "Three stops."
    assert shape(replies[0]) == [
        ("reasoning", "pick a tool"),
        ("function_call", "plan_route", arguments),
        ("function_call_output", "route from Vienna with 3 stops"),
        ("message", "Three stops."),
    ]
    assert [d for d in deltas if d[0] == "response.function_call_arguments.delta"] == [
        ("response.function_call_arguments.delta", encoded),
    ]
    second = provider.sent()[1]["input"]
    second[-2]["arguments"] = json.loads(second[-2]["arguments"])
    assert second[-2:] == [
        {
            "type": "function_call",
            "call_id": "call_9",
            "name": "plan_route",
            "arguments": arguments,
        },
        {
            "type": "function_call_output",
            "call_id": "call_9",
            "output": "route from Vienna with 3 stops",
        },
    ]
    assert provider.sent()[2]["input"][-1]["content"][0]["text"] == "again"


# --- direct connection: the user's own tab streams the provider's lines ------------------


def test_a_direct_connections_streamed_pieces_are_joined(streaming):
    admin = admin_of(streaming)
    function = {"name": "get_current_timestamp", "arguments": ""}
    clock = {"index": 0, "id": "call_1", "type": "function", "function": function}
    with answering(admin) as tab, admin.client() as client:
        tab.stream(
            chunk_line({"reasoning_content": "check "}),
            chunk_line({"reasoning_content": "clock"}),
            chunk_line({"tool_calls": [clock]}),
            chunk_line({"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]}),
            chunk_line({"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}),
            chunk_line({}, "tool_calls"),
        )
        tab.stream(
            chunk_line({"content": "It is "}),
            chunk_line({"content": "late."}),
            chunk_line({}, "stop"),
        )
        tab.stream(chunk_line({"content": "fine"}), chunk_line({}, "stop"))
        replies, deltas = converse(
            tab.session,
            client,
            "my-own-model",
            ["what time is it?", "again"],
            model_item=direct_model("my-own-model"),
            session_id=tab.session_id,
        )

    [_, _, result_item, _] = replies[0]["output"]
    result = result_item["output"][0]["text"]
    assert json.loads(result)["current_timestamp"] > 0
    assert replies[0]["content"] == "It is late.", "the streamed reply came back scrambled (#31953)"
    assert shape(replies[0])[:2] == [
        ("reasoning", "check clock"),
        ("function_call", "get_current_timestamp", {}),
    ]
    assert shape(replies[0])[3] == ("message", "It is late.")
    assert deltas == [
        ("response.reasoning_text.delta", "check "),
        ("response.reasoning_text.delta", "clock"),
        ("response.function_call_arguments.delta", ""),
        ("response.function_call_arguments.delta", "{"),
        ("response.function_call_arguments.delta", "}"),
        ("response.output_text.delta", "It is "),
        ("response.output_text.delta", "late."),
    ]
    asked = [request["form_data"]["messages"] for request in tab.requests]
    called = {"name": "get_current_timestamp", "arguments": "{}"}
    tool_round = [
        {"role": "user", "content": "what time is it?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": called}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": result},
    ]
    assert asked[0] == tool_round[:1]
    assert asked[1] == tool_round
    assert asked[2] == [
        *tool_round,
        {"role": "assistant", "content": "It is late."},
        {"role": "user", "content": "again"},
    ]


# --- tool sources: calls whose arguments arrive split across deltas ----------------------

NOTE_TOOL = '''
class Tools:
    def compose_note(self, title: str, lines: int, shout: bool = False) -> str:
        """Compose a note.
        :param title: The note's title
        :param lines: How many lines it has
        :param shout: Whether it is upper case
        """
        text = f"{title}:{lines}"
        return text.upper() if shout else text
'''


CREATED = '{\n  "id": "2",\n  "created": true\n}'


def in_pieces(text: str, size: int = 7) -> list[str]:
    return [text[start : start + size] for start in range(0, len(text), size)]


def tool_call_split(name: str, arguments: dict) -> list[dict]:
    """The deltas of a reply calling `name`, its JSON arguments arriving 7 characters at a time."""
    function = {"name": name, "arguments": ""}
    opened = {"index": 0, "id": "call_1", "type": "function", "function": function}
    pieces = in_pieces(json.dumps(arguments))
    return [
        {"tool_calls": [opened]},
        *({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]} for piece in pieces),
    ]


@pytest.fixture
def tooled(streaming, preserve, listener):
    """(script, client, socket): `script(*replies)` lines up the provider's chat completions."""
    preserve(OPENAI_CONFIG, TOOL_SERVERS, on=streaming)
    admin = admin_of(streaming)
    pending = []
    listener.route("POST", "/v1/chat/completions", lambda _request: pending.pop(0))
    with admin.client() as client, connected(admin) as socket:
        attach(client, listener, SECOND_MODEL)
        yield lambda *replies: pending.extend(replies), client, socket, listener


def tool_round(listener, name: str, arguments: dict, result: str) -> None:
    """The follow-up request carries the assistant's joined tool call and the tool's result."""
    first, second = [
        request.json()["messages"] for request in listener.requests_to("/v1/chat/completions")
    ][:2]
    joined = second[len(first) :]
    for call in joined[0].get("tool_calls", []):
        call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    called = {"name": name, "arguments": arguments}
    assert joined == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": called}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": result},
    ]


def call_and_answer(name: str, arguments: dict) -> list:
    return [
        sse(*tool_call_split(name, arguments), finish_reason="tool_calls"),
        sse({"content": "Do"}, {"content": "ne."}),
    ]


def test_a_python_tool_gets_the_arguments_streamed_in_pieces(tooled, streaming):
    script, client, socket, listener = tooled
    arguments = {"title": "Trip", "lines": 4, "shout": True}
    script(*call_and_answer("compose_note", arguments))

    with python_tool(admin_of(streaming), NOTE_TOOL) as tool_id:
        replies, deltas = converse(socket, client, SECOND_MODEL, ["write it"], tool_ids=[tool_id])

    assert replies[0]["content"] == "Done."
    assert shape(replies[0]) == [
        ("function_call", "compose_note", arguments),
        ("function_call_output", "TRIP:4"),
        ("message", "Done."),
    ]
    assert [delta for kind, delta in deltas if kind.endswith("arguments.delta")] == [
        "",
        *in_pieces(json.dumps(arguments)),
    ]
    tool_round(listener, "compose_note", arguments, "TRIP:4")


def test_an_openapi_tool_gets_its_query_and_body_from_split_arguments(tooled, streaming):
    script, client, socket, listener = tooled
    operation = {
        "operationId": "create_pet",
        "parameters": [{"name": "dry_run", "in": "query", "schema": {"type": "boolean"}}],
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    }
                }
            }
        },
        "responses": {"200": {"description": "ok"}},
    }
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Pets", "version": "1"},
        "paths": {"/pets": {"post": operation}},
    }
    listener.route("GET", "/pets-api/openapi.json", json_answer(spec))
    listener.route("POST", "/pets-api/pets", json_answer({"id": "2", "created": True}))
    connection = {
        "url": f"{listener.base_url}/pets-api",
        "path": "openapi.json",
        "auth_type": "none",
        "key": "",
        "config": {"enable": True},
        "info": {"id": "pets", "name": "Pets"},
    }
    saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
    assert saved.status_code == 200, saved.text
    arguments = {"dry_run": True, "name": "Rex the third", "age": 12}
    script(*call_and_answer("create_pet", arguments))

    replies, deltas = converse(socket, client, SECOND_MODEL, ["add Rex"], tool_ids=["server:pets"])

    [sent] = listener.requests_to("/pets-api/pets")
    assert sent.path.endswith("?dry_run=True")
    assert sent.json() == {"name": "Rex the third", "age": 12}
    assert shape(replies[0]) == [
        ("function_call", "create_pet", arguments),
        ("function_call_output", CREATED),
        ("message", "Done."),
    ]
    tool_round(listener, "create_pet", arguments, CREATED)


def test_an_mcp_tool_gets_the_text_streamed_in_pieces(tooled, streaming):
    script, client, socket, listener = tooled
    server_id = f"echo_{secrets.token_hex(4)}"
    arguments = {"text": "the quick brown fox jumps over the lazy dog"}
    script(*call_and_answer(f"{server_id}_echo", arguments))

    with serving_mcp() as url:
        connection = mcp_connection(url, server_id, EVERYONE_READS)
        saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
        assert saved.status_code == 200, saved.text
        replies, deltas = converse(
            socket, client, SECOND_MODEL, ["say it"], tool_ids=[f"server:mcp:{server_id}"]
        )

    assert shape(replies[0]) == [
        ("function_call", f"{server_id}_echo", arguments),
        ("function_call_output", arguments["text"]),
        ("message", "Done."),
    ]
    tool_round(listener, f"{server_id}_echo", arguments, arguments["text"])


def test_a_knowledge_tool_result_keeps_its_source(tooled, streaming):
    script, client, socket, listener = tooled
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": ("notes.txt", b"The meeting is at noon.", "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    arguments = {"file_id": uploaded.json()["id"]}
    script(*call_and_answer("view_knowledge_file", arguments))

    replies, deltas = converse(socket, client, SECOND_MODEL, ["when is the meeting?"])

    [call_item, result_item, answer_item] = replies[0]["output"]
    result = json.loads(result_item["output"][0]["text"])
    assert (result["filename"], result["content"]) == ("notes.txt", "The meeting is at noon.")
    assert (json.loads(call_item["arguments"]), answer_item["content"][0]["text"]) == (
        arguments,
        "Done.",
    )
    assert replies[0]["sources"] == [
        {
            "source": {"id": arguments["file_id"], "name": "notes.txt", "type": "file"},
            "document": ["The meeting is at noon."],
            "metadata": [
                {"file_id": arguments["file_id"], "name": "notes.txt", "source": "notes.txt"}
            ],
        }
    ]
    tool_round(listener, "view_knowledge_file", arguments, result_item["output"][0]["text"])
    [request, follow_up] = [
        r.json()["messages"] for r in listener.requests_to("/v1/chat/completions")
    ]
    assert request == [{"role": "user", "content": "when is the meeting?"}]
    file_id = arguments["file_id"]
    tag = f'<source id="1" name="notes.txt" resource-type="file" resource-id="{file_id}">'
    assert tag in follow_up[0]["content"]
