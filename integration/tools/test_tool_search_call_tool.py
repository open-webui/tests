"""Journey: deferred tools run through `call_tool` while the offered tools never change.

With Tool Search on and tools large enough to defer, the model is offered `search_tools` and
`call_tool(tool_name, params)` alone; `params` is a JSON-encoded object string, or since dev
d87837372 a plain object. Open WebUI unwraps the dispatcher into the target tool before approvals
and special handling, so the target keeps its own behaviour (approval prompt, `ask_user` pause,
access, MCP and builtin tools), and stores the model's original call so later turns replay the
dispatch, on Chat Completions and the Responses API; since dev 7f3e0d018 the replayed arguments
are normalized JSON, the same on every later turn. A malformed dispatch comes back as a failed
result starting `Error: ` and the turn goes on. A tool already named `search_tools` or
`call_tool`, a filter's included, turns the dispatcher off. What a dispatched tool leaves on the
reply matches a direct call with Tool Search off: the sources of the file and knowledge tools,
`fetch_url` and a tool's citation event, a tool's files event, an HTML embed, and pictures from
`generate_image` and `edit_image` stored as files. The source list the citations template adds
to a sourced round's tool result (since dev 8e2ba2937) is the same, and the next turn replays
`call_tool`. Added with dev 7b47c3b22.

The admin settings decide which tools are held back, where `search_tools` finds them. A tool is
held back only when its definition, as compact JSON, is longer than the Deferral Threshold in
characters; one at the threshold is sent in full. An Always Loaded pattern such as `github_*`
keeps the tools it matches in full, and with Defer Built-in Tools off the built-in tools stay in
full too. Since dev 768a83ae7 the `<available_tools>` block holds fixed search instructions and
names no held-back tool, so it stays the same for any tool set. `search_tools` searches only the
held-back tools and returns their full definitions, for an exact name or a keyword. With Tool
Search off, or no tool over the threshold, the model gets every tool, no dispatch tools and no
instructions. The admin chat settings store all four settings, with a negative threshold saved
as 0 and blank patterns dropped.

Approvals treat a dispatched tool as the tool itself: under every approval setting (Ask, Full
access, no choice, the admin switch off, a model preset that asks) it is asked exactly when a
direct call is. In a round mixing dispatched, direct and refused calls each call waits for its own
approval in order and only the allowed ones run; an allowed call can lead to a second round that
asks again; allowing a tool once allows no other call of it, dispatched or direct. Continue runs
nothing that is still pending, Regenerate asks anew, and a temporary chat asks over the socket.

Discriminates: in backend copies of dev 6276e81ce, unwrapping only at execution (not in the stream
handler) fails the run, replay, Responses, approval, ask_user, builtin and MCP tests; no unwrap at
all also fails every refused-dispatch case, the access, non-streamed resume and Ollama tests;
replaying the resolved target in place of the model's call fails the replay, Responses, approval,
ask_user and Ollama tests; leaving filter tools out of the name check fails the filter test; and
dropping the stored-item restore in the OpenAI router fails the relayed Responses test. Skipping
citation sources for dispatched calls fails the file tool, cited round and page tests; no unwrap
at all also fails the citation event, embed and image tests.


dropping the stored-item restore in the OpenAI router fails the relayed Responses test.
In backend copies of dev 81f5b8adf, skipping the Ask pause for a round holding a dispatched call
fails the ask and model-asks parity cases and every approval test of the module; skipping the
temporary chat's confirmation for a dispatched call fails the temporary chat test; marking every
pending call approved on one approval fails the mixed round and allow-once tests; letting a
Continue run pending calls fails the pending Continue test; and resuming with full access after
an approval fails the mixed round, second round and allow-once tests.

In backend copies of dev 948432b43, deferring a tool at the threshold fails the threshold test;
ignoring Always Loaded fails the pattern and search tests; deferring built-in tools regardless
fails the built-in test (the list those mutations also failed went with 768a83ae7); searching
every tool or returning only names and descriptions fails the search test; ignoring the Tool
Search switch fails the threshold, built-in, search and off tests; adding the dispatch tools with
nothing held back fails the nothing-over-the-threshold case; and saving the threshold and
patterns as sent fails the settings test.

Since dev 003299641 one `call_tool(calls=[{tool_name, params}, ...])` carries several tool calls.
Each runs as a call of its own, shown as its own entry with its own approval and result, and one
failing member stops none of the others. The provider gets the batch back as the one call it
sent, under its id, with one combined result listing each member's tool, status and output, on
Chat Completions, the Responses API and Ollama, on later turns too, and when a client relays the
stored reply. A batch an approval pause left half done is never sent back with a call lacking its
result. A malformed batch, like any malformed dispatch, reaches the model as a failed result and
the model keeps its tools to try again; on f659c0d12 a malformed batch or arguments that are not
a JSON object end tool use for the reply instead (the refused-dispatch cases not-an-object,
arguments-malformed and batch-*, red there; the first two pass on b3b07a658, before batches).

Since dev 0f3bd0a7e the admin picks how `search_tools` matches: BM25 by words, the Documents
embedding model by meaning (every held-back tool ranked, best first) or a decisions model asked
about every held-back tool, which keeps the tools at or over its relevance threshold, best first,
with the admin's prompt or the default one. Every engine keeps to the result count asked for and
finds an exact tool name without asking a model. A decisions or embedding model that fails reaches
the model as a search error. Saving a decisions setup in use with a bad address, no model, a
threshold outside 0 to 1 or an unknown engine is refused and nothing is saved; a setup not in use is
saved as typed; a save from an older client without the engine settings keeps the saved ones; the
settings also carry the default decisions prompt and the embedding model's name, and the environment
sets the engine and the decisions setup of a new instance.

Discriminates (dev f659c0d12 backend copies): replaying a batch as its members fails the batch
replay, Responses, approval pause, relayed batch and Ollama batch tests; relaying a half-done batch
fails the relayed half-done case (the approval pause test stays green there, as the chat history
drops the unpaired call on its own); searching with BM25 for every engine fails the embedding,
embedding failure and decisions tests; ignoring the engine's environment variable fails the
environment test; ignoring the threshold or sending a blank prompt in place of the default fails the
decisions test; no validation fails every bad setup case; validating a setup not in use fails the
saved-as-typed test; resetting omitted engine settings fails the older client test; BM25 returning
every tool fails the BM25 test.

Since 93ba628f3 (#32680, issue #32678) a picture the model draws after the user answered its
`ask_user` question or approved or rejected a tool call is stored on the reply. Resolving the
paused call saved the reply with empty fields that broke adding a file to it later in the turn,
so the image was drawn but never shown and the model was told the drawing failed. On dev
f36490bf2 with 93ba628f3 reverted in a backend copy every case of the paused-turn image test
fails (no image on the reply, the model told the operand error of the issue).
"""

from __future__ import annotations

import base64
import contextlib
import json
import re
import time
import uuid
from typing import Iterator

import httpx
import pytest

from harness import responses_provider as responses_api
from harness import upstream as reply
from harness.actors import Actor, admin_of, create_user
from harness.channel_chat import serve_openai_images
from harness.chat import ChatTurn, ask, send_message, wait_for_reply
from harness.image_engines import IMAGES_CONFIG, PNG_BASE64, save_image_settings
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.listener import Answer, ReceivedRequest, json_answer, text_answer
from harness.mcp_server import TOOL_SERVERS, mcp_connection, serving_mcp
from harness.ollama_provider import OLLAMA_CONFIG, chat_stream, connect_ollama, serve_ollama
from harness.plugins import installed_function
from harness.python_tools import EVERYONE_READS, python_tool
from harness.second_provider import OPENAI_CONFIG
from harness.socket_client import connected
from harness.terminal_server import read_grant
from harness.tool_calls import without_hidden_calls
from harness.upstream import MOCK_MODEL_ID, Reply
from harness.web_retrieval import (
    LOCAL_WEB_FETCH,
    RETRIEVAL_CONFIG,
    save_web_settings,
    serve_search_results,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
DISPATCH_NAMES = {"search_tools", "call_tool"}
OLLAMA_MODEL = "llama3:latest"
WAIT = 20.0

HARBOUR_TOOLS = '''
class Tools:
    def harbour_lights(self, harbour: str) -> str:
        """Report whether the lights of a harbour are lit.

        :param harbour: Name of the harbour
        """
        return f"lit at {harbour}"
'''


@pytest.fixture
def chat_settings(admin, preserve):
    """`chat_settings(**changes)` updates the admin chat settings for the rest of the test."""
    preserve(CHAT_CONFIG)

    def change(**changes) -> None:
        with admin.client() as client:
            current = client.get(CHAT_CONFIG[0]).json()
            client.post(CHAT_CONFIG[1], json={**current, **changes}).raise_for_status()

    return change


@pytest.fixture
def tool_search(chat_settings):
    """Tool Search on with every tool deferred, so the model sees only the two dispatch tools."""
    chat_settings(
        ENABLE_TOOL_SEARCH=True,
        TOOL_SEARCH_DEFER_THRESHOLD=0,
        TOOL_SEARCH_ALWAYS_LOADED=[],
        TOOL_SEARCH_DEFER_BUILTIN_TOOLS=True,
    )
    return chat_settings


def dispatch_call(call_id: str, arguments: dict | str) -> dict:
    encoded = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "call_tool", "arguments": encoded},
    }


def dispatch(tool_name: str, params: dict | str, call_id: str = "call_1") -> Reply:
    """The model calling `call_tool`, with `params` encoded the way the definition asks."""
    encoded = params if isinstance(params, str) else json.dumps(params)
    return Reply(tool_calls=[dispatch_call(call_id, {"tool_name": tool_name, "params": encoded})])


def offered(request: dict) -> list[str]:
    return [tool["function"]["name"] for tool in request.get("tools") or []]


def tool_messages(request: dict) -> dict[str, str]:
    return {
        entry["tool_call_id"]: entry["content"]
        for entry in reply.conversation(request)
        if entry["role"] == "tool"
    }


def replayed_calls(request: dict) -> dict[str, dict]:
    return {
        call["id"]: call["function"]
        for entry in reply.conversation(request)
        if entry["role"] == "assistant"
        for call in entry.get("tool_calls") or []
    }


def items(message: dict, kind: str) -> dict[str, dict]:
    output = without_hidden_calls(message.get("output") or [])
    return {item["call_id"]: item for item in output if item["type"] == kind}


def result_text(message: dict, call_id: str) -> str:
    return items(message, "function_call_output")[call_id]["output"][0]["text"]


def stored_output(client: httpx.Client, turn: ChatTurn) -> list[dict]:
    chat = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]
    return without_hidden_calls(
        chat["history"]["messages"][turn.assistant_message_id].get("output") or []
    )


def wait_for_status(client: httpx.Client, turn: ChatTurn, call_id: str, status: str) -> dict:
    deadline = time.monotonic() + WAIT
    calls: dict = {}
    while time.monotonic() < deadline:
        calls = {
            item.get("call_id"): item
            for item in stored_output(client, turn)
            if item.get("type") == "function_call"
        }
        if calls.get(call_id, {}).get("status") == status:
            return calls[call_id]
        time.sleep(0.1)
    raise AssertionError(f"{call_id} never reached {status}; the calls stood at {calls}")


def resolve(
    client: httpx.Client, turn: ChatTurn, call_id: str, action: str, answers: dict | None = None
) -> None:
    """Approve or reject a pending call, or answer an `ask_user` question with `answers`."""
    decision = {"call_id": call_id, "action": action}
    if answers is not None:
        decision["answers"] = answers
    resolved = client.post(
        f"/api/v1/chats/{turn.chat_id}/messages/{turn.assistant_message_id}/resolve",
        json=decision,
    )
    assert resolved.status_code == 200, f"{action} {call_id}: {resolved.text}"


@contextlib.contextmanager
def private_tool(admin: Actor, source: str) -> Iterator[str]:
    """A workspace tool nobody but the admin may use."""
    tool_id = f"tool_{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/tools/create",
            json={
                "id": tool_id,
                "name": "Private tools",
                "content": source,
                "meta": {"description": "Private tools"},
                "access_grants": [],
            },
        )
        assert created.status_code == 200, created.text
        try:
            yield tool_id
        finally:
            client.delete(f"/api/v1/tools/id/{tool_id}/delete")


def test_a_deferred_tool_runs_through_call_tool_and_later_turns_replay_the_dispatch(
    tool_search, admin, make_user, upstream
):
    sent_arguments = '{"tool_name":"harbour_lights",  "params":"{\\"harbour\\": \\"Pula\\"}"}'
    upstream.queue(
        reply.tool_call("search_tools", {"query": "harbour_lights"}),
        Reply(tool_calls=[dispatch_call("call_2", sent_arguments)]),
        reply.text("The lights are on."),
        reply.text("Still on."),
    )
    with python_tool(admin, HARBOUR_TOOLS) as tool_id, make_user().client() as client:
        turn, message = ask(client, "are the lights on in Pula?", tool_ids=[tool_id])
        ask(
            client,
            "and now?",
            tool_ids=[tool_id],
            chat_id=turn.chat_id,
            parent_id=turn.assistant_message_id,
        )

    first, after_search, after_call, next_turn = upstream.chat_requests()
    assert set(offered(first)) == DISPATCH_NAMES
    found = json.loads(tool_messages(after_search)["call_1"])["tools"]
    assert [tool["name"] for tool in found] == ["harbour_lights"]
    assert tool_messages(after_call)["call_2"] == "lit at Pula"
    assert message["content"] == "The lights are on."

    # the tools and the manifest stay byte for byte the same, so a prompt cache holds
    assert len({json.dumps(request["tools"]) for request in upstream.chat_requests()}) == 1
    manifests = {request["messages"][0]["content"] for request in upstream.chat_requests()}
    assert len(manifests) == 1 and "<available_tools>" in manifests.pop()

    shown = items(message, "function_call")["call_2"]
    assert (shown["name"], json.loads(shown["arguments"])) == (
        "harbour_lights",
        {"harbour": "Pula"},
    )
    assert shown["status"] == "completed"
    replays = [replayed_calls(request)["call_2"] for request in (after_call, next_turn)]
    assert replays[0] == replays[1], "the replayed dispatch changed between turns"
    assert replays[0]["name"] == "call_tool"
    assert json.loads(replays[0]["arguments"]) == json.loads(sent_arguments)
    assert tool_messages(next_turn) == tool_messages(after_call)


REFUSED_DISPATCHES = [
    pytest.param(
        {"tool_name": "get_current_timestamp"},
        "Error: call_tool requires tool_name and params.",
        id="params-missing",
    ),
    pytest.param(
        {"tool_name": "get_current_timestamp", "params": "{}", "extra": 1},
        "Error: call_tool requires tool_name and params.",
        id="extra-key",
    ),
    pytest.param("[]", "Error: Tool arguments must be a JSON object.", id="not-an-object"),
    pytest.param(
        {"tool_name": "", "params": "{}"},
        "Error: tool_name must be a non-empty string.",
        id="empty-tool-name",
    ),
    pytest.param(
        {"tool_name": "   ", "params": "{}"},
        "Error: tool_name must be a non-empty string.",
        id="blank-tool-name",
    ),
    pytest.param(
        {"tool_name": 7, "params": "{}"},
        "Error: tool_name must be a non-empty string.",
        id="tool-name-not-a-string",
    ),
    pytest.param(
        {"tool_name": "search_tools", "params": '{"query": "time"}'},
        "Error: call_tool cannot invoke search_tools or itself.",
        id="search-tools",
    ),
    pytest.param(
        {"tool_name": "call_tool", "params": "{}"},
        "Error: call_tool cannot invoke search_tools or itself.",
        id="itself",
    ),
    pytest.param(
        {"tool_name": "launch_rockets", "params": "{}"},
        'Error: Tool "launch_rockets" is not available. Search for an available tool.',
        id="unknown-tool",
    ),
    pytest.param(
        {"tool_name": "get_current_timestamp", "params": "[]"},
        "Error: Tool arguments must be a JSON object.",
        id="params-not-an-object",
    ),
    pytest.param(
        {"tool_name": "get_current_timestamp", "params": []},
        "Error: Tool arguments must be a JSON object.",
        id="params-a-list",
    ),
    pytest.param(
        {"tool_name": "get_current_timestamp", "params": "{broken"},
        "Error: ",
        id="params-malformed",
    ),
    pytest.param("{broken", "Error: ", id="arguments-malformed"),
    pytest.param(
        {"calls": []}, "Error: call_tool requires a non-empty calls array.", id="batch-empty"
    ),
    pytest.param(
        {"calls": "get_current_timestamp"},
        "Error: call_tool requires a non-empty calls array.",
        id="batch-not-a-list",
    ),
    pytest.param(
        {"calls": [{"tool_name": "get_current_timestamp", "params": "{}"}], "extra": 1},
        "Error: call_tool requires a non-empty calls array.",
        id="batch-extra-key",
    ),
    pytest.param(
        {"calls": [{"tool_name": "get_current_timestamp"}]},
        "Error: Each call requires tool_name and params.",
        id="batch-entry-without-params",
    ),
    pytest.param(
        {"calls": [{"tool_name": "get_current_timestamp", "params": "{}"}, "now"]},
        "Error: Each call requires tool_name and params.",
        id="batch-entry-not-an-object",
    ),
]


@pytest.mark.parametrize(("arguments", "expected"), REFUSED_DISPATCHES)
def test_a_refused_dispatch_reaches_the_model_as_a_failed_result(
    tool_search, user, upstream, arguments, expected
):
    upstream.queue(Reply(tool_calls=[dispatch_call("call_1", arguments)]), reply.text("retrying"))
    with user.client() as client:
        _, message = ask(client, "use a tool")

    first, follow_up = upstream.chat_requests()
    sent_back = tool_messages(follow_up)["call_1"]
    if expected == "Error: ":
        assert sent_back.startswith(expected) and len(sent_back) > len(expected), sent_back
    else:
        assert sent_back == expected
    assert result_text(message, "call_1") == sent_back
    assert items(message, "function_call_output")["call_1"]["status"] == "failed"
    assert items(message, "function_call")["call_1"]["status"] == "failed"
    assert replayed_calls(follow_up)["call_1"]["name"] == "call_tool"
    assert follow_up["tools"] == first["tools"]
    assert message["content"] == "retrying"


def batch_call(call_id: str, *calls: tuple[str, dict]) -> dict:
    """One `call_tool` call carrying several tool calls, each with JSON-encoded params."""
    entries = [{"tool_name": name, "params": json.dumps(params)} for name, params in calls]
    return dispatch_call(call_id, {"calls": entries})


def combined_results(content: str) -> list[tuple[str, str, str]]:
    """Tool name, status and text of each member in a batch's one combined result."""
    return [
        (entry["tool_name"], entry["status"], "".join(part["text"] for part in entry["output"]))
        for entry in json.loads(content)["results"]
    ]


def calls_and_results(request: dict) -> tuple[list[str], list[str]]:
    """The ids of every tool call the history carries, and of every tool result."""
    conversation = reply.conversation(request)
    calls = [
        call["id"]
        for entry in conversation
        if entry["role"] == "assistant"
        for call in entry.get("tool_calls") or []
    ]
    results = [entry["tool_call_id"] for entry in conversation if entry["role"] == "tool"]
    return calls, results


UNAVAILABLE_ROCKETS = 'Error: Tool "launch_rockets" is not available. Search for an available tool.'


def test_one_call_tool_batch_runs_each_tool_and_is_sent_back_as_one_call(
    tool_search, admin, make_user, upstream
):
    batch = batch_call(
        "call_1",
        ("harbour_lights", {"harbour": "Pula"}),
        ("launch_rockets", {}),
        (TIME_TOOL, {}),
    )
    upstream.queue(Reply(tool_calls=[batch]), reply.text("Checked."), reply.text("Still lit."))
    with python_tool(admin, HARBOUR_TOOLS) as tool_id, make_user().client() as client:
        turn, message = ask(client, "lights and time?", tool_ids=[tool_id])
        ask(
            client,
            "and now?",
            tool_ids=[tool_id],
            chat_id=turn.chat_id,
            parent_id=turn.assistant_message_id,
        )

    shown = list(items(message, "function_call").values())
    assert [(call["name"], call["status"]) for call in shown] == [
        ("harbour_lights", "completed"),
        ("call_tool", "failed"),
        (TIME_TOOL, "completed"),
    ], "the batch did not show as one entry per tool"
    assert shown[0]["call_id"] == "call_1"
    assert len({call["call_id"] for call in shown}) == 3
    outputs = items(message, "function_call_output")
    assert outputs["call_1"]["output"][0]["text"] == "lit at Pula"
    assert outputs[shown[1]["call_id"]]["output"][0]["text"] == UNAVAILABLE_ROCKETS
    assert "current_timestamp" in outputs[shown[2]["call_id"]]["output"][0]["text"]
    assert message["content"] == "Checked."

    _, after_batch, next_turn = upstream.chat_requests()
    for request in (after_batch, next_turn):
        assert calls_and_results(request) == (["call_1"], ["call_1"]), request["messages"]
        assert replayed_calls(request)["call_1"] == batch["function"]
        results = combined_results(tool_messages(request)["call_1"])
        assert [(name, status) for name, status, _ in results] == [
            ("harbour_lights", "completed"),
            ("launch_rockets", "failed"),
            (TIME_TOOL, "completed"),
        ], results
        assert results[0][2] == "lit at Pula" and results[1][2] == UNAVAILABLE_ROCKETS
        assert "current_timestamp" in results[2][2]
    assert tool_messages(next_turn) == tool_messages(after_batch)


def test_the_responses_api_gets_a_batch_back_as_one_call_with_one_combined_result(
    tool_search, admin, preserve, listener
):
    preserve(OPENAI_CONFIG)
    entries = [
        {"tool_name": TIME_TOOL, "params": "{}"},
        {"tool_name": "write_note", "params": json.dumps({"title": "Tide", "content": "low"})},
    ]
    with admin.client() as client:
        provider = responses_api.connect_responses(client, listener)
        provider.answer(
            responses_api.events_stream(
                *responses_api.function_call("call_tool", {"calls": entries}),
                responses_api.completed(),
            ),
            responses_api.events_stream(*responses_api.message("Done."), responses_api.completed()),
            responses_api.events_stream(
                *responses_api.message("Still done."), responses_api.completed()
            ),
        )
        turn, message = ask(client, "time and a note", model=responses_api.RESPONSES_MODEL)
        ask(
            client,
            "and now?",
            model=responses_api.RESPONSES_MODEL,
            chat_id=turn.chat_id,
            parent_id=turn.assistant_message_id,
        )

    shown = [(call["name"], call["status"]) for call in items(message, "function_call").values()]
    assert shown == [(TIME_TOOL, "completed"), ("write_note", "completed")], shown
    for request in provider.sent()[1:]:
        conversation = responses_api.conversation(request)
        calls = [entry for entry in conversation if entry.get("type") == "function_call"]
        outputs = [entry for entry in conversation if entry.get("type") == "function_call_output"]
        assert [(call["call_id"], call["name"]) for call in calls] == [("call_1", "call_tool")]
        assert json.loads(calls[0]["arguments"]) == {"calls": entries}
        assert [output["call_id"] for output in outputs] == ["call_1"]
        output = outputs[0]["output"]
        text = output if isinstance(output, str) else "".join(part["text"] for part in output)
        results = combined_results(text)
        assert [(name, status) for name, status, _ in results] == [
            (TIME_TOOL, "completed"),
            ("write_note", "completed"),
        ], results


def test_a_batch_paused_for_approval_is_never_sent_half_done(
    tool_search, admin, make_user, upstream
):
    tool_search(ENABLE_TOOL_PERMISSIONS=True)
    batch = batch_call("call_1", (TIME_TOOL, {}), ("harbour_lights", {"harbour": "Pula"}))
    upstream.queue(
        Reply(tool_calls=[batch]),
        reply.text("Both checked.", match=reply.answering("time and lights?")),
    )
    with python_tool(admin, HARBOUR_TOOLS) as tool_id, make_user().client() as client:
        turn = send_message(client, "time and lights?", tool_ids=[tool_id], params=ASK)
        wait_for_status(client, turn, "call_1", "pending")
        resolve(client, turn, "call_1", "approve")
        lights = next(
            call_id
            for call_id, call in items(stored_message(client, turn), "function_call").items()
            if call["name"] == "harbour_lights"
        )
        wait_for_status(client, turn, lights, "pending")
        half_done = stored_message(client, turn)

        upstream.queue(reply.text("Meanwhile.", match=reply.answering("meanwhile?")))
        ask(
            client,
            "meanwhile?",
            tool_ids=[tool_id],
            params=ASK,
            chat_id=turn.chat_id,
            parent_id=turn.assistant_message_id,
        )
        meanwhile = upstream.chat_requests()[-1]

        resolve(client, turn, lights, "approve")
        message = wait_for_reply(client, turn)

    assert "current_timestamp" in result_text(half_done, "call_1")
    assert lights not in items(half_done, "function_call_output")
    calls, results = calls_and_results(meanwhile)
    assert calls == results == [], f"a half-done batch was sent: {meanwhile['messages']}"

    finished = upstream.chat_requests()[-1]
    assert reply.answering("time and lights?")(finished)
    assert calls_and_results(finished) == (["call_1"], ["call_1"]), finished["messages"]
    results = combined_results(tool_messages(finished)["call_1"])
    assert [(name, status) for name, status, _ in results] == [
        (TIME_TOOL, "completed"),
        ("harbour_lights", "completed"),
    ], results
    assert results[1][2] == "lit at Pula"
    assert message["content"].endswith("Both checked.")


def test_params_sent_as_an_object_run_the_tool(tool_search, user, upstream):
    called = {"tool_name": "get_current_timestamp", "params": {}}
    upstream.queue(Reply(tool_calls=[dispatch_call("call_1", called)]), reply.text("It is late."))
    with user.client() as client:
        _, message = ask(client, "what time is it?")

    _, follow_up = upstream.chat_requests()
    assert "current_timestamp" in tool_messages(follow_up)["call_1"], follow_up["messages"]
    assert items(message, "function_call")["call_1"]["status"] == "completed"
    assert message["content"] == "It is late."


def test_a_refused_dispatch_does_not_stop_the_other_and_both_replay_as_sent(
    tool_search, user, upstream
):
    bad = dispatch_call("call_bad", {"tool_name": "nope", "params": "{}"})
    good = dispatch_call("call_good", {"tool_name": "get_current_timestamp", "params": "{}"})
    upstream.queue(Reply(tool_calls=[bad, good]), reply.text("one of two"), reply.text("ok"))
    with user.client() as client:
        turn, message = ask(client, "what time is it?")
        ask(client, "thanks", chat_id=turn.chat_id, parent_id=turn.assistant_message_id)

    _, follow_up, next_turn = upstream.chat_requests()
    results = tool_messages(follow_up)
    assert (
        results["call_bad"] == 'Error: Tool "nope" is not available. Search for an available tool.'
    )
    assert "current_timestamp" in results["call_good"]
    assert message["content"] == "one of two"
    for request in (follow_up, next_turn):
        [assistant] = [
            entry
            for entry in reply.conversation(request)
            if entry["role"] == "assistant" and entry.get("tool_calls")
        ]
        assert assistant["tool_calls"] == [bad, good]
    assert tool_messages(next_turn) == results


def test_the_responses_api_replays_the_dispatch_as_the_model_sent_it(
    tool_search, admin, preserve, listener
):
    preserve(OPENAI_CONFIG)
    dispatched = {"tool_name": "get_current_timestamp", "params": "{}"}
    with admin.client() as client:
        provider = responses_api.connect_responses(client, listener)
        provider.answer(
            responses_api.events_stream(
                *responses_api.function_call("call_tool", dispatched),
                responses_api.completed(),
            ),
            responses_api.events_stream(
                *responses_api.message("It is late."), responses_api.completed()
            ),
            responses_api.events_stream(
                *responses_api.message("Still late."), responses_api.completed()
            ),
        )
        turn, message = ask(client, "what time is it?", model=responses_api.RESPONSES_MODEL)
        ask(
            client,
            "and now?",
            model=responses_api.RESPONSES_MODEL,
            chat_id=turn.chat_id,
            parent_id=turn.assistant_message_id,
        )

    first, follow_up, next_turn = provider.sent()
    assert {tool["name"] for tool in first["tools"]} == DISPATCH_NAMES
    assert follow_up["tools"] == first["tools"] == next_turn["tools"]
    shown = items(message, "function_call")["call_1"]
    assert shown["name"] == "get_current_timestamp"
    assert "current_timestamp" in result_text(message, "call_1")
    for request in (follow_up, next_turn):
        [call] = [
            entry
            for entry in responses_api.conversation(request)
            if entry.get("type") == "function_call"
        ]
        assert (call["name"], call["arguments"]) == ("call_tool", json.dumps(dispatched))
        [output] = [
            entry
            for entry in responses_api.conversation(request)
            if entry.get("type") == "function_call_output"
        ]
        assert "current_timestamp" in output["output"]
    assert message["content"] == "It is late."


def test_a_dispatched_tool_still_asks_for_approval(tool_search, make_user, upstream):
    tool_search(ENABLE_TOOL_PERMISSIONS=True)
    upstream.queue(dispatch("get_current_timestamp", {}), reply.text("It is late."))
    with make_user().client() as client:
        turn = send_message(client, "what time is it?", params={"tool_approval_mode": "ask"})
        pending = wait_for_status(client, turn, "call_1", "pending")
        assert pending["name"] == "get_current_timestamp"
        time.sleep(0.5)  # nothing runs while the approval is open
        assert len(upstream.chat_requests()) == 1
        resolve(client, turn, "call_1", "approve")
        message = wait_for_reply(client, turn)

    follow_up = upstream.chat_requests()[-1]
    assert "current_timestamp" in tool_messages(follow_up)["call_1"]
    assert replayed_calls(follow_up)["call_1"]["name"] == "call_tool"
    assert message["content"].endswith("It is late.")


def test_a_rejected_dispatch_never_runs(tool_search, make_user, upstream):
    tool_search(ENABLE_TOOL_PERMISSIONS=True)
    upstream.queue(dispatch("get_current_timestamp", {}), reply.text("Fine."))
    with make_user().client() as client:
        turn = send_message(client, "what time is it?", params={"tool_approval_mode": "ask"})
        wait_for_status(client, turn, "call_1", "pending")
        resolve(client, turn, "call_1", "reject")
        message = wait_for_reply(client, turn)

    assert (
        tool_messages(upstream.chat_requests()[-1])["call_1"]
        == "Error: tool call rejected by user."
    )
    assert items(message, "function_call")["call_1"]["status"] == "rejected"


def test_an_approved_dispatch_resumes_without_streaming(tool_search, make_user, upstream):
    tool_search(ENABLE_TOOL_PERMISSIONS=True)
    upstream.queue(dispatch("get_current_timestamp", {}), reply.text("It is late."))
    with make_user().client() as client:
        turn = send_message(client, "what time is it?", params={"tool_approval_mode": "ask"})
        wait_for_status(client, turn, "call_1", "pending")
        client.post(
            f"/api/v1/chats/{turn.chat_id}",
            json={"chat": {"params": {"stream_response": False, "tool_approval_mode": "ask"}}},
        ).raise_for_status()
        resolve(client, turn, "call_1", "approve")
        message = wait_for_reply(client, turn)

    follow_up = upstream.chat_requests()[-1]
    assert follow_up["stream"] is False
    assert "current_timestamp" in tool_messages(follow_up)["call_1"]
    assert replayed_calls(follow_up)["call_1"]["name"] == "call_tool"
    assert message["content"].endswith("It is late.")


ASK = {"tool_approval_mode": "ask"}
TIME_TOOL = "get_current_timestamp"
REJECTED = "Error: tool call rejected by user."


def stored_message(client: httpx.Client, turn: ChatTurn) -> dict:
    chat = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]
    return chat["history"]["messages"][turn.assistant_message_id]


def asked_or_ran(client: httpx.Client, turn: ChatTurn, call_id: str = "call_1") -> str:
    """`asked` once the call waits for approval, else its status once the reply is done."""
    deadline = time.monotonic() + WAIT
    message: dict = {}
    while time.monotonic() < deadline:
        message = stored_message(client, turn)
        call = items(message, "function_call").get(call_id, {})
        if call.get("status") == "pending":
            return "asked"
        if message.get("done"):
            return call.get("status", "missing")
        time.sleep(0.1)
    raise AssertionError(f"{call_id} neither asked nor finished: {message}")


def note_titles(client: httpx.Client) -> list[str]:
    """The account's own notes; a note another test shared with everyone is listed too."""
    own_id = client.get("/api/v1/auths/").json()["id"]
    notes = client.get("/api/v1/notes/").json()
    return [note["title"] for note in notes if (note.get("user") or {}).get("id") == own_id]


@contextlib.contextmanager
def preset(admin: Actor, params: dict | None) -> Iterator[str]:
    """The scripted model, or a preset of it carrying `params` when given."""
    if params is None:
        yield reply.MOCK_MODEL_ID
        return
    model_id = f"preset-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": reply.MOCK_MODEL_ID,
                "name": model_id,
                "meta": {},
                "params": params,
                "access_grants": [EVERYONE_READS],
            },
        )
        assert created.status_code == 200, created.text
        try:
            client.get("/api/models", params={"refresh": "true"}).raise_for_status()
            yield model_id
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


APPROVAL_MODES = [
    pytest.param(True, ASK, None, "asked", id="ask"),
    pytest.param(True, {"tool_approval_mode": "full"}, None, "completed", id="full-access"),
    pytest.param(True, {}, None, "completed", id="no-choice"),
    pytest.param(False, ASK, None, "completed", id="permissions-off"),
    pytest.param(True, {}, ASK, "asked", id="model-asks"),
]


@pytest.mark.parametrize(("permissions", "params", "model_params", "expected"), APPROVAL_MODES)
def test_a_dispatched_tool_is_asked_exactly_when_a_direct_call_is(
    tool_search, admin, make_user, upstream, permissions, params, model_params, expected
):
    tool_search(ENABLE_TOOL_PERMISSIONS=permissions)
    outcomes = {}
    with preset(admin, model_params) as model, make_user().client() as client:
        for kind, call in (
            ("direct", reply.tool_call(TIME_TOOL, {})),
            ("dispatched", dispatch(TIME_TOOL, {})),
        ):
            upstream.queue(call, reply.text("done"))
            turn = send_message(client, "what time is it?", model=model, params=params)
            outcomes[kind] = asked_or_ran(client, turn)
            if outcomes[kind] == "asked":
                assert not items(stored_message(client, turn), "function_call_output")
                resolve(client, turn, "call_1", "reject")
            wait_for_reply(client, turn)

    assert outcomes == {"direct": expected, "dispatched": expected}


def test_a_mixed_round_asks_each_call_in_order_and_runs_only_the_allowed(
    tool_search, make_user, upstream
):
    tool_search(ENABLE_TOOL_PERMISSIONS=True)
    kept, dropped = f"kept {uuid.uuid4().hex[:6]}", f"dropped {uuid.uuid4().hex[:6]}"
    unavailable = 'Error: Tool "launch_rockets" is not available. Search for an available tool.'
    calls = [
        {"id": "call_1", "type": "function", "function": {"name": TIME_TOOL, "arguments": "{}"}},
        dispatch_call(
            "call_2",
            {"tool_name": "write_note", "params": json.dumps({"title": kept, "content": "lit"})},
        ),
        dispatch_call("call_3", {"tool_name": "launch_rockets", "params": "{}"}),
        dispatch_call(
            "call_4",
            {"tool_name": "write_note", "params": json.dumps({"title": dropped, "content": "x"})},
        ),
    ]
    upstream.queue(Reply(tool_calls=calls), reply.text("Two of four."))
    with make_user().client() as client:
        turn = send_message(client, "log the lights", params=ASK)
        wait_for_status(client, turn, "call_1", "pending")
        paused = stored_message(client, turn)
        resolve(client, turn, "call_1", "approve")
        wait_for_status(client, turn, "call_2", "pending")
        titles_before_second = note_titles(client)
        resolve(client, turn, "call_2", "approve")
        wait_for_status(client, turn, "call_4", "pending")
        titles_before_fourth = note_titles(client)
        resolve(client, turn, "call_4", "reject")
        message = wait_for_reply(client, turn)
        titles = note_titles(client)

    shown = items(paused, "function_call")
    assert [(item["name"], item["status"]) for item in shown.values()] == [
        (TIME_TOOL, "pending"),
        ("write_note", "queued"),
        ("call_tool", "failed"),
        ("write_note", "queued"),
    ]
    assert result_text(paused, "call_3") == unavailable
    assert list(items(paused, "function_call_output")) == ["call_3"]
    assert titles_before_second == [] and titles_before_fourth == titles == [kept]

    follow_up = upstream.chat_requests()[-1]
    [assistant] = [
        entry
        for entry in reply.conversation(follow_up)
        if entry["role"] == "assistant" and entry.get("tool_calls")
    ]
    assert assistant["tool_calls"] == calls
    results = tool_messages(follow_up)
    assert "current_timestamp" in results["call_1"]
    assert not results["call_2"].startswith("Error")
    assert (results["call_3"], results["call_4"]) == (unavailable, REJECTED)
    assert [item["status"] for item in items(message, "function_call").values()] == [
        "completed",
        "completed",
        "failed",
        "rejected",
    ]
    assert message["content"].endswith("Two of four.")


def test_an_approved_dispatch_leads_to_a_second_dispatch_that_asks_again(
    tool_search, admin, make_user, upstream
):
    tool_search(ENABLE_TOOL_PERMISSIONS=True)
    upstream.queue(
        dispatch("harbour_lights", {"harbour": "Pula"}),
        dispatch("harbour_lights", {"harbour": "Rovinj"}, call_id="call_2"),
        reply.text("Both lit."),
    )
    with python_tool(admin, HARBOUR_TOOLS) as tool_id, make_user().client() as client:
        turn = send_message(client, "check two harbours", tool_ids=[tool_id], params=ASK)
        wait_for_status(client, turn, "call_1", "pending")
        resolve(client, turn, "call_1", "approve")
        second = wait_for_status(client, turn, "call_2", "pending")
        time.sleep(0.5)  # nothing runs while the second approval is open
        requests_while_open = len(upstream.chat_requests())
        resolve(client, turn, "call_2", "approve")
        message = wait_for_reply(client, turn)

    assert (second["name"], json.loads(second["arguments"])) == (
        "harbour_lights",
        {"harbour": "Rovinj"},
    )
    assert requests_while_open == 2
    final = upstream.chat_requests()[-1]
    assert tool_messages(final) == {"call_1": "lit at Pula", "call_2": "lit at Rovinj"}
    assert {call["name"] for call in replayed_calls(final).values()} == {"call_tool"}
    assert message["content"].endswith("Both lit.")


def test_allowing_a_tool_once_allows_no_other_call_of_it(tool_search, make_user, upstream):
    tool_search(ENABLE_TOOL_PERMISSIONS=True)
    same_tool_twice = Reply(
        tool_calls=[
            dispatch_call("call_1", {"tool_name": TIME_TOOL, "params": "{}"}),
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": TIME_TOOL, "arguments": "{}"},
            },
        ]
    )
    upstream.queue(same_tool_twice, reply.text("first"))
    with make_user().client() as client:
        turn = send_message(client, "what time is it?", params=ASK)
        wait_for_status(client, turn, "call_1", "pending")
        resolve(client, turn, "call_1", "approve")
        wait_for_status(client, turn, "call_2", "pending")
        resolve(client, turn, "call_2", "reject")
        wait_for_reply(client, turn)

        later = []
        for call in (reply.tool_call(TIME_TOOL, {}), dispatch(TIME_TOOL, {})):
            upstream.queue(call, reply.text("again"))
            turn = send_message(
                client,
                "and now?",
                params=ASK,
                chat_id=turn.chat_id,
                parent_id=turn.assistant_message_id,
            )
            later.append(asked_or_ran(client, turn))
            resolve(client, turn, "call_1", "approve")
            wait_for_reply(client, turn)

    assert later == ["asked", "asked"]


def resend(
    client: httpx.Client, turn: ChatTurn, assistant_message_id: str, continuing: bool
) -> ChatTurn:
    """Regenerate the reply to `turn` under a new id, or continue it, as the web client does."""
    stored = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]["history"]["messages"]
    question = stored[turn.user_message_id]
    payload = {
        "model": reply.MOCK_MODEL_ID,
        "messages": [{"role": "user", "content": question["content"]}],
        "stream": True,
        "chat_id": turn.chat_id,
        "id": assistant_message_id,
        "user_message": {
            **question,
            "childrenIds": sorted({turn.assistant_message_id, assistant_message_id}),
        },
        "params": ASK,
        "session_id": f"harness-{uuid.uuid4().hex[:8]}",
        "background_tasks": {},
    }
    if continuing:
        payload["assistant_message_id"] = assistant_message_id
    accepted = client.post("/api/chat/completions", json=payload)
    assert accepted.status_code == 200, accepted.text
    return ChatTurn(turn.chat_id, turn.user_message_id, assistant_message_id)


def test_continuing_a_pending_dispatch_runs_nothing_until_it_is_allowed(
    tool_search, make_user, upstream
):
    tool_search(ENABLE_TOOL_PERMISSIONS=True)
    upstream.queue(dispatch(TIME_TOOL, {}), reply.text("It is late."))
    with make_user().client() as client:
        turn = send_message(client, "what time is it?", params=ASK)
        wait_for_status(client, turn, "call_1", "pending")
        resend(client, turn, turn.assistant_message_id, continuing=True)
        time.sleep(1)  # a continue must not run the call or call the model
        after_continue = stored_message(client, turn)
        requests_after_continue = len(upstream.chat_requests())
        resolve(client, turn, "call_1", "approve")
        message = wait_for_reply(client, turn)

    assert requests_after_continue == 1
    assert items(after_continue, "function_call")["call_1"]["status"] == "pending"
    assert not items(after_continue, "function_call_output")
    assert "current_timestamp" in tool_messages(upstream.chat_requests()[-1])["call_1"]
    assert message["content"].endswith("It is late.")


def test_continuing_a_rejected_dispatch_replays_the_rejection_and_runs_nothing(
    tool_search, make_user, upstream
):
    tool_search(ENABLE_TOOL_PERMISSIONS=True)
    upstream.queue(dispatch(TIME_TOOL, {}), reply.text("Fine."), reply.text(" Still fine."))
    with make_user().client() as client:
        turn = send_message(client, "what time is it?", params=ASK)
        wait_for_status(client, turn, "call_1", "pending")
        resolve(client, turn, "call_1", "reject")
        wait_for_reply(client, turn)
        resend(client, turn, turn.assistant_message_id, continuing=True)
        deadline = time.monotonic() + WAIT
        while len(upstream.chat_requests()) < 3 and time.monotonic() < deadline:
            time.sleep(0.1)
        message = wait_for_reply(client, turn)

    continued = upstream.chat_requests()[-1]
    assert len(upstream.chat_requests()) == 3
    assert replayed_calls(continued)["call_1"]["name"] == "call_tool"
    assert tool_messages(continued) == {"call_1": REJECTED}
    assert [item["status"] for item in items(message, "function_call_output").values()] == [
        "rejected"
    ]


def test_regenerating_a_pending_dispatch_asks_again_and_leaves_the_old_call_unrun(
    tool_search, make_user, upstream
):
    tool_search(ENABLE_TOOL_PERMISSIONS=True)
    upstream.queue(dispatch(TIME_TOOL, {}), dispatch(TIME_TOOL, {}), reply.text("Regenerated."))
    with make_user().client() as client:
        turn = send_message(client, "what time is it?", params=ASK)
        wait_for_status(client, turn, "call_1", "pending")
        regenerated = resend(client, turn, str(uuid.uuid4()), continuing=False)
        assert asked_or_ran(client, regenerated) == "asked"
        resolve(client, regenerated, "call_1", "approve")
        message = wait_for_reply(client, regenerated)
        old = stored_message(client, turn)

    assert message["content"].endswith("Regenerated.")
    assert items(old, "function_call")["call_1"]["status"] == "pending"
    assert not items(old, "function_call_output")
    assert len(upstream.chat_requests()) == 3


def test_a_temporary_chat_asks_over_the_socket_for_the_dispatched_tool(
    tool_search, admin, make_user, upstream
):
    tool_search(ENABLE_TOOL_PERMISSIONS=True)
    account = make_user()
    asked: list[dict] = []
    decisions = [False, False, True]
    turns = [
        dispatch("harbour_lights", {"harbour": "Pula"}),
        reply.tool_call("harbour_lights", {"harbour": "Pula"}),
        dispatch("harbour_lights", {"harbour": "Rovinj"}),
        dispatch("launch_rockets", {}),
    ]
    results = []
    with (
        python_tool(admin, HARBOUR_TOOLS) as tool_id,
        connected(account) as socket,
        account.client() as client,
    ):

        def answer(event: dict):
            socket.events.append(event)
            data = event.get("data") or {}
            if data.get("type") == "confirmation":
                asked.append(data["data"]["tool_call"])
                return decisions.pop(0)
            return None

        socket.client.on("events", answer)
        chat_id = f"temporary:{socket.client.get_sid()}"
        for count, call in enumerate(turns, start=1):
            upstream.queue(call, reply.text("ok"))
            send_message(
                client,
                "are the lights on?",
                chat_id=chat_id,
                session_id=socket.client.get_sid(),
                tool_ids=[tool_id],
                params=ASK,
            )
            deadline = time.monotonic() + WAIT
            while len(upstream.chat_requests()) < 2 * count and time.monotonic() < deadline:
                time.sleep(0.1)
            results.append(tool_messages(upstream.chat_requests()[-1])["call_1"])
        assert client.get("/api/v1/chats/").json() == []

    assert [(entry["name"], entry["arguments"]) for entry in asked] == [
        ("harbour_lights", {"harbour": "Pula"}),
        ("harbour_lights", {"harbour": "Pula"}),
        ("harbour_lights", {"harbour": "Rovinj"}),
    ]
    not_approved = "Error: tool call was not approved."
    assert results == [
        not_approved,
        not_approved,
        "lit at Rovinj",
        'Error: Tool "launch_rockets" is not available. Search for an available tool.',
    ]


def ask_user_question() -> dict:
    options = [{"label": "A", "description": "first"}, {"label": "B", "description": "second"}]
    return {"id": "q1", "question": "Which one?", "options": options}


def test_ask_user_through_call_tool_waits_for_the_answer(tool_search, make_user, upstream):
    tool_search(ENABLE_TOOL_PERMISSIONS=True)
    upstream.queue(
        dispatch("ask_user", {"questions": [ask_user_question()]}), reply.text("A it is.")
    )
    with make_user().client() as client:
        turn = send_message(client, "ask me something", params={"tool_approval_mode": "ask"})
        pending = wait_for_status(client, turn, "call_1", "pending")
        time.sleep(1)  # nothing more should reach the model while the question is open
        assert len(upstream.chat_requests()) == 1
        approved = client.post(
            f"/api/v1/chats/{turn.chat_id}/messages/{turn.assistant_message_id}/resolve",
            json={"call_id": "call_1", "action": "approve"},
        )
        assert approved.status_code == 400, "the question was treated as an approval prompt"
        resolve_answer = {"call_id": "call_1", "action": "answer", "answers": {"q1": ["A"]}}
        answered = client.post(
            f"/api/v1/chats/{turn.chat_id}/messages/{turn.assistant_message_id}/resolve",
            json=resolve_answer,
        )
        assert answered.status_code == 200, answered.text
        message = wait_for_reply(client, turn)

    arguments = json.loads(pending["arguments"])
    assert pending["name"] == "ask_user"
    assert arguments["timeout_ms"] == 120_000
    assert arguments["questions"][0]["options"][0] == {"label": "A", "description": "first"}
    follow_up = upstream.chat_requests()[-1]
    assert json.loads(tool_messages(follow_up)["call_1"])["answers"] == {"q1": ["A"]}
    assert replayed_calls(follow_up)["call_1"]["name"] == "call_tool"
    assert message["content"].endswith("A it is.")


def test_a_refused_ask_user_through_call_tool_matches_a_direct_call(tool_search, user, upstream):
    no_questions = {"questions": []}
    upstream.queue(
        reply.tool_call("ask_user", no_questions),
        reply.text("first"),
        dispatch("ask_user", no_questions),
        reply.text("second"),
    )
    with user.client() as client:
        _, direct = ask(client, "ask me something")
        _, dispatched = ask(client, "ask me something")

    direct_result = result_text(direct, "call_1")
    assert direct_result.startswith("Error:")
    assert result_text(dispatched, "call_1") == direct_result
    assert dispatched["content"] == "second"


def test_call_tool_reaches_only_tools_the_user_may_use_in_this_chat(
    tool_search, admin, make_user, upstream
):
    unavailable = 'Error: Tool "harbour_lights" is not available. Search for an available tool.'
    person = make_user()
    with (
        private_tool(admin, HARBOUR_TOOLS) as private_id,
        python_tool(admin, HARBOUR_TOOLS) as shared_id,
        person.client() as client,
    ):
        for tool_ids in ([private_id], []):
            upstream.queue(dispatch("harbour_lights", {"harbour": "Pula"}), reply.text("no"))
            _, message = ask(client, "are the lights on?", tool_ids=tool_ids)
            assert result_text(message, "call_1") == unavailable, tool_ids

        upstream.queue(dispatch("harbour_lights", {"harbour": "Pula"}), reply.text("yes"))
        _, message = ask(client, "are the lights on?", tool_ids=[shared_id])
    assert result_text(message, "call_1") == "lit at Pula"


FILTER_ADDS_TOOL = """
class Filter:
    async def inlet(self, body):
        parameters = {{"type": "object", "properties": {{}}}}
        spec = {{"name": "{name}", "description": "added by a filter", "parameters": parameters}}
        body["tools"] = [{{"type": "function", "function": spec}}]
        return body
"""


@pytest.mark.parametrize("name", ["call_tool", "search_tools"])
def test_a_filter_tool_with_a_dispatch_name_turns_tool_search_off(
    tool_search, admin, user, upstream, name
):
    with installed_function(admin, FILTER_ADDS_TOOL.format(name=name), is_global=True):
        upstream.queue(reply.text("ok"))
        with user.client() as client:
            ask(client, "hello")

    request = upstream.chat_requests()[-1]
    names = offered(request)
    assert names.count(name) == 1
    assert not DISPATCH_NAMES - {name} & set(names)
    assert "get_current_timestamp" in names
    assert "<available_tools>" not in json.dumps(request["messages"])


def test_a_filter_tool_with_another_name_keeps_tool_search_on(tool_search, admin, user, upstream):
    with installed_function(admin, FILTER_ADDS_TOOL.format(name="harbour_beacon"), is_global=True):
        upstream.queue(reply.text("ok"))
        with user.client() as client:
            ask(client, "hello")

    assert set(offered(upstream.chat_requests()[-1])) == DISPATCH_NAMES | {"harbour_beacon"}


TIDE_DESCRIPTION = (
    "Look up the tides of a harbour for a whole week, with the height and time of every high and"
    " low water, the moon phase, the swell and the wind at each one."
)
WEATHER_DESCRIPTION = (
    "Report the weather at a harbour: the wind, the sea state, the visibility"
    " and the rain over the hour."
)

HARBOUR_DESK_TOOLS = '''
class Tools:
    def harbour_lights(self, harbour: str) -> str:
        """Report whether the lights of a harbour are lit.

        :param harbour: Name of the harbour
        """
        return f"lit at {harbour}"

    def tide_table(self, harbour: str, days: int = 7) -> str:
        """TIDE_DESCRIPTION

        :param harbour: Name of the harbour
        :param days: How many days to list
        """
        return "high water at noon"

    def beacon_status(self, harbour: str) -> str:
        """Report whether the beacon
        of a harbour is flashing.

        :param harbour: Name of the harbour
        """
        return "flashing"

    def weather_report(self, harbour: str) -> str:
        """WEATHER_DESCRIPTION

        :param harbour: Name of the harbour
        """
        return "calm"
'''.replace("TIDE_DESCRIPTION", TIDE_DESCRIPTION).replace(
    "WEATHER_DESCRIPTION", WEATHER_DESCRIPTION
)
DESK_NAMES = ["beacon_status", "harbour_lights", "tide_table", "weather_report"]

FORGE_TOOLS = '''
class Tools:
    def github_create_issue(self, title: str) -> str:
        """Open an issue on GitHub.

        :param title: Title of the issue
        """
        return "opened"

    def github_list_repos(self) -> str:
        """List the GitHub repositories of the account."""
        return "none"

    def gitlab_create_issue(self, title: str) -> str:
        """Open an issue on GitLab.

        :param title: Title of the issue
        """
        return "opened"

    def my_github_sync(self) -> str:
        """Sync the mirror of a GitHub repository."""
        return "synced"
'''


def offered_specs(request: dict) -> dict[str, dict]:
    return {tool["function"]["name"]: tool["function"] for tool in request.get("tools") or []}


def schema_size(spec: dict) -> int:
    """Characters of the definition as compact JSON, the unit of the Deferral Threshold."""
    return len(json.dumps(spec, separators=(",", ":"), ensure_ascii=False))


def available_tools_block(request: dict) -> str | None:
    """The `<available_tools>` block of the system prompt, or None without one."""
    system = "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )
    block = re.search(r"<available_tools>\n.*?</available_tools>", system, re.DOTALL)
    return block.group(0) if block else None


def search_call(call_id: str, query: str) -> dict:
    arguments = json.dumps({"query": query})
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "search_tools", "arguments": arguments},
    }


def held_back(client: httpx.Client, upstream, tool_ids: list[str], names: list[str]) -> set[str]:
    """Which of `names` `search_tools` finds by exact name: the tools held back from the model."""
    searches = [search_call(f"find_{index}", name) for index, name in enumerate(names)]
    upstream.queue(Reply(tool_calls=searches), reply.text("found them"))
    ask(client, "which tools are held back?", tool_ids=tool_ids)
    results = tool_messages(upstream.chat_requests()[-1])
    found = set()
    for index, name in enumerate(names):
        tools = json.loads(results[f"find_{index}"])["tools"]
        if name in [tool["name"] for tool in tools]:
            found.add(name)
    return found


def request_with(client: httpx.Client, upstream, tool_ids: list[str]) -> dict:
    """The request the model receives for a fresh chat with these tools."""
    upstream.queue(reply.text("ok"))
    ask(client, "what is going on at the harbour?", tool_ids=tool_ids)
    return upstream.chat_requests()[-1]


def test_a_tool_is_held_back_only_when_its_definition_is_longer_than_the_threshold(
    chat_settings, admin, make_user, upstream
):
    with python_tool(admin, HARBOUR_DESK_TOOLS) as tool_id, make_user().client() as client:
        chat_settings(ENABLE_TOOL_SEARCH=False)
        full = offered_specs(request_with(client, upstream, [tool_id]))
        small, large = schema_size(full["harbour_lights"]), schema_size(full["tide_table"])
        assert small < large, (small, large)

        chat_settings(
            ENABLE_TOOL_SEARCH=True,
            TOOL_SEARCH_DEFER_THRESHOLD=small,
            TOOL_SEARCH_ALWAYS_LOADED=[],
            TOOL_SEARCH_DEFER_BUILTIN_TOOLS=False,
        )
        at_threshold = request_with(client, upstream, [tool_id])
        held_at_threshold = held_back(client, upstream, [tool_id], ["harbour_lights", "tide_table"])
        chat_settings(TOOL_SEARCH_DEFER_THRESHOLD=small - 1)
        over_threshold = request_with(client, upstream, [tool_id])
        held_over_threshold = held_back(client, upstream, [tool_id], ["harbour_lights"])

    sent = offered_specs(at_threshold)
    assert sent.get("harbour_lights") == full["harbour_lights"], (
        f"a {small}-character tool at a {small} threshold was not sent in full: {sorted(sent)}"
    )
    assert DISPATCH_NAMES <= set(sent) and "tide_table" not in sent, sorted(sent)
    assert held_at_threshold == {"tide_table"}, held_at_threshold

    sent = offered_specs(over_threshold)
    assert "harbour_lights" not in sent, (
        f"a {small}-character tool at a {small - 1} threshold was sent in full"
    )
    assert held_over_threshold == {"harbour_lights"}, held_over_threshold


def test_an_always_loaded_pattern_keeps_its_matching_tools_in_full(
    tool_search, admin, make_user, upstream
):
    tool_search(TOOL_SEARCH_ALWAYS_LOADED=["github_*"])
    forge_names = [
        "github_create_issue",
        "github_list_repos",
        "gitlab_create_issue",
        "my_github_sync",
    ]
    with python_tool(admin, FORGE_TOOLS) as tool_id, make_user().client() as client:
        request = request_with(client, upstream, [tool_id])
        held = held_back(client, upstream, [tool_id], forge_names)

    sent = offered_specs(request)
    assert {"github_create_issue", "github_list_repos"} <= set(sent), sorted(sent)
    assert sent["github_create_issue"]["parameters"]["properties"].keys() == {"title"}
    assert not {"gitlab_create_issue", "my_github_sync"} & set(sent), sorted(sent)
    assert held == {"gitlab_create_issue", "my_github_sync"}, held


def test_builtin_tools_stay_in_full_with_defer_builtin_tools_off(
    chat_settings, admin, make_user, upstream
):
    with python_tool(admin, HARBOUR_TOOLS) as tool_id, make_user().client() as client:
        chat_settings(ENABLE_TOOL_SEARCH=False)
        full = offered_specs(request_with(client, upstream, [tool_id]))
        builtins = set(full) - {"harbour_lights"}
        assert "get_current_timestamp" in builtins, sorted(full)

        chat_settings(
            ENABLE_TOOL_SEARCH=True,
            TOOL_SEARCH_DEFER_THRESHOLD=0,
            TOOL_SEARCH_ALWAYS_LOADED=[],
            TOOL_SEARCH_DEFER_BUILTIN_TOOLS=True,
        )
        candidates = sorted(builtins | {"harbour_lights"})
        deferring = request_with(client, upstream, [tool_id])
        held_deferring = held_back(client, upstream, [tool_id], candidates)
        chat_settings(TOOL_SEARCH_DEFER_BUILTIN_TOOLS=False)
        keeping = request_with(client, upstream, [tool_id])
        held_keeping = held_back(client, upstream, [tool_id], candidates)

    assert set(offered(deferring)) == DISPATCH_NAMES
    assert held_deferring == set(candidates), sorted(set(candidates) - held_deferring)

    kept = offered_specs(keeping)
    assert set(kept) == builtins | DISPATCH_NAMES, sorted(kept)
    assert {name: kept[name] for name in builtins} == {name: full[name] for name in builtins}
    assert held_keeping == {"harbour_lights"}, held_keeping


def test_the_search_instructions_name_no_held_back_tool_and_stay_the_same_for_any_tool_set(
    tool_search, admin, make_user, upstream
):
    tool_search(TOOL_SEARCH_DEFER_BUILTIN_TOOLS=False)
    with make_user().client() as client:
        with python_tool(admin, HARBOUR_DESK_TOOLS) as tool_id:
            desk = request_with(client, upstream, [tool_id])
        with python_tool(admin, FORGE_TOOLS) as tool_id:
            forge = request_with(client, upstream, [tool_id])

    block = available_tools_block(desk)
    assert block and "search_tools" in block and "call_tool" in block, desk["messages"]
    assert available_tools_block(forge) == block
    for name in [*DESK_NAMES, "gitlab_create_issue"]:
        assert name not in block, f"{name} was named in the instructions: {block}"
    assert TIDE_DESCRIPTION not in json.dumps(desk), "the held-back definition was sent"
    assert WEATHER_DESCRIPTION not in json.dumps(desk), "the held-back definition was sent"


def test_search_tools_returns_the_full_definition_by_exact_name_and_by_keyword(
    chat_settings, admin, make_user, upstream
):
    with python_tool(admin, HARBOUR_DESK_TOOLS) as tool_id, make_user().client() as client:
        chat_settings(ENABLE_TOOL_SEARCH=False)
        full = offered_specs(request_with(client, upstream, [tool_id]))

        chat_settings(
            ENABLE_TOOL_SEARCH=True,
            TOOL_SEARCH_DEFER_THRESHOLD=0,
            TOOL_SEARCH_ALWAYS_LOADED=["harbour_lights"],
            TOOL_SEARCH_DEFER_BUILTIN_TOOLS=False,
        )
        searches = [
            search_call("by_name", "tide_table"),
            search_call("by_keyword", "flashing"),
            search_call("sent_in_full", "harbour_lights"),
        ]
        upstream.queue(Reply(tool_calls=searches), reply.text("found them"))
        ask(client, "which tools are there?", tool_ids=[tool_id])

    results = {
        call_id: json.loads(content)
        for call_id, content in tool_messages(upstream.chat_requests()[-1]).items()
    }
    assert results["by_name"]["tools"] == [full["tide_table"]], results["by_name"]
    assert results["by_keyword"]["tools"] == [full["beacon_status"]], results["by_keyword"]
    # only held-back tools are searched, so its name only matches the others as keywords
    found = [tool["name"] for tool in results["sent_in_full"]["tools"]]
    assert found and "harbour_lights" not in found, found


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({"ENABLE_TOOL_SEARCH": False, "TOOL_SEARCH_DEFER_THRESHOLD": 0}, id="off"),
        pytest.param(
            {"ENABLE_TOOL_SEARCH": True, "TOOL_SEARCH_DEFER_THRESHOLD": 1_000_000},
            id="nothing-over-the-threshold",
        ),
    ],
)
def test_without_a_held_back_tool_every_tool_is_sent_and_there_is_no_search_or_list(
    chat_settings, admin, make_user, upstream, settings
):
    chat_settings(**settings, TOOL_SEARCH_ALWAYS_LOADED=[], TOOL_SEARCH_DEFER_BUILTIN_TOOLS=True)
    with python_tool(admin, HARBOUR_DESK_TOOLS) as tool_id, make_user().client() as client:
        request = request_with(client, upstream, [tool_id])

    names = set(offered(request))
    assert {*DESK_NAMES, "get_current_timestamp"} <= names, sorted(names)
    assert not DISPATCH_NAMES & names, sorted(names)
    assert "<available_tools>" not in json.dumps(request["messages"])


TOOL_SEARCH_SETTINGS = [
    "ENABLE_TOOL_SEARCH",
    "TOOL_SEARCH_DEFER_THRESHOLD",
    "TOOL_SEARCH_ALWAYS_LOADED",
    "TOOL_SEARCH_DEFER_BUILTIN_TOOLS",
]


def test_the_admin_chat_settings_store_the_tool_search_settings(admin, user, preserve):
    preserve(CHAT_CONFIG)
    saves = [
        (
            {
                "ENABLE_TOOL_SEARCH": True,
                "TOOL_SEARCH_DEFER_THRESHOLD": 1234,
                "TOOL_SEARCH_ALWAYS_LOADED": [" github_* ", "", "jira_create_issue"],
                "TOOL_SEARCH_DEFER_BUILTIN_TOOLS": False,
            },
            [True, 1234, ["github_*", "jira_create_issue"], False],
        ),
        (
            {
                "ENABLE_TOOL_SEARCH": False,
                "TOOL_SEARCH_DEFER_THRESHOLD": -5,
                "TOOL_SEARCH_ALWAYS_LOADED": [],
                "TOOL_SEARCH_DEFER_BUILTIN_TOOLS": True,
            },
            [False, 0, [], True],
        ),
    ]
    with admin.client() as client:
        for changes, expected in saves:
            current = client.get(CHAT_CONFIG[0]).json()
            saved = client.post(CHAT_CONFIG[1], json={**current, **changes})
            assert saved.status_code == 200, saved.text
            stored = client.get(CHAT_CONFIG[0]).json()
            assert [stored[key] for key in TOOL_SEARCH_SETTINGS] == expected, stored
            assert [saved.json()[key] for key in TOOL_SEARCH_SETTINGS] == expected

    with user.client() as client:
        assert client.get(CHAT_CONFIG[0]).status_code == 401
        refused = client.post(CHAT_CONFIG[1], json={"ENABLE_TOOL_SEARCH": True})
        assert refused.status_code == 401, refused.text


# --- the search engine: BM25, the embedding model or a decisions model ------------------------

EMBEDDING_SETTINGS = ("/api/v1/retrieval/embedding", "/api/v1/retrieval/embedding/update")
# a query sharing no word with any tool, so only meaning can find the beacon
MEANING_QUERY = "blinking"
# words that mean the same thing share one dimension of the stand-in embedding model
MEANINGS = [("flashing", "blinking"), ("tide", "water"), ("weather", "rain"), ("lit", "lamp")]
DECISIONS_KEY = "sk-decisions-test"


def meaning_vector(text: str) -> list[float]:
    lowered = text.lower()
    return [1.0 if any(word in lowered for word in words) else 0.0 for words in MEANINGS] + [0.05]


def embed_by_meaning(request: ReceivedRequest) -> Answer:
    inputs = request.json()["input"]
    texts = inputs if isinstance(inputs, list) else [inputs]
    data = [
        {"object": "embedding", "index": index, "embedding": meaning_vector(text)}
        for index, text in enumerate(texts)
    ]
    return json_answer({"object": "list", "data": data})


def found_by(client: httpx.Client, upstream, tool_id: str, *queries: str | dict) -> list[dict]:
    """What `search_tools` returned for each query (or its arguments), asked in one round."""
    searches = [
        {
            "id": f"search_{index}",
            "type": "function",
            "function": {
                "name": "search_tools",
                "arguments": json.dumps(query if isinstance(query, dict) else {"query": query}),
            },
        }
        for index, query in enumerate(queries)
    ]
    upstream.queue(Reply(tool_calls=searches), reply.text("searched"))
    _, message = ask(client, "which tool fits?", tool_ids=[tool_id])
    assert message["content"] == "searched"
    results = tool_messages(upstream.chat_requests()[-1])
    return [json.loads(results[f"search_{index}"]) for index in range(len(queries))]


def found_names(result: dict) -> list[str]:
    return [tool["name"] for tool in result.get("tools", [])]


@pytest.fixture
def desk_search(tool_search):
    """Tool Search holding back only the four harbour desk tools."""
    tool_search(TOOL_SEARCH_DEFER_BUILTIN_TOOLS=False)
    return tool_search


def decisions_service(listener, scores: dict[str, float] | None = None) -> None:
    """A decisions model at `/v1/decisions` scoring the tools with `scores`, or failing for
    `outage` (HTTP 500) and leaving a tool out for `partial`."""

    def decide(request: ReceivedRequest) -> Answer:
        body = request.json()
        if body["input"] == "outage":
            return json_answer({"error": "down"}, status=500)
        names = [question["name"] for question in body["questions"]]
        if body["input"] == "partial":
            names = names[1:]
        answers = [
            {"type": "predicate", "name": name, "probability": (scores or {}).get(name, 0.0)}
            for name in names
        ]
        return json_answer({"answers": answers})

    listener.route("POST", "/v1/decisions", decide)


def test_bm25_finds_a_tool_by_its_words_and_nothing_by_meaning_alone(
    desk_search, admin, make_user, upstream
):
    desk_search(TOOL_SEARCH_ENGINE="bm25")
    with python_tool(admin, HARBOUR_DESK_TOOLS) as tool_id, make_user().client() as client:
        by_word, by_meaning = found_by(client, upstream, tool_id, "flashing", MEANING_QUERY)

    assert found_names(by_word) == ["beacon_status"], by_word
    assert found_names(by_meaning) == [], by_meaning


def test_embedding_search_finds_a_tool_by_meaning_with_the_documents_embedding_model(
    desk_search, admin, make_user, upstream, preserve, listener
):
    preserve(EMBEDDING_SETTINGS)
    listener.route("POST", "/embeddings", embed_by_meaning)
    with admin.client() as client:
        current = client.get(EMBEDDING_SETTINGS[0]).json()
        saved = client.post(
            EMBEDDING_SETTINGS[1],
            json={
                "RAG_EMBEDDING_ENGINE": "openai",
                "RAG_EMBEDDING_MODEL": current["RAG_EMBEDDING_MODEL"],
                "openai_config": {"url": listener.base_url, "key": "sk-meaning"},
            },
        )
        assert saved.status_code == 200, saved.text
    desk_search(TOOL_SEARCH_ENGINE="embeddings")
    with python_tool(admin, HARBOUR_DESK_TOOLS) as tool_id, make_user().client() as client:
        result, best_only, by_name = found_by(
            client,
            upstream,
            tool_id,
            MEANING_QUERY,
            {"query": MEANING_QUERY, "count": 1},
            "tide_table",
        )

    names = found_names(result)
    assert names and names[0] == "beacon_status", result
    assert sorted(names) == DESK_NAMES, "embedding search ranks every held-back tool"
    assert found_names(best_only) == ["beacon_status"], best_only
    assert found_names(by_name) == ["tide_table"], by_name
    embedded = [
        text
        for request in listener.requests_to("/embeddings")
        for text in (request.json()["input"] if isinstance(request.json()["input"], list) else [])
    ]
    assert MEANING_QUERY in embedded, "the query never reached the embedding model"
    assert "tide_table" not in embedded, "an exact tool name was searched by meaning"
    assert sum("beacon_status" in text for text in embedded) == 1, embedded


def test_a_failing_embedding_model_reaches_the_model_as_a_search_error(
    desk_search, admin, make_user, upstream, preserve, listener
):
    preserve(EMBEDDING_SETTINGS)
    listener.route("POST", "/embeddings", json_answer({"error": "down"}, status=500))
    with admin.client() as client:
        current = client.get(EMBEDDING_SETTINGS[0]).json()
        saved = client.post(
            EMBEDDING_SETTINGS[1],
            json={
                "RAG_EMBEDDING_ENGINE": "openai",
                "RAG_EMBEDDING_MODEL": current["RAG_EMBEDDING_MODEL"],
                "openai_config": {"url": listener.base_url, "key": "sk-down"},
            },
        )
        assert saved.status_code == 200, saved.text
    desk_search(TOOL_SEARCH_ENGINE="embeddings")
    with python_tool(admin, HARBOUR_DESK_TOOLS) as tool_id, make_user().client() as client:
        failed, by_name = found_by(client, upstream, tool_id, MEANING_QUERY, "tide_table")

    assert listener.requests_to("/embeddings"), "the embedding model was never asked"
    assert set(failed) == {"error"} and "Embedding" in failed["error"], failed
    assert found_names(by_name) == ["tide_table"], by_name


def test_the_decisions_model_finds_the_tools_over_the_threshold_in_order(
    desk_search, admin, make_user, upstream, listener
):
    scores = {"beacon_status": 0.9, "harbour_lights": 0.75, "weather_report": 0.69}
    decisions_service(listener, scores)
    desk_search(
        TOOL_SEARCH_ENGINE="decisions",
        TOOL_SEARCH_DECISIONS_API_BASE_URL=f"{listener.base_url}/v1/",
        TOOL_SEARCH_DECISIONS_API_KEY=DECISIONS_KEY,
        TOOL_SEARCH_DECISIONS_MODEL=" judge-1 ",
        TOOL_SEARCH_DECISIONS_THRESHOLD=0.7,
        TOOL_SEARCH_DECISIONS_PROMPT="Rate how well this harbour tool fits.",
    )
    with python_tool(admin, HARBOUR_DESK_TOOLS) as tool_id, make_user().client() as client:
        at_seventy, best_only, by_name = found_by(
            client,
            upstream,
            tool_id,
            MEANING_QUERY,
            {"query": MEANING_QUERY, "count": 1},
            "tide_table",
        )
        desk_search(TOOL_SEARCH_DECISIONS_THRESHOLD=0.8, TOOL_SEARCH_DECISIONS_PROMPT="  ")
        [at_eighty] = found_by(client, upstream, tool_id, MEANING_QUERY)
    with admin.client() as client:
        default_prompt = client.get(CHAT_CONFIG[0]).json()["DEFAULT_TOOL_SEARCH_DECISIONS_PROMPT"]

    assert found_names(at_seventy) == ["beacon_status", "harbour_lights"], at_seventy
    assert found_names(best_only) == ["beacon_status"], best_only
    # an exact tool name is found without asking the decisions model
    assert found_names(by_name) == ["tide_table"], by_name
    assert found_names(at_eighty) == ["beacon_status"], at_eighty

    asked = listener.requests_to("/v1/decisions")
    assert len(asked) == 3, [request.json()["input"] for request in asked]
    typed_prompt = "Rate how well this harbour tool fits."
    for request, prompt in zip(asked, [typed_prompt, typed_prompt, default_prompt]):
        assert request.headers.get("Authorization") == f"Bearer {DECISIONS_KEY}"
        body = request.json()
        assert (body["model"], body["input"]) == ("judge-1", MEANING_QUERY)
        assert [question["name"] for question in body["questions"]] == DESK_NAMES
        for question in body["questions"]:
            assert question["type"] == "predicate"
            assert question["instructions"].startswith(prompt), question["instructions"]
            assert f'"name":"{question["name"]}"' in question["instructions"].replace(" ", "")


def test_a_failing_decisions_model_reaches_the_model_as_a_search_error(
    desk_search, admin, make_user, upstream, listener
):
    decisions_service(listener, {"beacon_status": 0.9})
    desk_search(
        TOOL_SEARCH_ENGINE="decisions",
        TOOL_SEARCH_DECISIONS_API_BASE_URL=f"{listener.base_url}/v1",
        TOOL_SEARCH_DECISIONS_MODEL="judge-1",
        TOOL_SEARCH_DECISIONS_THRESHOLD=0.5,
    )
    with python_tool(admin, HARBOUR_DESK_TOOLS) as tool_id, make_user().client() as client:
        outage, partial, working = found_by(
            client, upstream, tool_id, "outage", "partial", MEANING_QUERY
        )

    assert outage == {"error": "Decisions API request failed: HTTP 500"}
    assert partial == {"error": "Decisions API omitted tool relevance scores."}
    assert found_names(working) == ["beacon_status"], working


BAD_DECISIONS_SETUPS = [
    pytest.param({"TOOL_SEARCH_DECISIONS_API_BASE_URL": "ftp://decide.test/v1"}, id="not-http"),
    pytest.param({"TOOL_SEARCH_DECISIONS_API_BASE_URL": "decide.test/v1"}, id="no-scheme"),
    pytest.param(
        {"TOOL_SEARCH_DECISIONS_API_BASE_URL": "https://user:pw@decide.test/v1"},
        id="credentials-in-url",
    ),
    pytest.param(
        {"TOOL_SEARCH_DECISIONS_API_BASE_URL": "https://decide.test/v1?key=1"}, id="query"
    ),
    pytest.param(
        {"TOOL_SEARCH_DECISIONS_API_BASE_URL": "https://decide.test/v1#top"}, id="fragment"
    ),
    pytest.param(
        {"TOOL_SEARCH_DECISIONS_API_BASE_URL": "https://decide.test:99999/v1"}, id="bad-port"
    ),
    pytest.param({"TOOL_SEARCH_DECISIONS_MODEL": "   "}, id="no-model"),
    pytest.param({"TOOL_SEARCH_DECISIONS_THRESHOLD": 1.5}, id="threshold-over-one"),
    pytest.param({"TOOL_SEARCH_DECISIONS_THRESHOLD": -0.1}, id="threshold-negative"),
    pytest.param({"TOOL_SEARCH_ENGINE": "magic"}, id="unknown-engine"),
]
GOOD_DECISIONS = {
    "ENABLE_TOOL_SEARCH": True,
    "TOOL_SEARCH_ENGINE": "decisions",
    "TOOL_SEARCH_DECISIONS_API_BASE_URL": "https://decide.test/v1",
    "TOOL_SEARCH_DECISIONS_API_KEY": "sk-kept",
    "TOOL_SEARCH_DECISIONS_MODEL": "judge-1",
    "TOOL_SEARCH_DECISIONS_THRESHOLD": 0.6,
    "TOOL_SEARCH_DECISIONS_PROMPT": "Rate the tool.",
}
ENGINE_SETTINGS = [key for key in GOOD_DECISIONS if key != "ENABLE_TOOL_SEARCH"]


def engine_settings(config: dict) -> dict:
    return {key: config[key] for key in ENGINE_SETTINGS}


@pytest.mark.parametrize("bad", BAD_DECISIONS_SETUPS)
def test_a_bad_decisions_setup_is_refused_and_nothing_is_saved(admin, preserve, bad):
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        saved = client.post(CHAT_CONFIG[1], json={**current, **GOOD_DECISIONS})
        assert saved.status_code == 200, saved.text
        refused = client.post(
            CHAT_CONFIG[1],
            json={**saved.json(), **bad, "TOOL_SEARCH_DEFER_THRESHOLD": 777},
        )
        stored = client.get(CHAT_CONFIG[0]).json()

    assert refused.status_code == 422, refused.text
    assert engine_settings(stored) == engine_settings(GOOD_DECISIONS)
    assert stored["TOOL_SEARCH_DEFER_THRESHOLD"] == current["TOOL_SEARCH_DEFER_THRESHOLD"]


def test_decisions_settings_not_in_use_are_saved_as_typed(admin, preserve):
    preserve(CHAT_CONFIG)
    draft = {"TOOL_SEARCH_DECISIONS_API_BASE_URL": "decide.test", "TOOL_SEARCH_DECISIONS_MODEL": ""}
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        for unused in (
            {"ENABLE_TOOL_SEARCH": False, "TOOL_SEARCH_ENGINE": "decisions"},
            {"ENABLE_TOOL_SEARCH": True, "TOOL_SEARCH_ENGINE": "bm25"},
        ):
            saved = client.post(CHAT_CONFIG[1], json={**current, **unused, **draft})
            assert saved.status_code == 200, f"{unused}: {saved.text}"
            stored = client.get(CHAT_CONFIG[0]).json()
            assert stored["TOOL_SEARCH_ENGINE"] == unused["TOOL_SEARCH_ENGINE"]
            assert {key: stored[key] for key in draft} == draft


def test_a_save_without_the_engine_settings_keeps_the_saved_ones(admin, preserve):
    preserve(CHAT_CONFIG)
    typed = {
        **GOOD_DECISIONS,
        "TOOL_SEARCH_DECISIONS_API_BASE_URL": " https://decide.test/v1/ ",
        "TOOL_SEARCH_DECISIONS_MODEL": " judge-1 ",
    }
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        saved = client.post(CHAT_CONFIG[1], json={**current, **typed})
        assert saved.status_code == 200, saved.text
        assert engine_settings(saved.json()) == engine_settings(GOOD_DECISIONS)

        older_client = {
            key: value
            for key, value in saved.json().items()
            if key not in ENGINE_SETTINGS
            and key not in ("DEFAULT_TOOL_SEARCH_DECISIONS_PROMPT", "TOOL_SEARCH_EMBEDDING_MODEL")
        }
        resaved = client.post(
            CHAT_CONFIG[1], json={**older_client, "TOOL_SEARCH_DEFER_THRESHOLD": 4321}
        )
        assert resaved.status_code == 200, resaved.text
        stored = client.get(CHAT_CONFIG[0]).json()
        embedding_model = client.get(EMBEDDING_SETTINGS[0]).json()["RAG_EMBEDDING_MODEL"]

    assert engine_settings(stored) == engine_settings(GOOD_DECISIONS), "an older client reset them"
    assert stored["TOOL_SEARCH_DEFER_THRESHOLD"] == 4321
    assert stored["DEFAULT_TOOL_SEARCH_DECISIONS_PROMPT"].strip(), stored
    assert stored["TOOL_SEARCH_EMBEDDING_MODEL"] == embedding_model


def test_a_builtin_tool_writes_a_note_through_call_tool(tool_search, make_user, upstream):
    title = f"Harbour log {uuid.uuid4().hex[:6]}"
    upstream.queue(
        dispatch("write_note", {"title": title, "content": "lights lit at dusk"}),
        reply.text("Saved."),
    )
    with make_user().client() as client:
        _, message = ask(client, "note that the lights are lit")
        titles = note_titles(client)

    assert items(message, "function_call")["call_1"]["name"] == "write_note"
    assert items(message, "function_call_output")["call_1"]["status"] == "completed"
    assert titles == [title]


def test_an_mcp_tool_runs_through_call_tool(tool_search, admin, preserve, make_user, upstream):
    preserve(TOOL_SERVERS)
    person = make_user()
    server_id = f"pier_{uuid.uuid4().hex[:8]}"
    with serving_mcp() as url:
        with admin.client() as client:
            saved = client.post(
                TOOL_SERVERS[1],
                json={
                    "TOOL_SERVER_CONNECTIONS": [
                        mcp_connection(url, server_id, [read_grant(person.id)])
                    ]
                },
            )
            assert saved.status_code == 200, saved.text
        upstream.queue(
            dispatch(f"{server_id}_echo", {"text": "the lamp is lit"}), reply.text("Echoed.")
        )
        with person.client() as client:
            _, message = ask(client, "echo it", tool_ids=[f"server:mcp:{server_id}"])

    first, follow_up = upstream.chat_requests()
    assert set(offered(first)) == DISPATCH_NAMES
    assert tool_messages(follow_up)["call_1"] == "the lamp is lit"
    assert items(message, "function_call")["call_1"]["name"] == f"{server_id}_echo"


def test_an_ollama_model_dispatches_through_call_tool(tool_search, admin, preserve, listener):
    preserve(OLLAMA_CONFIG)
    server = serve_ollama(listener, OLLAMA_MODEL)
    dispatched = {"tool_name": "get_current_timestamp", "params": "{}"}
    call = {"function": {"name": "call_tool", "arguments": dispatched}}
    with admin.client() as client:
        connect_ollama(client, listener)
        client.get("/api/models").raise_for_status()
        server.queue_chat(
            chat_stream(OLLAMA_MODEL, {"tool_calls": [call]}),
            chat_stream(OLLAMA_MODEL, {"content": "It is late."}),
        )
        _, message = ask(client, "what time is it?", model=OLLAMA_MODEL)

    first, follow_up = server.chat_requests()
    assert {tool["function"]["name"] for tool in first["tools"]} == DISPATCH_NAMES
    assert follow_up["tools"] == first["tools"]
    [stored] = items(message, "function_call").values()
    assert stored["name"] == "get_current_timestamp"
    assistant, tool = follow_up["messages"][-2:]
    assert [entry["function"] for entry in assistant["tool_calls"]] == [call["function"]]
    assert "current_timestamp" in tool["content"]
    assert message["content"] == "It is late."


def test_an_ollama_model_batches_through_call_tool_and_gets_one_combined_result(
    tool_search, admin, preserve, listener
):
    preserve(OLLAMA_CONFIG)
    server = serve_ollama(listener, OLLAMA_MODEL)
    batch = {
        "calls": [
            {"tool_name": "get_current_timestamp", "params": "{}"},
            {"tool_name": "launch_rockets", "params": "{}"},
        ]
    }
    call = {"function": {"name": "call_tool", "arguments": batch}}
    with admin.client() as client:
        connect_ollama(client, listener)
        client.get("/api/models").raise_for_status()
        server.queue_chat(
            chat_stream(OLLAMA_MODEL, {"tool_calls": [call]}),
            chat_stream(OLLAMA_MODEL, {"content": "One of two."}),
        )
        _, message = ask(client, "time and rockets?", model=OLLAMA_MODEL)

    _, follow_up = server.chat_requests()
    shown = [(item["name"], item["status"]) for item in items(message, "function_call").values()]
    assert shown == [("get_current_timestamp", "completed"), ("call_tool", "failed")], shown
    assistant, tool = follow_up["messages"][-2:]
    assert [entry["function"] for entry in assistant["tool_calls"]] == [call["function"]]
    results = combined_results(tool["content"])
    assert [(name, status) for name, status, _ in results] == [
        ("get_current_timestamp", "completed"),
        ("launch_rockets", "failed"),
    ], results
    assert follow_up["tools"], "the model lost its tools after a batch with one failed call"
    assert message["content"] == "One of two."


@pytest.mark.slow
def test_the_search_engine_and_decisions_setup_can_be_set_by_environment(instance_with):
    launched = instance_with(
        {
            "ENABLE_TOOL_SEARCH": "true",
            "TOOL_SEARCH_ENGINE": "decisions",
            "TOOL_SEARCH_DECISIONS_API_BASE_URL": "https://decide.test/v1",
            "TOOL_SEARCH_DECISIONS_API_KEY": "sk-from-env",
            "TOOL_SEARCH_DECISIONS_MODEL": "judge-env",
            "TOOL_SEARCH_DECISIONS_THRESHOLD": "0.3",
            "TOOL_SEARCH_DECISIONS_PROMPT": "Rate it from the environment.",
        }
    )
    with admin_of(launched).client() as client:
        config = client.get(CHAT_CONFIG[0]).json()

    assert engine_settings(config) == {
        "TOOL_SEARCH_ENGINE": "decisions",
        "TOOL_SEARCH_DECISIONS_API_BASE_URL": "https://decide.test/v1",
        "TOOL_SEARCH_DECISIONS_API_KEY": "sk-from-env",
        "TOOL_SEARCH_DECISIONS_MODEL": "judge-env",
        "TOOL_SEARCH_DECISIONS_THRESHOLD": 0.3,
        "TOOL_SEARCH_DECISIONS_PROMPT": "Rate it from the environment.",
    }


def test_a_stored_dispatch_relayed_to_the_responses_api_keeps_the_models_call(
    admin, preserve, listener
):
    preserve(OPENAI_CONFIG)
    sent_arguments = '{"tool_name": "get_current_timestamp", "params": "{}"}'
    stored_call = {
        "type": "function_call",
        "id": "call_1",
        "call_id": "call_1",
        "name": "get_current_timestamp",
        "arguments": "{}",
        "status": "completed",
        "provider_function": {"name": "call_tool", "arguments": sent_arguments},
    }
    stored_result = {
        "type": "function_call_output",
        "id": "fco_1",
        "call_id": "call_1",
        "output": [{"type": "input_text", "text": "noon"}],
        "status": "completed",
    }
    with admin.client() as client:
        provider = responses_api.connect_responses(client, listener)
        provider.answer(
            responses_api.events_stream(*responses_api.message("Noon."), responses_api.completed())
        )
        relayed = client.post(
            "/openai/chat/completions",
            json={
                "model": responses_api.RESPONSES_MODEL,
                "stream": True,
                "messages": [
                    {"role": "user", "content": "what time is it?"},
                    {"role": "assistant", "content": "", "output": [stored_call, stored_result]},
                    {"role": "user", "content": "and now?"},
                ],
            },
        )
        assert relayed.status_code == 200, relayed.text

    [call] = [item for item in provider.sent()[-1]["input"] if item.get("type") == "function_call"]
    assert (call["name"], call["arguments"]) == ("call_tool", sent_arguments)
    assert "provider_function" not in call


def stored_batch_member(call_id: str, tool_name: str, status: str) -> dict:
    batch_function = {
        "name": "call_tool",
        "arguments": json.dumps(
            {
                "calls": [
                    {"tool_name": TIME_TOOL, "params": "{}"},
                    {"tool_name": "write_note", "params": '{"title": "Tide"}'},
                ]
            }
        ),
    }
    return {
        "type": "function_call",
        "id": call_id,
        "call_id": call_id,
        "name": tool_name,
        "arguments": "{}",
        "status": status,
        "provider_batch": {
            "call_id": "call_1",
            "function": batch_function,
            "size": 2,
            "tool_name": tool_name,
        },
    }


def stored_result(call_id: str, text: str) -> dict:
    return {
        "type": "function_call_output",
        "id": f"fco_{call_id}",
        "call_id": call_id,
        "output": [{"type": "input_text", "text": text}],
        "status": "completed",
    }


@pytest.mark.parametrize("finished", [True, False], ids=["finished", "half-done"])
def test_a_stored_batch_relayed_to_the_responses_api_is_one_call_or_none(
    admin, preserve, listener, finished
):
    output = [
        stored_batch_member("call_1", TIME_TOOL, "completed"),
        stored_result("call_1", "noon"),
        stored_batch_member("call_2", "write_note", "completed" if finished else "pending"),
    ]
    if finished:
        output.append(stored_result("call_2", "saved"))
    preserve(OPENAI_CONFIG)
    with admin.client() as client:
        provider = responses_api.connect_responses(client, listener)
        provider.answer(
            responses_api.events_stream(*responses_api.message("Noon."), responses_api.completed())
        )
        relayed = client.post(
            "/openai/chat/completions",
            json={
                "model": responses_api.RESPONSES_MODEL,
                "stream": True,
                "messages": [
                    {"role": "user", "content": "time and a note"},
                    {"role": "assistant", "content": "", "output": output},
                    {"role": "user", "content": "and now?"},
                ],
            },
        )
        assert relayed.status_code == 200, relayed.text

    sent = provider.sent()[-1]["input"]
    calls = [item for item in sent if item.get("type") == "function_call"]
    outputs = [item for item in sent if item.get("type") == "function_call_output"]
    if not finished:
        assert calls == outputs == [], f"a half-done batch was relayed: {sent}"
        return
    assert [(call["call_id"], call["name"]) for call in calls] == [("call_1", "call_tool")]
    assert [item["call_id"] for item in outputs] == ["call_1"]
    output_text = outputs[0]["output"]
    if not isinstance(output_text, str):
        output_text = "".join(part["text"] for part in output_text)
    assert combined_results(output_text) == [
        (TIME_TOOL, "completed", "noon"),
        ("write_note", "completed", "saved"),
    ]


# --- what a dispatched tool leaves on the reply: sources, files, embeds -----------------------

VIAS = ("direct", "call_tool")
NOTES = "Grey herons nest in colonies near water and hunt fish in the shallows."
LEDGER = "The heron survey budget for the spring is four thousand."
SOURCE_TAG = re.compile(r'<source id="\d+" name="[^"]+"')

REPORTING_TOOLS = '''
from fastapi.responses import HTMLResponse


class Tools:
    async def harbour_report(self, harbour: str, __event_emitter__=None) -> str:
        """Report on a harbour, citing the harbour log.

        :param harbour: Name of the harbour
        """
        await __event_emitter__(
            {
                "type": "citation",
                "data": {
                    "document": [f"The lights of {harbour} were lit at dusk."],
                    "metadata": [{"source": "harbour log"}],
                    "source": {"name": "Harbour log", "url": "https://example.com/log"},
                },
            }
        )
        await __event_emitter__(
            {"type": "files", "data": {"files": [{"type": "image", "url": "/static/favicon.png"}]}}
        )
        return f"report for {harbour}"

    def harbour_chart(self, harbour: str) -> HTMLResponse:
        """Show a chart of a harbour.

        :param harbour: Name of the harbour
        """
        page = f'<p id="chart">Chart of {harbour}</p>'
        return HTMLResponse(content=page, headers={"Content-Disposition": "inline"})
'''


def model_calls(tool_search, via: str, name: str, arguments: dict) -> Reply:
    """The model calling `name`: offered as itself with Tool Search off, or through call_tool."""
    tool_search(ENABLE_TOOL_SEARCH=via == "call_tool")
    if via == "direct":
        return reply.tool_call(name, arguments)
    return dispatch(name, arguments)


def system_prompts(request: dict) -> list:
    return [entry["content"] for entry in request["messages"] if entry["role"] == "system"]


def assert_ran_as(message: dict, name: str) -> None:
    [call] = items(message, "function_call").values()
    assert (call["name"], call["status"]) == (name, "completed"), call


@pytest.fixture
def library(admin, make_user):
    """Two text files in a knowledge base a fresh reader may read; yields reader, base, files."""
    reader = make_user()
    tag = uuid.uuid4().hex[:8]
    with (
        admin.client() as client,
        knowledge_base(client, f"Birds {tag}", [read_grant(reader.id)]) as knowledge_id,
    ):
        notes = add_text_file(client, knowledge_id, f"herons-{tag}.txt", NOTES)
        ledger = add_text_file(client, knowledge_id, f"ledger-{tag}.txt", LEDGER)
        yield reader, knowledge_id, [notes, ledger]


@pytest.fixture
def file_reader(admin):
    """A preset that reads a chat's files through tools, so the chat file tools are offered."""
    model_id = f"file-reader-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": "File reader",
        "meta": {"capabilities": {"file_upload": True, "file_context": False}},
        "params": {},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        client.post("/api/v1/models/create", json=form).raise_for_status()
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


FILE_TOOLS = ["query_knowledge_files", "view_knowledge_file", "query_chat_files", "view_file"]


@pytest.mark.parametrize("tool", FILE_TOOLS)
def test_a_file_tool_through_call_tool_cites_what_a_direct_call_cites(
    tool_search, library, file_reader, upstream, tool
):
    reader, knowledge_id, file_ids = library
    arguments = {
        "query_knowledge_files": {"query": "heron", "knowledge_ids": [knowledge_id], "count": 10},
        "view_knowledge_file": {"file_id": file_ids[0]},
        "query_chat_files": {"query": "heron"},
        "view_file": {"file_id": file_ids[0]},
    }[tool]
    chat = {}
    if tool in ("query_chat_files", "view_file"):
        files = [{"type": "file", "id": file_id, "name": file_id} for file_id in file_ids]
        chat = {"model": file_reader, "files": files}
    sources = {}
    with reader.client() as client:
        for via in VIAS:
            upstream.queue(model_calls(tool_search, via, tool, arguments), reply.text("cited"))
            _, message = ask(client, f"what do my notes say about herons? ({via})", **chat)
            assert_ran_as(message, tool)
            sources[via] = message.get("sources")

    assert sources["direct"], f"{tool} cited nothing on a direct call"
    assert sources["call_tool"] == sources["direct"]
    assert NOTES in json.dumps(sources["call_tool"])


@pytest.mark.parametrize("via", VIAS)
def test_an_approved_file_tool_cites_what_an_unasked_one_cites(tool_search, library, upstream, via):
    """Sources must not be lost when the reply resumes after an approval (#32436)."""
    reader, knowledge_id, _ = library
    tool_search(ENABLE_TOOL_PERMISSIONS=True)
    arguments = {"query": "heron", "knowledge_ids": [knowledge_id], "count": 10}
    sources = {}
    with reader.client() as client:
        for mode in ("full", "ask"):
            upstream.queue(
                model_calls(tool_search, via, "query_knowledge_files", arguments),
                reply.text("cited"),
            )
            turn = send_message(
                client, f"what about herons? ({via}, {mode})", params={"tool_approval_mode": mode}
            )
            if mode == "ask":
                wait_for_status(client, turn, "call_1", "pending")
                resolve(client, turn, "call_1", "approve")
            message = wait_for_reply(client, turn)
            assert_ran_as(message, "query_knowledge_files")
            sources[mode] = message.get("sources")

    assert sources["full"], "the unasked call cited nothing"
    assert sources["ask"] == sources["full"], "the approved call lost its sources"


def test_a_round_cited_through_call_tool_gets_the_source_list_and_replays_next_turn(
    tool_search, library, upstream
):
    reader, knowledge_id, _ = library
    arguments = {"query": "heron", "knowledge_ids": [knowledge_id], "count": 10}
    rewritten = {}
    with reader.client() as client:
        for via in VIAS:
            question = f"what do my notes say about herons? ({via})"
            upstream.queue(
                model_calls(tool_search, via, "query_knowledge_files", arguments),
                reply.text("Herons nest [1]."),
                reply.text("Still herons."),
            )
            turn, _ = ask(client, question)
            ask(client, "and now?", chat_id=turn.chat_id, parent_id=turn.assistant_message_id)
            first, after_call, next_turn = upstream.chat_requests()[-3:]
            [user_prompt] = [entry for entry in after_call["messages"] if entry["role"] == "user"]
            assert user_prompt["content"] == question
            rewritten[via] = tool_messages(after_call)["call_1"].replace(f" ({via})", "")
            assert after_call["tools"] == first["tools"] == next_turn["tools"]
            assert system_prompts(after_call) == system_prompts(first)
            assert tool_messages(next_turn) == tool_messages(after_call)
            replayed = replayed_calls(next_turn)["call_1"]["name"]
            assert replayed == ("call_tool" if via == "call_tool" else "query_knowledge_files")

    assert len(SOURCE_TAG.findall(rewritten["call_tool"])) == 2, rewritten["call_tool"]
    assert rewritten["call_tool"] == rewritten["direct"]


def test_a_workspace_tool_through_call_tool_keeps_its_citation_and_files(
    tool_search, admin, make_user, upstream
):
    stored = {}
    with python_tool(admin, REPORTING_TOOLS) as tool_id, make_user().client() as client:
        for via in VIAS:
            upstream.queue(
                model_calls(tool_search, via, "harbour_report", {"harbour": "Pula"}),
                reply.text("Reported."),
            )
            _, message = ask(client, "report on Pula", tool_ids=[tool_id])
            assert_ran_as(message, "harbour_report")
            stored[via] = (message.get("sources"), message.get("files"))

    sources, files = stored["call_tool"]
    assert [source["source"]["name"] for source in sources] == ["Harbour log"]
    assert files == [{"type": "image", "url": "/static/favicon.png"}]
    assert stored["call_tool"] == stored["direct"]


def test_an_html_embed_through_call_tool_matches_a_direct_call(
    tool_search, admin, make_user, upstream
):
    shown = {}
    with python_tool(admin, REPORTING_TOOLS) as tool_id, make_user().client() as client:
        for via in VIAS:
            upstream.queue(
                model_calls(tool_search, via, "harbour_chart", {"harbour": "Pula"}),
                reply.text("Charted."),
            )
            _, message = ask(client, "chart Pula", tool_ids=[tool_id])
            assert_ran_as(message, "harbour_chart")
            [result] = items(message, "function_call_output").values()
            told = tool_messages(upstream.chat_requests()[-1])["call_1"]
            shown[via] = (result.get("embeds"), told)

    embeds, told = shown["call_tool"]
    assert embeds == ['<p id="chart">Chart of Pula</p>']
    assert "harbour_chart: Embedded UI result is active" in told
    assert shown["call_tool"] == shown["direct"]


@pytest.fixture
def image_engine(admin, listener, preserve):
    """Image generation and editing on, the images drawn by the listener."""
    preserve(IMAGES_CONFIG)
    with admin.client() as client:
        save_image_settings(client, **serve_openai_images(listener))
    return listener


DRAWINGS = {
    "generate_image": ({"prompt": "a lighthouse"}, "/images/generations"),
    "edit_image": (
        {"prompt": "a lighthouse", "image_urls": [f"data:image/png;base64,{PNG_BASE64}"]},
        "/images/edits",
    ),
}


@pytest.mark.parametrize("tool", DRAWINGS)
def test_an_image_drawn_through_call_tool_is_stored_on_the_reply(
    tool_search, make_user, upstream, image_engine, tool
):
    arguments, endpoint = DRAWINGS[tool]
    drawn = {}
    with make_user().client() as client:
        for via in VIAS:
            asked = len(image_engine.requests_to(endpoint))
            upstream.queue(model_calls(tool_search, via, tool, arguments), reply.text("Drawn."))
            _, message = ask(client, "draw it", features={"image_generation": True})
            assert_ran_as(message, tool)
            assert len(image_engine.requests_to(endpoint)) == asked + 1, f"{via} never drew"
            [image] = message.get("files") or [{}]
            assert image.get("type") == "image" and image["url"].startswith("/api/v1/files/")
            stored = client.get(image["url"])
            assert stored.content == base64.b64decode(PNG_BASE64), f"{via}: {image}"
            told = json.loads(tool_messages(upstream.chat_requests()[-1])["call_1"])
            assert [entry["url"] for entry in told["images"]] == [image["url"]]
            drawn[via] = (told["status"], told["message"])

    assert drawn["call_tool"] == drawn["direct"]


def drawing(call_id: str) -> Reply:
    return reply.tool_call("generate_image", {"prompt": "a lighthouse"}, call_id)


def paused_turn(resolution: str) -> tuple[dict, list[Reply], str, str, bool]:
    """Mode, the model's tool rounds, the user's action on call_1, the drawing's call id and
    whether the drawing waits for approval as well."""
    if resolution == "approved":
        return ASK, [drawing("call_1")], "approve", "call_1", False
    if resolution == "rejected":
        return ASK, [reply.tool_call(TIME_TOOL, {}), drawing("call_2")], "reject", "call_2", True
    question = reply.tool_call("ask_user", {"questions": [ask_user_question()]})
    return {"tool_approval_mode": "full"}, [question, drawing("call_2")], "answer", "call_2", False


@pytest.mark.parametrize("resolution", ["approved", "rejected", "answered"])
def test_an_image_drawn_after_the_user_resolved_a_paused_call_is_stored_on_the_reply(
    chat_settings, make_user, upstream, image_engine, resolution
):
    """#32680: the image was drawn but never reached the reply."""
    mode, rounds, action, drawing_id, drawing_asks = paused_turn(resolution)
    chat_settings(ENABLE_TOOL_SEARCH=False, ENABLE_TOOL_PERMISSIONS=True)
    upstream.queue(*rounds, reply.text("Drawn."))
    drawn_before = len(image_engine.requests_to("/images/generations"))
    with make_user().client() as client:
        turn = send_message(client, "draw it", params=mode, features={"image_generation": True})
        wait_for_status(client, turn, "call_1", "pending")
        resolve(
            client, turn, "call_1", action, answers={"q1": ["A"]} if action == "answer" else None
        )
        if drawing_asks:
            wait_for_status(client, turn, drawing_id, "pending")
            resolve(client, turn, drawing_id, "approve")
        message = wait_for_reply(client, turn)
        assert len(image_engine.requests_to("/images/generations")) == drawn_before + 1
        told = tool_messages(upstream.chat_requests()[-1])[drawing_id]
        images = [file for file in message.get("files") or [] if file.get("type") == "image"]
        assert images, f"the drawn image never reached the reply; the model was told {told}"
        [image] = images
        assert client.get(image["url"]).content == base64.b64decode(PNG_BASE64)

    assert [entry["url"] for entry in json.loads(told)["images"]] == [image["url"]], told
    assert message["content"].endswith("Drawn.")


@pytest.fixture
def web_search(instance_with, preserve, listener):
    """The local-fetch instance searching the listener, Tool Search on, a page to read."""
    launched = instance_with(LOCAL_WEB_FETCH)
    preserve(RETRIEVAL_CONFIG, CHAT_CONFIG, on=launched)
    page_url = f"{listener.base_url}/herons"
    listener.route("GET", "/herons", text_answer(PAGE))
    with admin_of(launched).client() as client:
        save_web_settings(client, **serve_search_results(listener, [page_url]))

    def settings(**changes) -> None:
        with admin_of(launched).client() as client:
            current = client.get(CHAT_CONFIG[0]).json()
            client.post(CHAT_CONFIG[1], json={**current, **changes}).raise_for_status()

    settings(
        TOOL_SEARCH_DEFER_THRESHOLD=0,
        TOOL_SEARCH_ALWAYS_LOADED=[],
        TOOL_SEARCH_DEFER_BUILTIN_TOOLS=True,
    )
    return launched, settings, page_url


PAGE_TEXT = "Two grey herons stand at the jetty every morning."
PAGE = f"<html><body><h1>Heron watch</h1><p>{PAGE_TEXT}</p></body></html>"


@pytest.mark.slow
def test_a_page_read_through_call_tool_is_cited_like_a_direct_read(web_search, listener):
    launched, settings, page_url = web_search
    cited = {}
    with create_user(launched).client() as client:
        for via in VIAS:
            searches = len(listener.requests_to("/search"))
            launched.upstream.queue(
                model_calls(settings, via, "search_web", {"query": "herons jetty"}),
                model_calls(settings, via, "fetch_url", {"url": page_url}),
                reply.text("Two herons [1]."),
            )
            _, message = ask(client, "how many herons?", features={"web_search": True})
            assert len(listener.requests_to("/search")) == searches + 1, f"{via} never searched"
            names = [item["name"] for item in items(message, "function_call").values()]
            assert names == ["search_web", "fetch_url"], names
            cited[via] = message.get("sources")

    [source] = cited["call_tool"]
    assert source["source"]["id"] == page_url
    assert PAGE_TEXT in source["document"][0]
    assert cited["call_tool"] == cited["direct"]
