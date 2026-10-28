"""Journey: deferred tools run through `call_tool` while the offered tools never change.

With Tool Search on and tools large enough to defer, the model is offered `search_tools` and
`call_tool(tool_name, params)` alone; `params` is a JSON-encoded object string. Open WebUI unwraps
the dispatcher into the target tool before approvals and special handling, so the target keeps
its own behaviour (approval prompt, `ask_user` pause, access, MCP and builtin tools), and stores
the model's original call so later turns replay what the model actually sent, on Chat Completions
and the Responses API. A malformed dispatch comes back as a failed result starting `Error: ` and
the turn goes on. A tool already named `search_tools` or `call_tool`, a filter's included, turns
the dispatcher off. What a dispatched tool leaves on the reply matches a direct call with Tool
Search off: the sources of the file and knowledge tools, `fetch_url` and a tool's citation event,
a tool's files event, an HTML embed, and pictures from `generate_image` and `edit_image` stored
as files. The source list the citations template adds after a sourced round is the same, and
the next turn replays `call_tool`. Added with dev 7b47c3b22.

The admin settings decide which tools are held back. A tool goes into the `<available_tools>`
list only when its definition, as compact JSON, is longer than the Deferral Threshold in
characters; one at the threshold is sent in full. An Always Loaded pattern such as `github_*`
keeps the tools it matches in full, and with Defer Built-in Tools off the built-in tools stay in
full too. The list names each held-back tool in name order with its description on one line, cut
to 100 characters ending in `…`. `search_tools` searches only the held-back tools and returns
their full definitions, for an exact name or a keyword. With Tool Search off, or no tool over the
threshold, the model gets every tool, no dispatch tools and no list. The admin chat settings
store all four settings, with a negative threshold saved as 0 and blank patterns dropped.

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
fails the built-in and list tests; a 200-character cut fails the list test and an uncollapsed
description the built-in and list tests; searching every tool or returning only names and
descriptions fails the search test; ignoring the Tool Search switch fails the threshold,
built-in, search and off tests; adding the dispatch tools with nothing held back fails the
nothing-over-the-threshold case; and saving the threshold and patterns as sent fails the
settings test.

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
from harness.listener import text_answer
from harness.mcp_server import TOOL_SERVERS, mcp_connection, serving_mcp
from harness.ollama_provider import OLLAMA_CONFIG, chat_stream, connect_ollama, serve_ollama
from harness.plugins import installed_function
from harness.python_tools import EVERYONE_READS, python_tool
from harness.second_provider import OPENAI_CONFIG
from harness.socket_client import connected
from harness.terminal_server import read_grant
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
        for entry in request["messages"]
        if entry["role"] == "tool"
    }


def replayed_calls(request: dict) -> dict[str, dict]:
    return {
        call["id"]: call["function"]
        for entry in request["messages"]
        if entry["role"] == "assistant"
        for call in entry.get("tool_calls") or []
    }


def items(message: dict, kind: str) -> dict[str, dict]:
    return {item["call_id"]: item for item in message.get("output") or [] if item["type"] == kind}


def result_text(message: dict, call_id: str) -> str:
    return items(message, "function_call_output")[call_id]["output"][0]["text"]


def stored_output(client: httpx.Client, turn: ChatTurn) -> list[dict]:
    chat = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]
    return chat["history"]["messages"][turn.assistant_message_id].get("output") or []


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
    for request in (after_call, next_turn):
        assert replayed_calls(request)["call_2"] == {
            "name": "call_tool",
            "arguments": sent_arguments,
        }
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
    pytest.param("[]", "Error: call_tool requires tool_name and params.", id="not-an-object"),
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
        {"tool_name": "get_current_timestamp", "params": {}},
        "Error: params must be a JSON-encoded object string.",
        id="params-not-a-string",
    ),
    pytest.param(
        {"tool_name": "get_current_timestamp", "params": "[]"},
        "Error: params must decode to a JSON object.",
        id="params-not-an-object",
    ),
    pytest.param(
        {"tool_name": "get_current_timestamp", "params": "{broken"},
        "Error: ",
        id="params-malformed",
    ),
    pytest.param("{broken", "Error: ", id="arguments-malformed"),
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
            for entry in request["messages"]
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
        [call] = [entry for entry in request["input"] if entry.get("type") == "function_call"]
        assert (call["name"], call["arguments"]) == ("call_tool", json.dumps(dispatched))
        [output] = [
            entry for entry in request["input"] if entry.get("type") == "function_call_output"
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
        for entry in follow_up["messages"]
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


def listed_tools(request: dict) -> list[tuple[str, str]] | None:
    """The (name, description) pairs of the `<available_tools>` list, or None without a list."""
    system = "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )
    block = re.search(r"<available_tools>\n.*?</available_tools>", system, re.DOTALL)
    if block is None:
        return None
    return re.findall(
        r"<tool>\n<name>(.*?)</name>\n<description>(.*?)</description>\n</tool>\n",
        block.group(0),
    )


def listed_names(request: dict) -> list[str]:
    listed = listed_tools(request)
    assert listed is not None, f"the request has no <available_tools> list: {request['messages']}"
    return [name for name, _ in listed]


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
        chat_settings(TOOL_SEARCH_DEFER_THRESHOLD=small - 1)
        over_threshold = request_with(client, upstream, [tool_id])

    sent = offered_specs(at_threshold)
    assert sent.get("harbour_lights") == full["harbour_lights"], (
        f"a {small}-character tool at a {small} threshold was not sent in full: {sorted(sent)}"
    )
    assert DISPATCH_NAMES <= set(sent) and "tide_table" not in sent, sorted(sent)
    assert "harbour_lights" not in listed_names(at_threshold)
    assert "tide_table" in listed_names(at_threshold)

    sent = offered_specs(over_threshold)
    assert "harbour_lights" not in sent, (
        f"a {small}-character tool at a {small - 1} threshold was sent in full"
    )
    assert "harbour_lights" in listed_names(over_threshold)


def test_an_always_loaded_pattern_keeps_its_matching_tools_in_full(
    tool_search, admin, make_user, upstream
):
    tool_search(TOOL_SEARCH_ALWAYS_LOADED=["github_*"])
    with python_tool(admin, FORGE_TOOLS) as tool_id, make_user().client() as client:
        request = request_with(client, upstream, [tool_id])

    sent = offered_specs(request)
    assert {"github_create_issue", "github_list_repos"} <= set(sent), sorted(sent)
    assert sent["github_create_issue"]["parameters"]["properties"].keys() == {"title"}
    assert not {"gitlab_create_issue", "my_github_sync"} & set(sent), sorted(sent)
    listed = listed_names(request)
    assert {"gitlab_create_issue", "my_github_sync"} <= set(listed), listed
    assert not {"github_create_issue", "github_list_repos"} & set(listed), listed


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
        deferring = request_with(client, upstream, [tool_id])
        chat_settings(TOOL_SEARCH_DEFER_BUILTIN_TOOLS=False)
        keeping = request_with(client, upstream, [tool_id])

    assert set(offered(deferring)) == DISPATCH_NAMES
    assert set(listed_names(deferring)) == builtins | {"harbour_lights"}

    kept = offered_specs(keeping)
    assert set(kept) == builtins | DISPATCH_NAMES, sorted(kept)
    assert {name: kept[name] for name in builtins} == {name: full[name] for name in builtins}
    assert listed_names(keeping) == ["harbour_lights"]


def test_the_list_names_each_held_back_tool_with_its_description_cut_to_100_characters(
    tool_search, admin, make_user, upstream
):
    tool_search(TOOL_SEARCH_DEFER_BUILTIN_TOOLS=False)
    with python_tool(admin, HARBOUR_DESK_TOOLS) as tool_id, make_user().client() as client:
        request = request_with(client, upstream, [tool_id])

    cut = TIDE_DESCRIPTION[:99] + "…"
    assert len(cut) == 100
    assert listed_tools(request) == [
        ("beacon_status", "Report whether the beacon of a harbour is flashing."),
        ("harbour_lights", "Report whether the lights of a harbour are lit."),
        ("tide_table", cut),
        ("weather_report", WEATHER_DESCRIPTION),
    ]
    assert TIDE_DESCRIPTION not in json.dumps(request), "the held-back definition was sent"


def search_call(call_id: str, query: str) -> dict:
    arguments = json.dumps({"query": query})
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "search_tools", "arguments": arguments},
    }


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
            rewritten[via] = user_prompt["content"].replace(f" ({via})", "")
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
