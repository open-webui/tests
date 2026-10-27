"""A `display_file` terminal tool call without `inline` still resolves to a file entry.

open-webui 0.11.2 `64e6c9f01` (a `refac`): the terminal file result was built only when the
call's `inline` was exactly `True`, so a `display_file` call made without it reached the model
as the terminal's raw answer, with no file name or mime type resolved. The fix builds the
`{'type': 'file', 'source': 'open_terminal', ...}` entry for every call that names an existing
file on a terminal and marks it `displayed` only for `inline: true`, the flag the frontend reads
to show the file inline. The model is sent that entry as the tool's result.

With #33263 the terminal's `display_file` also takes an optional `file_id` (`path` no longer
required): a call with one shows the chat or knowledge base file as the built-in tool does
and never reaches the terminal, while a call with a path still goes to the terminal.

A connected fake terminal serves the tools; the scripted model calls them and the test reads
the tool result Open WebUI sent back to the provider.

Twin of unit/chat/test_terminal_file_tool_result.py.

Discriminates: passes on dev ef67cc3fa; with `inline is not True` restored to the bail-out, the
calls without `inline: true` reach the model as the raw terminal answer. Against the #33263
branch 73fca05b3, the terminal tool left without `file_id`, or a `file_id` call routed to the
terminal, turned the file_id test red; it fails on dev until #33263 merges.
"""

from __future__ import annotations

import json
import uuid

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.listener import json_answer
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, serving_terminal
from harness.tool_calls import run_tool

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

REPORT = "/workspace/report.png"


def _operation(operation_id: str) -> dict:
    properties = {
        "path": {"type": "string"},
        "inline": {"type": "boolean"},
        "page": {"type": "integer"},
    }
    body = {"type": "object", "properties": properties, "required": ["path"]}
    return {
        "operationId": operation_id,
        "requestBody": {"required": True, "content": {"application/json": {"schema": body}}},
        "responses": {"200": {"description": "ok"}},
    }


SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "terminal", "version": "1"},
    "paths": {
        "/files/display": {"post": _operation("display_file")},
        "/files/view": {"post": _operation("read_file")},
    },
}


@pytest.fixture
def terminal(admin, preserve):
    """A connected terminal whose `display_file` answers with `terminal.answer`."""
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.answer = {"exists": True, "path": REPORT}
        server.route("POST", "/files/display", lambda _request: json_answer(server.answer))
        server.route("POST", "/files/view", json_answer({"exists": True, "path": REPORT}))
        connection = server.connection()
        with admin.client() as client:
            configure_terminals(client, connection)
        yield server, connection["id"]


def _tool_result(terminal, admin, upstream, name: str, arguments: dict) -> dict:
    _, terminal_id = terminal
    with admin.client() as client:
        sent_back = run_tool(client, upstream, name, arguments, terminal_id=terminal_id)
    return json.loads(sent_back)


@pytest.mark.parametrize(
    "inline", [False, None, "true", 1], ids=["false", "absent", "string", "one"]
)
def test_a_display_file_call_without_inline_true_still_resolves_the_file(
    terminal, admin, upstream, inline
):
    arguments = {"path": REPORT, **({"inline": inline} if inline is not None else {})}

    entry = _tool_result(terminal, admin, upstream, "display_file", arguments)

    assert entry.get("type") == "file", (
        f"a display_file call without inline=true reached the model unresolved: {entry}"
    )
    assert "displayed" not in entry, "a file never shown inline was flagged as displayed"
    assert {key: entry.get(key) for key in ("source", "name", "mime_type", "path")} == {
        "source": "open_terminal",
        "name": "report.png",
        "mime_type": "image/png",
        "path": REPORT,
    }
    assert entry["terminal_id"] == entry["terminal_selector"] == terminal[1]


def test_an_inline_call_is_marked_displayed(terminal, admin, upstream):
    entry = _tool_result(
        terminal, admin, upstream, "display_file", {"path": REPORT, "inline": True}
    )

    assert entry.get("type") == "file"
    assert entry.get("displayed") is True


@pytest.mark.parametrize(
    ("answer", "arguments", "expected"),
    [
        ({"exists": True, "path": "/w/blob.zzz"}, {}, {"mime_type": "application/octet-stream"}),
        (
            {"exists": True, "path": REPORT, "content_type": "image/webp"},
            {},
            {"mime_type": "image/webp"},
        ),
        ({"exists": True, "path": REPORT}, {"page": 3}, {"page": 3}),
        ({"exists": True}, {}, {"path": REPORT, "name": "report.png"}),
    ],
    ids=["unknown-extension", "terminal-content-type", "page-from-the-call", "path-from-the-call"],
)
def test_the_file_fields_are_resolved(terminal, admin, upstream, answer, arguments, expected):
    terminal[0].answer = answer

    entry = _tool_result(terminal, admin, upstream, "display_file", {"path": REPORT, **arguments})

    assert {key: entry.get(key) for key in expected} == expected, entry


@pytest.mark.parametrize(
    ("name", "answer"),
    [
        ("display_file", {"exists": False, "path": "/workspace/gone.png"}),
        ("read_file", None),
    ],
    ids=["missing-file", "another-tool"],
)
def test_calls_that_name_no_terminal_file_are_left_alone(terminal, admin, upstream, name, answer):
    if answer is not None:
        terminal[0].answer = answer

    entry = _tool_result(terminal, admin, upstream, name, {"path": REPORT})

    assert "type" not in entry and "source" not in entry, entry


def test_the_terminal_display_file_also_shows_a_chat_or_knowledge_file(terminal, admin, upstream):
    server, terminal_id = terminal
    with admin.client() as client:
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "true", "process_in_background": "false"},
            files={"file": ("berths.txt", b"berth 3 is free", "text/plain")},
        )
        assert uploaded.status_code == 200, uploaded.text
        file_id = uploaded.json()["id"]

        prompt = f"which tools are there? ({uuid.uuid4().hex[:8]})"
        upstream.queue(reply.text("done", match=reply.answering(prompt)))
        ask(client, prompt, terminal_id=terminal_id)
    sent = next(request for request in upstream.chat_requests() if reply.answering(prompt)(request))
    offered = [tool["function"] for tool in sent.get("tools") or []]
    display = [tool for tool in offered if tool["name"] == "display_file"]
    assert len(display) == 1, f"expected one display_file tool, got {display}"
    parameters = display[0]["parameters"]
    assert {"path", "file_id"} <= set(parameters["properties"]), parameters
    assert "path" not in parameters.get("required", []), parameters

    server.clear()
    by_id = _tool_result(terminal, admin, upstream, "display_file", {"file_id": file_id, "page": 2})
    terminal_calls = server.requests_to("/files/display")
    by_path = _tool_result(terminal, admin, upstream, "display_file", {"path": REPORT})

    assert (by_id.get("source"), by_id.get("id"), by_id.get("page")) == (
        "open_webui",
        file_id,
        2,
    ), f"a display_file call with a file_id did not show the Open WebUI file: {by_id}"
    assert not terminal_calls, "a display_file call with a file_id went to the terminal"
    assert (by_path.get("source"), by_path.get("path")) == ("open_terminal", REPORT), by_path
