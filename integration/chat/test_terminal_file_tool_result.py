"""A `display_file` terminal tool call without `inline` still resolves to a file entry.

open-webui 0.11.2 `64e6c9f01` (a `refac`): the terminal file result was built only when the
call's `inline` was exactly `True`, so a `display_file` call made without it reached the model
as the terminal's raw answer, with no file name or mime type resolved. The fix builds the
`{'type': 'file', 'source': 'open_terminal', ...}` entry for every call that names an existing
file on a terminal and marks it `displayed` only for `inline: true`, the flag the frontend reads
to show the file inline (since `f2cdbbebe` the flag is always present, `False` without
`inline: true`). The model is sent that entry as the tool's result.

Since `f2cdbbebe` the terminal's `display_file` also takes a chat or knowledge base file, named
in `path` by its id or its `/api/v1/files/<id>/content` URL: such a call shows the Open WebUI
file as the built-in tool does and never reaches the terminal, while a terminal path still goes
to the terminal.

A terminal's `read_file` of a PNG answers with the image itself, which the model is sent as an
image. With the model's Vision capability off it must not be (open-webui/open-webui#32720, PR
#33345 open): the follow-up carries no image part and the tool message notes `[image]`. That
test is red on dev until #33345 merges.

A connected fake terminal serves the tools; the scripted model calls them and the test reads
the tool result Open WebUI sent back to the provider.

Twin of unit/chat/test_terminal_file_tool_result.py.

Discriminates: passes on dev ef67cc3fa and 05e115366; with `inline is not True` restored to the
bail-out, the calls without `inline: true` reach the model as the raw terminal answer, and with
`displayed` set from a truthy `inline`, the string and 1 calls are flagged as displayed. On dev
c5981aaea, a file id call routed to the terminal instead of the built-in tool turned the chat or
knowledge file test red. The Vision-off read passes on the #33345 branch 1a02165c1 and fails on
dev 3b8205505 (the PNG is sent to a model that cannot read it).
"""

from __future__ import annotations

import base64
import json
import uuid

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.listener import json_answer
from harness.model_presets import preset_with_capabilities
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, serving_terminal
from harness.tool_calls import run_tool

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

REPORT = "/workspace/report.png"
CHART = "/workspace/chart.png"
CHART_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


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


def _view_file(request):
    """The terminal's file viewer: an image file answers with its bytes, as Open Terminal does."""
    if json.loads(request.body).get("path") == CHART:
        return 200, {"Content-Type": "image/png"}, CHART_PNG
    return json_answer({"exists": True, "path": REPORT})


@pytest.fixture
def terminal(admin, preserve):
    """A connected terminal whose `display_file` answers with `terminal.answer`."""
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.answer = {"exists": True, "path": REPORT}
        server.route("POST", "/files/display", lambda _request: json_answer(server.answer))
        server.route("POST", "/files/view", _view_file)
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
    # The chat shows the file inline for any truthy `displayed`; False or absent opens the viewer.
    assert not entry.get("displayed"), (
        f"a file never shown inline was flagged as displayed: {entry.get('displayed')!r}"
    )
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
    assert "path" in display[0]["parameters"]["properties"], display[0]["parameters"]

    content_url = f"/api/v1/files/{file_id}/content"
    server.clear()
    by_id = _tool_result(terminal, admin, upstream, "display_file", {"path": file_id, "page": 2})
    by_url = _tool_result(terminal, admin, upstream, "display_file", {"path": content_url})
    terminal_calls = server.requests_to("/files/display")
    by_path = _tool_result(terminal, admin, upstream, "display_file", {"path": REPORT})

    for entry in (by_id, by_url):
        assert (entry.get("source"), entry.get("path"), entry.get("name")) == (
            "open_webui",
            content_url,
            "berths.txt",
        ), f"a display_file call naming an uploaded file did not show it: {entry}"
    assert by_id.get("page") == 2, by_id
    assert not terminal_calls, (
        f"a display_file call naming an uploaded file went to the terminal: {terminal_calls}"
    )
    assert (by_path.get("source"), by_path.get("path")) == ("open_terminal", REPORT), by_path


def _image_parts(messages: list[dict]) -> list[dict]:
    return [
        part
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") in ("image_url", "input_image")
    ]


def _read_the_chart(terminal, admin, upstream, capabilities: dict) -> list[dict]:
    """Have a model with `capabilities` read CHART; returns the follow-up request's messages."""
    _, terminal_id = terminal
    prompt = f"read the chart ({uuid.uuid4().hex[:8]})"
    upstream.queue(
        reply.tool_call("read_file", {"path": CHART}, match=reply.mentioning(prompt)),
        reply.text("a chart", match=reply.mentioning(prompt)),
    )
    with preset_with_capabilities(admin, capabilities) as model_id, admin.client() as client:
        ask(client, prompt, model=model_id, terminal_id=terminal_id)
    requests = [
        request for request in upstream.chat_requests() if reply.mentioning(prompt)(request)
    ]
    follow_up = requests[-1]["messages"] if requests else []
    assert any(entry["role"] == "tool" for entry in follow_up), f"read_file never ran: {follow_up}"
    return follow_up


def test_with_vision_off_a_terminal_image_read_sends_no_image(terminal, admin, upstream):
    follow_up = _read_the_chart(terminal, admin, upstream, {"vision": False})

    assert _image_parts(follow_up) == [], (
        f"a model with Vision off was sent the PNG the terminal read (#32720): "
        f"{_image_parts(follow_up)}"
    )
    [tool_message] = [entry for entry in follow_up if entry["role"] == "tool"]
    assert "[image]" in str(tool_message["content"]), tool_message


def test_with_vision_on_a_terminal_image_read_sends_the_image(terminal, admin, upstream):
    follow_up = _read_the_chart(terminal, admin, upstream, {"vision": True})

    assert len(_image_parts(follow_up)) == 1, f"the PNG the terminal read was not sent: {follow_up}"
