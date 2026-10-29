"""Journey: an image a model generates or edits is also copied into the selected terminal.

open-webui 0.12.5 `b420726ff`: with a terminal selected in the chat, the `generate_image` and
`edit_image` tools upload each new picture to that terminal's working folder, named after its
file id so two pictures never overwrite each other. The tool result the model reads and the
image stored on the reply carry the picture's `terminal_path`, so the model can work on it with
the terminal's tools. A failed copy leaves the picture in the chat and carries a
`terminal_error` instead; without a terminal nothing is sent to one. A personal terminal, which
only the user's browser can reach, is handed the upload through the user's open tab.

A fake terminal answers the working folder and takes the uploads, a socket client plays the tab
for the personal terminal, and the image engine is a listener answering as OpenAI's image
endpoints. The legacy handler that generates before the model answers runs only in Legacy
function calling mode and is out of scope here.

Discriminates: passes on dev 05e115366; each mutation below, made in a backend copy, turns
exactly the named tests red. The `upload_files_to_terminal` calls removed from both image tools
fail the copy, failure and personal-terminal tests; the uploaded name built from the image's own
name in place of its file id, and the `directory` query left out of the upload path, each fail
the copy and personal-terminal tests; the file left out of the tab's `request:terminal` fails
the personal-terminal test; `upload_files_to_terminal` running without a selected terminal
fails the no-terminal test.
"""

from __future__ import annotations

import base64
import json
import uuid
from contextlib import contextmanager
from email.parser import BytesParser
from email.policy import HTTP
from typing import Iterator
from urllib.parse import parse_qs, urlsplit

import pytest

from harness import upstream as reply
from harness.channel_chat import serve_openai_images
from harness.chat import ask
from harness.image_engines import IMAGES_CONFIG, PNG_BASE64, save_image_settings
from harness.listener import json_answer
from harness.socket_client import connected
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    TerminalRequest,
    configure_terminals,
    read_grant,
    serving_terminal,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

GENERATED_PNG = base64.b64decode(PNG_BASE64)
# a space shows the folder survives the trip through the query string
WORKDIR = "/home/user/my work"
# one tool, as a terminal offering none is refused for the chat
SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "terminal", "version": "1"},
    "paths": {
        "/execute": {
            "post": {
                "operationId": "run_command",
                "responses": {"200": {"description": "ok"}},
            }
        }
    },
}
# a personal terminal's address, which only the user's browser can reach
UNREACHABLE = "http://127.0.0.1:9"
RUN_COMMAND = {
    "name": "run_command",
    "description": "Run a shell command.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
}
UPLOAD_FAILED = "File saved in Open WebUI, but upload to the selected terminal failed."
TERMINAL_UNAVAILABLE = "File saved in Open WebUI, but the selected terminal is unavailable."
# the tool the model calls, its arguments and the engine endpoint it reaches
DRAWINGS = {
    "generate_image": ({"prompt": "a lighthouse"}, "/images/generations"),
    "edit_image": (
        {"prompt": "a lighthouse", "image_urls": [f"data:image/png;base64,{PNG_BASE64}"]},
        "/images/edits",
    ),
}


def _uploaded_file(request: TerminalRequest) -> tuple[str, bytes]:
    """The file name and bytes of a multipart upload the terminal received."""
    head = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
    form = BytesParser(policy=HTTP).parsebytes(head + request.body)
    [part] = [
        part
        for part in form.iter_parts()
        if part.get_param("name", header="content-disposition") == "file"
    ]
    return part.get_filename(), part.get_payload(decode=True)


def _upload_directory_of(path: str) -> list[str]:
    return parse_qs(urlsplit(path).query).get("directory", [])


def _save_upload(request: TerminalRequest):
    """Answer as Open Terminal does: the path the file was written to."""
    name, content = _uploaded_file(request)
    [directory] = _upload_directory_of(request.path)
    return json_answer({"path": f"{directory}/{name}", "size": len(content)})


@pytest.fixture
def image_engine(admin, listener, preserve):
    preserve(IMAGES_CONFIG)
    with admin.client() as client:
        save_image_settings(client, **serve_openai_images(listener))
    return listener


@pytest.fixture
def painter(make_user):
    return make_user()


@pytest.fixture
def terminal(admin, preserve, painter):
    """A terminal the painter may use, working in WORKDIR and saving every upload."""
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.route("GET", "/files/cwd", json_answer({"cwd": WORKDIR}))
        server.route("POST", "/files/upload", _save_upload)
        connection = server.connection()
        connection["config"]["access_grants"] = [read_grant(painter.id)]
        with admin.client() as client:
            configure_terminals(client, connection)
        server.clear()
        yield server, connection["id"]


def _draw(painter, upstream, image_engine, tool: str, **options) -> tuple[dict, dict, bytes]:
    """Have the model call `tool`; returns the image the model was told about, the image stored
    on the reply and the stored picture's bytes."""
    arguments, endpoint = DRAWINGS[tool]
    prompt = f"draw it ({uuid.uuid4().hex[:8]})"
    mentioned = reply.mentioning(prompt)
    upstream.queue(
        reply.tool_call(tool, arguments, match=mentioned), reply.text("Drawn.", match=mentioned)
    )
    drawn_before = len(image_engine.requests_to(endpoint))
    with painter.client() as client:
        _, message = ask(client, prompt, features={"image_generation": True}, **options)
        stored_images = [file for file in message.get("files") or [] if file.get("type") == "image"]
        assert len(stored_images) == 1, f"{tool} left no single image on the reply: {message}"
        picture = client.get(stored_images[0]["url"])
        assert picture.status_code == 200, picture.text
    assert len(image_engine.requests_to(endpoint)) == drawn_before + 1, f"{tool} never drew"

    follow_ups = [request for request in upstream.chat_requests() if mentioned(request)]
    told = [entry["content"] for entry in follow_ups[-1]["messages"] if entry["role"] == "tool"]
    assert told, f"{tool} never ran; the provider was sent {follow_ups[-1]['messages']}"
    result = json.loads(told[-1])
    assert len(result.get("images") or []) == 1, f"the model was told {result}"
    return result["images"][0], stored_images[0], picture.content


@pytest.mark.parametrize("tool", DRAWINGS)
def test_a_drawn_image_is_copied_into_the_terminal_working_folder(
    painter, upstream, image_engine, terminal, tool
):
    server, terminal_id = terminal

    told, stored, picture = _draw(painter, upstream, image_engine, tool, terminal_id=terminal_id)

    assert picture == GENERATED_PNG
    uploads = server.requests_to("/files/upload")
    assert len(uploads) == 1, f"expected one upload to the terminal, got {len(uploads)}"
    assert _upload_directory_of(uploads[0].path) == [WORKDIR], (
        f"the picture was not put in the terminal's working folder: {uploads[0].path}"
    )
    name, content = _uploaded_file(uploads[0])
    assert name == f"{told['id']}.png", f"the copy is not named after its file id: {name}"
    assert content == GENERATED_PNG, "the terminal got different bytes than the chat shows"

    expected_path = f"{WORKDIR}/{told['id']}.png"
    assert told.get("terminal_path") == expected_path, f"the model was not told the path: {told}"
    assert stored.get("terminal_path") == expected_path, f"the reply's image has no path: {stored}"
    assert "terminal_error" not in told


@pytest.mark.parametrize("failing", ["upload", "working-folder"])
@pytest.mark.parametrize("tool", DRAWINGS)
def test_a_failed_copy_is_reported_and_the_image_still_shows(
    painter, upstream, image_engine, terminal, tool, failing
):
    server, terminal_id = terminal
    if failing == "upload":
        server.route("POST", "/files/upload", json_answer({"detail": "disk full"}, 500))
        expected_error = UPLOAD_FAILED
    else:
        server.route("GET", "/files/cwd", json_answer({"detail": "starting"}, 503))
        expected_error = TERMINAL_UNAVAILABLE

    told, stored, picture = _draw(painter, upstream, image_engine, tool, terminal_id=terminal_id)

    assert picture == GENERATED_PNG, "the picture was lost with the failed copy"
    assert told.get("terminal_error") == expected_error, f"the model was told {told}"
    assert stored.get("terminal_error") == expected_error, f"the reply's image: {stored}"
    assert "terminal_path" not in told and "terminal_path" not in stored


@contextmanager
def _browser_tab(actor) -> Iterator[tuple[str, list[dict]]]:
    """A tab that uploads to the personal terminal for the server; yields its sid and the asks."""
    asked: list[dict] = []

    def handle(event: dict):
        data = event.get("data") or {}
        if data.get("type") != "request:terminal":
            return None
        request = data.get("data") or {}
        asked.append(request)
        if request.get("path") == "/files/cwd":
            return {"data": {"cwd": WORKDIR}}
        if request.get("file"):
            return {"data": {"path": f"{WORKDIR}/{request['file']['name']}"}}
        return {"data": None}

    with connected(actor) as session:
        session.client.on("events", handle)
        yield session.client.get_sid(namespace="/"), asked


@pytest.mark.parametrize("tool", DRAWINGS)
def test_a_personal_terminal_gets_the_image_through_the_browser(
    make_user, upstream, image_engine, tool
):
    owner = make_user(role="admin")
    personal = {"url": UNREACHABLE, "key": "", "is_terminal": True, "specs": [RUN_COMMAND]}

    with _browser_tab(owner) as (session_id, asked):
        told, stored, picture = _draw(
            owner,
            upstream,
            image_engine,
            tool,
            session_id=session_id,
            terminal_id=UNREACHABLE,
            tool_servers=[personal],
        )

    assert picture == GENERATED_PNG
    uploads = [request for request in asked if request.get("file")]
    assert len(uploads) == 1, f"the tab was not asked once to upload the image: {asked}"
    assert _upload_directory_of(uploads[0]["path"]) == [WORKDIR], uploads[0]
    assert uploads[0]["file"] == {"id": told["id"], "name": f"{told['id']}.png"}, uploads[0]
    expected_path = f"{WORKDIR}/{told['id']}.png"
    assert told.get("terminal_path") == expected_path, f"the model was not told the path: {told}"
    assert stored.get("terminal_path") == expected_path, f"the reply's image has no path: {stored}"


@pytest.mark.parametrize("tool", DRAWINGS)
def test_without_a_terminal_nothing_is_copied(painter, upstream, image_engine, terminal, tool):
    server, _ = terminal

    told, stored, picture = _draw(painter, upstream, image_engine, tool)

    assert picture == GENERATED_PNG
    assert server.received == [], f"a chat without a terminal reached it: {server.received}"
    for image in (told, stored):
        assert "terminal_path" not in image and "terminal_error" not in image, image
