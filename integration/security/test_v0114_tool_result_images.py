"""Regression: base64 images inside tool results reached the model as text.

open-webui 0.11.4, PR #29665 (issue #29208), two commits in `utils/middleware.py`:

* `afda09454`: only a tool result that was exactly one image data URI was moved out of the
  model's context. An image inside a returned object or list was serialised into the tool
  message as base64 text, so one screenshot could cost hundreds of thousands of tokens. Every
  string that is entirely one image data URI is now attached as an image and replaced by an
  `[image]` marker, wherever it sits. The list branch also stopped removing entries from the
  list it iterated, which had skipped every second data URI.
* `d372bec70`: a tool image in a saved chat is written to a file and referred to by URL instead
  of sitting inline in the chat JSON, falling back to the inline data URL whenever storing is not
  applicable or fails.

An admin's Python tool, readable by every account, returns the images; the scripted model calls
it (native function calling, the default) and a fresh user checks what the provider is sent
next, what the chat keeps and which files were stored.

Twin of unit/security/test_v0114_tool_result_images.py.

Discriminates: passes on bbfa876af. With `afda09454` reverted every test fails except the bare
screenshot row and the deliberate-limits test (base64 in the tool message, nothing attached);
with `d372bec70` reverted the saved chat test and the three broad rows fail (the image stays
inline, no file is stored). Dropping the fallback on a storage error, the fallback on an empty
stored URL or the saved-chat check each turns exactly its own nearby test red.
"""

from __future__ import annotations

import base64
import json
import re
import time
import uuid

import pytest

from harness import upstream as reply
from harness.chat import ChatTurn, ask, send_message
from harness.python_tools import python_tool

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def data_url(mime_type: str, payload: bytes) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(payload).decode()}"


PNG = data_url("image/png", PNG_BYTES)
JPEG = data_url("image/jpeg", b"jpeg-bytes-of-a-screenshot")
PDF = data_url("application/pdf", b"%PDF-1.4 a report")
SVG = data_url("image/svg+xml", b"<svg xmlns='http://www.w3.org/2000/svg'/>")
STORED_FILE_URL = re.compile(r"^/api/v1/files/[^/]+/content$")

TOOL_SOURCE = f'''
import base64


class Tools:
    def render_chart(self) -> dict:
        """Render the sales chart."""
        return {{"chart": {PNG!r}, "summary": "chart of sales"}}

    def take_screenshot(self) -> str:
        """Take one screenshot."""
        return {PNG!r}

    def take_screenshots(self) -> list:
        """Take two screenshots."""
        return [{PNG!r}, {JPEG!r}]

    def render_frames(self) -> dict:
        """Render the animation frames."""
        return {{"frames": ({{"shot": {PNG!r}}}, {{"shot": {JPEG!r}}})}}

    def export_report(self) -> list:
        """Export the report with its chart."""
        return [{PNG!r}, {PDF!r}, "plain text"]

    def describe_documents(self) -> dict:
        """Describe the documents."""
        return {{"doc": {PDF!r}, "caption": "see {PNG} above", "count": 3, "none": None}}

    def draw_diagram(self) -> dict:
        """Draw the diagram."""
        return {{"diagram": {SVG!r}}}

    def capture_screen(self) -> dict:
        """Capture the whole screen."""
        return {{"screen": "data:image/png;base64," + base64.b64encode(bytes(1_200_000)).decode()}}
'''


@pytest.fixture(scope="module")
def image_tool(admin):
    with python_tool(admin, TOOL_SOURCE, name="Image tools") as tool_id:
        yield tool_id


@pytest.fixture
def one_megabyte_file_limit(admin):
    with admin.client() as client:
        before = client.get("/api/v1/retrieval/config").json()["FILE_MAX_SIZE"]
        limited = client.post("/api/v1/retrieval/config/update", json={"FILE_MAX_SIZE": 1})
        assert limited.status_code == 200, limited.text
        yield
        # `preserve` cannot restore this one: posting back null leaves the limit in place
        restored = client.post(
            "/api/v1/retrieval/config/update",
            json={"FILE_MAX_SIZE": "" if before is None else before},
        )
        assert restored.status_code == 200, restored.text


def run_tool(client, upstream, tool_id: str, method: str) -> tuple[ChatTurn, dict]:
    upstream.queue(reply.tool_call(method, {}), reply.text("done"))
    turn, message = ask(client, f"please run {method}", tool_ids=[tool_id])
    assert message["content"] == "done", message
    return turn, message


def request_after_the_tool(upstream) -> dict:
    deadline = time.monotonic() + 30
    while len(upstream.chat_requests()) < 2 and time.monotonic() < deadline:
        time.sleep(0.1)
    requests = upstream.chat_requests()
    assert len(requests) == 2, f"expected the tool round trip, got {len(requests)} requests"
    return requests[-1]


def tool_message(request: dict) -> str:
    return next(entry["content"] for entry in request["messages"] if entry["role"] == "tool")


def attached_images(request: dict) -> list[str]:
    return [
        part["image_url"]["url"]
        for entry in request["messages"]
        if isinstance(entry["content"], list)
        for part in entry["content"]
        if part.get("type") == "image_url"
    ]


def payload_of(url: str) -> str:
    return url.split(",", 1)[1]


def tool_output(message: dict) -> dict:
    return next(item for item in message["output"] if item["type"] == "function_call_output")


def kept_image_urls(message: dict) -> list[str]:
    """The tool images as the chat keeps them for the next model call."""
    parts = tool_output(message)["output"]
    return [part["image_url"] for part in parts if part["type"] == "input_image"]


def stored_files(client) -> list[dict]:
    listing = client.get("/api/v1/files/")
    assert listing.status_code == 200, listing.text
    return listing.json()["items"]


def test_an_image_inside_a_returned_object_is_attached_not_serialised(
    make_user, upstream, image_tool
):
    with make_user().client() as client:
        run_tool(client, upstream, image_tool, "render_chart")
    sent = request_after_the_tool(upstream)

    assert payload_of(PNG) not in tool_message(sent), "the image reached the model as text"
    assert json.loads(tool_message(sent)) == {"chart": "[image]", "summary": "chart of sales"}
    assert attached_images(sent) == [PNG]


def test_a_saved_chat_refers_to_the_stored_image_instead_of_carrying_it(
    make_user, upstream, image_tool
):
    with make_user().client() as client:
        turn, message = run_tool(client, upstream, image_tool, "render_chart")
        saved_chat = client.get(f"/api/v1/chats/{turn.chat_id}").text
        [image_url] = kept_image_urls(message)
        served = client.get(image_url)

    assert STORED_FILE_URL.match(image_url), f"the chat keeps the image inline: {image_url[:40]}"
    assert payload_of(PNG) not in saved_chat
    assert served.status_code == 200
    assert served.content == PNG_BYTES


@pytest.mark.parametrize(
    "method, images",
    [
        ("take_screenshot", [PNG]),
        ("take_screenshots", [PNG, JPEG]),
        ("render_frames", [PNG, JPEG]),
    ],
)
def test_every_image_in_a_tool_result_is_attached_stored_and_kept_out_of_the_text(
    make_user, upstream, image_tool, method, images
):
    with make_user().client() as client:
        run_tool(client, upstream, image_tool, method)
        files = stored_files(client)
    sent = request_after_the_tool(upstream)

    assert not [image for image in images if payload_of(image) in tool_message(sent)]
    assert attached_images(sent) == images
    assert len(files) == len(images)


def test_a_non_image_data_uri_in_a_list_is_filed_as_data_beside_the_image(
    make_user, upstream, image_tool
):
    with make_user().client() as client:
        _, message = run_tool(client, upstream, image_tool, "export_report")
    sent = request_after_the_tool(upstream)

    assert json.loads(tool_message(sent)) == {"results": ["[image]", "plain text"]}
    assert attached_images(sent) == [PNG]
    assert tool_output(message)["files"] == [{"type": "data", "content": PDF}]


def test_only_values_that_are_entirely_an_image_data_uri_are_taken_out(
    make_user, upstream, image_tool
):
    """A PDF data URI and an image data URI wrapped in text stay where they are, by design."""
    with make_user().client() as client:
        run_tool(client, upstream, image_tool, "describe_documents")
        files = stored_files(client)
    sent = request_after_the_tool(upstream)

    assert json.loads(tool_message(sent)) == {
        "doc": PDF,
        "caption": f"see {PNG} above",
        "count": 3,
        "none": None,
    }
    assert attached_images(sent) == []
    assert files == []


def test_a_temporary_chat_attaches_the_image_without_storing_a_file(
    make_user, upstream, image_tool
):
    session_id = f"harness-{uuid.uuid4().hex[:8]}"
    upstream.queue(reply.tool_call("render_chart", {}), reply.text("done"))
    with make_user().client() as client:
        send_message(
            client,
            "please run render_chart",
            chat_id=f"temporary:{session_id}",
            session_id=session_id,
            tool_ids=[image_tool],
        )
        sent = request_after_the_tool(upstream)
        files = stored_files(client)

    assert "[image]" in tool_message(sent)
    assert attached_images(sent) == [PNG]
    assert files == [], "a temporary chat wrote the tool image to a file"


def test_an_image_the_store_cannot_handle_stays_inline(make_user, upstream, image_tool):
    with make_user().client() as client:
        _, message = run_tool(client, upstream, image_tool, "draw_diagram")
        files = stored_files(client)

    assert kept_image_urls(message) == [SVG]
    assert attached_images(request_after_the_tool(upstream)) == [SVG]
    assert files == []


def test_an_image_refused_by_the_file_size_limit_stays_inline(
    make_user, upstream, image_tool, one_megabyte_file_limit
):
    with make_user().client() as client:
        _, message = run_tool(client, upstream, image_tool, "capture_screen")
        files = stored_files(client)

    [image_url] = kept_image_urls(message)
    assert image_url.startswith("data:image/png;base64,")
    assert files == []
