"""An image a model generates inside a channel is attached to its channel message.

0.11.2 `aeb126b95` (a `refac` across `routers/images.py`, `socket/main.py`, `tools/builtin.py`,
`utils/files.py` and `utils/middleware.py`). A channel turn runs with `chat_id` set to
`channel:<id>`, and every image path passed that through as a chat id while handing on nothing
but the image URL, so the generated file was never registered on the channel and the channel
message never learned about it. `upload_image` now returns the file's id, url, name and content
type, channel turns are routed by `channel_id`, and the channel event emitter registers each
emitted file and stores it on the message.

Both image paths are driven: the `generate_image` tool a native function-calling model calls,
and the legacy handler that generates before the model answers. The image engine is a listener
answering `/images/generations`.

Twin of unit/chat/test_channel_image_files.py.

Discriminates: passes on dev bbfa876af; the three channel paths fail with the channel emitter
ignoring `files` events (nothing stored) and with `upload_image` returning only the URL (entries
without id or name); the two-image test fails when the emitter drops the files already stored.
"""

from __future__ import annotations

import pytest

from harness import channel_chat
from harness import upstream as reply
from harness.chat import ask
from harness.image_engines import IMAGES_CONFIG, PNG_BASE64, save_image_settings

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def image_engine(admin, listener, preserve):
    """Channels and image generation on, the images drawn by the listener."""
    preserve("admin_config", IMAGES_CONFIG)
    with admin.client() as client:
        channel_chat.enable_channels(client)
        save_image_settings(client, **channel_chat.serve_openai_images(listener))
    return listener


PHOTO = f"data:image/png;base64,{PNG_BASE64}"
# function calling, the model's tool call, the image endpoint it reaches
DRAWINGS = {
    "generate-tool": (
        "native",
        reply.tool_call("generate_image", {"prompt": "a cat"}),
        "/images/generations",
    ),
    "edit-tool": (
        "native",
        reply.tool_call("edit_image", {"prompt": "a cat", "image_urls": [PHOTO]}),
        "/images/edits",
    ),
    "legacy-handler": ("legacy", None, "/images/generations"),
}


@pytest.mark.parametrize("drawing", DRAWINGS)
def test_a_generated_image_is_stored_on_the_channel_message(admin, upstream, image_engine, drawing):
    function_calling, tool_call, endpoint = DRAWINGS[drawing]
    upstream.queue(*filter(None, [tool_call, reply.text("done")]))
    with admin.client() as client, channel_chat.drawing_model(client, function_calling) as model:
        channel_id, _ = channel_chat.mention(client, model, "draw a cat")
        answer = channel_chat.model_reply(client, channel_id)
        files = channel_chat.stored_files(client, channel_id, answer["id"])

    assert image_engine.requests_to(endpoint), f"{endpoint} was never called"
    assert files, "the generated image never reached the channel message (aeb126b95)"
    assert [entry["type"] for entry in files] == ["image"]
    assert all(entry.get("id") and entry.get("name") for entry in files), files


def test_a_second_image_is_stored_beside_the_first(admin, upstream, image_engine):
    draw = DRAWINGS["generate-tool"][1]
    upstream.queue(draw, draw, reply.text("two cats"))
    with admin.client() as client, channel_chat.drawing_model(client) as model:
        channel_id, _ = channel_chat.mention(client, model, "draw two cats")
        answer = channel_chat.model_reply(client, channel_id)
        files = channel_chat.stored_files(client, channel_id, answer["id"])

    stored_ids = {entry["id"] for entry in files}
    assert len(stored_ids) == 2, f"the second image replaced the first: {files}"


def test_a_generated_image_still_lands_on_a_saved_chat_reply(admin, upstream, image_engine):
    upstream.queue(DRAWINGS["generate-tool"][1], reply.text("done"))
    with admin.client() as client, channel_chat.drawing_model(client) as model:
        _, message = ask(client, "draw a cat", model=model, features={"image_generation": True})

    files = message.get("files") or []
    assert [entry["type"] for entry in files] == ["image"], message
    assert files[0]["url"].startswith("/api/v1/files/")
