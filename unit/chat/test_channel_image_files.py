"""Regression: an image generated inside a channel must be attached to that channel,
not filed against a chat that does not exist.

open-webui 0.11.2 commit `aeb126b95` (labelled "refac"), across
`routers/images.py`, `socket/main.py`, `tools/builtin.py`, `utils/files.py` and
`utils/middleware.py`. All five changes serve one behaviour: a channel conversation
carries a `chat_id` of the form `channel:<id>`, and every image path used to pass that
straight through as `metadata['chat_id']` while handing downstream nothing but a URL,
so the generated file was never registered on the channel and the channel message never
learned about it. The fix makes `upload_image` return the full file descriptor
(`id`, `url`, `name`, `content_type`), routes channel turns as
`metadata['channel_id']` with the prefix stripped, spreads the descriptor into the
emitted file entries, and teaches the channel event emitter a `files` branch that
registers each file on the channel and persists it onto the message.

Discriminates: passes on v0.11.3, fails on v0.11.1 (`upload_image` returns a bare URL
string, the image handlers pass `chat_id='channel:<id>'`, the built-in image tools pass
no metadata and emit URL-only file entries, and the channel emitter ignores `files`
events entirely).
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

CHANNEL_ID = "chan-1"
CHANNEL_CHAT_ID = f"channel:{CHANNEL_ID}"
SAVED_CHAT_ID = "chat-1"
MESSAGE_ID = "msg-1"
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(b"fake-png").decode()


@pytest.fixture(scope="session")
def images_module(owui_module):
    return owui_module("open_webui.routers.images")


@pytest.fixture(scope="session")
def files_utils_module(owui_module):
    return owui_module("open_webui.utils.files")


@pytest.fixture(scope="session")
def middleware_module(owui_module):
    return owui_module("open_webui.utils.middleware")


@pytest.fixture(scope="session")
def builtin_module(owui_module):
    return owui_module("open_webui.tools.builtin")


@pytest.fixture(scope="session")
def socket_main(owui_module):
    return owui_module("open_webui.socket.main")


@pytest.fixture(scope="session")
def messages_module(owui_module):
    return owui_module("open_webui.models.messages")


def _user(role="user"):
    return SimpleNamespace(id="u-1", role=role, email="u@example.com", name="u")


def _request():
    app = SimpleNamespace(url_path_for=lambda name, id: f"/api/v1/files/{id}/content")
    return SimpleNamespace(app=app)


def _file_item(file_id="file-1", filename="generated-image.png", meta=None):
    return SimpleNamespace(id=file_id, filename=filename, meta=meta)


async def _upload(images_module, metadata, file_item=None, insert=None):
    """Drive the real `upload_image` with only the storage and DB boundary stubbed."""
    handler = AsyncMock(return_value=file_item or _file_item())
    with (
        patch.object(images_module, "upload_file_handler", handler),
        patch.object(images_module.Chats, "insert_chat_files", insert or AsyncMock()),
    ):
        return await images_module.upload_image(
            _request(), b"fake-png", "image/png", metadata, _user()
        )


# Narrow: `upload_image` hands back a file descriptor, not a bare URL.


@pytest.mark.asyncio
async def test_upload_image_returns_a_full_file_descriptor(images_module):
    meta = {"name": "cat.png", "content_type": "image/png"}
    _, descriptor = await _upload(images_module, {}, file_item=_file_item(meta=meta))

    assert descriptor == {
        "id": "file-1",
        "url": "/api/v1/files/file-1/content",
        "name": "cat.png",
        "content_type": "image/png",
    }, (
        "upload_image returned only a URL, so nothing downstream could register the "
        "generated file against a channel by id (aeb126b95)"
    )


@pytest.mark.asyncio
async def test_upload_image_falls_back_to_the_stored_filename(images_module):
    _, descriptor = await _upload(images_module, {}, file_item=_file_item(meta=None))

    assert isinstance(descriptor, dict), (
        f"upload_image returned a bare URL instead of a file descriptor (aeb126b95): {descriptor!r}"
    )
    assert descriptor["name"] == "generated-image.png"
    assert descriptor["content_type"] is None


# Narrow: a channel turn is routed by channel_id, and the emitted files carry the descriptor.


async def _run_image_handler(middleware_module, chat_id, images, editing=False):
    """Drive the real `chat_image_generation_handler` and collect the emitted events."""
    events = []

    async def emitter(event):
        events.append(event)

    generations = AsyncMock(return_value=images)
    edits = AsyncMock(return_value=images)
    config = SimpleNamespace(
        get=AsyncMock(side_effect=lambda key: key != "image_generation.prompt.enable")
    )
    form_data = {
        "model": "m-1",
        "messages": [{"role": "user", "content": "draw a cat"}],
    }
    if editing:
        form_data["messages"][0]["files"] = [{"type": "image", "url": PNG_DATA_URL}]

    extra_params = {
        "__metadata__": {"chat_id": chat_id, "message_id": MESSAGE_ID},
        "__event_emitter__": emitter,
    }
    with (
        patch.object(middleware_module, "image_generations", generations),
        patch.object(middleware_module, "image_edits", edits),
        patch.object(middleware_module, "Config", config),
    ):
        result = await middleware_module.chat_image_generation_handler(
            SimpleNamespace(), form_data, extra_params, _user()
        )

    return result, events, (edits if editing else generations)


def _descriptor(file_id="file-1"):
    return {
        "id": file_id,
        "url": f"/api/v1/files/{file_id}/content",
        "name": "cat.png",
        "content_type": "image/png",
    }


@pytest.mark.asyncio
async def test_channel_turn_is_routed_by_channel_id(middleware_module):
    _, _, call = await _run_image_handler(middleware_module, CHANNEL_CHAT_ID, [_descriptor()])

    metadata = call.await_args.kwargs["metadata"]
    assert metadata == {"message_id": MESSAGE_ID, "channel_id": CHANNEL_ID}, (
        "a channel image generation was filed under a chat_id of 'channel:<id>', which "
        f"is not a chat, so the file was never attached to the channel (aeb126b95): {metadata}"
    )


@pytest.mark.asyncio
async def test_channel_files_event_carries_the_file_descriptor(middleware_module):
    _, events, _ = await _run_image_handler(middleware_module, CHANNEL_CHAT_ID, [_descriptor()])

    files = [event for event in events if event["type"] == "files"]
    assert len(files) == 1
    assert files[0]["data"]["files"] == [{"type": "image", **_descriptor()}], (
        "the emitted file entry carried only a URL, so the channel emitter had no file "
        f"id to register on the channel (aeb126b95): {files[0]['data']['files']}"
    )


# Narrow: the built-in image tools route a channel the same way.


async def _run_builtin_image_tool(builtin_module, tool_name, chat_id, images, emit=False):
    """Drive the real built-in image tool and collect the router call and emitted events."""
    events = []

    async def emitter(event):
        events.append(event)

    call = AsyncMock(return_value=images)
    patched = "image_generations" if tool_name == "generate_image" else "image_edits"
    kwargs = {} if tool_name == "generate_image" else {"image_urls": [PNG_DATA_URL]}
    if emit:
        kwargs["__event_emitter__"] = emitter

    with patch.object(builtin_module, patched, call):
        await getattr(builtin_module, tool_name)(
            prompt="a cat",
            __request__=SimpleNamespace(),
            __user__=None,
            __chat_id__=chat_id,
            __message_id__=MESSAGE_ID,
            **kwargs,
        )

    assert call.await_count == 1, (
        "the built-in image tool swallowed an error instead of calling the router"
    )
    return events, call


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["generate_image", "edit_image"])
async def test_builtin_image_tools_route_a_channel_by_channel_id(builtin_module, tool_name):
    _, call = await _run_builtin_image_tool(
        builtin_module, tool_name, CHANNEL_CHAT_ID, [_descriptor()]
    )

    metadata = call.await_args.kwargs.get("metadata")
    assert metadata == {"channel_id": CHANNEL_ID, "message_id": MESSAGE_ID}, (
        "a built-in image tool run inside a channel passed no channel metadata, so the "
        f"generated file was never attached to the channel (aeb126b95): {metadata}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["generate_image", "edit_image"])
async def test_builtin_image_tools_emit_file_entries_that_keep_their_id(builtin_module, tool_name):
    descriptors = [_descriptor("file-1"), _descriptor("file-2")]

    events, _ = await _run_builtin_image_tool(
        builtin_module, tool_name, CHANNEL_CHAT_ID, descriptors, emit=True
    )

    emitted = [event for event in events if event["type"] == "chat:message:files"]
    assert len(emitted) == 1
    files = emitted[0]["data"]["files"]
    assert [entry.get("id") for entry in files] == ["file-1", "file-2"], (
        "a built-in image tool emitted file entries carrying only a URL, so the channel "
        f"emitter had no file id to register on the channel (aeb126b95): {files}"
    )
    assert all(entry["type"] == "image" for entry in files)
    assert all(entry["name"] == "cat.png" for entry in files)


# Narrow: the channel emitter learned to handle files events.


class FakeMessage:
    def __init__(self, content="here you go", data=None):
        self.id = MESSAGE_ID
        self.channel_id = CHANNEL_ID
        self.user_id = "u-1"
        self.content = content
        self.data = data
        self.meta = None

    def model_dump(self):
        return {
            "id": self.id,
            "channel_id": self.channel_id,
            "content": self.content,
            "data": self.data,
        }


async def _emit_to_channel(socket_main, messages_module, event, message=None):
    """Drive the real channel emitter and record the channel and message writes."""
    msg = message or FakeMessage()
    updates = []
    channel_calls = []

    async def update_message_by_id(message_id, form, *args, **kwargs):
        updates.append((message_id, form))
        msg.data = form.data
        return msg

    channels = SimpleNamespace(
        add_file_to_channel_by_id=AsyncMock(
            side_effect=lambda *a: channel_calls.append(("add", *a))
        ),
        set_file_message_id_in_channel_by_id=AsyncMock(
            side_effect=lambda *a: channel_calls.append(("set", *a))
        ),
    )
    with (
        patch.object(messages_module.Messages, "get_message_by_id", AsyncMock(return_value=msg)),
        patch.object(messages_module.Messages, "update_message_by_id", update_message_by_id),
        patch.object(socket_main, "Channels", channels),
        patch.object(socket_main.sio, "emit", AsyncMock()),
    ):
        emitter = await socket_main._make_channel_emitter(
            {"chat_id": CHANNEL_CHAT_ID, "message_id": MESSAGE_ID}
        )
        await emitter(event)

    return updates, channel_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["files", "chat:message:files"])
async def test_channel_emitter_registers_generated_files_on_the_channel(
    socket_main, messages_module, event_type
):
    event = {"type": event_type, "data": {"files": [{"type": "image", **_descriptor()}]}}

    updates, channel_calls = await _emit_to_channel(socket_main, messages_module, event)

    assert channel_calls == [
        ("add", CHANNEL_ID, "file-1", "u-1"),
        ("set", CHANNEL_ID, "file-1", MESSAGE_ID),
    ], (
        "a generated image was never registered on the channel, so it did not appear in "
        f"the channel's file list (aeb126b95): {channel_calls}"
    )
    assert len(updates) == 1, (
        f"the channel message was never updated with the generated file (aeb126b95): {updates}"
    )
    assert updates[0][1].data == {"files": [{"type": "image", **_descriptor(), "url": "file-1"}]}
    assert updates[0][1].content == "here you go", "the files update overwrote the message content"


@pytest.mark.asyncio
async def test_channel_emitter_keeps_files_already_on_the_message(socket_main, messages_module):
    existing = {"type": "image", "id": "file-0", "url": "file-0"}
    message = FakeMessage(data={"files": [existing]})
    event = {"type": "files", "data": {"files": [{"type": "image", **_descriptor()}]}}

    updates, _ = await _emit_to_channel(socket_main, messages_module, event, message=message)

    assert len(updates) == 1
    assert existing in updates[0][1].data["files"], (
        "a second generated image replaced the first instead of being appended (aeb126b95)"
    )
    assert len(updates[0][1].data["files"]) == 2


# Broad: every path that turns a generated image into a file entry keeps the id, and no
# path leaks the 'channel:' prefix into a chat_id.


@pytest.mark.asyncio
@pytest.mark.parametrize("editing", [False, True])
async def test_no_image_path_files_a_channel_turn_under_a_chat_id(middleware_module, editing):
    _, _, call = await _run_image_handler(
        middleware_module, CHANNEL_CHAT_ID, [_descriptor()], editing=editing
    )

    metadata = call.await_args.kwargs["metadata"]
    assert "chat_id" not in metadata
    assert metadata["channel_id"] == CHANNEL_ID


@pytest.mark.asyncio
@pytest.mark.parametrize("editing", [False, True])
async def test_every_emitted_image_entry_keeps_its_file_id(middleware_module, editing):
    descriptors = [_descriptor("file-1"), _descriptor("file-2")]

    _, events, _ = await _run_image_handler(
        middleware_module, CHANNEL_CHAT_ID, descriptors, editing=editing
    )

    emitted = [event for event in events if event["type"] == "files"][0]["data"]["files"]
    assert [entry.get("id") for entry in emitted] == ["file-1", "file-2"]
    assert all(entry["type"] == "image" for entry in emitted)


# Nearby: the saved-chat path and the URL-only consumers are unchanged.


@pytest.mark.asyncio
@pytest.mark.parametrize("editing", [False, True])
async def test_a_saved_chat_turn_is_still_routed_by_chat_id(middleware_module, editing):
    message = {"id": MESSAGE_ID, "role": "user", "content": "draw a cat", "parentId": None}
    if editing:
        message["files"] = [{"type": "image", "url": PNG_DATA_URL}]
    chat = SimpleNamespace(
        chat={"history": {"currentId": MESSAGE_ID, "messages": {MESSAGE_ID: message}}}
    )
    with patch.object(
        middleware_module.Chats, "get_chat_by_id_and_user_id", AsyncMock(return_value=chat)
    ):
        _, _, call = await _run_image_handler(
            middleware_module, SAVED_CHAT_ID, [_descriptor()], editing=editing
        )

    metadata = call.await_args.kwargs["metadata"]
    assert metadata["chat_id"] == SAVED_CHAT_ID
    assert metadata["message_id"] == MESSAGE_ID
    assert "channel_id" not in metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["generate_image", "edit_image"])
async def test_builtin_image_tools_send_no_channel_metadata_for_a_saved_chat(
    builtin_module, tool_name
):
    with patch.object(
        builtin_module.Chats,
        "add_message_files_by_id_and_message_id",
        AsyncMock(return_value=None),
    ):
        _, call = await _run_builtin_image_tool(
            builtin_module, tool_name, SAVED_CHAT_ID, [_descriptor()]
        )

    assert call.await_args.kwargs.get("metadata") is None


@pytest.mark.asyncio
async def test_base64_image_helper_still_returns_a_plain_url(files_utils_module, images_module):
    """`get_image_url_from_base64` feeds markdown image rewriting, so it must keep
    returning the URL string and not the new descriptor."""
    handler = AsyncMock(return_value=_file_item())
    with (
        patch.object(images_module, "upload_file_handler", handler),
        patch.object(images_module.Chats, "insert_chat_files", AsyncMock()),
    ):
        url = await files_utils_module.get_image_url_from_base64(
            _request(), PNG_DATA_URL, {}, _user()
        )

    assert url == "/api/v1/files/file-1/content"


@pytest.mark.asyncio
async def test_base64_image_helper_ignores_a_non_image_string(files_utils_module):
    assert (
        await files_utils_module.get_image_url_from_base64(_request(), "not an image", {}, _user())
        is None
    )


@pytest.mark.asyncio
async def test_upload_image_still_links_a_saved_chat_message(images_module):
    insert = AsyncMock()
    await _upload(
        images_module, {"chat_id": SAVED_CHAT_ID, "message_id": MESSAGE_ID}, insert=insert
    )

    assert insert.await_count == 1
    assert insert.await_args.kwargs["chat_id"] == SAVED_CHAT_ID
    assert insert.await_args.kwargs["file_ids"] == ["file-1"]


@pytest.mark.asyncio
async def test_upload_image_does_not_link_a_channel_turn_to_a_chat(images_module):
    insert = AsyncMock()
    await _upload(
        images_module, {"channel_id": CHANNEL_ID, "message_id": MESSAGE_ID}, insert=insert
    )

    assert insert.await_count == 0


@pytest.mark.asyncio
async def test_upload_image_rejects_missing_image_data(images_module):
    with pytest.raises(ValueError):
        await images_module.upload_image(_request(), None, None, {}, _user())


@pytest.mark.asyncio
async def test_channel_emitter_ignores_an_empty_files_event(socket_main, messages_module):
    updates, channel_calls = await _emit_to_channel(
        socket_main, messages_module, {"type": "files", "data": {"files": []}}
    )

    assert updates == []
    assert channel_calls == []


@pytest.mark.asyncio
async def test_channel_emitter_still_streams_completion_content(socket_main, messages_module):
    event = {"type": "chat:completion", "data": {"content": "hello", "done": True}}

    updates, channel_calls = await _emit_to_channel(socket_main, messages_module, event)

    assert len(updates) == 1
    assert updates[0][1].content == "hello"
    assert updates[0][1].meta == {"done": True}
    assert channel_calls == []
