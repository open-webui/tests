"""Regression: base64 images inside tool results must be attached, not serialised.

open-webui 0.11.4, PR #29665 (issue #29208), two commits in
`open_webui/utils/middleware.py`:

* `afda09454` — when a tool's result was exactly one image data URI the backend
  already moved it out of the model's context, but an image tucked inside a
  returned object or list was serialised into the tool message as raw base64
  text, so one screenshot could cost hundreds of thousands of tokens and push
  the rest of the conversation out. `extract_base64_images` now walks the whole
  result structure, moves every string that is entirely one image data URI into
  the result's files and leaves a short `[image]` marker behind. Detection is
  deliberately limited to values that are entirely a data URI. The OpenAPI branch
  also stopped removing entries from the list it was iterating, which had
  skipped every second `data:` entry.
* `d372bec70` — a tool image in a saved chat is now written to a file and
  referred to by URL instead of sitting inline in the chat JSON
  (`store_tool_result_image`), falling back to the inline data URL whenever
  storage is not applicable or fails.

These tests pin the backend extraction and storing contract only (what the
model context and the emitted file list look like), not the rendering.

Discriminates: passes on dev 344ea5306, fails on `afda09454^` and on
`d372bec70^` (no structure walk, so nested images reach the serialized tool
message as base64 text; and no `store_tool_result_image`, so a saved-chat tool
image stays inline).
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression

PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(b"fake-png").decode()
JPEG_DATA_URL = "data:image/jpeg;base64," + base64.b64encode(b"fake-jpeg").decode()
PDF_DATA_URL = "data:application/pdf;base64," + base64.b64encode(b"fake-pdf").decode()
SAVED_CHAT_ID = "chat-1"
TEMPORARY_CHAT_ID = "temporary:abc"
CHANNEL_CHAT_ID = "channel:chan-1"


@pytest.fixture(scope="session")
def middleware_module(owui_module):
    return owui_module("open_webui.utils.middleware")


def _request():
    return SimpleNamespace()


def _metadata(chat_id=SAVED_CHAT_ID):
    return {
        "chat_id": chat_id,
        "message_id": "msg-1",
        "session_id": "sess-1",
    }


def _user():
    return SimpleNamespace(id="u-1", role="user")


# ---------------------------------------------------------------------------
# narrow: extraction from nested structures (afda09454)
# ---------------------------------------------------------------------------


def test_an_image_inside_a_dict_is_moved_to_files(middleware_module):
    files: list = []
    result = middleware_module.extract_base64_images({"chart": PNG_DATA_URL, "note": "hi"}, files)

    assert result == {"chart": "[image]", "note": "hi"}
    assert files == [{"type": "image", "url": PNG_DATA_URL}]


def test_an_image_inside_a_list_is_moved_to_files(middleware_module):
    files: list = []
    result = middleware_module.extract_base64_images(["intro", PNG_DATA_URL], files)

    assert result == ["intro", "[image]"]
    assert files == [{"type": "image", "url": PNG_DATA_URL}]


def test_images_nested_deep_are_moved_to_files(middleware_module):
    files: list = []
    result = middleware_module.extract_base64_images(
        {"frames": ({"shot": PNG_DATA_URL}, {"shot": JPEG_DATA_URL})}, files
    )

    assert result == {"frames": ({"shot": "[image]"}, {"shot": "[image]"})}
    assert files == [
        {"type": "image", "url": PNG_DATA_URL},
        {"type": "image", "url": JPEG_DATA_URL},
    ]


def test_adjacent_images_in_an_openapi_list_are_all_moved(middleware_module):
    """Pins the iterate-while-removing fix: every second entry used to be skipped."""
    files: list = []
    result = middleware_module.extract_base64_images([PNG_DATA_URL, JPEG_DATA_URL], files)

    assert result == ["[image]", "[image]"]
    assert len(files) == 2


# ---------------------------------------------------------------------------
# broad: the invariant the bug was an instance of
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_tool_result_attaches_a_nested_image(middleware_module):
    tool_result, files, embeds = await middleware_module.process_tool_result(
        _request(),
        "render_chart",
        {"chart": PNG_DATA_URL, "summary": "chart of sales"},
        "openapi",
        False,
        _metadata(),
        _user(),
    )

    assert PNG_DATA_URL not in tool_result, "the base64 image reached the model context as text"
    assert "[image]" in tool_result
    assert files == [{"type": "image", "url": PNG_DATA_URL}]
    assert embeds == []


@pytest.mark.asyncio
async def test_process_tool_result_moves_a_bare_image_with_a_summary(middleware_module):
    tool_result, files, _ = await middleware_module.process_tool_result(
        _request(), "take_screenshot", PNG_DATA_URL, "openapi", False, _metadata(), _user()
    )

    assert tool_result == "take_screenshot: Image file read successfully."
    assert files == [{"type": "image", "url": PNG_DATA_URL}]


@pytest.mark.asyncio
async def test_process_tool_result_files_both_data_types_in_one_list(middleware_module):
    tool_result, files, _ = await middleware_module.process_tool_result(
        _request(),
        "mixed_output",
        [PNG_DATA_URL, PDF_DATA_URL, "plain text"],
        "openapi",
        False,
        _metadata(),
        _user(),
    )

    image_urls = [entry["url"] for entry in files if entry["type"] == "image"]
    data_entries = [entry for entry in files if entry["type"] == "data"]
    assert image_urls == [PNG_DATA_URL], "the image must be attached as an image, not as data"
    assert data_entries == [{"type": "data", "content": PDF_DATA_URL}]
    assert PNG_DATA_URL not in tool_result
    assert PDF_DATA_URL not in tool_result


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_type", ["external", "action", "terminal"])
async def test_process_tool_result_extraction_applies_across_tool_types(
    middleware_module, tool_type
):
    tool_result, files, _ = await middleware_module.process_tool_result(
        _request(),
        "make_graph",
        {"graph": PNG_DATA_URL},
        tool_type,
        False,
        _metadata(),
        _user(),
    )

    assert files == [{"type": "image", "url": PNG_DATA_URL}]
    assert "[image]" in tool_result


# ---------------------------------------------------------------------------
# narrow: storing in a saved chat (d372bec70)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_saved_chat_tool_image_is_stored_and_referred_to(middleware_module, monkeypatch):
    async def fake_store(request, image_url, metadata, user):
        assert image_url == PNG_DATA_URL
        assert metadata == {"chat_id": SAVED_CHAT_ID, "message_id": "msg-1", "session_id": "sess-1"}
        return "/api/v1/files/file-1/content"

    monkeypatch.setattr(middleware_module, "get_file_url_from_base64", fake_store)

    stored = await middleware_module.store_tool_result_image(
        _request(), PNG_DATA_URL, _metadata(), _user()
    )

    assert stored == "/api/v1/files/file-1/content"


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_id", [TEMPORARY_CHAT_ID, CHANNEL_CHAT_ID, None])
async def test_a_non_saved_chat_keeps_the_image_inline(middleware_module, monkeypatch, chat_id):
    async def unexpected_store(request, image_url, metadata, user):
        raise AssertionError("a temporary or channel chat must not write files")

    monkeypatch.setattr(middleware_module, "get_file_url_from_base64", unexpected_store)

    inline = await middleware_module.store_tool_result_image(
        _request(), PNG_DATA_URL, _metadata(chat_id), _user()
    )

    assert inline == PNG_DATA_URL


@pytest.mark.asyncio
async def test_a_storage_failure_falls_back_to_the_inline_image(middleware_module, monkeypatch):
    async def failing_store(request, image_url, metadata, user):
        raise RuntimeError("disk full")

    monkeypatch.setattr(middleware_module, "get_file_url_from_base64", failing_store)

    inline = await middleware_module.store_tool_result_image(
        _request(), PNG_DATA_URL, _metadata(), _user()
    )

    assert inline == PNG_DATA_URL


@pytest.mark.asyncio
async def test_an_empty_stored_url_falls_back_to_the_inline_image(
    middleware_module, monkeypatch
):
    async def empty_store(request, image_url, metadata, user):
        return None

    monkeypatch.setattr(middleware_module, "get_file_url_from_base64", empty_store)

    inline = await middleware_module.store_tool_result_image(
        _request(), PNG_DATA_URL, _metadata(), _user()
    )

    assert inline == PNG_DATA_URL


# ---------------------------------------------------------------------------
# nearby: behaviour that was already correct, and the deliberate limits
# ---------------------------------------------------------------------------


def test_plain_text_is_untouched(middleware_module):
    files: list = []
    result = middleware_module.extract_base64_images("just words", files)

    assert result == "just words"
    assert files == []


def test_a_non_image_data_uri_is_not_extracted_as_an_image(middleware_module):
    files: list = []
    result = middleware_module.extract_base64_images({"doc": PDF_DATA_URL}, files)

    assert result == {"doc": PDF_DATA_URL}
    assert files == []


def test_a_data_uri_wrapped_in_text_is_left_in_place(middleware_module):
    """Deliberate limit: scanning inside longer strings truncates payloads."""
    files: list = []
    value = f"before {PNG_DATA_URL} after"
    result = middleware_module.extract_base64_images(value, files)

    assert result == value
    assert files == []


def test_a_non_string_value_is_untouched(middleware_module):
    files: list = []
    result = middleware_module.extract_base64_images({"count": 3, "ok": True, "none": None}, files)

    assert result == {"count": 3, "ok": True, "none": None}
    assert files == []


@pytest.mark.asyncio
async def test_a_non_image_tool_result_passes_through_unchanged(middleware_module):
    tool_result, files, embeds = await middleware_module.process_tool_result(
        _request(),
        "add_numbers",
        {"answer": 42},
        "openapi",
        False,
        _metadata(),
        _user(),
    )

    assert '"answer": 42' in tool_result
    assert files == []
    assert embeds == []
