"""Regression tests for four 0.11.0 message-text extraction fixes.

Four independent defects, all in how the backend pulls usable text (and
images) out of a stored assistant turn:

1. Reply text missing (#26799, issue #26436). `get_content_from_message`
   in `utils/misc.py` only read `message['content']`. An assistant reply
   produced by the structured-output path stores its text under
   `message['output']` and leaves `content` empty, so copying, exporting,
   searching and re-using such a chat returned nothing. 0.11.0 adds
   `get_output_text()` and falls back to it, and `reconcile_tool_pairs`
   switched to `get_content_from_message(message) or ''` so such a message
   is no longer dropped as an empty orphan.

2. Memories from structured replies (commit 3fe0358, #26705, issue
   #26651, `utils/memory.py`). `review_memory_after_turn` and
   `_review_memory` read the raw `content` and returned early when it was
   not a non-empty str, so a structured-output turn skipped the whole
   background memory review. Both now go through
   `get_content_from_message`.

3. Tool images (commit dd86b98, `convert_output_to_messages`). Images a
   tool returned were emitted as a multimodal `role='tool'` message,
   which OpenAI-compatible providers reject or ignore. The new
   `flatten_tool_images=True` keyword makes the tool message text-only
   and re-emits the images as a following `role='user'` message. 0.11.1
   (commit 7d99b2716, tool approval) additionally requires the
   `function_call` item to carry a terminal `status` and a matching
   output before it is replayed at all, so the fixtures here spell that
   status out; 0.11.0 and earlier ignore the field.

4. Unusual characters (commit 43e7eef, #27201, issue #27081,
   `sanitize_text_for_db` and friends). The pre-fix fast path returned
   early whenever there was no null byte, so a lone UTF-16 surrogate
   survived and the chat failed to encode on save and then failed to
   load. 0.11.0 adds `SURROGATE_RE`, strips surrogates in dict keys too,
   and forces the recursive walk when the serialized form will not encode
   as UTF-8.

Discriminates: passes on v0.11.0 and v0.11.1, fails on v0.10.2 (no `get_output_text`,
so structured-output text is dropped everywhere and the memory review is
skipped for such a turn; no `flatten_tool_images` keyword; surrogates
survive into the value handed to the database).
"""

from __future__ import annotations

import asyncio
import inspect
import json
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.regression

LONE_SURROGATE = chr(0xD800)
LOW_SURROGATE = chr(0xDFFF)
NUL = chr(0)
EMOJI = chr(0x1F600)


@pytest.fixture(scope="session")
def memory_module(owui_module) -> ModuleType:
    """`open_webui.utils.memory` (review_memory_after_turn, _review_memory)."""
    return owui_module("open_webui.utils.memory")


def _structured_reply(text: str) -> dict:
    """An assistant turn whose text lives only in `output`."""
    return {
        "role": "assistant",
        "content": None,
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


# -----------------------------------------------------------------------------
# 1. Response text where it was missing (#26799, issue #26436)
# -----------------------------------------------------------------------------


def test_structured_output_text_is_returned(misc_module: ModuleType) -> None:
    """NARROW #26436: text stored under `output` must be readable."""
    message = _structured_reply("The capital of France is Paris.")
    assert misc_module.get_content_from_message(message) == "The capital of France is Paris."


def test_multiple_output_message_items_are_joined(misc_module: ModuleType) -> None:
    """NARROW #26436: several `message` items join with newlines, blanks and
    non-message items dropped."""
    message = {
        "role": "assistant",
        "content": "",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "first"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "   "}]},
            {"type": "reasoning", "content": [{"type": "output_text", "text": "hidden"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "second"}]},
        ],
    }
    assert misc_module.get_content_from_message(message) == "first\nsecond"


def test_structured_output_assistant_survives_orphan_tool_calls(misc_module: ModuleType) -> None:
    """NARROW #26436: an all-orphan assistant turn whose text is in `output`
    must be kept, not dropped as empty."""
    message = {
        **_structured_reply("Here is what I found."),
        "tool_calls": [{"id": "call_orphan", "type": "function", "function": {"name": "s"}}],
    }
    reconciled = misc_module.reconcile_tool_pairs([{"role": "user", "content": "hi"}, message])

    assistants = [m for m in reconciled if m.get("role") == "assistant"]
    assert len(assistants) == 1, "structured-output assistant turn was dropped"
    assert "tool_calls" not in assistants[0]
    assert misc_module.get_content_from_message(assistants[0]) == "Here is what I found."


def test_broad_every_message_shape_yields_its_text(misc_module: ModuleType) -> None:
    """BROAD: whichever field a turn parks its text in, one accessor finds it."""
    shapes = [
        ({"role": "assistant", "content": "plain"}, "plain"),
        ({"role": "user", "content": [{"type": "text", "text": "parts"}]}, "parts"),
        (_structured_reply("structured"), "structured"),
    ]
    for message, expected in shapes:
        assert misc_module.get_content_from_message(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": None, "output": None},
        {"role": "assistant", "content": None, "output": []},
        {"role": "assistant", "content": None, "output": "not-a-list"},
        {"role": "assistant", "content": None, "output": ["bare-string", 7]},
        {"role": "assistant", "content": None, "output": [{"type": "message", "content": None}]},
        {"role": "assistant", "content": None},
        {"role": "assistant"},
    ],
)
def test_nearby_empty_shapes_are_falsy_and_do_not_raise(
    misc_module: ModuleType, message: dict
) -> None:
    """NEARBY: nothing to extract returns a falsy value rather than raising."""
    assert not misc_module.get_content_from_message(message)


def test_nearby_content_wins_over_output(misc_module: ModuleType) -> None:
    """NEARBY: a normal reply is unaffected by the new fallback."""
    message = {**_structured_reply("from output"), "content": "from content"}
    assert misc_module.get_content_from_message(message) == "from content"


def test_nearby_well_formed_tool_pairs_untouched(misc_module: ModuleType) -> None:
    """NEARBY: reconciliation still leaves a matched pair alone."""
    messages = [
        {
            "role": "assistant",
            "content": "looking",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "s"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]
    assert misc_module.reconcile_tool_pairs(messages) == messages


# -----------------------------------------------------------------------------
# 2. Memories from structured replies (commit 3fe0358, #26705, issue #26651)
# -----------------------------------------------------------------------------


class _FakeConfig:
    """Stand-in for the config store read by review_memory_after_turn."""

    @staticmethod
    async def get_many(*keys: str) -> dict:
        return {
            "memories.background_review.enable": True,
            "memories.review_interval_turns": 1,
        }


@pytest.mark.asyncio
async def test_structured_reply_triggers_memory_review(
    memory_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NARROW issue #26651: the review must run for a reply whose text is
    only in `output`."""
    started = asyncio.Event()

    async def fake_review(**kwargs) -> None:
        started.set()

    monkeypatch.setattr(memory_module, "Config", _FakeConfig)
    monkeypatch.setattr(memory_module, "_review_memory", fake_review)

    await memory_module.review_memory_after_turn(
        request=SimpleNamespace(),
        user=SimpleNamespace(id="u1"),
        model={"id": "m"},
        metadata={"features": {"memory": True}},
        form_data={},
        assistant_message=_structured_reply("I moved to Vienna last year."),
        messages=[{"role": "user", "content": "where do I live now?"}],
    )

    try:
        await asyncio.wait_for(started.wait(), timeout=5)
    except asyncio.TimeoutError:
        pytest.fail("memory review was skipped for a structured-output assistant reply")


@pytest.mark.asyncio
async def test_structured_reply_text_reaches_the_review_prompt(
    memory_module: ModuleType, owui_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NARROW issue #26651: `_review_memory` must put the structured reply's
    text into the transcript it sends to the model."""
    chat_module = owui_module("open_webui.utils.chat")
    captured: list[str] = []

    async def fake_generate_chat_completion(request, form_data, user=None, **kwargs) -> dict:
        captured.append(form_data["messages"][-1]["content"])
        return {"choices": [{"message": {"content": '{"operations": []}'}}]}

    async def fake_get_memories(user_id: str) -> list:
        return []

    monkeypatch.setattr(chat_module, "generate_chat_completion", fake_generate_chat_completion)
    monkeypatch.setattr(memory_module.Memories, "get_memories_by_user_id", fake_get_memories)

    await asyncio.wait_for(
        memory_module._review_memory(
            request=SimpleNamespace(),
            user=SimpleNamespace(id="u1"),
            model={"id": "m"},
            metadata={"chat_id": "c", "message_id": "msg"},
            form_data={},
            assistant_message=_structured_reply("I moved to Vienna last year."),
            messages=[{"role": "user", "content": "where do I live now?"}],
        ),
        timeout=10,
    )

    assert captured, "_review_memory never reached the model call"
    assert "I moved to Vienna last year." in captured[0]


@pytest.mark.asyncio
async def test_nearby_memory_review_skipped_when_feature_off(
    memory_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEARBY: the fallback did not make the review unconditional."""
    started = asyncio.Event()

    async def fake_review(**kwargs) -> None:
        started.set()

    monkeypatch.setattr(memory_module, "Config", _FakeConfig)
    monkeypatch.setattr(memory_module, "_review_memory", fake_review)

    await memory_module.review_memory_after_turn(
        request=SimpleNamespace(),
        user=SimpleNamespace(id="u1"),
        model={"id": "m"},
        metadata={"features": {"memory": False}},
        form_data={},
        assistant_message=_structured_reply("something durable"),
        messages=[{"role": "user", "content": "hi"}],
    )

    await asyncio.sleep(0)
    assert not started.is_set()


# -----------------------------------------------------------------------------
# 3. Images returned by a tool (commit dd86b98)
# -----------------------------------------------------------------------------

_IMAGE_URL = "data:image/png;base64,iVBORw0KGgo="


def _filters_unapproved_tool_calls(misc_module: ModuleType) -> bool:
    """Whether the ref only replays a function_call that reached a terminal status
    (0.11.1's tool-approval gate, commit 7d99b2716)."""
    return "completed_call_ids" in inspect.getsource(misc_module.convert_output_to_messages)


def _tool_image_output(call_id: str = "call_img", status: str = "completed") -> list[dict]:
    return [
        {
            "type": "function_call",
            "call_id": call_id,
            "name": "render",
            "arguments": "{}",
            "status": status,
        },
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": [
                {"type": "input_text", "text": "chart rendered"},
                {"type": "input_image", "image_url": _IMAGE_URL},
            ],
        },
    ]


def test_flatten_moves_tool_images_into_a_user_message(misc_module: ModuleType) -> None:
    """NARROW dd86b98: with flattening on, the tool message is text-only and
    the images arrive as a following user message."""
    messages = misc_module.convert_output_to_messages(
        _tool_image_output(), flatten_tool_images=True
    )

    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "chart rendered"

    tool_index = messages.index(tool_messages[0])
    following_users = [m for m in messages[tool_index + 1 :] if m.get("role") == "user"]
    assert following_users, "tool images were dropped instead of re-emitted"

    parts = following_users[0]["content"]
    assert parts[0]["type"] == "text"
    assert [p for p in parts if p.get("type") == "image_url"] == [
        {"type": "image_url", "image_url": {"url": _IMAGE_URL}}
    ]


def test_flatten_emits_images_before_the_next_assistant_turn(misc_module: ModuleType) -> None:
    """NARROW dd86b98: buffered images flush ahead of the next item, so the
    assistant reply that follows the tool call stays after them."""
    output = [
        *_tool_image_output(),
        {"type": "message", "content": [{"type": "output_text", "text": "here it is"}]},
    ]
    messages = misc_module.convert_output_to_messages(output, flatten_tool_images=True)

    assert [m["role"] for m in messages] == ["assistant", "tool", "user", "assistant"]
    assert messages[-1]["content"] == "here it is"
    assert any(p.get("type") == "image_url" for p in messages[2]["content"])


def test_flatten_is_a_no_op_for_imageless_tool_output(misc_module: ModuleType) -> None:
    """NARROW dd86b98: flattening changes nothing when no tool image exists."""
    output = [
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "s",
            "arguments": "{}",
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": [{"type": "input_text", "text": "plain"}],
        },
    ]
    both = misc_module.convert_output_to_messages(output, flatten_tool_images=True)
    assert [m["role"] for m in both] == ["assistant", "tool"]
    assert both == misc_module.convert_output_to_messages(output)


def test_nearby_flatten_off_keeps_legacy_multimodal_tool_message(
    misc_module: ModuleType,
) -> None:
    """NEARBY: the default is unchanged, so the Responses path still sees the
    multimodal tool message it expects."""
    messages = misc_module.convert_output_to_messages(_tool_image_output())

    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == [
        {"type": "input_text", "text": "chart rendered"},
        {"type": "input_image", "image_url": _IMAGE_URL},
    ]
    assert not [m for m in messages if m.get("role") == "user"]


def test_nearby_tool_call_awaiting_approval_is_not_replayed(misc_module: ModuleType) -> None:
    """NEARBY 0.11.1 (7d99b2716): a call that never reached a terminal status is held
    back entirely, so an unapproved tool never reaches the model."""
    if not _filters_unapproved_tool_calls(misc_module):
        pytest.skip("ref predates tool approval, every function_call is replayed")

    assert (
        misc_module.convert_output_to_messages(
            _tool_image_output(status="pending"), flatten_tool_images=True
        )
        == []
    )


# -----------------------------------------------------------------------------
# 4. Chats containing unusual characters (commit 43e7eef, #27201, issue #27081)
# -----------------------------------------------------------------------------


def test_lone_surrogate_is_stripped_from_text(misc_module: ModuleType) -> None:
    """NARROW issue #27081: a surrogate with no null byte alongside it must
    still be removed."""
    assert misc_module.sanitize_text_for_db(f"a{LONE_SURROGATE}b") == "ab"


def test_sanitized_structure_round_trips_to_utf8(misc_module: ModuleType) -> None:
    """NARROW issue #27081: keys and values both get cleaned, and the result
    is storable."""
    cleaned = misc_module.sanitize_data_for_db(
        {f"k{LONE_SURROGATE}": [f"v{LONE_SURROGATE}", {"nested": f"n{LONE_SURROGATE}"}]}
    )

    assert cleaned == {"k": ["v", {"nested": "n"}]}
    json.dumps(cleaned, ensure_ascii=False).encode("utf-8")


def test_broad_no_surrogate_survives_any_sanitized_shape(misc_module: ModuleType) -> None:
    """BROAD: whatever the container, nothing the sanitizer returns can fail to
    encode as UTF-8, and no null byte is left behind."""
    shapes = [
        f"top{LONE_SURROGATE}level",
        [f"a{LONE_SURROGATE}", f"b{NUL}{LONE_SURROGATE}"],
        {f"key{LONE_SURROGATE}": f"value{LONE_SURROGATE}"},
        {"messages": [{"content": f"hi {LOW_SURROGATE} there", "role": f"user{NUL}"}]},
        [{"a": [{"b": [f"deep{LONE_SURROGATE}"]}]}],
    ]
    for shape in shapes:
        cleaned = misc_module.sanitize_data_for_db(shape)
        serialized = json.dumps(cleaned, ensure_ascii=False)
        serialized.encode("utf-8")
        assert NUL not in serialized


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("plain text", "plain text"),
        ("", ""),
        (f"emoji {EMOJI} and cafe", f"emoji {EMOJI} and cafe"),
        (f"nul{NUL}byte", "nulbyte"),
        (None, None),
        (42, 42),
    ],
)
def test_nearby_sanitize_text_leaves_valid_input_alone(
    misc_module: ModuleType, value, expected
) -> None:
    """NEARBY: the widened fast path did not start mangling normal text."""
    assert misc_module.sanitize_text_for_db(value) == expected


def test_nearby_clean_structure_is_returned_unchanged(misc_module: ModuleType) -> None:
    """NEARBY: the fast path still short-circuits, returning the same object."""
    payload = {"messages": [{"role": "user", "content": f"hello {EMOJI}"}]}
    assert misc_module.sanitize_data_for_db(payload) is payload


def test_nearby_unserializable_structure_still_walks(misc_module: ModuleType) -> None:
    """NEARBY: a value json.dumps cannot handle falls through to the recursive
    walk rather than raising."""
    cleaned = misc_module.sanitize_data_for_db({"fn": object, "text": f"x{NUL}y"})
    assert cleaned["text"] == "xy"
