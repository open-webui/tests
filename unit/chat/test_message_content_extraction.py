"""Stored-turn text for the memory review, and the extraction paths no HTTP request reaches.

Regressions from 0.11.0 in how the backend pulls text out of a stored assistant turn:

1. Memories from structured replies (commit 3fe0358, #26705, issue #26651, `utils/memory.py`).
   `review_memory_after_turn` and `_review_memory` read the raw `content` and gave up when it
   was not a non-empty string, so a reply whose text lives only in `output` skipped the
   background memory review. Both now go through `get_content_from_message`. A live reply with
   its text only in `output` needs a provider speaking the Responses API, so this stays here.
2. `reconcile_tool_pairs` (#26799) keeps an assistant turn whose tool calls were all orphaned
   when its text is in `output`; the replay path only feeds it messages rebuilt from `output`,
   which carry their text in `content`, so only a direct call shows it.
3. `convert_output_to_messages` without `flatten_tool_images` keeps the multimodal tool message
   the Responses path expects, and the sanitizer's edges no JSON request body can carry.

The task-prompt text, tool-image and surrogate fixes are pinned over HTTP in
integration/chat/test_message_content_extraction.py.

Discriminates: passes on dev bbfa876af; reverting 3fe0358 fails the memory review test, and
reverting the `output` fallback in `get_content_from_message` fails it and the orphan-tool test.
The remaining nearby tests pass on both.
"""

from __future__ import annotations

import asyncio
from types import ModuleType
from unittest.mock import AsyncMock, patch

import pytest
from starlette.requests import Request

pytestmark = pytest.mark.regression

NUL = chr(0)
IMAGE_URL = "data:image/png;base64,iVBORw0KGgo="
NO_OPERATIONS = {"choices": [{"message": {"content": '{"operations": []}'}}]}
REVIEW_ENABLED = {
    "memories.enable": True,
    "memories.background_review.enable": True,
    "memories.review_interval_turns": 1,
}


@pytest.fixture(scope="session")
def memory_module(owui_module) -> ModuleType:
    return owui_module("open_webui.utils.memory")


@pytest.fixture(scope="session")
def reviewer(owui_module):
    return owui_module("open_webui.models.users").UserModel(
        id="memory-review-user",
        email="reviewer@example.com",
        name="Reviewer",
        role="admin",
        last_active_at=0,
        updated_at=0,
        created_at=0,
    )


def structured_reply(text: str) -> dict:
    """An assistant turn whose text lives only in `output`."""
    output = [{"type": "message", "content": [{"type": "output_text", "text": text}]}]
    return {"role": "assistant", "content": None, "output": output}


async def review_prompts(memory_module, owui_module, user, *, memory_feature: bool) -> list[str]:
    """Run the public review entry point; return the prompts it sent to the model."""
    chat_module = owui_module("open_webui.utils.chat")
    model_call = AsyncMock(return_value=NO_OPERATIONS)
    with (
        patch.object(memory_module.Config, "get_many", AsyncMock(return_value=REVIEW_ENABLED)),
        patch.object(chat_module, "generate_chat_completion", model_call),
    ):
        await memory_module.review_memory_after_turn(
            request=Request({"type": "http"}),
            user=user,
            model={"id": "m"},
            metadata={"features": {"memory": memory_feature}},
            form_data={},
            assistant_message=structured_reply("I moved to Vienna last year."),
            messages=[{"role": "user", "content": "where do I live now?"}],
        )
        # The review runs as a background task; let it finish before reading the calls.
        spawned = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        await asyncio.wait_for(asyncio.gather(*spawned), timeout=10)
    return [
        call.kwargs["form_data"]["messages"][-1]["content"] for call in model_call.await_args_list
    ]


@pytest.mark.asyncio
async def test_a_structured_reply_is_reviewed_with_its_text(memory_module, owui_module, reviewer):
    prompts = await review_prompts(memory_module, owui_module, reviewer, memory_feature=True)

    assert len(prompts) == 1, "the memory review was skipped for a reply stored only as output"
    assert "I moved to Vienna last year." in prompts[0]


@pytest.mark.asyncio
async def test_the_review_still_needs_the_memory_feature(memory_module, owui_module, reviewer):
    prompts = await review_prompts(memory_module, owui_module, reviewer, memory_feature=False)

    assert prompts == []


def test_an_orphaned_structured_turn_keeps_its_text(misc_module):
    orphaned = {
        **structured_reply("Here is what I found."),
        "tool_calls": [{"id": "call_orphan", "type": "function", "function": {"name": "s"}}],
    }

    reconciled = misc_module.reconcile_tool_pairs(
        messages=[{"role": "user", "content": "hi"}, orphaned]
    )

    assistants = [message for message in reconciled if message["role"] == "assistant"]
    assert len(assistants) == 1, "the structured-output turn was dropped as empty"
    assert "tool_calls" not in assistants[0]
    assert misc_module.get_content_from_message(assistants[0]) == "Here is what I found."


def test_unflattened_tool_images_stay_in_the_tool_message(misc_module):
    output = [
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "render",
            "arguments": "{}",
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": [
                {"type": "input_text", "text": "chart rendered"},
                {"type": "input_image", "image_url": IMAGE_URL},
            ],
        },
    ]

    messages = misc_module.convert_output_to_messages(output=output)

    assert [message["role"] for message in messages] == ["assistant", "tool"]
    assert messages[1]["content"] == [
        {"type": "input_text", "text": "chart rendered"},
        {"type": "input_image", "image_url": IMAGE_URL},
    ]


@pytest.mark.parametrize("value", [None, 42])
def test_sanitizing_leaves_non_text_alone(misc_module, value):
    assert misc_module.sanitize_text_for_db(value) == value


def test_a_structure_json_cannot_serialize_is_still_cleaned(misc_module):
    cleaned = misc_module.sanitize_data_for_db({"fn": object, "text": f"x{NUL}y"})

    assert cleaned["text"] == "xy"
