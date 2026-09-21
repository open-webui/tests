"""Regression: memory tool replies and memory rows carried the whole model entry.

open-webui 0.11.4 commit `e9a0164690` (a `refac`): the `/memories/update` route
stamped each new memory's meta with `metadata.get('model')`, which on the chat
path is the full resolved model dict (id, name, info, access control, params and
everything else a workspace model carries), and both the update response and
`/memories/path` serialized memories with a plain `model_dump()`, handing the
model the stored `meta` of every memory too. The fix records only
`model['id']` in the meta and serializes memory rows with
`model_dump(exclude={'meta'})`, so what crosses the model boundary is the id and
the memory's own fields.

Discriminates: passes on 344ea5306 (0.11.4), fails on e9a0164690^ (the stored
meta is the whole model dict and every returned memory leaks its meta).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def memories_router(owui_module):
    return owui_module("open_webui.routers.memories")


@pytest.fixture(scope="module")
def user():
    return SimpleNamespace(id="alice", role="user")


def _request(memories_router, metadata: dict) -> SimpleNamespace:
    embedding = AsyncMock(return_value=[0.1])
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(EMBEDDING_FUNCTION=embedding)),
        state=SimpleNamespace(metadata=metadata),
    )


def _stored_memory(memories_router, memory_id="mem-1"):
    return memories_router.MemoryModel(
        id=memory_id,
        user_id="alice",
        content="prefers tea over coffee",
        path="work/preferences",
        meta={"created_by": "tool", "chat_id": "c-1", "model": "llama-3"},
        updated_at=1,
        created_at=1,
    )


async def _run_update(memories_router, request, user):
    form = memories_router.UpdateMemoriesForm(
        operations=[memories_router.MemoryOperationModel(action="add", content="prefers tea")],
        source="tool",
    )
    created = _stored_memory(memories_router)

    async def fake_apply(user_id, operations):
        captured.append((user_id, operations))
        return [{"action": "add", "status": "created", "memory": created}]

    captured = []
    with (
        patch.object(memories_router.Memories, "apply_memory_operations", fake_apply),
        patch.object(memories_router, "publish_event", AsyncMock()),
    ):
        response = await memories_router.update_memories(request, form, user)
    return captured[0][1][0], response[0]


# --- narrow -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_meta_records_the_model_id_not_the_model_entry(
    memories_router, user
):
    """The chat path resolves `metadata['model']` to the whole model dict; only the id is stored."""
    model_entry = {"id": "llama-3", "name": "Llama 3", "info": {"meta": {"hidden": "secret"}}}
    request = _request(
        memories_router, {"chat_id": "c-1", "message_id": "m-1", "model": model_entry}
    )

    operation, _ = await _run_update(memories_router, request, user)

    assert operation["meta"]["model"] == "llama-3", (
        "the memory row stored the full model entry (name, info, params) instead of "
        "the model id, so every later read of that row carried it (e9a0164690)"
    )


@pytest.mark.asyncio
async def test_update_response_serializes_memories_without_their_meta(
    memories_router, user
):
    """The memory tool reads this response; the stored meta is not the model's business."""
    request = _request(
        memories_router, {"chat_id": "c-1", "message_id": "m-1", "model": {"id": "llama-3"}}
    )

    _, result = await _run_update(memories_router, request, user)

    assert "meta" not in result["memory"], (
        "the update response handed the model each memory's stored metadata, including "
        "the chat and message ids of whatever turn wrote it (e9a0164690)"
    )


@pytest.mark.asyncio
async def test_read_memory_path_drops_the_meta_from_every_row(memories_router, user):
    """`/memories/path` feeds the same tool surface; the exclusion must hold there too."""
    memory = _stored_memory(memories_router)
    with patch.object(
        memories_router.Memories, "get_memories_by_user_id", AsyncMock(return_value=[memory])
    ):
        result = await memories_router.read_memory_path(
            memories_router.ReadMemoryPathForm(path="work/preferences"), user
        )

    serialized = result["memories"][0]
    assert "meta" not in serialized, (
        "the path listing serialized each memory with its stored meta, which on this "
        "path is what the read_memory_path tool passes to the model (e9a0164690)"
    )


# --- nearby -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_dict_model_metadata_records_no_model(memories_router, user):
    """A metadata dict without a model entry stores None, matching the pre-fix shape."""
    request = _request(memories_router, {"chat_id": "c-1", "message_id": "m-1"})

    operation, _ = await _run_update(memories_router, request, user)

    assert operation["meta"]["model"] is None
    assert operation["meta"]["chat_id"] == "c-1"
    assert operation["meta"]["message_id"] == "m-1"
    assert operation["meta"]["created_by"] == "tool"


@pytest.mark.asyncio
async def test_background_review_source_is_kept_in_the_meta(memories_router, user):
    """Nearby: the source discriminator travels in meta and must survive the fix."""
    form = memories_router.UpdateMemoriesForm(
        operations=[memories_router.MemoryOperationModel(action="add", content="prefers tea")],
        source="background_review",
    )
    request = _request(memories_router, {})
    created = _stored_memory(memories_router)
    captured = []

    async def fake_apply(user_id, operations):
        captured.append(operations)
        return [{"action": "add", "status": "created", "memory": created}]

    with (
        patch.object(memories_router.Memories, "apply_memory_operations", fake_apply),
        patch.object(memories_router, "publish_event", AsyncMock()),
    ):
        await memories_router.update_memories(request, form, user)

    assert captured[0][0]["meta"]["created_by"] == "background_review"


@pytest.mark.asyncio
async def test_serialized_memory_keeps_its_own_fields(memories_router, user):
    """Nearby: excluding meta must not strip the fields the tool reads."""
    request = _request(memories_router, {"model": {"id": "llama-3"}})
    _, result = await _run_update(memories_router, request, user)

    memory = result["memory"]
    assert memory["id"] == "mem-1"
    assert memory["content"] == "prefers tea over coffee"
    assert memory["path"] == "work/preferences"
    assert memory["user_id"] == "alice"
