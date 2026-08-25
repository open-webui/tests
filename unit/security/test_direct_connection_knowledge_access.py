"""Regression: a direct-connection request could name any knowledge it liked.

open-webui 0.11.0 fix `305880f2e` (#26723): on a direct connection the model
object is not looked up server-side, it arrives in the request body as
`model_item`. `/api/chat/completions`, `/api/chat/completed` and
`/api/chat/actions/{id}` copied it straight onto `request.state.model`, and the
chat pipeline then read `model["info"]["meta"]["knowledge"]` and retrieved every
collection, file and note listed there. A crafted `model_item` therefore pulled
documents the caller had no read access to. The fix routes all three entrypoints
through `_set_direct_model`, which runs the claimed knowledge through
`get_accessible_folder_files` before anything downstream sees it.

Discriminates: passes on v0.11.0, fails on v0.10.2 (unfiltered `model_item`
reaches the pipeline, so unauthorized ids are still queued for retrieval).
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

CALLER = SimpleNamespace(id="alice", role="user", name="alice")

READABLE_COLLECTION = "collection-alice-can-read"
FOREIGN_COLLECTION = "collection-owned-by-bob"
FOREIGN_FILE = "file-owned-by-bob"
FOREIGN_NOTE = "note-owned-by-bob"


class _FakeRequest:
    def __init__(self):
        self.state = SimpleNamespace()
        self.app = SimpleNamespace(state=SimpleNamespace(MODELS={}))
        self.headers = {}


def _direct_model_item(knowledge: list[dict] | None) -> dict:
    return {
        "id": "direct-model",
        "name": "Direct Model",
        "direct": True,
        "info": {"meta": {"knowledge": knowledge} if knowledge is not None else {}},
    }


def _patch_access_boundary(
    readable_collection_ids: set[str],
    owned_file_ids: set[str],
    owned_note_ids: set[str] = frozenset(),
):
    """Stub the DB model classes the access filter reads. Nothing is readable
    unless listed, so any id that survives got there without a check."""
    import open_webui.models.files as files_model
    import open_webui.models.groups as groups_model
    import open_webui.models.knowledge as knowledge_model
    import open_webui.models.notes as notes_model

    # 0.11.1 (#28810) resolves group membership once and passes it down as user_group_ids.
    async def check_collection(id, user_id, permission="read", db=None, user_group_ids=None):
        return id in readable_collection_ids

    async def get_file(id, db=None):
        if id in owned_file_ids:
            return SimpleNamespace(id=id, user_id=CALLER.id, meta={})
        return None

    async def get_note(id, db=None):
        if id in owned_note_ids:
            return SimpleNamespace(id=id, user_id=CALLER.id)
        return None

    return (
        patch.object(groups_model.Groups, "get_groups_by_member_id", AsyncMock(return_value=[])),
        patch.object(knowledge_model.Knowledges, "check_access_by_user_id", check_collection),
        patch.object(files_model.Files, "get_file_by_id", get_file),
        patch.object(notes_model.Notes, "get_note_by_id", get_note),
    )


def _knowledge_recorder():
    """Captures the knowledge ids the chat pipeline is handed."""
    seen: list[str] = []

    async def handler(request, *args, **kwargs):
        model = getattr(request.state, "model", None) or {}
        for item in (model.get("info", {}).get("meta", {}) or {}).get("knowledge") or []:
            seen.append(item.get("id"))
        return {"ok": True}

    return seen, handler


async def _run_completed(
    main_module, model_item, readable=frozenset(), owned_files=frozenset(), owned_notes=frozenset()
):
    seen, handler = _knowledge_recorder()
    request = _FakeRequest()
    groups, knowledge, files, notes = _patch_access_boundary(
        set(readable), set(owned_files), set(owned_notes)
    )
    with groups, knowledge, files, notes:
        with patch.object(main_module, "chat_completed_handler", handler):
            await main_module.chat_completed(request, {"model_item": model_item}, CALLER)
    return request, seen


async def _run_action(main_module, model_item, readable=frozenset()):
    seen, handler = _knowledge_recorder()

    async def action_handler(request, action_id, *args, **kwargs):
        return await handler(request)

    request = _FakeRequest()
    groups, knowledge, files, notes = _patch_access_boundary(set(readable), set())
    with groups, knowledge, files, notes:
        with patch.object(main_module, "chat_action_handler", action_handler):
            await main_module.chat_action(
                request, "some-action", {"model_item": model_item}, CALLER
            )
    return request, seen


# --- Narrow: the claimed-knowledge escalation itself -------------------------


@pytest.mark.asyncio
async def test_claimed_knowledge_is_filtered_on_chat_completed(owui_module):
    main_module = owui_module("open_webui.main")
    model_item = _direct_model_item(
        [
            {"type": "collection", "id": FOREIGN_COLLECTION, "name": "Bob's KB"},
            {"type": "file", "id": FOREIGN_FILE, "name": "bob-salaries.pdf"},
            {"type": "collection", "id": READABLE_COLLECTION, "name": "Shared KB"},
        ]
    )

    _, retrieved_ids = await _run_completed(main_module, model_item, readable={READABLE_COLLECTION})

    assert retrieved_ids == [READABLE_COLLECTION], (
        "a browser-supplied direct-connection model named knowledge the caller "
        f"cannot read and it still reached retrieval ({retrieved_ids}): #26723 is back"
    )


@pytest.mark.asyncio
async def test_claimed_knowledge_is_filtered_on_chat_action(owui_module):
    main_module = owui_module("open_webui.main")
    model_item = _direct_model_item(
        [
            {"type": "collection", "id": FOREIGN_COLLECTION, "name": "Bob's KB"},
            {"type": "collection", "id": READABLE_COLLECTION, "name": "Shared KB"},
        ]
    )

    _, retrieved_ids = await _run_action(main_module, model_item, readable={READABLE_COLLECTION})

    assert retrieved_ids == [READABLE_COLLECTION], (
        "the chat action route accepted unauthorized knowledge off the client's "
        f"model object ({retrieved_ids}): #26723 is back"
    )


# --- Broad: nothing off the client-supplied model is an authorization decision


@pytest.mark.asyncio
async def test_no_claimed_entry_type_bypasses_the_access_check(owui_module):
    """Files, collections and notes are all attachable, so all three must be
    checked. An entry type nobody thought about must be dropped, not trusted."""
    main_module = owui_module("open_webui.main")
    model_item = _direct_model_item(
        [
            {"type": "collection", "id": FOREIGN_COLLECTION},
            {"type": "file", "id": FOREIGN_FILE},
            {"type": "note", "id": FOREIGN_NOTE},
            {"type": "something-new", "id": "unclassified-entry"},
        ]
    )

    _, retrieved_ids = await _run_completed(main_module, model_item)

    assert retrieved_ids == [], (
        "at least one claimed knowledge entry reached retrieval without an "
        f"access check ({retrieved_ids}), #26723"
    )


def test_every_direct_connection_entrypoint_uses_the_shared_filter(owui_module):
    """The chokepoint: any route that trusts `model_item` must go through
    `_set_direct_model`, otherwise it reopens the hole on its own."""
    main_module = owui_module("open_webui.main")
    tree = ast.parse(Path(main_module.__file__).read_text(encoding="utf-8"))

    assigning_functions = set()
    filtering_functions = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Assign) and any(
                isinstance(target, ast.Attribute)
                and target.attr == "model"
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "state"
                for target in inner.targets
            ):
                assigning_functions.add(node.name)
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "_set_direct_model"
            ):
                filtering_functions.add(node.name)

    assert assigning_functions == {"_set_direct_model"}, (
        "a route copies the browser-supplied model onto request.state without "
        f"filtering its claimed knowledge, {sorted(assigning_functions)} (#26723)"
    )
    assert {"chat_completion", "chat_completed", "chat_action"} <= filtering_functions, (
        "a direct-connection entrypoint stopped routing through the shared "
        f"access filter; only {sorted(filtering_functions)} do (#26723)"
    )


# --- Nearby: the fix must not break direct connections that are legitimate ---


@pytest.mark.asyncio
async def test_readable_knowledge_still_reaches_retrieval(owui_module):
    main_module = owui_module("open_webui.main")
    model_item = _direct_model_item(
        [
            {"type": "collection", "id": READABLE_COLLECTION},
            {"type": "file", "id": "file-alice-owns"},
            {"type": "note", "id": "note-alice-owns"},
        ]
    )

    _, retrieved_ids = await _run_completed(
        main_module,
        model_item,
        readable={READABLE_COLLECTION},
        owned_files={"file-alice-owns"},
        owned_notes={"note-alice-owns"},
    )

    assert retrieved_ids == [READABLE_COLLECTION, "file-alice-owns", "note-alice-owns"]


@pytest.mark.asyncio
async def test_direct_request_without_knowledge_is_accepted(owui_module):
    main_module = owui_module("open_webui.main")

    request, retrieved_ids = await _run_completed(main_module, _direct_model_item(None))

    assert retrieved_ids == []
    assert request.state.direct is True
    assert request.state.model["id"] == "direct-model"


@pytest.mark.asyncio
async def test_non_direct_request_never_sets_a_client_model(owui_module):
    main_module = owui_module("open_webui.main")
    model_item = {
        "id": "workspace-model",
        "info": {"meta": {"knowledge": [{"type": "collection", "id": FOREIGN_COLLECTION}]}},
    }

    request, retrieved_ids = await _run_completed(main_module, model_item)

    assert retrieved_ids == []
    assert not hasattr(request.state, "model"), (
        "a request that never claimed a direct connection still installed the "
        "client's model object"
    )
