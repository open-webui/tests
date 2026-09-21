"""Regression: a file's knowledge-base access follows its actual attachment alone.

open-webui 0.11.4 fix `fbc489726` (#29937): `has_access_to_file` in
`open_webui/utils/access_control/files.py` granted access through a
`collection_name` value left behind on the file's `meta`, without the file
being attached to that knowledge base. The value is a leftover from older
uploads and is not an attachment: a user who could open the knowledge base
named there gained read, write and delete on a file they had no other route
to, and a revoked attachment still conferred access for as long as the stale
value sat on the record. The fix removes the whole branch; the association
table (`get_knowledges_by_file_id`) is now the only knowledge route.

The tests stub the model boundary and give the stale `collection_name`
every chance to grant access (an existing, openable knowledge base owned by
the file's owner, so the ownership side-gate cannot save the fix either).

Discriminates: passes on v0.11.4, fails on v0.11.3 (the stale meta value
grants read, and write through the owner-of-the-knowledge-base clause).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("fastapi")

pytestmark = pytest.mark.regression

OWNER = "alice"
VIEWER = "mallory"

KB_ID = "kb-stale"
FILE_ID = "file-1"


def _user(user_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=user_id, role="user")


def _file(user_id: str, meta: dict | None) -> SimpleNamespace:
    return SimpleNamespace(id=FILE_ID, user_id=user_id, meta=meta)


def _knowledge(kb_id: str, owner_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=kb_id, user_id=owner_id)


def _unrelated_route_checks():
    """Deny on every other route, so only the stale meta value can decide."""
    return {
        "knowledges_by_file": AsyncMock(return_value=[]),
        "channels": AsyncMock(return_value=[]),
        "shared_chat_ids": AsyncMock(return_value=[]),
        "model_owners": AsyncMock(return_value={}),
    }


@pytest.fixture
def boundary(owui_module):
    """Patch the model layer under `has_access_to_file` with denials everywhere."""
    module = owui_module("open_webui.utils.access_control.files")
    knowledge_model = owui_module("open_webui.models.knowledge")
    groups_model = owui_module("open_webui.models.groups")
    access_grants = owui_module("open_webui.models.access_grants")
    files_model = owui_module("open_webui.models.files")
    channels_model = owui_module("open_webui.models.channels")
    chats_model = owui_module("open_webui.models.chats")
    models_model = owui_module("open_webui.models.models")

    checks = _unrelated_route_checks()
    knowledge_base = _knowledge(KB_ID, OWNER)
    patches = [
        patch.object(files_model.Files, "get_file_by_id", AsyncMock(return_value=None)),
        patch.object(
            knowledge_model.Knowledges,
            "get_knowledges_by_file_id",
            checks["knowledges_by_file"],
        ),
        patch.object(
            knowledge_model.Knowledges,
            "get_knowledge_by_id",
            AsyncMock(return_value=knowledge_base),
        ),
        patch.object(
            groups_model.Groups, "get_groups_by_member_id", AsyncMock(return_value=[])
        ),
        patch.object(
            access_grants.AccessGrants,
            "has_access",
            AsyncMock(return_value=True),
        ),
        patch.object(
            channels_model.Channels,
            "get_channels_by_file_id_and_user_id",
            checks["channels"],
        ),
        patch.object(
            chats_model.Chats,
            "get_shared_chat_ids_by_file_id",
            checks["shared_chat_ids"],
        ),
        patch.object(
            models_model.Models,
            "get_model_owner_ids_by_file_id",
            checks["model_owners"],
        ),
    ]
    for p in patches:
        p.start()
    try:
        yield module, checks
    finally:
        for p in reversed(patches):
            p.stop()


async def _has_access(boundary, file_meta, access_type, viewer=VIEWER):
    module, _ = boundary
    file = _file(OWNER, file_meta)
    with patch.object(
        module.Files, "get_file_by_id", AsyncMock(return_value=file)
    ):
        return await module.has_access_to_file(FILE_ID, access_type, _user(viewer))


# ── Narrow: the stale meta value must not grant access ──────────────────────


@pytest.mark.asyncio
async def test_stale_collection_name_does_not_grant_read_access(boundary):
    """The exact bug: `meta.collection_name` points at an existing, openable
    knowledge base, but the file is not attached to it."""
    assert await _has_access(boundary, {"collection_name": KB_ID}, "read") is False, (
        "a leftover meta.collection_name granted read access to a file that is "
        "not attached to the knowledge base it names (#29937)"
    )


@pytest.mark.asyncio
async def test_stale_collection_name_does_not_grant_write_access(boundary):
    """Write is the sharper edge: the stale value plus an openable knowledge
    base owned by the file's owner let a viewer mutate or delete the file."""
    assert await _has_access(boundary, {"collection_name": KB_ID}, "write") is False, (
        "a leftover meta.collection_name granted write access to a file that is "
        "not attached to the knowledge base it names (#29937)"
    )


# ── Narrow: an actual attachment still grants access ────────────────────────


@pytest.mark.asyncio
async def test_real_attachment_still_grants_read_access(boundary):
    module, checks = boundary
    checks["knowledges_by_file"].return_value = [_knowledge(KB_ID, OWNER)]

    assert await _has_access(boundary, {"collection_name": KB_ID}, "read") is True, (
        "removing the meta branch must not break access through a real attachment"
    )


@pytest.mark.asyncio
async def test_real_attachment_still_grants_write_access(boundary):
    module, checks = boundary
    checks["knowledges_by_file"].return_value = [_knowledge(KB_ID, OWNER)]

    assert await _has_access(boundary, {"collection_name": KB_ID}, "write") is True, (
        "the attachment route still confers write when the knowledge base's "
        "owner owns the file"
    )


# ── Broad: no meta residue of any shape opens a route ───────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("meta", [None, {}, {"collection_name": KB_ID}])
@pytest.mark.parametrize("access_type", ["read", "write"])
async def test_no_meta_shape_grants_access_without_an_attachment(
    boundary, meta, access_type
):
    assert await _has_access(boundary, meta, access_type) is False, (
        f"meta={meta!r} opened a {access_type} route to an unattached file"
    )


# ── Nearby: the owner check ahead of the meta branch is untouched ───────────


@pytest.mark.asyncio
async def test_file_owner_keeps_access_regardless_of_meta(boundary):
    assert await _has_access(boundary, {"collection_name": KB_ID}, "read", viewer=OWNER) is True
    assert await _has_access(boundary, None, "write", viewer=OWNER) is True


@pytest.mark.asyncio
async def test_missing_file_is_denied(boundary):
    module, _ = boundary
    with patch.object(module.Files, "get_file_by_id", AsyncMock(return_value=None)):
        assert (
            await module.has_access_to_file("file-that-never-existed", "read", _user(VIEWER))
            is False
        )
