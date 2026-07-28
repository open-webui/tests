"""Regression: knowledge-base sync cleanup must only delete things that belong
to the knowledge base named in the URL.

open-webui 0.11.0 fix `707efeaed` (#26722): `POST /knowledge/{id}/sync/cleanup`
verified write access to the knowledge base in the URL, then acted on the
caller-supplied `file_ids` and `dir_ids` without checking they belong to it.
Anyone with write access to one knowledge base could pass another knowledge
base's directory id to delete its directory subtree, or another knowledge
base's file id to drop that file's `file-{file_id}` vector collection and its
stored blob. The fix skips directories whose `knowledge_id` differs from the
URL id, and gates the per-file cleanup on `Knowledges.has_file(id, file_id)`.

The tests capture every destructive call at the model / vector-client boundary
and assert on the exact ids passed, because the bug was never a crash: the
calls succeeded, they just hit the wrong knowledge base.

Discriminates: passes on v0.11.0, fails on v0.10.2 (a foreign directory id and
a foreign file id are deleted instead of skipped).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.regression

TARGET_KB = "kb-alice"
FOREIGN_KB = "kb-bob"

ALICE = SimpleNamespace(id="alice", role="user")


class DeletionLog:
    """Every destructive call the router makes, in order."""

    def __init__(self):
        self.unlinked_files = []  # (knowledge_id, file_id)
        self.vector_deletes = []  # (collection_name, filter)
        self.dropped_collections = []
        self.deleted_file_rows = []
        self.deleted_directories = []
        self.deleted_storage_paths = []
        self.moved_files = []  # (knowledge_id, file_id, directory_id)
        self.updated_directories = []


def _directory(id: str, knowledge_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=id, knowledge_id=knowledge_id, parent_id=None, name=id, user_id="alice"
    )


def _file(id: str) -> SimpleNamespace:
    # user_id is alice everywhere so the owner check can never be what saves a
    # foreign file from deletion.
    return SimpleNamespace(
        id=id,
        user_id="alice",
        hash=f"hash-{id}",
        path=f"uploads/{id}",
        filename=f"{id}.txt",
        meta={},
    )


class World:
    """Two knowledge bases, one directory and one file each, wired into the
    router's model layer so the real endpoint code runs unmodified."""

    def __init__(self, router_module):
        self.router = router_module
        self.log = DeletionLog()
        self.knowledge_bases = {
            TARGET_KB: SimpleNamespace(id=TARGET_KB, user_id="alice", meta={}),
            FOREIGN_KB: SimpleNamespace(id=FOREIGN_KB, user_id="bob", meta={}),
        }
        self.directories = {
            "dir-alice": _directory("dir-alice", TARGET_KB),
            "dir-bob": _directory("dir-bob", FOREIGN_KB),
        }
        self.files = {"file-alice": _file("file-alice"), "file-bob": _file("file-bob")}
        self.memberships = {(TARGET_KB, "file-alice"), (FOREIGN_KB, "file-bob")}

    async def get_knowledge_by_id(self, id=None, db=None):
        return self.knowledge_bases.get(id)

    async def get_file_by_id(self, id=None, db=None):
        return self.files.get(id)

    async def has_file(self, knowledge_id, file_id, db=None):
        return (knowledge_id, file_id) in self.memberships

    async def remove_file_from_knowledge_by_id(self, knowledge_id, file_id, db=None):
        self.log.unlinked_files.append((knowledge_id, file_id))
        return True

    async def get_directory_by_id(self, directory_id, db=None):
        return self.directories.get(directory_id)

    async def delete_directory(self, directory_id, move_files_to_parent=True, db=None):
        self.log.deleted_directories.append(directory_id)
        return True

    async def update_directory(self, directory_id, name=None, parent_id="__unset__", db=None):
        self.log.updated_directories.append(directory_id)
        return self.directories.get(directory_id)

    async def move_file_to_directory(self, knowledge_id, file_id, directory_id=None, db=None):
        self.log.moved_files.append((knowledge_id, file_id, directory_id))
        return True

    async def delete_file_by_id(self, id, db=None):
        self.log.deleted_file_rows.append(id)
        return True

    def delete_storage_file(self, path):
        self.log.deleted_storage_paths.append(path)

    async def vector_delete(self, collection_name=None, filter=None, **kwargs):
        self.log.vector_deletes.append((collection_name, filter))

    async def vector_has_collection(self, collection_name=None, **kwargs):
        return True

    async def vector_delete_collection(self, collection_name=None, **kwargs):
        self.log.dropped_collections.append(collection_name)


@pytest.fixture
def world(owui_module):
    router_module = owui_module("open_webui.routers.knowledge")
    fixture = World(router_module)
    vector_client = SimpleNamespace(
        delete=fixture.vector_delete,
        has_collection=fixture.vector_has_collection,
        delete_collection=fixture.vector_delete_collection,
    )
    patches = [
        patch.object(router_module.Knowledges, "get_knowledge_by_id", fixture.get_knowledge_by_id),
        patch.object(router_module.Knowledges, "has_file", fixture.has_file),
        patch.object(
            router_module.Knowledges,
            "remove_file_from_knowledge_by_id",
            fixture.remove_file_from_knowledge_by_id,
        ),
        patch.object(router_module.Knowledges, "get_directory_by_id", fixture.get_directory_by_id),
        patch.object(router_module.Knowledges, "delete_directory", fixture.delete_directory),
        patch.object(router_module.Knowledges, "update_directory", fixture.update_directory),
        patch.object(
            router_module.Knowledges, "move_file_to_directory", fixture.move_file_to_directory
        ),
        patch.object(
            router_module.Knowledges, "get_file_metadatas_by_id", AsyncMock(return_value=[])
        ),
        patch.object(router_module.Files, "get_file_by_id", fixture.get_file_by_id),
        patch.object(router_module.Files, "delete_file_by_id", fixture.delete_file_by_id),
        patch.object(router_module.Storage, "delete_file", fixture.delete_storage_file),
        patch.object(router_module.AccessGrants, "has_access", AsyncMock(return_value=False)),
        patch.object(router_module, "ASYNC_VECTOR_DB_CLIENT", vector_client),
        patch.object(router_module, "publish_event", AsyncMock()),
    ]
    for p in patches:
        p.start()
    try:
        yield fixture
    finally:
        for p in reversed(patches):
            p.stop()


async def _run_cleanup(world, file_ids=(), dir_ids=(), knowledge_id=TARGET_KB, user=ALICE):
    form = world.router.SyncCleanupForm(file_ids=list(file_ids), dir_ids=list(dir_ids))
    return await world.router.sync_knowledge_cleanup(
        id=knowledge_id, form_data=form, user=user, db=None
    )


# ── Narrow: the bug itself ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_foreign_directory_is_not_deleted(world):
    await _run_cleanup(world, dir_ids=["dir-bob"])
    assert world.log.deleted_directories == [], (
        "cleanup of kb-alice deleted directory 'dir-bob', which belongs to kb-bob: "
        "write access to one knowledge base destroys another's folders (#26722)"
    )


@pytest.mark.asyncio
async def test_foreign_file_is_not_unlinked_or_purged(world):
    await _run_cleanup(world, file_ids=["file-bob"])
    assert world.log.dropped_collections == [], (
        "cleanup of kb-alice dropped the vector collection of 'file-bob', which "
        "belongs to kb-bob: another knowledge base loses its search data (#26722)"
    )
    assert world.log.unlinked_files == [], (
        "cleanup of kb-alice unlinked a file it does not contain (#26722)"
    )
    assert world.log.deleted_file_rows == [], (
        "cleanup of kb-alice deleted the file row of 'file-bob' (#26722)"
    )
    assert world.log.deleted_storage_paths == [], (
        "cleanup of kb-alice deleted the stored blob of 'file-bob' (#26722)"
    )


@pytest.mark.asyncio
async def test_mixed_request_deletes_only_the_target_knowledge_base_ids(world):
    """The exact-id assertion: A's own entries go, B's stay, in one request."""
    await _run_cleanup(
        world,
        file_ids=["file-alice", "file-bob"],
        dir_ids=["dir-alice", "dir-bob"],
    )
    assert world.log.deleted_directories == ["dir-alice"], (
        f"expected only kb-alice's directory to be deleted, got "
        f"{world.log.deleted_directories} (#26722)"
    )
    assert world.log.dropped_collections == ["file-file-alice"], (
        f"expected only kb-alice's file collection to be dropped, got "
        f"{world.log.dropped_collections} (#26722)"
    )
    assert world.log.unlinked_files == [(TARGET_KB, "file-alice")], (
        f"expected only kb-alice's file to be unlinked, got {world.log.unlinked_files} (#26722)"
    )
    assert world.log.deleted_file_rows == ["file-alice"], (
        f"expected only kb-alice's file row to be deleted, got "
        f"{world.log.deleted_file_rows} (#26722)"
    )
    assert all(collection == TARGET_KB for collection, _ in world.log.vector_deletes), (
        f"vector deletes escaped the target collection: {world.log.vector_deletes} (#26722)"
    )


# ── Broad: every destructive endpoint in the router carries the scope ───────


@pytest.mark.asyncio
async def test_delete_directory_endpoint_refuses_foreign_directory(world):
    with pytest.raises(HTTPException) as raised:
        await world.router.delete_knowledge_directory(
            request=None, id=TARGET_KB, dir_id="dir-bob", user=ALICE, db=None
        )
    assert raised.value.status_code == 404
    assert world.log.deleted_directories == [], (
        "the explicit directory-delete endpoint deleted another knowledge base's directory"
    )


@pytest.mark.asyncio
async def test_update_directory_endpoint_refuses_foreign_directory(world):
    form = world.router.KnowledgeDirectoryUpdateForm(name="renamed")
    with pytest.raises(HTTPException) as raised:
        await world.router.update_knowledge_directory(
            request=None, id=TARGET_KB, dir_id="dir-bob", form_data=form, user=ALICE, db=None
        )
    assert raised.value.status_code == 404
    assert world.log.updated_directories == [], (
        "the directory-update endpoint mutated another knowledge base's directory"
    )


@pytest.mark.asyncio
async def test_remove_file_endpoint_refuses_foreign_file(world):
    form = world.router.KnowledgeFileIdForm(file_id="file-bob")
    with pytest.raises(HTTPException):
        await world.router.remove_file_from_knowledge_by_id(
            request=None, id=TARGET_KB, form_data=form, user=ALICE, db=None
        )
    assert world.log.unlinked_files == []
    assert world.log.dropped_collections == []
    assert world.log.deleted_file_rows == [], (
        "the file-remove endpoint deleted a file belonging to another knowledge base"
    )


@pytest.mark.asyncio
async def test_move_file_endpoint_refuses_foreign_file(world):
    form = world.router.KnowledgeFileMoveForm(file_id="file-bob", directory_id="dir-alice")
    with pytest.raises(HTTPException) as raised:
        await world.router.move_file_in_knowledge(
            request=None, id=TARGET_KB, form_data=form, user=ALICE, db=None
        )
    assert raised.value.status_code == 404
    assert world.log.moved_files == [], (
        "the file-move endpoint pulled a file out of another knowledge base"
    )


@pytest.mark.asyncio
async def test_move_file_endpoint_refuses_foreign_target_directory(world):
    form = world.router.KnowledgeFileMoveForm(file_id="file-alice", directory_id="dir-bob")
    with pytest.raises(HTTPException) as raised:
        await world.router.move_file_in_knowledge(
            request=None, id=TARGET_KB, form_data=form, user=ALICE, db=None
        )
    assert raised.value.status_code == 404
    assert world.log.moved_files == [], (
        "the file-move endpoint moved a file into another knowledge base's directory"
    )


# ── Nearby: the cleanup still does its job ──────────────────────────────────


@pytest.mark.asyncio
async def test_own_file_is_fully_cleaned_up(world):
    result = await _run_cleanup(world, file_ids=["file-alice"])
    assert result == {"status": True}
    assert world.log.unlinked_files == [(TARGET_KB, "file-alice")]
    assert world.log.dropped_collections == ["file-file-alice"]
    assert world.log.deleted_file_rows == ["file-alice"]
    assert world.log.deleted_storage_paths == ["uploads/file-alice"]
    assert world.log.vector_deletes == [
        (TARGET_KB, {"file_id": "file-alice"}),
        (TARGET_KB, {"hash": "hash-file-alice"}),
    ]


@pytest.mark.asyncio
async def test_own_directory_is_deleted_without_moving_files_to_parent(world):
    await _run_cleanup(world, dir_ids=["dir-alice"])
    assert world.log.deleted_directories == ["dir-alice"]


@pytest.mark.asyncio
async def test_empty_cleanup_is_a_no_op(world):
    result = await _run_cleanup(world)
    assert result == {"status": True}
    assert vars(world.log) == vars(DeletionLog())


@pytest.mark.asyncio
async def test_unknown_file_id_is_skipped(world):
    await _run_cleanup(world, file_ids=["file-that-never-existed"])
    assert world.log.unlinked_files == []
    assert world.log.dropped_collections == []


@pytest.mark.asyncio
async def test_unknown_knowledge_base_is_refused_before_any_deletion(world):
    with pytest.raises(HTTPException) as raised:
        await _run_cleanup(
            world,
            file_ids=["file-alice"],
            dir_ids=["dir-alice"],
            knowledge_id="kb-that-never-existed",
        )
    assert raised.value.status_code == 404
    assert vars(world.log) == vars(DeletionLog()), (
        "a cleanup naming a nonexistent knowledge base still deleted things"
    )


@pytest.mark.asyncio
async def test_user_without_write_access_is_refused(world):
    stranger = SimpleNamespace(id="mallory", role="user")
    with pytest.raises(HTTPException) as raised:
        await _run_cleanup(world, dir_ids=["dir-alice"], user=stranger)
    assert raised.value.status_code == 403
    assert world.log.deleted_directories == []
